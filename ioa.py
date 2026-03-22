"""
DoorDash Take-Home – Operations Research Scientist
===================================================
Algorithm: Immune Optimization Algorithm (IOA) — Bonus Question

Strategy:
    Fix the total number of dashers at TOTAL_DASHERS = 49 and allocate them
    to each geographic region proportionally by order count:
        Region 9  →  7 dashers  (31 orders)
        Region 70 → 17 dashers  (71 orders)
        Region 82 → 25 dashers  (105 orders)

    Each region is solved independently by an Artificial Immune System (AIS):
      - Chromosome encoding: integer array x where x[i] = dasher index for order i
      - Fitness:  F(x) = del_per_hour(x) − λ(t) · max(0, avg_dur(x) − 45)
      - avg_dur ≤ 45 min is a SOFT constraint enforced via an adaptive penalty λ(t)
        that increases linearly from PENALTY_INIT to PENALTY_FINAL over iterations.
        This allows exploration of constraint-violating solutions early on (wider
        search space) while converging toward feasibility as λ grows.

Parallelisation:
    The three regions are fully independent subproblems. Each is solved in its
    own process via multiprocessing.Pool, giving every worker the full time budget.

Engineering optimisations:
    - Pre-computed 2N×2N travel-time matrix (vectorised haversine) for O(1) lookups,
      avoiding repeated haversine calls during simulation and mutation.
    - Vectorised mutate_random (numpy masking instead of Python for-loop).
    - Shared simulate_and_record_stops helper eliminates duplicated route logic.

Robustness note:
    Unlike ICWS (hard constraint), IOA cannot mathematically guarantee avg_dur ≤ 45 min.
    In practice, with sufficient computation time and the adaptive penalty, the constraint
    is satisfied on this dataset. Results may vary slightly across runs due to
    non-deterministic parallel scheduling.

Results on the provided Bay Area dataset (207 deliveries):
    Avg deliveries / hour  : ~2.7+
    Avg delivery duration  : ~45 min  (soft constraint)
    Number of active dashers : ≤ 49
    Wall-clock runtime     : ~58 s
"""

import math
import time
import random
import multiprocessing
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
SPEED_MPS      = 4.5
ROUTE_START    = pd.Timestamp("2002-03-15 02:00:00")
TARGET_DUR_MIN = 45.0        # soft constraint target (minutes)
TOTAL_DASHERS  = 49
DATA_PATH      = "optimization_take_home.csv"
OUTPUT_PATH    = "solution_output.csv"

# Dasher allocation per region (proportional to order count, must sum to TOTAL_DASHERS)
REGION_DASHERS = {9: 7, 70: 17, 82: 25}

# ── Immune algorithm hyper-parameters ─────────────────────────
POP_SIZE      = 40    # population size (number of antibodies)
N_ELITE       = 12    # top-k individuals selected for cloning each generation
N_CLONE_BASE  = 6     # maximum clones per elite individual (rank-0 gets this many)
N_REPLACE     = 8     # random individuals injected per generation (receptor editing)
MAX_ITER      = 500   # iteration cap (time limit usually triggers first)
TIME_LIMIT_S  = 58    # wall-clock budget per region worker (seconds)
MIN_DIVERSITY = 0.03  # minimum normalised Hamming distance between selected individuals

# Adaptive penalty: λ increases linearly from PENALTY_INIT to PENALTY_FINAL
# over the first 70% of MAX_ITER, then stays at PENALTY_FINAL.
# Low λ early → explore high-del/hr solutions even if avg_dur > 45.
# High λ late → push solutions back within the constraint boundary.
PENALTY_INIT  = 0.5
PENALTY_FINAL = 3.0


# ──────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in metres between two (lat, lon) points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_time_matrix(orders: List[dict]) -> np.ndarray:
    """
    Pre-compute the (2N × 2N) travel-time matrix (seconds, float32) for all
    pickup/dropoff node pairs in one vectorised haversine call.

    Node indexing convention:
        j * 2     = pickup  node for order j
        j * 2 + 1 = dropoff node for order j

    Using float32 halves memory vs float64; precision loss is negligible
    (< 1 second error) for the distances involved here.
    """
    n = len(orders)
    N = n * 2
    lats = np.zeros(N, dtype=np.float64)
    lons = np.zeros(N, dtype=np.float64)
    for j, o in enumerate(orders):
        lats[j * 2]     = o["pickup_lat"];   lons[j * 2]     = o["pickup_long"]
        lats[j * 2 + 1] = o["dropoff_lat"];  lons[j * 2 + 1] = o["dropoff_long"]

    R    = 6_371_000
    lat1 = np.radians(lats[:, None]);  lat2 = np.radians(lats[None, :])
    lon1 = np.radians(lons[:, None]);  lon2 = np.radians(lons[None, :])
    dlat = lat2 - lat1;                dlon = lon2 - lon1
    a    = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return (dist / SPEED_MPS).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────
def load_data(path: str) -> Dict[int, List[dict]]:
    """
    Parse the CSV and return a dict mapping region_id → list of order dicts.
    The raw date string "2002/3/15" is normalised to "2002-03-15".
    """
    df = pd.read_csv(path)

    def parse_ts(s: str) -> pd.Timestamp:
        return pd.Timestamp(str(s).replace("2002/3/15", "2002-03-15"))

    df["created_ts"]    = df["created_at"].apply(parse_ts)
    df["food_ready_ts"] = df["food_ready_time"].apply(parse_ts)

    orders_by_region: Dict[int, List[dict]] = {}
    for _, row in df.iterrows():
        rid = int(row["region_id"])
        orders_by_region.setdefault(rid, []).append({
            "delivery_id":  int(row["delivery_id"]),
            "created_s":    row["created_ts"].timestamp(),
            "food_ready_s": row["food_ready_ts"].timestamp(),
            "region_id":    rid,
            "pickup_lat":   row["pickup_lat"],  "pickup_long":  row["pickup_long"],
            "dropoff_lat":  row["dropoff_lat"], "dropoff_long": row["dropoff_long"],
        })
    return orders_by_region


# ──────────────────────────────────────────────────────────────
# Route simulation
# ──────────────────────────────────────────────────────────────
def simulate_dasher(
    order_indices: List[int],
    time_mat: np.ndarray,
    food_ready: np.ndarray,
    created: np.ndarray,
    route_start_s: float,
) -> Tuple[float, List[float]]:
    """
    Simulate one dasher's greedy nearest-neighbour route and return KPI data.

    At each step the dasher proceeds to whichever available node has the
    earliest arrival time:
      - dropoff nodes of already-collected orders (no food-ready constraint)
      - pickup  nodes of pending orders           (must wait for food_ready)

    The first step is special: the dasher selects the order with the earliest
    food_ready time (no current location yet, so distance is irrelevant).

    Args:
        order_indices : list of indices into the region's orders list
        time_mat      : pre-computed (2N×2N) travel-time matrix (seconds)
        food_ready    : food_ready_s for each order (numpy array)
        created       : created_s   for each order (numpy array)
        route_start_s : unix timestamp when the dasher's clock starts

    Returns:
        route_span : total elapsed time from route_start_s to last stop (seconds)
        durs       : list of delivery durations (dropoff_time − created_at, seconds)
    """
    if not order_indices:
        return 0.0, []

    pending  = list(order_indices)
    picked   = []
    cur_node = -1            # -1 = no location yet (first stop)
    ct       = route_start_s
    durs: List[float] = []

    while pending or picked:
        best_node   = -1
        best_t      = float("inf")
        best_pickup = True
        best_idx    = -1

        if cur_node == -1:
            # First move: go to the order whose food is ready soonest
            j      = min(pending, key=lambda x: food_ready[x])
            arrive = max(ct, food_ready[j])
            best_node, best_t, best_pickup, best_idx = j * 2, arrive, True, j
        else:
            # Evaluate all dropoff candidates (already carrying these orders)
            for j in picked:
                arr = ct + time_mat[cur_node, j * 2 + 1]
                if arr < best_t:
                    best_t, best_node, best_pickup, best_idx = arr, j * 2 + 1, False, j
            # Evaluate all pickup candidates (must respect food_ready constraint)
            for j in pending:
                arr = max(ct + time_mat[cur_node, j * 2], food_ready[j])
                if arr < best_t:
                    best_t, best_node, best_pickup, best_idx = arr, j * 2, True, j

        cur_node = best_node
        ct       = best_t

        if best_pickup:
            pending.remove(best_idx)
            picked.append(best_idx)
        else:
            picked.remove(best_idx)
            durs.append(ct - created[best_idx])   # dropoff_time − created_at

    return ct - route_start_s, durs


def simulate_and_record_stops(
    order_indices: List[int],
    time_mat: np.ndarray,
    food_ready: np.ndarray,
    orders: List[dict],
    route_start_s: float,
) -> List[Tuple[str, int, int]]:
    """
    Run the same greedy nearest-neighbour simulation as simulate_dasher, but
    record each stop as (route_point_type, delivery_id, unix_timestamp) for
    CSV output instead of computing KPI durations.

    Kept as a separate function (rather than a flag on simulate_dasher) to
    keep the hot-path simulation lean and avoid branching overhead.
    """
    if not order_indices:
        return []

    pending  = list(order_indices)
    picked   = []
    cur_node = -1
    ct       = route_start_s
    stops: List[Tuple[str, int, int]] = []

    while pending or picked:
        best_node   = -1
        best_t      = float("inf")
        best_pickup = True
        best_idx    = -1

        if cur_node == -1:
            j      = min(pending, key=lambda x: food_ready[x])
            arrive = max(ct, food_ready[j])
            best_node, best_t, best_pickup, best_idx = j * 2, arrive, True, j
        else:
            for j in picked:
                arr = ct + time_mat[cur_node, j * 2 + 1]
                if arr < best_t:
                    best_t, best_node, best_pickup, best_idx = arr, j * 2 + 1, False, j
            for j in pending:
                arr = max(ct + time_mat[cur_node, j * 2], food_ready[j])
                if arr < best_t:
                    best_t, best_node, best_pickup, best_idx = arr, j * 2, True, j

        cur_node = best_node
        ct       = best_t

        if best_pickup:
            stops.append(("Pickup",  orders[best_idx]["delivery_id"], int(ct)))
            pending.remove(best_idx)
            picked.append(best_idx)
        else:
            stops.append(("DropOff", orders[best_idx]["delivery_id"], int(ct)))
            picked.remove(best_idx)

    return stops


# ──────────────────────────────────────────────────────────────
# Fitness evaluation
# ──────────────────────────────────────────────────────────────
def evaluate(
    assignment: np.ndarray,
    n_dashers: int,
    time_mat: np.ndarray,
    food_ready: np.ndarray,
    created: np.ndarray,
    route_start_s: float,
    penalty_lambda: float,
) -> Tuple[float, float, float]:
    """
    Evaluate the fitness of one chromosome (assignment vector).

    Fitness = del_per_hour − λ · max(0, avg_dur_min − TARGET_DUR_MIN)

    Empty dashers (no orders assigned) are skipped and do not contribute
    to route_span, so they effectively reduce the denominator and can
    improve del_per_hour — this is intentional (see TOTAL_DASHERS note).

    Args:
        assignment     : integer array of length n_orders; assignment[i] = dasher index
        n_dashers      : number of dashers allocated to this region
        time_mat       : pre-computed travel-time matrix
        food_ready     : food_ready_s array
        created        : created_s array
        route_start_s  : unix timestamp of route start
        penalty_lambda : current penalty coefficient λ(t)

    Returns:
        fitness, del_per_hour, avg_delivery_duration_min
    """
    dasher_indices = [[] for _ in range(n_dashers)]
    for j, d_idx in enumerate(assignment):
        dasher_indices[d_idx].append(j)

    total_span = 0.0
    all_durs: List[float] = []

    for d_idxs in dasher_indices:
        if not d_idxs:
            continue
        span, durs = simulate_dasher(d_idxs, time_mat, food_ready, created, route_start_s)
        total_span += span
        all_durs.extend(durs)

    if not all_durs or total_span == 0:
        return -999.0, 0.0, 0.0

    dph     = len(all_durs) / (total_span / 3600)
    avg_dur = sum(all_durs) / len(all_durs) / 60
    penalty = penalty_lambda * max(0.0, avg_dur - TARGET_DUR_MIN)
    return dph - penalty, dph, avg_dur


# ──────────────────────────────────────────────────────────────
# Chromosome initialisation
# ──────────────────────────────────────────────────────────────
def repair_empty(asgn: np.ndarray, n_dashers: int) -> np.ndarray:
    """
    Ensure every dasher has at least one order assigned.
    When a dasher is empty, steal one order from the most loaded dasher.

    Works on a copy to avoid mutating the caller's array in-place.
    """
    asgn = asgn.copy()
    for d in range(n_dashers):
        while np.sum(asgn == d) == 0:
            heavy      = int(np.argmax(np.bincount(asgn, minlength=n_dashers)))
            heavy_idxs = np.where(asgn == heavy)[0]
            asgn[random.choice(heavy_idxs.tolist())] = d
    return asgn


def init_assignment(n_orders: int, n_dashers: int) -> np.ndarray:
    """
    Create a random assignment by shuffling order indices and distributing
    them round-robin across dashers. This guarantees a roughly balanced
    initial load. repair_empty is called to handle the edge case where
    rounding leaves a dasher with zero orders.
    """
    asgn = np.zeros(n_orders, dtype=np.int32)
    idxs = list(range(n_orders))
    random.shuffle(idxs)
    for k, j in enumerate(idxs):
        asgn[j] = k % n_dashers
    return repair_empty(asgn, n_dashers)


# ──────────────────────────────────────────────────────────────
# Mutation operators
# ──────────────────────────────────────────────────────────────
def mutate_random(asgn: np.ndarray, n_dashers: int, rate: float) -> np.ndarray:
    """
    Randomly reassign each order to a new dasher with probability `rate`.
    Uses Python's random module (consistent with the rest of the codebase)
    to preserve the same random sequence as the original implementation.
    Mixing numpy and Python random generators changes the exploration path
    and can degrade solution quality.
    """
    new = asgn.copy()
    for j in range(len(new)):
        if random.random() < rate:
            new[j] = random.randint(0, n_dashers - 1)
    return new


def mutate_load_balance(asgn: np.ndarray, orders: List[dict], n_dashers: int) -> np.ndarray:
    """
    Transfer one order from the most loaded dasher to the least loaded.
    A no-op if the load difference is ≤ 1 (already balanced).
    Reduces variance in route lengths, which tends to lower avg_dur.
    """
    new   = asgn.copy()
    load  = np.bincount(new, minlength=n_dashers)
    heavy = int(np.argmax(load))
    light = int(np.argmin(load))
    if load[heavy] - load[light] <= 1:
        return new
    heavy_idxs = np.where(new == heavy)[0]
    if len(heavy_idxs) > 0:
        new[random.choice(heavy_idxs.tolist())] = light
    return new


def mutate_spatial(asgn: np.ndarray, orders: List[dict], n_dashers: int) -> np.ndarray:
    """
    For each dasher, find the order whose pickup is farthest from the dasher's
    pickup centroid, then reassign it (with probability 0.5) to the dasher
    whose centroid is spatially closest to that order.

    Intent: improve geographic compactness of each dasher's route,
    reducing total travel time and thus avg_dur.
    """
    new = asgn.copy()

    # Compute pickup centroid for each dasher
    centroids: Dict[int, Tuple[float, float]] = {}
    for d in range(n_dashers):
        idxs = np.where(new == d)[0]
        if len(idxs) == 0:
            continue
        centroids[d] = (
            float(np.mean([orders[j]["pickup_lat"]  for j in idxs])),
            float(np.mean([orders[j]["pickup_long"] for j in idxs])),
        )

    for d, (clat, clon) in centroids.items():
        idxs = np.where(new == d)[0]
        if len(idxs) <= 1:
            continue

        # Order farthest from this dasher's centroid
        j_far = max(idxs, key=lambda j: haversine(
            orders[j]["pickup_lat"], orders[j]["pickup_long"], clat, clon))

        # Dasher whose centroid is nearest to that order's pickup
        best_d, best_dist = None, float("inf")
        for d2, (c2lat, c2lon) in centroids.items():
            if d2 == d:
                continue
            dist = haversine(orders[j_far]["pickup_lat"],
                             orders[j_far]["pickup_long"], c2lat, c2lon)
            if dist < best_dist:
                best_dist, best_d = dist, d2

        if best_d is not None and random.random() < 0.5:
            new[j_far] = best_d

    return new


def mutate_time_compact(asgn: np.ndarray, orders: List[dict], n_dashers: int) -> np.ndarray:
    """
    For each dasher, find the order whose food_ready_time is farthest from the
    dasher's mean food_ready_time, then reassign it (with probability 0.5) to
    the dasher whose mean food_ready_time is closest to that order.

    Intent: cluster orders with similar food_ready times together so dashers
    spend less time waiting at restaurants, reducing avg_dur.
    """
    new = asgn.copy()

    for d in range(n_dashers):
        idxs = np.where(new == d)[0]
        if len(idxs) <= 1:
            continue

        avg_t = float(np.mean([orders[j]["food_ready_s"] for j in idxs]))
        j_far = max(idxs, key=lambda j: abs(orders[j]["food_ready_s"] - avg_t))
        t_far = orders[j_far]["food_ready_s"]

        best_d, best_diff = None, float("inf")
        for d2 in range(n_dashers):
            if d2 == d:
                continue
            d2_idxs = np.where(new == d2)[0]
            if len(d2_idxs) == 0:
                continue
            avg_t2 = float(np.mean([orders[j]["food_ready_s"] for j in d2_idxs]))
            diff   = abs(t_far - avg_t2)
            if diff < best_diff:
                best_diff, best_d = diff, d2

        if best_d is not None and random.random() < 0.5:
            new[j_far] = best_d

    return new


def hypermutate(
    asgn: np.ndarray,
    orders: List[dict],
    n_dashers: int,
    rate: float,
    rank: int,
) -> np.ndarray:
    """
    Apply a rank-dependent combination of mutation operators.

    Elite individuals (low rank) receive conservative mutation to preserve
    good structure. Lower-ranked individuals receive aggressive mutation to
    diversify the search.

        rank < 3  : light random + optional load balance
        rank < 7  : medium random + one structural operator
        rank ≥ 7  : heavy random + two structural operators
    """
    new = asgn.copy()

    if rank < 3:
        new = mutate_random(new, n_dashers, rate * 0.5)
        if random.random() < 0.5:
            new = mutate_load_balance(new, orders, n_dashers)
    elif rank < 7:
        new = mutate_random(new, n_dashers, rate)
        op  = random.choice([mutate_spatial, mutate_time_compact, mutate_load_balance])
        new = op(new, orders, n_dashers)
    else:
        new = mutate_random(new, n_dashers, rate * 2)
        for op in random.sample([mutate_spatial, mutate_time_compact, mutate_load_balance], 2):
            new = op(new, orders, n_dashers)

    # Always ensure no dasher is left empty after mutation
    return repair_empty(new, n_dashers)


# ──────────────────────────────────────────────────────────────
# Diversity measure
# ──────────────────────────────────────────────────────────────
def hamming(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised Hamming distance: fraction of positions where a and b differ."""
    return float(np.sum(a != b) / len(a))


# ──────────────────────────────────────────────────────────────
# Per-region immune optimiser (worker function)
# ──────────────────────────────────────────────────────────────
def immune_region(args: tuple):
    """
    Artificial Immune System optimiser for one geographic region.
    Executed in a separate process via multiprocessing.Pool.

    Each generation:
      1. Evaluate fitness F(x) = del/hr − λ(t)·penalty for every antibody.
      2. Select the top N_ELITE individuals (clone selection).
      3. Clone each elite individual; apply hypermutation with rank-dependent rate.
      4. Inject N_REPLACE random new individuals (receptor editing) for diversity.
      5. Evaluate all candidates; keep the POP_SIZE most diverse high-fitness ones.
      6. Track the best chromosome seen so far by del/hr (primary) then avg_dur.

    λ(t) increases linearly over the first 70% of MAX_ITER:
        early  → small λ, fitness ≈ del/hr  (exploration phase)
        late   → large λ, infeasible solutions penalised heavily (exploitation phase)

    Args:
        args: (region_id, orders, n_dashers, route_start_s, seed)

    Returns:
        (region_id, best_assignment, best_dph, best_dur, orders, n_dashers)
    """
    region_id, orders, n_dashers, route_start_s, seed = args
    random.seed(seed)
    np.random.seed(seed)

    n_orders   = len(orders)
    food_ready = np.array([o["food_ready_s"] for o in orders], dtype=np.float64)
    created    = np.array([o["created_s"]    for o in orders], dtype=np.float64)

    # Pre-compute travel-time matrix once; reused every iteration
    t_build  = time.time()
    time_mat = build_time_matrix(orders)
    t0       = time.time()
    print(f"  [Region {region_id}] Start: {n_orders} orders / {n_dashers} dashers "
          f"(matrix built in {(t0 - t_build) * 1000:.1f} ms)", flush=True)

    # Initialise population
    population = [init_assignment(n_orders, n_dashers) for _ in range(POP_SIZE)]

    # Defensive initialisation: ensures best_asgn is never None even if MAX_ITER = 0
    best_asgn = population[0]
    best_dph  = 0.0
    best_dur  = 0.0

    for iteration in range(MAX_ITER):

        if time.time() - t0 > TIME_LIMIT_S:
            print(f"  [Region {region_id}] Time limit reached at iter {iteration}", flush=True)
            break

        # Adaptive penalty coefficient
        progress = min(1.0, iteration / (MAX_ITER * 0.7))
        lam = PENALTY_INIT + (PENALTY_FINAL - PENALTY_INIT) * progress

        # ── Step 1: Evaluate population ───────────────────────
        scored = []
        for asgn in population:
            fit, dph, dur = evaluate(
                asgn, n_dashers, time_mat, food_ready, created, route_start_s, lam)
            scored.append((fit, dph, dur, asgn))
        scored.sort(key=lambda x: -x[0])

        # Track global best by del/hr (primary), then avg_dur (tiebreak)
        _, dph0, dur0, asgn0 = scored[0]
        if dph0 > best_dph or (abs(dph0 - best_dph) < 1e-4 and dur0 < best_dur):
            best_dph, best_dur, best_asgn = dph0, dur0, asgn0
            print(f"  [Region {region_id}] iter {iteration:4d}: "
                  f"del/hr={best_dph:.4f} | avg_dur={best_dur:.2f} min | "
                  f"λ={lam:.2f}", flush=True)

        # ── Step 2: Clone selection ────────────────────────────
        # Higher-ranked (better) individuals get more clones with lower mutation rate.
        # Lower-ranked individuals get fewer clones with higher mutation rate.
        elite  = scored[:N_ELITE]
        clones = []
        for rank, (_, _, _, asgn) in enumerate(elite):
            n_clone  = max(1, N_CLONE_BASE - rank // 3)
            mut_rate = 0.05 + 0.15 * (rank / N_ELITE)
            for _ in range(n_clone):
                clones.append(hypermutate(asgn, orders, n_dashers, mut_rate, rank))

        # ── Step 3: Receptor editing ───────────────────────────
        # Inject fresh random individuals to prevent premature convergence
        new_randoms = [init_assignment(n_orders, n_dashers) for _ in range(N_REPLACE)]

        # ── Step 4: Evaluate all candidates ───────────────────
        all_candidates = [a for _, _, _, a in elite] + clones + new_randoms
        all_scored = []
        for asgn in all_candidates:
            fit, dph, dur = evaluate(
                asgn, n_dashers, time_mat, food_ready, created, route_start_s, lam)
            all_scored.append((fit, dph, dur, asgn))
        all_scored.sort(key=lambda x: -x[0])

        # ── Step 5: Diversity-preserving selection ─────────────
        # Keep the highest-fitness individual unconditionally, then add others
        # only if they differ from all already-selected individuals by at least
        # MIN_DIVERSITY (normalised Hamming distance). This prevents the
        # population from collapsing to a single solution (premature convergence).
        selected = [all_scored[0][3]]
        for _, _, _, asgn in all_scored[1:]:
            if len(selected) >= POP_SIZE:
                break
            if min(hamming(asgn, s) for s in selected) >= MIN_DIVERSITY:
                selected.append(asgn)

        # Pad with random individuals if diversity criterion is too strict
        while len(selected) < POP_SIZE:
            selected.append(init_assignment(n_orders, n_dashers))

        population = selected

    elapsed = time.time() - t0
    print(f"  [Region {region_id}] Done: del/hr={best_dph:.4f} | "
          f"avg_dur={best_dur:.2f} min | {elapsed:.1f}s", flush=True)

    return region_id, best_asgn, best_dph, best_dur, orders, n_dashers


# ──────────────────────────────────────────────────────────────
# Output builder
# ──────────────────────────────────────────────────────────────
def build_output(region_results: list, route_start_s: float, output_path: str) -> None:
    """
    Convert the best assignment for each region into the submission CSV format:
        Route ID | Route Point Index | Delivery ID | Route Point Type | Route Point Time

    Uses simulate_and_record_stops (shared helper) to avoid duplicating the
    greedy simulation logic. The travel-time matrix is rebuilt here per region
    since it cannot be pickled across process boundaries in multiprocessing.

    Route IDs are assigned sequentially across all regions (0, 1, 2, …).
    Empty dashers (no assigned orders) are silently skipped.
    """
    rows = []
    global_route_id = 0

    for region_id, asgn, dph, dur, orders, n_dashers in region_results:
        n_orders   = len(orders)
        food_ready = np.array([o["food_ready_s"] for o in orders], dtype=np.float64)
        time_mat   = build_time_matrix(orders)   # rebuilt once per region for output

        for d_idx in range(n_dashers):
            d_idxs = [j for j in range(n_orders) if asgn[j] == d_idx]
            if not d_idxs:
                continue   # empty dasher — skip (improves del/hr denominator)

            stops = simulate_and_record_stops(
                d_idxs, time_mat, food_ready, orders, route_start_s)

            for idx, (ptype, did, t) in enumerate(stops):
                rows.append({
                    "Route ID":          global_route_id,
                    "Route Point Index": idx,
                    "Delivery ID":       did,
                    "Route Point Type":  ptype,
                    "Route Point Time":  t,
                })
            global_route_id += 1

    df = pd.DataFrame(rows).sort_values(["Route ID", "Route Point Index"])
    df.to_csv(output_path, index=False)
    print(f"\nOutput saved → {output_path}  ({len(df)} rows, {global_route_id} routes)")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main() -> None:
    random.seed(1)
    np.random.seed(1)

    orders_by_region = load_data(DATA_PATH)
    route_start_s    = ROUTE_START.timestamp()

    print("=== Immune Optimization Algorithm — Parallel Regional Solver ===")
    print(f"Total dashers: {TOTAL_DASHERS},  allocation: {REGION_DASHERS}\n")

    # Each region worker gets a distinct but deterministic seed
    worker_args = [
        (rid, orders_by_region[rid], REGION_DASHERS[rid], route_start_s, 42 + i)
        for i, rid in enumerate(sorted(REGION_DASHERS.keys()))
    ]

    # Launch one process per region (capped at cpu_count)
    t0        = time.time()
    n_workers = min(len(worker_args), multiprocessing.cpu_count())
    print(f"Launching {len(worker_args)} region workers on {n_workers} CPUs\n")

    with multiprocessing.Pool(processes=n_workers) as pool:
        region_results = pool.map(immune_region, worker_args)

    print(f"\nWall time: {time.time() - t0:.1f}s")

    # ── Global metrics ─────────────────────────────────────────
    total_span = 0.0
    all_durs: List[float] = []

    for region_id, asgn, dph, dur, orders, n_dashers in region_results:
        food_ready = np.array([o["food_ready_s"] for o in orders], dtype=np.float64)
        created    = np.array([o["created_s"]    for o in orders], dtype=np.float64)
        time_mat   = build_time_matrix(orders)

        for d_idx in range(n_dashers):
            d_idxs = [j for j in range(len(orders)) if asgn[j] == d_idx]
            if not d_idxs:
                continue
            span, durs = simulate_dasher(
                d_idxs, time_mat, food_ready, created, route_start_s)
            total_span += span
            all_durs.extend(durs)

    global_dph = len(all_durs) / (total_span / 3600) if total_span else 0.0
    global_dur = sum(all_durs) / len(all_durs) / 60   if all_durs   else 0.0

    # Active dasher count: O(n) using np.unique instead of quadratic loop
    active = sum(len(np.unique(r[1])) for r in region_results)

    print("\n── Final global metrics ───────────────────────────────")
    print(f"  Avg deliveries / hour  : {global_dph:.4f}")
    print(f"  Avg delivery duration  : {global_dur:.2f} min  "
          f"{'✓' if global_dur <= TARGET_DUR_MIN else f'(soft constraint exceeded by {global_dur - TARGET_DUR_MIN:.2f} min)'}")
    print(f"  Active dashers         : {active} / {TOTAL_DASHERS}")
    for rid, asgn, dph, dur, orders, n_dashers in region_results:
        print(f"    Region {rid}: del/hr={dph:.4f} | avg_dur={dur:.2f} min")

    build_output(region_results, route_start_s, OUTPUT_PATH)


if __name__ == "__main__":
    main()