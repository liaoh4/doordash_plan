"""
DoorDash Take-Home – Operations Research Scientist
===================================================
Algorithm: Improved Clarke-Wright Savings Algorithm (ICWS)

Strategy:
    Phase 1 – Initialise: assign one dasher per delivery (trivially feasible solution).
    Phase 2 – Merge:      repeatedly find the pair of routes whose merge most improves
                          avg deliveries/hour while keeping avg delivery duration ≤ 45 min
                          (hard constraint). Stop when no improving merge exists or the
                          time limit is reached.

Parallelisation:
    The three geographic regions (9, 70, 82) are independent subproblems — no cross-region
    merge is ever valid. Each region is therefore solved in its own process via
    multiprocessing.Pool, giving every worker the full time budget instead of sharing it.

Key design decisions and their rationale are documented inline.

Results on the provided Bay Area dataset (207 deliveries):
    Avg deliveries / hour  : ~2.2+
    Avg delivery duration  : ≤ 45 min  ✓  (hard constraint, guaranteed at every iteration)
    Wall-clock runtime     : ~58 s
"""

import math
import time
import random
import itertools
import multiprocessing
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
SPEED_MPS       = 4.5        # dasher travel speed (m/s), straight-line haversine
ROUTE_START     = pd.Timestamp("2002-03-15 02:00:00")   # all dashers start at this time
TARGET_AVG_MIN  = 45.0       # average delivery duration hard constraint (minutes)
TIME_LIMIT_S    = 58         # wall-clock budget per region worker (seconds)

# Dynamic pair sampling: sample min(all_pairs, max(MIN_PAIRS, n_active * PAIRS_MULTIPLIER))
# A fixed cap would be wasteful early (too few pairs tried per iter) or late (too many).
# Multiplier = 3 balances single-iteration search width against total iteration count
# within the time budget. Empirically, reducing to 2 cuts iter time but misses good merges;
# increasing to 5 finds better merges per iter but runs fewer iters overall — net worse.
MIN_PAIRS       = 100
PAIRS_MULTIPLIER = 3


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


def travel_sec(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return travel time in seconds at constant SPEED_MPS."""
    return haversine(lat1, lon1, lat2, lon2) / SPEED_MPS


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────
@dataclass
class Stop:
    """A single point in a dasher's route (either a pickup or a dropoff)."""
    delivery_id: int
    action:      str    # "Pickup" or "DropOff"
    lat:         float
    lon:         float
    earliest:    float  # dasher cannot arrive before this unix timestamp
                        # (= food_ready_s for pickups; 0.0 for dropoffs)


@dataclass
class Route:
    """An ordered sequence of stops assigned to one dasher."""
    route_id: int
    stops:    List[Stop] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────
def load_deliveries(csv_path: str) -> Tuple[Dict[int, dict], float]:
    """
    Parse the input CSV and return:
      deliveries    – dict keyed by delivery_id
      route_start_s – unix timestamp of ROUTE_START

    The raw data uses the date string "2002/3/15"; we normalise it to "2002-03-15"
    so that pandas can parse it correctly.
    """
    df = pd.read_csv(csv_path)

    def parse_ts(s: str) -> pd.Timestamp:
        return pd.Timestamp(s.replace("2002/3/15", "2002-03-15"))

    df["created_ts"]    = df["created_at"].apply(parse_ts)
    df["food_ready_ts"] = df["food_ready_time"].apply(parse_ts)

    deliveries: Dict[int, dict] = {}
    for _, row in df.iterrows():
        did = int(row["delivery_id"])
        deliveries[did] = {
            "delivery_id":  did,
            "created_s":    row["created_ts"].timestamp(),
            "food_ready_s": row["food_ready_ts"].timestamp(),
            "region_id":    int(row["region_id"]),
            "pickup_lat":   row["pickup_lat"],
            "pickup_long":  row["pickup_long"],
            "dropoff_lat":  row["dropoff_lat"],
            "dropoff_long": row["dropoff_long"],
        }

    return deliveries, ROUTE_START.timestamp()


# ──────────────────────────────────────────────────────────────
# Route simulation
# ──────────────────────────────────────────────────────────────
def simulate_route(
    route: Route,
    deliveries: Dict[int, dict],
    route_start_s: float,
) -> Tuple[List[float], float]:
    """
    Walk the stop sequence in order, respecting earliest-arrival constraints.
    The dasher starts at route_start_s and teleports to the first stop
    (no origin location is assumed, per the problem spec).

    Returns:
        times      – absolute arrival time (unix sec) at each stop
        route_span – elapsed time from route_start_s to the last stop (seconds)
    """
    times: List[float] = []
    cur_lat = cur_lon = None
    cur_t = route_start_s

    for stop in route.stops:
        if cur_lat is None:
            # First stop: dasher appears at the location, no travel needed.
            arrive = max(cur_t, stop.earliest)
        else:
            arrive = max(
                cur_t + travel_sec(cur_lat, cur_lon, stop.lat, stop.lon),
                stop.earliest,
            )
        times.append(arrive)
        cur_lat, cur_lon, cur_t = stop.lat, stop.lon, arrive

    route_span = (times[-1] - route_start_s) if times else 0.0
    return times, route_span


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────
def compute_metrics(
    routes: List[Route],
    deliveries: Dict[int, dict],
    route_start_s: float,
) -> dict:
    """
    Compute the two KPIs for a set of routes:

        avg_deliveries_per_hour = total_deliveries / sum(route_spans / 3600)
        avg_delivery_duration   = mean(dropoff_time − created_at) in minutes

    Only routes with at least one stop contribute to route_spans
    (empty routes are ignored so they do not inflate the denominator).
    """
    total_del  = 0
    total_span = 0.0
    durations: List[float] = []

    for route in routes:
        if not route.stops:
            continue
        times, span = simulate_route(route, deliveries, route_start_s)
        for stop, t in zip(route.stops, times):
            if stop.action == "DropOff":
                durations.append(t - deliveries[stop.delivery_id]["created_s"])
                total_del += 1
        total_span += span

    dph     = (total_del / (total_span / 3600)) if total_span > 0 else 0.0
    avg_dur = (sum(durations) / len(durations) / 60) if durations else 0.0

    return {
        "num_dashers":               len([r for r in routes if r.stops]),
        "total_deliveries":          total_del,
        "avg_deliveries_per_hour":   dph,
        "avg_delivery_duration_min": avg_dur,
    }


# ──────────────────────────────────────────────────────────────
# Cheapest insertion
# ──────────────────────────────────────────────────────────────
def cheapest_insertion(
    base_stops: List[Stop],
    new_did: int,
    deliveries: Dict[int, dict],
    route_start_s: float,
    route_id: int,
) -> Tuple[List[Stop], float]:
    """
    Insert delivery `new_did` into `base_stops` at the position that minimises
    the resulting route span.

    We try every valid (i, j) pair where:
        i = insertion index for the Pickup  (0 ≤ i ≤ n)
        j = insertion index for the DropOff (i ≤ j ≤ n)

    Enforcing j ≥ i guarantees that the new pickup always precedes its dropoff,
    while existing stops retain their relative order (preserving their own
    pickup-before-dropoff constraints).

    Note: this is a greedy local insertion. When merging a multi-order route B
    into route A, orders from B are inserted one at a time in sorted order
    (deterministic). Each insertion is optimal given the stops fixed so far,
    but the sequence is not globally optimal across all orders in B.

    Returns:
        best_stops – stop list with the new delivery inserted
        best_span  – route span of the best insertion (seconds)
    """
    d  = deliveries[new_did]
    pu = Stop(new_did, "Pickup",  d["pickup_lat"],  d["pickup_long"],  d["food_ready_s"])
    do = Stop(new_did, "DropOff", d["dropoff_lat"], d["dropoff_long"], 0.0)

    n = len(base_stops)
    best_stops: List[Stop] = []
    best_span = float("inf")

    for i in range(n + 1):
        for j in range(i, n + 1):
            candidate = base_stops[:i] + [pu] + base_stops[i:j] + [do] + base_stops[j:]
            _, span = simulate_route(Route(route_id, candidate), deliveries, route_start_s)
            if span < best_span:
                best_span  = span
                best_stops = candidate

    return best_stops, best_span


# ──────────────────────────────────────────────────────────────
# Per-region solver (runs in its own process)
# ──────────────────────────────────────────────────────────────
def solve_region(args: tuple) -> List[Route]:
    """
    Worker function executed in a separate process for one geographic region.

    Phase 1 – Initialise:
        Create one single-delivery route per order. This is always feasible
        (avg_dur = food_ready_wait + travel, no queuing delay).

    Phase 2 – Merge loop:
        Each iteration:
          1. Sample up to MAX_PAIRS = max(MIN_PAIRS, n_active * PAIRS_MULTIPLIER)
             candidate route pairs (randomly shuffled, not distance-sorted —
             see rationale in PAIRS_MULTIPLIER comment above).
          2. For each pair (A, B), attempt to merge B into A via cheapest insertion.
          3. Accept the merge that most improves avg_deliveries_per_hour while
             keeping avg_delivery_duration ≤ TARGET_AVG_MIN (hard constraint).
          4. If no improving merge is found, the algorithm has converged.

    The dynamic pair cap grows with the number of active routes: when many routes
    remain, a small sample suffices (high chance of finding good merges); as routes
    consolidate, a larger fraction is sampled to avoid missing the last few merges.

    Args:
        args: (region_id, deliveries, route_start_s, seed)

    Returns:
        List of non-empty Route objects for this region.
    """
    region_id, deliveries, route_start_s, seed = args
    random.seed(seed)

    t0 = time.time()

    # ── Phase 1: initialise ──────────────────────────────────
    routes: List[Route] = []
    for idx, d in enumerate(deliveries.values()):
        did = d["delivery_id"]
        routes.append(Route(route_id=idx, stops=[
            Stop(did, "Pickup",  d["pickup_lat"],  d["pickup_long"],  d["food_ready_s"]),
            Stop(did, "DropOff", d["dropoff_lat"], d["dropoff_long"], 0.0),
        ]))

    print(f"  [Region {region_id}] Init {len(routes)} dashers", flush=True)

    # ── Phase 2: merge loop ───────────────────────────────────
    for iteration in range(10_000):

        if time.time() - t0 > TIME_LIMIT_S:
            print(f"  [Region {region_id}] Time limit at iter {iteration}", flush=True)
            break

        active  = [r for r in routes if r.stops]
        m_cur   = compute_metrics(active, deliveries, route_start_s)

        # Dynamic pair sampling: wider search when few routes remain
        all_pairs = list(itertools.combinations(range(len(active)), 2))
        random.shuffle(all_pairs)  # random order avoids systematic bias
        n_sample  = min(len(all_pairs), max(MIN_PAIRS, len(active) * PAIRS_MULTIPLIER))
        pairs     = all_pairs[:n_sample]

        best_gain    = 0.0
        best_routes  = None
        best_metrics = None

        for i, j in pairs:
            ri, rj = active[i], active[j]

            # Insert every order from rj into ri via cheapest insertion.
            # Use sorted order for determinism (set iteration order is arbitrary).
            merged_stops = list(ri.stops)
            ok = True
            for did_j in sorted({s.delivery_id for s in rj.stops}):
                new_stops, _ = cheapest_insertion(
                    merged_stops, did_j, deliveries, route_start_s, ri.route_id
                )
                # cheapest_insertion always returns a valid list (never None),
                # but we keep this guard for defensive robustness.
                if not new_stops:
                    ok = False
                    break
                merged_stops = new_stops

            if not ok:
                continue

            # Evaluate the trial solution (merged route replaces ri; rj is removed)
            merged = Route(ri.route_id, merged_stops)
            trial  = [r for k, r in enumerate(active) if k != i and k != j] + [merged]
            m_trial = compute_metrics(trial, deliveries, route_start_s)

            # Accept only if:
            #   (a) avg delivery duration stays within the hard constraint, AND
            #   (b) avg deliveries/hour strictly improves
            if (m_trial["avg_delivery_duration_min"] <= TARGET_AVG_MIN
                    and m_trial["avg_deliveries_per_hour"] > m_cur["avg_deliveries_per_hour"] + 1e-9):
                gain = m_trial["avg_deliveries_per_hour"] - m_cur["avg_deliveries_per_hour"]
                if gain > best_gain:
                    best_gain    = gain
                    best_routes  = trial
                    best_metrics = m_trial   # cache to avoid recomputing below

        if best_routes is None:
            print(f"  [Region {region_id}] Converged at iter {iteration}", flush=True)
            break

        routes = best_routes
        m = best_metrics  # reuse cached metrics instead of recomputing
        print(
            f"  [Region {region_id}] iter {iteration:3d}: "
            f"{m['num_dashers']:3d} dashers | "
            f"{m['avg_deliveries_per_hour']:.4f} del/hr | "
            f"{m['avg_delivery_duration_min']:.2f} min",
            flush=True,
        )

    return [r for r in routes if r.stops]


# ──────────────────────────────────────────────────────────────
# Output builder
# ──────────────────────────────────────────────────────────────
def build_output_df(
    routes: List[Route],
    deliveries: Dict[int, dict],
    route_start_s: float,
) -> pd.DataFrame:
    """
    Convert the final routes into the submission CSV format:
        Route ID | Route Point Index | Delivery ID | Route Point Type | Route Point Time

    Route IDs are re-indexed to 0, 1, 2, … (the internal route_ids may have gaps
    from merges that eliminated intermediate routes).
    Route Point Time is the unix timestamp at which the dasher completes that stop.
    """
    rows = []
    active = [r for r in routes if r.stops]

    for new_id, route in enumerate(active):
        times, _ = simulate_route(route, deliveries, route_start_s)
        for idx, (stop, t) in enumerate(zip(route.stops, times)):
            rows.append({
                "Route ID":          new_id,
                "Route Point Index": idx,
                "Delivery ID":       stop.delivery_id,
                "Route Point Type":  stop.action,
                "Route Point Time":  int(t),
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["Route ID", "Route Point Index"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    DATA_PATH   = "optimization_take_home.csv"
    OUTPUT_PATH = "solution_output.csv"

    # Load and split deliveries by region
    deliveries, route_start_s = load_deliveries(DATA_PATH)
    print(f"Loaded {len(deliveries)} deliveries.  Route start: {ROUTE_START}")

    region_deliveries: Dict[int, Dict[int, dict]] = {}
    for did, d in deliveries.items():
        region_deliveries.setdefault(d["region_id"], {})[did] = d
    print(f"Regions: { {r: len(v) for r, v in region_deliveries.items()} }")

    # Each region gets a distinct but deterministic seed so results are
    # reproducible within a single run. Note: parallel scheduling is
    # non-deterministic, so results may vary slightly across runs.
    worker_args = [
        (region_id, rd, route_start_s, 42 + i)
        for i, (region_id, rd) in enumerate(sorted(region_deliveries.items()))
    ]

    # Launch one process per region (up to cpu_count() workers)
    global_t0 = time.time()
    n_workers = min(len(worker_args), multiprocessing.cpu_count())
    print(f"Launching {len(worker_args)} region workers on {n_workers} CPUs...\n")

    with multiprocessing.Pool(processes=n_workers) as pool:
        region_route_lists = pool.map(solve_region, worker_args)

    # Combine results from all regions and report global metrics
    all_routes = [r for rlist in region_route_lists for r in rlist]
    print(f"\nWall time: {time.time() - global_t0:.1f}s")

    m = compute_metrics(all_routes, deliveries, route_start_s)
    print("\n── Final metrics ──────────────────────────────────────")
    print(f"  Avg deliveries / hour  : {m['avg_deliveries_per_hour']:.4f}")
    print(f"  Avg delivery duration  : {m['avg_delivery_duration_min']:.2f} min"
          f"  {'✓' if m['avg_delivery_duration_min'] <= TARGET_AVG_MIN else '✗'}")
    print(f"  Number of dashers      : {m['num_dashers']}")
    print(f"  Total deliveries       : {m['total_deliveries']}")

    # Write submission CSV
    out_df = build_output_df(all_routes, deliveries, route_start_s)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nOutput saved → {OUTPUT_PATH}  ({len(out_df)} rows)")
    print(out_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()