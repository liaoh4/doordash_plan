"""
免疫优化算法 — 分区并行版
==========================
将49个dasher按订单比例分配给3个region：
  Region 9  → 7个dasher
  Region 70 → 17个dasher
  Region 82 → 25个dasher

三个region完全独立，用multiprocessing.Pool并行运行，
每个worker各自用满时间预算，相当于计算时间扩大3倍。
"""

import math, time, random, multiprocessing
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime

# ── 常量 ──────────────────────────────────────────────────────
SPEED_MPS      = 4.5
ROUTE_START    = pd.Timestamp("2002-03-15 02:00:00")
TARGET_DUR_MIN = 45.0
TOTAL_DASHERS  = 49
DATA_PATH      = "optimization_take_home.csv"
OUTPUT_PATH    = "solution_output.csv"

# 各region的dasher分配（按订单比例，合计49）
REGION_DASHERS = {9: 7, 70: 17, 82: 25}

# 免疫算法参数
POP_SIZE       = 40
N_ELITE        = 12
N_CLONE_BASE   = 6
N_REPLACE      = 8
MAX_ITER       = 500
TIME_LIMIT_S   = 58
MIN_DIVERSITY  = 0.03
PENALTY_INIT   = 0.5
PENALTY_FINAL  = 3.0

# ── 工具函数 ──────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def travel_sec(lat1, lon1, lat2, lon2):
    return haversine(lat1, lon1, lat2, lon2) / SPEED_MPS

# ── 数据加载 ──────────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path)
    def parse_ts(s):
        return pd.Timestamp(str(s).replace("2002/3/15","2002-03-15"))
    df["created_ts"]    = df["created_at"].apply(parse_ts)
    df["food_ready_ts"] = df["food_ready_time"].apply(parse_ts)

    orders_by_region = {}
    for _, row in df.iterrows():
        rid = int(row["region_id"])
        orders_by_region.setdefault(rid, []).append({
            "delivery_id":  int(row["delivery_id"]),
            "created_s":    row["created_ts"].timestamp(),
            "food_ready_s": row["food_ready_ts"].timestamp(),
            "region_id":    rid,
            "pickup_lat":   row["pickup_lat"],
            "pickup_long":  row["pickup_long"],
            "dropoff_lat":  row["dropoff_lat"],
            "dropoff_long": row["dropoff_long"],
        })
    return orders_by_region

# ── 路线仿真 ──────────────────────────────────────────────────
def simulate_dasher(order_list, route_start_s):
    if not order_list:
        return 0.0, []

    pending = list(order_list)
    picked  = []
    cl = cn = None
    ct = route_start_s
    durs = []

    while pending or picked:
        best_act = best_o = arrive = None
        best_cost = float("inf")

        if cl is None:
            best_o   = min(pending, key=lambda o: o["food_ready_s"])
            best_act = "pickup"
            arrive   = max(ct, best_o["food_ready_s"])
        else:
            for o in picked:
                arr = ct + travel_sec(cl, cn, o["dropoff_lat"], o["dropoff_long"])
                if arr < best_cost:
                    best_cost, best_act, best_o, arrive = arr, "dropoff", o, arr
            for o in pending:
                arr = max(ct + travel_sec(cl, cn, o["pickup_lat"], o["pickup_long"]),
                          o["food_ready_s"])
                if arr < best_cost:
                    best_cost, best_act, best_o, arrive = arr, "pickup", o, arr

        if best_act == "pickup":
            cl, cn = best_o["pickup_lat"], best_o["pickup_long"]
            ct = arrive; pending.remove(best_o); picked.append(best_o)
        else:
            cl, cn = best_o["dropoff_lat"], best_o["dropoff_long"]
            ct = arrive; picked.remove(best_o)
            durs.append(arrive - best_o["created_s"])

    return ct - route_start_s, durs

# ── 适应度评估 ────────────────────────────────────────────────
def evaluate(assignment, orders, n_dashers, route_start_s, penalty_lambda):
    """
    assignment: np.array shape (n_orders,)，值域 [0, n_dashers-1]
    仅针对单个region内的订单和dasher
    """
    dasher_orders = [[] for _ in range(n_dashers)]
    for j, d_idx in enumerate(assignment):
        dasher_orders[d_idx].append(orders[j])

    total_span = 0.0
    all_durs   = []
    for d_orders in dasher_orders:
        if not d_orders:
            continue
        span, durs = simulate_dasher(d_orders, route_start_s)
        total_span += span
        all_durs.extend(durs)

    if not all_durs or total_span == 0:
        return -999.0, 0.0, 0.0

    dph     = len(all_durs) / (total_span / 3600)
    avg_dur = sum(all_durs) / len(all_durs) / 60
    penalty = penalty_lambda * max(0, avg_dur - TARGET_DUR_MIN)
    return dph - penalty, dph, avg_dur

# ── 初始化 ────────────────────────────────────────────────────
def init_assignment(n_orders, n_dashers):
    """随机均匀分配"""
    asgn = np.zeros(n_orders, dtype=np.int32)
    idxs = list(range(n_orders))
    random.shuffle(idxs)
    for k, j in enumerate(idxs):
        asgn[j] = k % n_dashers
    return asgn

# ── 变异算子 ──────────────────────────────────────────────────
def mutate_random(asgn, n_dashers, rate):
    new = asgn.copy()
    for j in range(len(new)):
        if random.random() < rate:
            new[j] = random.randint(0, n_dashers - 1)
    return new

def mutate_load_balance(asgn, orders, n_dashers):
    new = asgn.copy()
    load = np.bincount(new, minlength=n_dashers)
    heavy = int(np.argmax(load))
    light = int(np.argmin(load))
    if load[heavy] - load[light] <= 1:
        return new
    heavy_idxs = np.where(new == heavy)[0]
    if len(heavy_idxs) > 0:
        new[random.choice(heavy_idxs)] = light
    return new

def mutate_spatial(asgn, orders, n_dashers):
    new = asgn.copy()
    n   = len(orders)
    # 每个dasher的pickup重心
    centroids = {}
    for d in range(n_dashers):
        idxs = np.where(new == d)[0]
        if len(idxs) == 0:
            continue
        centroids[d] = (
            np.mean([orders[j]["pickup_lat"]  for j in idxs]),
            np.mean([orders[j]["pickup_long"] for j in idxs]),
        )
    # 找每dasher中离重心最远的订单，转给最近的dasher
    for d, (clat, clon) in centroids.items():
        idxs = np.where(new == d)[0]
        if len(idxs) <= 1:
            continue
        j_far = max(idxs, key=lambda j: haversine(
            orders[j]["pickup_lat"], orders[j]["pickup_long"], clat, clon))
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

def mutate_time_compact(asgn, orders, n_dashers):
    new = asgn.copy()
    for d in range(n_dashers):
        idxs = np.where(new == d)[0]
        if len(idxs) <= 1:
            continue
        avg_t = np.mean([orders[j]["food_ready_s"] for j in idxs])
        j_far = max(idxs, key=lambda j: abs(orders[j]["food_ready_s"] - avg_t))
        t_far = orders[j_far]["food_ready_s"]
        best_d, best_diff = None, float("inf")
        for d2 in range(n_dashers):
            if d2 == d:
                continue
            d2_idxs = np.where(new == d2)[0]
            if len(d2_idxs) == 0:
                continue
            avg_t2 = np.mean([orders[j]["food_ready_s"] for j in d2_idxs])
            diff = abs(t_far - avg_t2)
            if diff < best_diff:
                best_diff, best_d = diff, d2
        if best_d is not None and random.random() < 0.5:
            new[j_far] = best_d
    return new

def hypermutate(asgn, orders, n_dashers, rate, rank):
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
    return new

def hamming(a, b):
    return np.sum(a != b) / len(a)

# ── 单region免疫优化（worker函数） ───────────────────────────
def immune_region(args):
    region_id, orders, n_dashers, route_start_s, seed = args
    random.seed(seed)
    np.random.seed(seed)

    n_orders = len(orders)
    t0 = time.time()
    print(f"  [Region {region_id}] 启动: {n_orders}单 / {n_dashers}dasher", flush=True)

    population = [init_assignment(n_orders, n_dashers) for _ in range(POP_SIZE)]

    best_asgn = None
    best_dph  = 0.0
    best_dur  = 0.0

    for iteration in range(MAX_ITER):
        if time.time() - t0 > TIME_LIMIT_S:
            print(f"  [Region {region_id}] 时间到 iter {iteration}", flush=True)
            break

        progress = min(1.0, iteration / (MAX_ITER * 0.7))
        lam = PENALTY_INIT + (PENALTY_FINAL - PENALTY_INIT) * progress

        # 评估
        scored = []
        for asgn in population:
            fit, dph, dur = evaluate(asgn, orders, n_dashers, route_start_s, lam)
            scored.append((fit, dph, dur, asgn))
        scored.sort(key=lambda x: -x[0])

        # 更新最优（以del/hr为主，avg_dur为辅）
        _, dph0, dur0, asgn0 = scored[0]
        if dph0 > best_dph or (abs(dph0 - best_dph) < 1e-4 and dur0 < best_dur):
            best_dph, best_dur, best_asgn = dph0, dur0, asgn0
            print(f"  [Region {region_id}] iter {iteration:4d}: "
                  f"del/hr={best_dph:.4f} | avg_dur={best_dur:.2f}min | "
                  f"λ={lam:.2f}", flush=True)

        # 克隆 + 超变异
        elite  = scored[:N_ELITE]
        clones = []
        for rank, (fit, dph, dur, asgn) in enumerate(elite):
            n_clone  = max(1, N_CLONE_BASE - rank // 3)
            mut_rate = 0.05 + 0.15 * (rank / N_ELITE)
            for _ in range(n_clone):
                clones.append(hypermutate(asgn, orders, n_dashers, mut_rate, rank))

        # 受体编辑
        new_randoms = [init_assignment(n_orders, n_dashers) for _ in range(N_REPLACE)]

        # 合并评估 + 多样性选择
        all_candidates = [a for _,_,_,a in elite] + clones + new_randoms
        all_scored = []
        for asgn in all_candidates:
            fit, dph, dur = evaluate(asgn, orders, n_dashers, route_start_s, lam)
            all_scored.append((fit, dph, dur, asgn))
        all_scored.sort(key=lambda x: -x[0])

        selected = [all_scored[0][3]]
        for _, _, _, asgn in all_scored[1:]:
            if len(selected) >= POP_SIZE:
                break
            if min(hamming(asgn, s) for s in selected) >= MIN_DIVERSITY:
                selected.append(asgn)
        while len(selected) < POP_SIZE:
            selected.append(init_assignment(n_orders, n_dashers))

        population = selected

    elapsed = time.time() - t0
    print(f"  [Region {region_id}] 完成: del/hr={best_dph:.4f} | "
          f"avg_dur={best_dur:.2f}min | {elapsed:.1f}s", flush=True)
    return region_id, best_asgn, best_dph, best_dur, orders, n_dashers

# ── 输出CSV ───────────────────────────────────────────────────
def build_output(region_results, route_start_s, output_path):
    rows = []
    global_route_id = 0

    for region_id, asgn, dph, dur, orders, n_dashers in region_results:
        for d_idx in range(n_dashers):
            d_orders = [orders[j] for j in range(len(orders)) if asgn[j] == d_idx]
            if not d_orders:
                continue

            pending = list(d_orders); picked = []
            cl = cn = None; ct = route_start_s
            stops = []

            while pending or picked:
                best_act = best_o = arrive = None
                best_cost = float("inf")

                if cl is None:
                    best_o   = min(pending, key=lambda o: o["food_ready_s"])
                    best_act = "pickup"
                    arrive   = max(ct, best_o["food_ready_s"])
                else:
                    for o in picked:
                        arr = ct + travel_sec(cl, cn, o["dropoff_lat"], o["dropoff_long"])
                        if arr < best_cost:
                            best_cost, best_act, best_o, arrive = arr, "dropoff", o, arr
                    for o in pending:
                        arr = max(ct + travel_sec(cl, cn, o["pickup_lat"], o["pickup_long"]),
                                  o["food_ready_s"])
                        if arr < best_cost:
                            best_cost, best_act, best_o, arrive = arr, "pickup", o, arr

                if best_act == "pickup":
                    stops.append(("Pickup",  best_o["delivery_id"], int(arrive)))
                    cl, cn = best_o["pickup_lat"],  best_o["pickup_long"]
                    ct = arrive; pending.remove(best_o); picked.append(best_o)
                else:
                    stops.append(("DropOff", best_o["delivery_id"], int(arrive)))
                    cl, cn = best_o["dropoff_lat"], best_o["dropoff_long"]
                    ct = arrive; picked.remove(best_o)

            for idx, (ptype, did, t) in enumerate(stops):
                rows.append({"Route ID": global_route_id,
                             "Route Point Index": idx,
                             "Delivery ID": did,
                             "Route Point Type": ptype,
                             "Route Point Time": t})
            global_route_id += 1

    df = pd.DataFrame(rows).sort_values(["Route ID", "Route Point Index"])
    df.to_csv(output_path, index=False)
    print(f"\nOutput saved → {output_path}  ({len(df)} rows, {global_route_id} routes)")

# ── 主函数 ────────────────────────────────────────────────────
def main():
    random.seed(42)
    np.random.seed(42)
    start_time = datetime.now()

    orders_by_region = load_data(DATA_PATH)
    route_start_s    = ROUTE_START.timestamp()

    print("=== 免疫优化算法 — 分区并行版 ===")
    print(f"总Dasher={TOTAL_DASHERS}, 各区分配: {REGION_DASHERS}\n")

    # 构造worker参数
    worker_args = [
        (rid, orders_by_region[rid], REGION_DASHERS[rid], route_start_s, 42 + i)
        for i, rid in enumerate(sorted(REGION_DASHERS.keys()))
    ]

    # 并行运行
    t0 = time.time()
    n_workers = min(len(worker_args), multiprocessing.cpu_count())
    print(f"并行启动 {len(worker_args)} 个region worker（{n_workers} CPUs）\n")

    with multiprocessing.Pool(processes=n_workers) as pool:
        region_results = pool.map(immune_region, worker_args)

    print(f"\nWall time: {time.time()-t0:.1f}s")

    # 汇总全局指标
    total_span = 0.0
    all_durs   = []
    for region_id, asgn, dph, dur, orders, n_dashers in region_results:
        for d_idx in range(n_dashers):
            d_orders = [orders[j] for j in range(len(orders)) if asgn[j] == d_idx]
            if not d_orders:
                continue
            span, durs = simulate_dasher(d_orders, route_start_s)
            total_span += span
            all_durs.extend(durs)

    global_dph = len(all_durs) / (total_span / 3600) if total_span else 0
    global_dur = sum(all_durs) / len(all_durs) / 60   if all_durs   else 0
    active     = sum(
        sum(1 for d in range(r[5]) if any(r[1][j]==d for j in range(len(r[4]))))
        for r in region_results
    )
    end_time = datetime.now()
    print(f"计算时间:               {end_time - start_time}")

    print(f"\n── 全局最终指标 ──────────────────────────")
    print(f"  Avg del/hr   : {global_dph:.4f}")
    print(f"  Avg dur      : {global_dur:.2f} min  "
          f"({'✓' if global_dur<=45 else f'软约束超{global_dur-45:.2f}min'})")
    print(f"  Active dashers: {active} / {TOTAL_DASHERS}")
    for rid, asgn, dph, dur, orders, n_dashers in region_results:
        print(f"    Region {rid}: del/hr={dph:.4f} | avg_dur={dur:.2f}min")

    build_output(region_results, route_start_s, OUTPUT_PATH)

if __name__ == "__main__":
    main()