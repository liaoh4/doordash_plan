"""
免疫优化算法 v2
==============
目标：固定50个dasher，最大化 avg deliveries/hour
约束：avg_dur ≤ 45 min 为软约束（惩罚项）

改进点：
1. 适应度函数：主目标是del/hr，45min作惩罚，但惩罚自适应（初期宽松后期严格）
2. 染色体表示：直接用 assignment[j] = dasher_idx（整数编码，更紧凑）
   等价于原二维0/1矩阵，但操作更高效
3. 更有意义的变异算子：
   - 负载均衡变异：把重载dasher的边缘订单转给轻载dasher
   - 时空紧凑变异：把时间/位置离群的订单重新分配给更近的dasher
4. 克隆数量与亲和度正相关，超变异率与亲和度负相关
5. 多样性维护：基于汉明距离的选择压制
"""
from datetime import datetime
import math, time, random, copy
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

# ── 常量 ──────────────────────────────────────────────────────
SPEED_MPS      = 4.5
ROUTE_START    = pd.Timestamp("2002-03-15 02:00:00")
TARGET_DUR_MIN = 45.0
N_DASHERS      = 50
DATA_PATH      = "optimization_take_home.csv"
OUTPUT_PATH    = "solution_output.csv"

# 免疫算法参数
POP_SIZE       = 40
N_ELITE        = 12
N_CLONE_BASE   = 6
N_REPLACE      = 8
MAX_ITER       = 300
TIME_LIMIT_S   = 60
MIN_DIVERSITY  = 0.03   # 最小汉明距离比例

# 惩罚参数：自适应，随迭代线性增大
PENALTY_INIT   = 0.5    # 初始惩罚系数
PENALTY_FINAL  = 3.0    # 最终惩罚系数

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
    df = df.sort_values("region_id").reset_index(drop=True)

    orders = []
    for _, row in df.iterrows():
        orders.append({
            "delivery_id":  int(row["delivery_id"]),
            "created_s":    row["created_ts"].timestamp(),
            "food_ready_s": row["food_ready_ts"].timestamp(),
            "region_id":    int(row["region_id"]),
            "pickup_lat":   row["pickup_lat"],
            "pickup_long":  row["pickup_long"],
            "dropoff_lat":  row["dropoff_lat"],
            "dropoff_long": row["dropoff_long"],
        })

    # region分段：{region_id: [order_indices]}
    region_order_idx = {}
    for j, o in enumerate(orders):
        region_order_idx.setdefault(o["region_id"], []).append(j)

    return orders, region_order_idx

# ── 路线仿真（贪心最近邻） ────────────────────────────────────
def simulate_dasher(order_list, route_start_s):
    """
    给定订单列表，用贪心最近邻执行。
    返回 (route_span_s, [delivery_duration_s, ...])
    """
    if not order_list:
        return 0.0, []

    pending = list(order_list)
    picked  = []
    cl = cn = None
    ct = route_start_s
    durs = []

    while pending or picked:
        best_act = best_o = None
        best_cost = float("inf")

        if cl is None:
            # 第一单：选food_ready最早的
            best_o   = min(pending, key=lambda o: o["food_ready_s"])
            best_act = "pickup"
            arrive   = max(ct, best_o["food_ready_s"])
        else:
            # 候选：所有dropoff + 所有pickup
            for o in picked:
                t = travel_sec(cl, cn, o["dropoff_lat"], o["dropoff_long"])
                arrive = ct + t
                if arrive < best_cost:
                    best_cost, best_act, best_o, _arrive = arrive, "dropoff", o, arrive
            for o in pending:
                t = travel_sec(cl, cn, o["pickup_lat"], o["pickup_long"])
                arrive = max(ct + t, o["food_ready_s"])
                if arrive < best_cost:
                    best_cost, best_act, best_o, _arrive = arrive, "pickup", o, arrive
            arrive = _arrive

        if best_act == "pickup":
            cl, cn = best_o["pickup_lat"], best_o["pickup_long"]
            ct = arrive
            pending.remove(best_o)
            picked.append(best_o)
        else:
            cl, cn = best_o["dropoff_lat"], best_o["dropoff_long"]
            ct = arrive
            picked.remove(best_o)
            durs.append(arrive - best_o["created_s"])

    return ct - route_start_s, durs

# ── 适应度评估 ────────────────────────────────────────────────
def evaluate(assignment, orders, region_order_idx, route_start_s, penalty_lambda):
    """
    assignment: np.array shape (n_orders,), 值为dasher_idx (0~N_DASHERS-1)
    返回 (fitness, del_per_hour, avg_dur_min)
    """
    # 按dasher分组
    dasher_orders = [[] for _ in range(N_DASHERS)]
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
    fitness = dph - penalty

    return fitness, dph, avg_dur

# ── 染色体初始化 ──────────────────────────────────────────────
def init_assignment(orders, region_order_idx):
    """
    整数编码：assignment[j] = 该订单分给哪个dasher
    按region比例分配dasher名额，region内均匀分配
    """
    n_orders = len(orders)
    assignment = np.zeros(n_orders, dtype=np.int32)

    total = n_orders
    rids  = sorted(region_order_idx.keys())
    # 按订单比例分配dasher数量
    dasher_per_region = {}
    used = 0
    for k, rid in enumerate(rids):
        cnt = len(region_order_idx[rid])
        if k < len(rids) - 1:
            n = max(1, round(N_DASHERS * cnt / total))
        else:
            n = N_DASHERS - used
        dasher_per_region[rid] = list(range(used, used + n))
        used += n

    for rid, idxs in region_order_idx.items():
        dashers = dasher_per_region[rid]
        idxs_s  = idxs[:]
        random.shuffle(idxs_s)
        for k, j in enumerate(idxs_s):
            assignment[j] = dashers[k % len(dashers)]

    return assignment

# ── 变异算子 ──────────────────────────────────────────────────
def mutate_random(asgn, orders, region_order_idx, rate):
    """随机变异：以rate概率把订单重新分配给同region另一dasher"""
    new_asgn = asgn.copy()
    # 构建region → dasher集合
    region_dashers = {}
    for rid, idxs in region_order_idx.items():
        ds = set(new_asgn[idxs].tolist())
        region_dashers[rid] = list(ds) if len(ds) > 1 else list(range(N_DASHERS))

    for j in range(len(orders)):
        if random.random() < rate:
            rid  = orders[j]["region_id"]
            cands = [d for d in region_dashers[rid] if d != new_asgn[j]]
            if cands:
                new_asgn[j] = random.choice(cands)
    return new_asgn

def mutate_load_balance(asgn, orders, region_order_idx):
    """
    负载均衡变异：找最重载的dasher，把它的一个订单
    转给同region最轻载的dasher
    """
    new_asgn = asgn.copy()
    for rid, idxs in region_order_idx.items():
        region_asgn = new_asgn[idxs]
        # 统计各dasher的订单数
        load = {}
        for d in region_asgn:
            load[d] = load.get(d, 0) + 1
        if len(load) < 2:
            continue
        heavy = max(load, key=load.get)
        light = min(load, key=load.get)
        if load[heavy] - load[light] <= 1:
            continue
        # 把heavy的一个随机订单给light
        heavy_orders = [j for j in idxs if new_asgn[j] == heavy]
        if heavy_orders:
            j = random.choice(heavy_orders)
            new_asgn[j] = light
    return new_asgn

def mutate_spatial(asgn, orders, region_order_idx):
    """
    时空紧凑变异：找pickup位置离当前dasher重心最远的订单，
    重新分配给pickup位置更近的dasher
    """
    new_asgn = asgn.copy()
    for rid, idxs in region_order_idx.items():
        # 各dasher的pickup重心
        dasher_orders = {}
        for j in idxs:
            d = new_asgn[j]
            dasher_orders.setdefault(d, []).append(j)

        if len(dasher_orders) < 2:
            continue

        # 找每个dasher中距重心最远的订单
        for d, d_idxs in dasher_orders.items():
            if len(d_idxs) <= 1:
                continue
            clat = sum(orders[j]["pickup_lat"]  for j in d_idxs) / len(d_idxs)
            clon = sum(orders[j]["pickup_long"] for j in d_idxs) / len(d_idxs)
            # 最远的订单
            j_far = max(d_idxs,
                        key=lambda j: haversine(orders[j]["pickup_lat"],
                                                orders[j]["pickup_long"], clat, clon))
            # 找最近的其他dasher
            best_d, best_dist = None, float("inf")
            for d2, d2_idxs in dasher_orders.items():
                if d2 == d or not d2_idxs:
                    continue
                c2lat = sum(orders[j]["pickup_lat"]  for j in d2_idxs) / len(d2_idxs)
                c2lon = sum(orders[j]["pickup_long"] for j in d2_idxs) / len(d2_idxs)
                dist = haversine(orders[j_far]["pickup_lat"],
                                 orders[j_far]["pickup_long"], c2lat, c2lon)
                if dist < best_dist:
                    best_dist, best_d = dist, d2
            if best_d is not None and random.random() < 0.5:
                new_asgn[j_far] = best_d
    return new_asgn

def mutate_time_compact(asgn, orders, region_order_idx):
    """
    时间紧凑变异：把food_ready_time离当前dasher平均时间最远的订单
    重新分配给时间上更接近的dasher
    """
    new_asgn = asgn.copy()
    for rid, idxs in region_order_idx.items():
        dasher_orders = {}
        for j in idxs:
            dasher_orders.setdefault(new_asgn[j], []).append(j)
        if len(dasher_orders) < 2:
            continue
        for d, d_idxs in dasher_orders.items():
            if len(d_idxs) <= 1:
                continue
            avg_t = sum(orders[j]["food_ready_s"] for j in d_idxs) / len(d_idxs)
            j_far = max(d_idxs, key=lambda j: abs(orders[j]["food_ready_s"] - avg_t))
            t_far = orders[j_far]["food_ready_s"]
            best_d, best_diff = None, float("inf")
            for d2, d2_idxs in dasher_orders.items():
                if d2 == d or not d2_idxs:
                    continue
                avg_t2 = sum(orders[j]["food_ready_s"] for j in d2_idxs) / len(d2_idxs)
                diff = abs(t_far - avg_t2)
                if diff < best_diff:
                    best_diff, best_d = diff, d2
            if best_d is not None and random.random() < 0.5:
                new_asgn[j_far] = best_d
    return new_asgn

# ── 超变异（组合多种算子） ─────────────────────────────────────
def hypermutate(asgn, orders, region_order_idx, rate, rank):
    """根据rank选择不同强度的变异策略"""
    new_asgn = asgn.copy()
    # 精英用轻变异；低rank用重变异
    if rank < 3:
        # 精英：只做小幅随机 + 负载均衡
        new_asgn = mutate_random(new_asgn, orders, region_order_idx, rate * 0.5)
        if random.random() < 0.5:
            new_asgn = mutate_load_balance(new_asgn, orders, region_order_idx)
    elif rank < 7:
        # 中等：随机 + 时空紧凑
        new_asgn = mutate_random(new_asgn, orders, region_order_idx, rate)
        op = random.choice([mutate_spatial, mutate_time_compact, mutate_load_balance])
        new_asgn = op(new_asgn, orders, region_order_idx)
    else:
        # 低rank：激进变异（多次）
        new_asgn = mutate_random(new_asgn, orders, region_order_idx, rate * 2)
        for op in random.sample([mutate_spatial, mutate_time_compact, mutate_load_balance], 2):
            new_asgn = op(new_asgn, orders, region_order_idx)
    return new_asgn

# ── 汉明距离（多样性度量） ────────────────────────────────────
def hamming(a, b):
    return np.sum(a != b) / len(a)

# ── 输出CSV ───────────────────────────────────────────────────
def build_output(assignment, orders, route_start_s, output_path):
    rows = []
    route_id = 0
    for d_idx in range(N_DASHERS):
        d_orders = [orders[j] for j in range(len(orders)) if assignment[j] == d_idx]
        if not d_orders:
            continue

        pending = list(d_orders); picked = []
        cl = cn = None; ct = route_start_s
        stops = []

        while pending or picked:
            best_act = best_o = None; best_cost = float("inf")
            if cl is None:
                best_o = min(pending, key=lambda o: o["food_ready_s"])
                best_act = "pickup"
                arrive = max(ct, best_o["food_ready_s"])
            else:
                for o in picked:
                    t = travel_sec(cl, cn, o["dropoff_lat"], o["dropoff_long"])
                    arr = ct + t
                    if arr < best_cost:
                        best_cost, best_act, best_o, arrive = arr, "dropoff", o, arr
                for o in pending:
                    t = travel_sec(cl, cn, o["pickup_lat"], o["pickup_long"])
                    arr = max(ct + t, o["food_ready_s"])
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
            rows.append({"Route ID": route_id, "Route Point Index": idx,
                         "Delivery ID": did, "Route Point Type": ptype,
                         "Route Point Time": t})
        route_id += 1

    pd.DataFrame(rows).sort_values(["Route ID","Route Point Index"]).to_csv(output_path, index=False)
    print(f"Output saved → {output_path}")

# ── 主算法 ────────────────────────────────────────────────────
def immune_optimize(orders, region_order_idx, route_start_s):
    n_orders = len(orders)
    print(f"订单={n_orders}, Dasher={N_DASHERS}, 种群={POP_SIZE}, 上限={MAX_ITER}代/{TIME_LIMIT_S}s")
    print(f"Region分布: { {rid: len(idxs) for rid,idxs in region_order_idx.items()} }\n")

    t0 = time.time()

    # 初始化种群
    population = [init_assignment(orders, region_order_idx) for _ in range(POP_SIZE)]

    best_fitness = -999.0
    best_asgn    = None
    best_dph     = 0.0
    best_dur     = 0.0

    for iteration in range(MAX_ITER):
        if time.time() - t0 > TIME_LIMIT_S:
            print(f"时间到，iter {iteration}")
            break

        # 自适应惩罚系数（随迭代线性增大）
        progress = min(1.0, iteration / (MAX_ITER * 0.7))
        lam = PENALTY_INIT + (PENALTY_FINAL - PENALTY_INIT) * progress

        # 计算亲和度
        scored = []
        for asgn in population:
            fit, dph, dur = evaluate(asgn, orders, region_order_idx, route_start_s, lam)
            scored.append((fit, dph, dur, asgn))
        scored.sort(key=lambda x: -x[0])

        # 更新全局最优（用固定惩罚系数评估，便于比较）
        fit0, dph0, dur0, asgn0 = scored[0]
        if dph0 > best_dph or (dph0 == best_dph and dur0 < best_dur):
            best_fitness, best_dph, best_dur, best_asgn = fit0, dph0, dur0, asgn0
            print(f"iter {iteration:4d} [λ={lam:.2f}]: "
                  f"del/hr={best_dph:.4f} | avg_dur={best_dur:.2f}min | "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

        # 克隆选择：精英个体按rank克隆
        elite  = scored[:N_ELITE]
        clones = []
        for rank, (fit, dph, dur, asgn) in enumerate(elite):
            n_clone  = max(1, N_CLONE_BASE - rank // 3)
            mut_rate = 0.05 + 0.15 * (rank / N_ELITE)  # 精英变异率低，末位变异率高
            for _ in range(n_clone):
                clone = hypermutate(asgn, orders, region_order_idx, mut_rate, rank)
                clones.append(clone)

        # 受体编辑：注入新随机个体增加多样性
        new_randoms = [init_assignment(orders, region_order_idx) for _ in range(N_REPLACE)]

        # 合并评估
        all_candidates = [a for _,_,_,a in elite] + clones + new_randoms
        all_scored = []
        for asgn in all_candidates:
            fit, dph, dur = evaluate(asgn, orders, region_order_idx, route_start_s, lam)
            all_scored.append((fit, dph, dur, asgn))
        all_scored.sort(key=lambda x: -x[0])

        # 多样性选择：保留适应度高且彼此不太相似的个体
        selected = [all_scored[0][3]]
        for fit, dph, dur, asgn in all_scored[1:]:
            if len(selected) >= POP_SIZE:
                break
            min_h = min(hamming(asgn, s) for s in selected)
            if min_h >= MIN_DIVERSITY:
                selected.append(asgn)

        # 补充随机个体
        while len(selected) < POP_SIZE:
            selected.append(init_assignment(orders, region_order_idx))

        population = selected

    elapsed = time.time() - t0
    print(f"\n完成，耗时 {elapsed:.1f}s")
    return best_asgn, best_dph, best_dur

# ── 入口 ──────────────────────────────────────────────────────
def main():
    random.seed(42)
    np.random.seed(42)
    start_time = datetime.now()

    orders, region_order_idx = load_data(DATA_PATH)
    route_start_s = ROUTE_START.timestamp()

    print("=== 免疫优化算法 v2 ===\n")
    best_asgn, dph, dur = immune_optimize(orders, region_order_idx, route_start_s)
    end_time = datetime.now()
    print(f"计算时间:               {end_time - start_time}")
    active = sum(1 for d in range(N_DASHERS)
                 if any(best_asgn[j] == d for j in range(len(orders))))
    print(f"\n── 最终结果 ──────────────────────────────")
    print(f"  Avg del/hr   : {dph:.4f}")
    print(f"  Avg dur      : {dur:.2f} min  ({'✓' if dur<=45 else f'软约束超{dur-45:.1f}min'})")
    print(f"  Active dashers: {active} / {N_DASHERS}")

    build_output(best_asgn, orders, route_start_s, OUTPUT_PATH)

if __name__ == "__main__":
    main()
