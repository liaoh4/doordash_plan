import math
from datetime import datetime
from datetime import timedelta

def haversine(lat1, lon1, lat2, lon2):
    # 题目要求：Assume dashers travel in straight lines [cite: 34]
    R = 6371  # 地球半径 (km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c * 1000  # 返回单位：米 (m)

def parse_time(time_str):
    # 转换 ISO 格式的 UTC 字符串 [cite: 12]
    return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')

def fix_to_original_year(dt):
    """
    不再强制改为 2015，而是确保日期与原始 CSV 解析出的年份一致。
    如果 pd.to_datetime 将 2/3/15 解析成了 2002-03-15，
    那我们就用 2002-03-15 作为基准。
    """
    # 只要确保小时、分钟和秒数正确即可
    return dt


def calculate_metrics(final_dashers, total_orders_count):
    # 1. 计算平均送货时长 (Average Delivery Duration) [cite: 24]
    total_duration_min = 0
    for dasher in final_dashers:
        for order in dasher.route:
            duration = (order.dropoff_time - order.created_at).total_seconds() / 60.0
            total_duration_min += duration

    avg_duration = total_duration_min / total_orders_count

    # 2. 计算每小时平均配送量 (Deliveries/Hour) [cite: 28]
    # 公式：总单数 / sum(每个骑手的路径时长)
    # 每个骑手的路径时长 = 最后一次送达时间 - 2:00 UTC [cite: 30, 31]
    start_time = datetime(2002, 3, 15, 2, 0, 0)
    total_route_hours = 0

    for dasher in final_dashers:
        route_duration_sec = (dasher.current_time - start_time).total_seconds()
        total_route_hours += (route_duration_sec / 3600.0)

    efficiency = total_orders_count / total_route_hours

    print("\n--- 最终指标统计 ---")
    print(f"平均送货时长: {avg_duration:.2f} 分钟 (目标 < 45)")
    print(f"平均效率 (Deliveries/Hour): {efficiency:.4f}")
    print(f"使用骑手总数: {len(final_dashers)}")

    # 验证是否满足题目目标
    if avg_duration < 45:
        print("✅ 满足服务水平约束。")
    else:
        print("❌ 警告：平均送货时长超过 45 分钟！")

    if len(final_dashers) < 50:
        print("⭐ 达成加分项：使用骑手少于 50 名。")


def simulate_route(test_nodes, start_time):
    current_time = start_time
    current_lat, current_lng = None, None
    node_times = []
    order_dropoffs = {}

    for node_type, order in test_nodes:
        target_lat = order.p_lat if node_type == 'P' else order.d_lat
        target_lng = order.p_lng if node_type == 'P' else order.d_lng

        # 恢复逻辑：第一点不计距离，后续点计入
        if current_lat is None:
            travel_sec = 0
        else:
            dist = haversine(current_lat, current_lng, target_lat, target_lng)
            travel_sec = dist / 4.5

        arrival_time = current_time + timedelta(seconds=travel_sec)
        actual_time = max(arrival_time, order.food_ready_time) if node_type == 'P' else arrival_time

        node_times.append(actual_time)
        current_time = actual_time
        current_lat, current_lng = target_lat, target_lng
        if node_type == 'D': order_dropoffs[order.id] = actual_time

    # 计算平均时长
    unique_orders = {o for t, o in test_nodes}
    avg_dur = sum((order_dropoffs[o.id] - o.created_at).total_seconds() for o in unique_orders) / (
                len(unique_orders) * 60.0)
    return avg_dur < 45.0, avg_dur, current_time, node_times