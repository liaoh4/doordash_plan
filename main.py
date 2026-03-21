import pandas as pd
import os
import heapq
from datetime import datetime
from Class.order import Order
from Class.dasher import Dasher
from Utilities.utility import fix_to_original_year, calculate_metrics,simulate_route
from Class.dashernode import DasherNode
from Solver.merge import try_merge_general
from Utilities.plot import plot_dasher_routes


def save_to_csv(final_dashers, output_file='output.csv'):
    rows = []
    for route_id, dasher in enumerate(final_dashers):
        # dasher.route_nodes 存储格式为 [('P', order1), ('D', order1), ('P', order2)...]
        for idx, (node_type, order) in enumerate(dasher.route_nodes):
            # 获取对应的实际时间 (我们在 confirm 时存入 order 对象或 nodes 记录中)
            # 这里统一使用存入 order 的时间戳
            point_time = order.pickup_time if node_type == 'P' else order.dropoff_time

            rows.append({
                'Delivery ID': order.id,
                'Route ID': route_id,
                'Route Point Index': idx,
                'Route Point Type': 'Pickup' if node_type == 'P' else 'DropOff',
                'Route Point Time': int(point_time.timestamp())
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_file, index=False)
    print(f"DEBUG: 生成了 {len(rows)} 个路径点，涉及 {len(final_dashers)} 名骑手。")
def main():
    data_path = os.path.join(os.getcwd(), 'data', 'optimization_take_home.csv')

    try:
        # 1. 数据读取与预处理
        df = pd.read_csv(data_path)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['food_ready_time'] = pd.to_datetime(df['food_ready_time'])
        df['created_at'] = df['created_at'].apply(fix_to_original_year)
        df['food_ready_time'] = df['food_ready_time'].apply(fix_to_original_year)

        orders = [Order(row) for index, row in df.iterrows()]
        print(f"成功读取 {len(orders)} 条订单数据。")

        # 2. “一单一车”初始化 [cite: 32, 36]
        dashers = []
        for i, order in enumerate(orders):
            d = Dasher(dasher_id=i)
            # 初始化节点序列：每个单车骑手路径只有两个点 [P, D]
            nodes = [('P', order), ('D', order)]

            # 使用你新写的 simulate_route 计算时间
            valid, avg_dur, final_t, times = simulate_route(nodes, d.START_TIME)

            # 记录时间并确认
            order.pickup_time = times[0]
            order.dropoff_time = times[1]
            d.route_nodes = nodes
            d.current_time = final_t
            d.route = [order]
            dashers.append(d)

            # 2. 准备优先队列
        pq = [DasherNode(d) for d in dashers]
        heapq.heapify(pq)

        # 3. 核心合并循环
        final_dashers = []
        while pq:
            node_a = heapq.heappop(pq)
            dasher_a = node_a.dasher
            merged = False

            # 搜索宽度恢复到 100 左右平衡速度与质量
            for i in range(min(len(pq), 100)):
                if try_merge_general(dasher_a, pq[i].dasher):
                    pq.pop(i)
                    heapq.heapify(pq)
                    heapq.heappush(pq, DasherNode(dasher_a))
                    merged = True
                    break
            if not merged:
                final_dashers.append(dasher_a)

        # 4. 指标计算与保存
        calculate_metrics(final_dashers, len(orders))
        save_to_csv(final_dashers)


    except FileNotFoundError:
        print(f"无法找到文件: {data_path}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"发生错误: {e}")


if __name__ == "__main__":
    plot_dasher_routes(final_dashers)