from datetime import timedelta
from Utilities.utility import haversine,simulate_route


def try_merge_general(main_dasher, extra_dasher):
    orders_to_add = extra_dasher.route
    current_temp_nodes = list(main_dasher.route_nodes)

    for next_order in orders_to_add:
        best_avg = 43.0
        best_config = None
        n = len(current_temp_nodes)

        # 全位置暴力搜索：i 为 Pickup 位置，j 为 DropOff 位置
        for i in range(n + 1):
            for j in range(i + 1, n + 2):
                test_nodes = current_temp_nodes[:i] + [('P', next_order)] + \
                             current_temp_nodes[i:j - 1] + [('D', next_order)] + \
                             current_temp_nodes[j - 1:]

                valid, avg_dur, final_t, times = simulate_route(test_nodes, main_dasher.START_TIME)
                if valid and avg_dur < best_avg:
                    best_avg = avg_dur
                    best_config = (test_nodes, times)

        if best_config:
            current_temp_nodes, new_times = best_config
            # 同步更新订单时间戳
            for idx, (nt, o) in enumerate(current_temp_nodes):
                if nt == 'P':
                    o.pickup_time = new_times[idx]
                else:
                    o.dropoff_time = new_times[idx]
        else:
            return False  # 只要一单插不进，整个合并取消

    main_dasher.route_nodes = current_temp_nodes
    main_dasher.route = list(set(main_dasher.route + orders_to_add))
    main_dasher.current_time = simulate_route(current_temp_nodes, main_dasher.START_TIME)[2]
    return True