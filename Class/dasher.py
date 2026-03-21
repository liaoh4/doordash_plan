from datetime import datetime, timedelta
from Utilities.utility import haversine


class Dasher:
    # 所有骑手从 2015-02-03 02:00 UTC 开始
    START_TIME = datetime(2002, 3, 15, 2, 0, 0)  # 对应 2/3/15 02:00
    SPEED = 4.5

    def __init__(self, dasher_id):
        self.id = dasher_id
        self.current_time = self.START_TIME
        self.current_lat = None
        self.current_lng = None
        self.route = []  # 存储已完成的订单对象

    def try_add_order(self, order):
        """
        核心逻辑：返回 (实际取货时间, 实际送达时间, 该单送货时长)
        """
        # 1. 确定去餐厅的行驶时间
        if self.current_lat is None:
            # 第一单：直接从餐厅开始，不需要行驶时间
            travel_to_p = 0
        else:
            dist_to_pickup = haversine(self.current_lat, self.current_lng, order.p_lat, order.p_lng)
            travel_to_p = dist_to_pickup / self.SPEED

        # 2. 计算实际取货时间 (Actual Pickup Time)
        # 到达餐厅的时间
        arrival_at_p = self.current_time + timedelta(seconds=travel_to_p)
        # 实际取货时间：不能早于食物准备好时间，且骑手至少从 02:00 开始工作
        actual_p_time = max(arrival_at_p, order.food_ready_time, self.START_TIME)

        # 3. 计算实际送达时间 (Actual Dropoff Time)
        dist_to_dropoff = haversine(order.p_lat, order.p_lng, order.d_lat, order.d_lng)
        travel_to_d = dist_to_dropoff / self.SPEED
        actual_d_time = actual_p_time + timedelta(seconds=travel_to_d)

        # 4. 计算送货时长 (Duration)
        duration = (actual_d_time - order.created_at).total_seconds() / 60.0

        return actual_p_time, actual_d_time, duration

    def confirm_order(self, order, pickup_time, dropoff_time):
        """
        更新骑手状态并记录订单的时间戳
        """
        # 记录订单的两个关键时间点，供后续生成 CSV 使用
        order.pickup_time = pickup_time
        order.dropoff_time = dropoff_time

        # 更新骑手当前状态
        self.current_time = dropoff_time
        self.current_lat = order.d_lat
        self.current_lng = order.d_lng
        self.route.append(order)