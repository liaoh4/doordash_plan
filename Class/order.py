from Utilities.utility import parse_time, haversine


class Order:
    def __init__(self, data):
        self.id = data['delivery_id']
        # data['created_at'] 现在已经是 Timestamp 对象了，直接赋值即可
        self.created_at = data['created_at']
        self.food_ready_time = data['food_ready_time']
        self.p_lat = data['pickup_lat']
        self.p_lng = data['pickup_long']
        self.d_lat = data['dropoff_lat']
        self.d_lng = data['dropoff_long']
        self.dropoff_time = None
        self.pickup_time = None
    def get_duration(self):
        # 送货时长 = (送达时间) - (创建时间) [cite: 23]
        if self.dropoff_time:
            return max(0.0, (self.dropoff_time - self.created_at).total_seconds() / 60.0)
        return 0