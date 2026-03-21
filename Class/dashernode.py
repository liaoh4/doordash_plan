class DasherNode:
    def __init__(self, dasher):
        self.dasher = dasher
        # 优先级 = 结束时间戳
        # 我们希望最早结束任务的骑手排在最前面，去尝试接新单
        self.priority = self.dasher.current_time.timestamp()

    def __lt__(self, other):
        return self.priority < other.priority