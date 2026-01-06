class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.stats = {
            "total_requests": 0,
            "accepted_requests": 0,
            "total_revenue": 0.0,
            "acceptance_rate": 0.0
        }

    def update(self, accepted: bool, revenue: float = 0.0):
        self.stats["total_requests"] += 1
        if accepted:
            self.stats["accepted_requests"] += 1
            self.stats["total_revenue"] += revenue

        if self.stats["total_requests"] > 0:
            self.stats["acceptance_rate"] = (
                    self.stats["accepted_requests"] / self.stats["total_requests"]
            )

    def get_stats(self):
        return self.stats