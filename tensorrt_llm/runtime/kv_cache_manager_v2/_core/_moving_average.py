class MovingAverage:
    __slots__ = ("decay", "avg", "weight", "num_updates")
    decay: float
    avg: float
    weight: float
    num_updates: int

    def __init__(self, decay: float = 0.9999):
        self.decay = decay
        self.avg = 0.0
        self.weight = 0.0
        self.num_updates = 0

    def update(self, value: int | float) -> float:
        self.weight = 1.0 + self.decay * self.weight
        self.avg += (value - self.avg) / self.weight
        self.num_updates += 1
        return self.avg

    @property
    def value(self) -> float:
        return self.avg


class Average:
    __slots__ = ("sum", "count")
    sum: float
    count: int

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: int | float) -> None:
        self.sum += value
        self.count += 1

    @property
    def value(self) -> float:
        return self.sum / self.count
