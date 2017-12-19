import time


class Timer:
    start = None
    end = None

    def __init__(self):
        self.start = None
        self.end = None

    def tick(self):
        self.start = time.clock()

    def tock(self):
        self.end = time.clock()
        return self.end - self.start
