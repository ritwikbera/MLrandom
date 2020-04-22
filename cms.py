import numpy as np
import mmh3

class CountMinSketch:

    def __init__(self, width, depth, seeds):
        self.width = width
        self.depth = depth
        self.table = np.zeros([depth, width]) 
        self.seed = seeds # np.random.randint(w, size = d) # create some seeds

    def increment(self, key):
        for i in range(0, self.depth):
            index = mmh3.hash(key, self.seed[i]) % self.width
            self.table[i, index] = self.table[i, index]+1

    def estimate(self, key):
        min_est = float('inf')
        for i in range(0, self.depth):
            index = mmh3.hash(key, self.seed[i]) % self.width
            if self.table[i, index] < min_est:
                min_est = self.table[i, index]
        return min_est
