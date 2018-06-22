
class Fenwick():
    def update(self, i, x):                     #add x to the ith position
        while i <= self.N:
            self.BIT[i-1] += x          #because we're working with an 1-based array
            i += i & (-i)                       #magic! don't touch!
    def query(self, i):                         #find the ith prefix sum
        s = 0
        while i > 0:
            s += self.BIT[i-1]
            i -= i & (-i)
        return s
    def __init__(self, l=[]):                   #initialize the fenwick tree
        self.N = len(l)
        self.BIT = [0 for i in range(self.N)]
        for i in range(1,self.N+1):
            self.update(i, l[i-1])

