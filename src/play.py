# for i in range(0,3):
#     for j in range(0,3):
#         if j == i:
#             continue
#         for k in range(0,3):
#             if k in (i,j):
#                 continue
#             print(i,j,k)
import copy
import itertools

class Combinator:
    def __init__(self, sz):
        self.sz = sz
        self.used = set()
        self.numbers = [i for i in range(0,sz)]
        for i in self.numbers:
            self.used.add(i)
    def __next__(self):
        res = copy.copy(self.numbers)
        last_num = self.numbers.pop()
        self.used.remove(last_num)
        # if last_num == 9
        return res


    def __iter__(self):
        return self


# c = Combinator(3)
# for x in c:
#     print(x)

def permutations(r,n):
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # permutations(range(3)) --> 012 021 102 120 201 210
    indices = list(range(n))
    cycles = list(range(n, n-r, -1))
    yield tuple(i for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(i for i in indices[:r])
                break
        else:
            return

for x in permutations(3, 10):
    print(x)



