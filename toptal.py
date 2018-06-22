
import math
def isPrime(N):
    sq = int(math.sqrt(N)) + 1

    if N < 2: return False
    elif N == 2: return True
    else:
        for i in range(3, sq, 2):
            if N%i == 0:
                return False

    return True

isPrime(15)

def x():
    if 2 in {3, 4, 2}:
        return True
    else:
        return False


# def solution(A, B):
#     m = 100001
#     available = set()
#     for i in range(len(A)):
#         if A[i] == B[i]:
#             available.add(A[i])
#         else:
#             if A[i] > B[i]:
#                 available.add(A[i])
#                 if (int(min(B[i], m)) in available):
#                     continue
#                 else:
#                     m = min
#             else:
#                 available.add(B[i])
#                 if (int(min(A[i], m)) in available):
#                     continue
#                 else:
#                     m = min
#
#     return m
#
#
# def solution(A, B):
#     m = 100001
#     available = set()
#     for i in range(len(A)):
#         if A[i] == B[i]:
#             available.add(A[i])
#         else:
#             if A[i] > B[i]:
#                 available.add(A[i])
#                 x = min(B[i], m)
#                 if (x in available):
#                     continue
#                 else:
#                     m = min
#             else:
#                 available.add(B[i])
#                 x = min(A[i], m)
#                 if (x in available):
#                     continue
#                 else:
#                     m = min
#
#     return m

# [1, 2], [1, 2] -> 3
# [1,3,6,4], [1,2,3,4]
def solution(A, B):
    m = 100001
    available = set()
    for i in range(len(A)):
        if A[i] == B[i]:
            available.add(A[i])
        else:
            if A[i] > B[i]:
                available.add(A[i])
                x = min(B[i], m)
                if (x in available):
                    continue
                else:
                    m = x
            else:
                available.add(B[i])
                x = min(A[i], m)
                if (x in available):
                    continue
                else:
                    m = x

    if m == 100001:
        return max(A) + 1
    else:
        return m

# solution([1,3,6,4], [1,2,3,4]) # 2
# solution([1, 2], [1, 2]) # 3
solution([4,5,7,9,0,3,6,2,7], [7,3,6,45,32,45,8,12,4]) # 3
solution([4,5,7,9,1,3,6,2,7], [7,3,6,45,32,45,8,12,4])