
# this is for contigiuous positive, with only one missing
def Solution(A):
    l = len(A)
    found = [-1]*(l+1)
    m = 1
    pcnt = 0

    for i in range(0, l):
        if A[i] <= 0:
            continue
        else:
            if m == A[i]:
              m += 1

            if found[A[i]] != -1:
                continue
            else:
                found[A[i]] = A[i]
                pcnt += 1

    print(pcnt)
    print(m)

    if pcnt == 0:
        return 1

    if pcnt == m - 1:
        return m

    n = m
    for k in range(m, pcnt+1):
        if found[k] == -1:
            return k
        else:
            n = max(n, k)

    return (n+1)

Solution([1, 4, 3, 2])
Solution([-1, -3, 2])


def solution(A):
    occurrence = [False] * (len(A) + 1)
    for item in A:
        if 1 <= item <= len(A) + 1:
            occurrence[item - 1] = True

    for index in range(len(A) + 1):
        if occurrence[index] == False:
            return index + 1

    return -1


# MissingInteger

def solution(A):
    l = len(A)
    available = [False] * l
    m = 1

    for i in A:
        if i > l:
            return 0
        else:
            available[i-1] = True

            if m == i:
                m += 1

    for j in range(m-1, l):
        if not available[j]:
            return 0

    return 1


# MaxCounters

# Naive Implementation
def solution(N, A):
    x = [0] * N

    for i in A:
        if 1 <= i <= N:
            x[i] += 1
        else:
          x[:] = [max(x)]*N

    return x


def solution(N, A):
    curr_max, max_counter = 0, 0
    res = [0]*N

    for i in A:
        if 1 <= i <= N:
            if max_counter > res[i-1]:
                res[i-1] = max_counter

            res[i-1] += 1

            if curr_max < res[i-1]:
                curr_max = res[i-1]

        else:
            max_counter = curr_max

    for i in range(0, N):
        if res[i] < max_counter:
            res[i] = max_counter

    return res


# Lesson 5: CountDiv
def solution(A, B, K):
    result = B//K - A//K

    if A % K == 0:
        result += 1

    return result


# Lesson 5: PassingCars
def solution(A):
    pass


def factorial(n):
    if n <= 1: return 1
    else:
        return n * factorial(n-1)


from math import factorial

print("The factorial of 23 is : ", end="")
print(factorial(23))


# RuntimeError: maximum recursion depth exceeded in comparison
# This happens because python stop calling recursive function
# after 1000 calls by default. To change this behavior you need
# to amend the code as follows.

import sys
sys.setrecursionlimit(3000)


def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)


# An iterative solution for the problem is also easy to write,
# though the recursive solution looks more like the mathematical
# definition:
def fibi(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a


# solution: memoiztion
memo = {0: 0, 1: 1}  # <- this is the way to initialize a dict
def fibm(n):
    if not n in memo:
        memo[n] = fibm(n-1) + fibm(n-2)
    return memo[n]

# [0, 2] + [4, 5]
# [0, 2, 4, 5]
def pascal(n):
    if n == 1:
        return [1]
    else:
        p_line = pascal(n-1)
        line = [p_line[i]+p_line[i+1] for i in range(len(p_line)-1)]
        line.insert(0, 1)
        line.append(1)
    return line


# Advantages of Recursion
# Recursive functions make the code look clean and elegant.
# A complex task can be broken down into simpler sub-problems using recursion.
# Sequence generation is easier with recursion than using some nested iteration.
#
# Disadvantages of Recursion
# Sometimes the logic behind recursion is hard to follow through.
# Recursive calls are expensive (inefficient) as they take up a lot of memory and time.
# Recursive functions are hard to debug.


# importing operator for operator functions
from operator import mul
from functools import reduce
reduce(mul, [(10-i) for i in range(0,3)]) # 720

import itertools
lis = [ 1, 3, 4, 10, 4 ]
print ("The summation of list using accumulate is :",end="")
print (list(itertools.accumulate(lis,lambda x,y : x+y)))
# The summation of list using accumulate is :[1, 4, 8, 18, 22]

# priting summation using reduce()
print ("The summation of list using reduce is :",end="")
print (reduce(lambda x,y:x+y,lis))
# The summation of list using reduce is :22

# [1 0 0 0 1 1 0 0 1 0] 6 6 1 -> 13
def solution(A):
    west = 0  # The number of west-driving cars so far
    passing = 0  # The number of passing

    for index in range(len(A) - 1, -1, -1):
        # Travel the list from the end to the beginning
        if A[index] == 0:  # A east-driving car
            passing += west
            if passing > 1000000000:
                return -1
        else:  # A west-driving car
            west += 1

    return passing


# Lesson 05: GenomicRangeQuery
# https://app.codility.com/programmers/lessons/5-prefix_sums/genomic_range_query/
# last seen from right
def solution(S, P, Q):
    result = []
    DNA_len = len(S)
    mapping = {"A": 1, "C": 2, "G": 3, "T": 4}
    # next_nucl is used to store the position information
    # next_nucl[0] is about the "A" nucleotides, [1] about "C"
    #    [2] about "G", and [3] about "T"
    # next_nucl[i][j] = k means: for the corresponding nucleotides i,
    #    at position j, the next corresponding nucleotides appears
    #    at position k (including j)
    # k == -1 means: the next corresponding nucleotides does not exist
    next_nucl = [[-1] * DNA_len, [-1] * DNA_len, [-1] * DNA_len, [-1] * DNA_len]
    next_nucl = [[-1, -1, -1, -1]] * DNA_len
    # Scan the whole DNA sequence, and retrieve the position information
    next_nucl[mapping[S[-1]] - 1][-1] = DNA_len - 1
    for index in range(DNA_len - 2, -1, -1):
        next_nucl[0][index] = next_nucl[0][index + 1]
        next_nucl[1][index] = next_nucl[1][index + 1]
        next_nucl[2][index] = next_nucl[2][index + 1]
        next_nucl[3][index] = next_nucl[3][index + 1]
        next_nucl[mapping[S[index]] - 1][index] = index

    for index in range(DNA_len-1, -1, -1):
        for j in range(4):
            next_nucl[j][index] = next_nucl[j][index + 1]

        next_nucl[mapping[S[index]] - 1][index] = index

    # If the distance between Q and P is lower than the distance to the last seen genome,
    # we have found the right candidate.
    for index in range(0, len(P)):
        if next_nucl[0][P[index]] != -1 and next_nucl[0][P[index]] <= Q[index]:
            result[index] = 1
        elif next_nucl[1][P[index]] != -1 and next_nucl[1][P[index]] <= Q[index]:
            result[index] = 2
        elif next_nucl[2][P[index]] != -1 and next_nucl[2][P[index]] <= Q[index]:
            result[index] = 3
        else:
            result[index] = 4

    for index in range(0, len(P)):
        if -1 < next_nucl[0][P[index]] <= Q[index]:
            result[index] = 1
        elif -1 < next_nucl[1][P[index]] <= Q[index]:
            result[index] = 2
        elif -1 < next_nucl[2][P[index]] <= Q[index]:
            result[index] = 3
        else:
            result[index] = 4

    return result

# last seen from left
def writeCharToList(S, last_seen, c, idx):
    if S[idx] == c:
        last_seen[idx] = idx
    elif idx > 0:
        last_seen[idx] = last_seen[idx - 1]


def solution(S, P, Q):
    if len(P) != len(Q):
        raise Exception("Invalid input")

    last_seen_A = [-1] * len(S)
    last_seen_C = [-1] * len(S)
    last_seen_G = [-1] * len(S)
    last_seen_T = [-1] * len(S)

    for idx in range(len(S)):
        writeCharToList(S, last_seen_A, 'A', idx)
        writeCharToList(S, last_seen_C, 'C', idx)
        writeCharToList(S, last_seen_G, 'G', idx)
        writeCharToList(S, last_seen_T, 'T', idx)

    solution = [0] * len(Q)

    # P[idx] <= last_seen_A[Q[idx]] <= Q[idx] -> A which is 1
    for idx in range(len(Q)):
        if last_seen_A[Q[idx]] >= P[idx]:
            solution[idx] = 1
        elif last_seen_C[Q[idx]] >= P[idx]:
            solution[idx] = 2
        elif last_seen_G[Q[idx]] >= P[idx]:
            solution[idx] = 3
        elif last_seen_T[Q[idx]] >= P[idx]:
            solution[idx] = 4
        else:
            raise Exception("Should never happen")

    return solution

# TODO
# Not there yet
# Example test:    ('CAGCCTA', [2, 5, 0], [4, 5, 6])
# WRONG ANSWER  (got [1, 1, 1] expected [2, 4, 1])
# Terse version
def solution(S, P, Q):
    ls, lq = len(S), len(Q)
    last_seen = [[-1, -1, -1, -1]] * ls
    mapping = "ACGT"
    solution = [0] * lq

    def writeCharToList(j, idx):
        if S[idx] == mapping[j]:
            last_seen[idx][j] = idx
        elif idx > 0:
            last_seen[idx][j] = last_seen[idx - 1][j]

    for idx in range(ls):
        for j in range(4):
            writeCharToList(j, idx)

    # P[idx] <= last_seen_A[Q[idx]] <= Q[idx] -> A which is 1
    for idx in range(lq):
        for j in range(4):
            if last_seen[Q[idx]][j] >= P[idx]:
                solution[idx] = j+1
                break

    return solution


# MinAvgTwoSlice
# Simple solution with prefix sum
from itertools import accumulate
from operator import add

def solution(A):
    N = len(A)

    pos = 0
    min_avg_value = (A[0] + A[1]) / 2.0
    for ind in range(N - 1):
        avg2 = (A[ind] + A[ind + 1]) / 2.0

        if avg2 < min_avg_value:
            min_avg_value = avg2
            pos = ind

        if ind <= N - 3:
            avg3 = (A[ind] + A[ind + 1] + A[ind + 2]) / 3.0

            if avg3 < min_avg_value:
                min_avg_value = avg3
                pos = ind

    return pos


# MaxProductOfThree
# After sorting the largest product can be found as a combination
# of the last three elements. Additionally, two negative numbers add to a positive,
# so by multiplying the two largest negatives with the largest positive,
# we get another candidate. If all numbers are negative,
# the three largest (closest to 0) still get the largest element!

def solution(A):
    if len(A) < 3:
        raise Exception("Invalid input")

    A.sort()

    return max(A[0] * A[1] * A[-1], A[-1] * A[-2] * A[-3])



# Distinct

def solution(A):
    S = set()
    cnt = 0
    for i in range(len(A)):
        if A[i] in S:
            continue
        else:
            cnt += 1
            S.add(A[i])

    return cnt



# Lession 6: Triangle
# On one hand, there is no false triangular. Since the array is sorted,
# we already know A[index] < = A[index+1] <= A[index+2],
# and all values are positive. A[index] <= A[index+2],
# so it must be true that A[index] < A[index+1] + A[index+2].
# Similarly, A[index+1] < A[index] + A[index+2]. Finally, we ONLY
# need to check A[index]+A[index+1] > A[index+2] to confirm the existence of triangular.
#
# On the other hand, there is no underreporting triangular. If the inequality can hold for
#     three out-of-order elements, to say, A[index]+A[index+m] > A[index+n], where n>m>1.
#     Again, because the array is sorted, we must have A[index] < = A[index+m-1]
#     and A[index+m+1] <= A[index + n]. So A[index+m-1] +A[index+m] >= A[index]+A[index+m] >
#     A[index+n] >= A[index+m+1]. After simplification, A[index+m-1] +A[index+m] >
#     A[index+m+1]. In other words, if we have any inequality holding
#     for out-of-order elements, we MUST have AT LEAST an inequality holding
#     for three consecutive elements.
def Solution(A):
    l = len(A)
    A.sort()

    for i in range(0, l-2):
        if A[i] + A[i+1] > A[i+2]:
            return 1

    return 0


# CountTriangles
def solution(A):
    n = len(A)
    result = 0

    A.sort()

    for first in range(n - 2):
        third = first + 2
        for second in range(first + 1, n - 1):
            while third < n and A[first] + A[second] > A[third]:
                third += 1
            result += third - second - 1

    return result


# Lesson 7: Fish
def solution(A, B):
    survived = 0
    stack = []

    for i in range(0, len(A)):
        if B[i] == 0:
            # the upstream can keep eating as long as it is bigger or
            # there are no downstream fishes
            while len(stack) != 0:
                if stack[-1] > A[i]:
                    break
                else:
                    stack.pop()
            else:
                survived += 1
        else:
            stack.append(A[i])

    survived += len(stack)

    return survived

# For prepending lists, insert is the best
# In [1]: %timeit ([1]*1000000).insert(0, 0)
# 100 loops, best of 3: 4.62 ms per loop
#
# In [2]: %timeit ([1]*1000000)[0:0] = [0]
# 100 loops, best of 3: 4.55 ms per loop
#
# In [3]: %timeit [0] + [1]*1000000
# 100 loops, best of 3: 8.04 ms per loop

# Lesson 7: Nesting
def solution(S):
    cnt = 0
    for c in S:
       if c == '(':
           cnt += 1
       elif c == ')':
           cnt -= 1

    return 1 if cnt == 0 else 0

import math
# here it marks non-prime as True
def sieveOfEratosthenis(N):
    result = [2]
    sq = int(math.sqrt(N)) + 1
    sieve = list(range(4, N+1, 2))
    for i in range(3, N+1, 2):
        if i < sq:
            for j in range(i*i, N, 2*i):
                sieve.append(j)

        if i not in sieve:
            result.append(i)

    return result

def isPrime(N):
    sq = int(math.sqrt(N)) + 1

    if N < 2: return False
    elif N == 2: return True
    else:
        for i in range(3, sq, 2):
            if N%i == 0:
                return False

    return True


# geekforgeeks
def SieveOfEratosthenes(n):
    # Create a boolean array "prime[0..n]" and initialize
    #  all entries it as true. A value in prime[i] will
    # finally be false if i is Not a prime, else true.
    # here it marks prim as true
    prime = [True for i in range(n + 1)]
    p = 2
    while (p * p <= n):
        # If prime[p] is not changed, then it is a prime
        if (prime[p] == True):
            # Update all multiples of p
            for i in range(p * 2, n + 1, p):
                prime[i] = False
        p += 1

    # Print all prime numbers
    return list(filter(lambda x: prime[x], range(2, n+1)))


def semi_sieve(N):
    semi = set()
    sieve = [True] * (N + 1)
    sieve[0] = sieve[1] = False

    i = 2
    while (i * i <= N):
        if sieve[i] == True:
            for j in range(i * i, N + 1, i):
                sieve[j] = False
        i += 1

    i = 2
    while (i * i <= N):
        if sieve[i] == True:
            for j in range(i * i, N + 1, i):
                if (j % i == 0 and sieve[int(j / i)] == True):
                    semi.add(j)
        i += 1

    return semi

reduce(mul, [(10-i) for i in range(0,3)]) # 720
import itertools
lis = [ 1, 3, 4, 10, 4 ]
print ("The summation of list using accumulate is :",end="")
print (list(itertools.accumulate(lis,lambda x,y : x+y)))

lis = [ 1, 3, 4, 10, 4 ]
list(accumulate(lis,lambda x,y : x+2*y))
[1, 7, 15, 35, 43]
list(accumulate(lis,lambda x,y : 2*x+y))
[1, 5, 14, 38, 80]

def solution(N, P, Q):
    semi = semi_sieve(N)
    # {4, 6, 9, 10, 14, 15, 21, 22, 25, 26, 33, 34, 35, 38, 39, 46, 49, 51, 55, 57, 58, 62, 65, 69, 74, 77, 82, 85, 86, 87, 91, 93, 94, 95}
    from itertools import accumulate
    def acc(x, y):
        if y in semi:
            x + 1
        else:
            x
    # not working
    # prefix_sum = accumulate(range(N+1), acc)
    prefix_sum = list(accumulate(range(N + 1), lambda x, y: x+1 if (y in semi) else x))

    # accumulate(range(100 + 1), lambda x, y: x + 1 if (y in sem) else x)
    # < itertools.accumulate
    # object
    # at
    # 0x7f5643dc2988 >
    # list(accumulate(range(100 + 1), lambda x, y: x + 1 if (y in sem) else x))
    # [0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 7, 8, 8, 8, 9, 10, 10, 10, 10, 10, 10, 10, 11, 12,
    #  13, 13, 13, 14, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 17, 17, 18, 18, 18, 18, 19, 19, 20, 21, 21, 21, 21, 22, 22,
    #  22, 23, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27, 28, 29, 30, 30, 30, 30, 31, 31,
    #  32, 33, 34, 34, 34, 34, 34, 34]
    P = [1, 4, 16]
    Q = [26, 10, 20]
    result = []
    for i in range(len(P)):
        result.append(prefix_sum[Q[i]], prefix_sum[P[i]-1])

    # map object
    map(lambda i: prefix_sum[Q[i]] - prefix_sum[P[i]-1], range(len(P)))

    result = [prefix_sum[Q[i]] - prefix_sum[P[i]-1] for i in range(len(P))]
    return result

def solution(N, P, Q):
    semi = semi_sieve(N)
    from itertools import accumulate
    prefix_sum = list(accumulate(range(N + 1), lambda x, y: x+1 if (y in semi) else x))

    return [prefix_sum[Q[i]] - prefix_sum[P[i]-1] for i in range(len(P))]

# CountNonDivisible
def solution(A):
    A_max = max(A)

    count = {}
    for element in A:
        if element not in count:
            count[element] = 1
        else:
            count[element] += 1

    divisors = {}
    for element in A:
        divisors[element] = set([1, element])

    # start the Sieve of Eratosthenes
    divisor = 2
    while divisor * divisor <= A_max:
        element_candidate = divisor
        while element_candidate <= A_max:
            if element_candidate in divisors and not divisor in divisors[element_candidate]:
                divisors[element_candidate].add(divisor)
                divisors[element_candidate].add(element_candidate // divisor)
            element_candidate += divisor
        divisor += 1

    result = [0] * len(A)
    for idx, element in enumerate(A):
        result[idx] = (len(A) - sum([count.get(divisor, 0) for divisor in divisors[element]]))

    return result





























