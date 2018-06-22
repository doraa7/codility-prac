
# Collinear

class Point2DObject(object):
    x = 0
    y = 0

def isCollinear(a, b, c):
    (b.y - a.y)/(b.x - a.x) == (c.y - c.y)/ (b.y - a.y)



def collinear(p0, p1, p2):
    x1, y1 = p1.x - p0.x, p1.y - p0.y
    x2, y2 = p2.x - p0.x, p2.y - p0.y
    return abs(x1 * y2 - x2 * y1) < 1e-12

def collinear(p0, p1, p2):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0]- p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < 1e-12

def solution(A):
    all_collinear_points = set()
    for i in range(len(A)):
        for j in range(len(A)):
            S = set()
            for p in range(len(A)):
                if collinear(A[i], A[j], A[p]):
                    S.add(A[p])
                else:
                    continue
            if len(S) >= 3:
                all_collinear_points.add(frozenset(S))

    return len(all_collinear_points)

# [[0, 0], [1, 1], [2, 2], [3, 3], [3, 2], [4, 2], [5, 1]]
solution([(0, 0), (1, 1), (2, 2), (3, 3), (3, 2), (4, 2), (5, 1)])

