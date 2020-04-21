import functools

# functools total ordering automatically generator other comparator functions
# if lt and eq are defined

@functools.total_ordering
class Point:
    def __init__(self, coord):
        self.x = coord[0]
        self.y = coord[1]

    __lt__ = lambda self, other: self.y < other.y if (self.x == other.x) else self.x < other.x
    __eq__ = lambda self, other: self.x == other.x and self.y == other.y

def sort_points(points):

    def slope(point):
        return (ref_point.y - point.y)/(ref_point.x - point.x + 1e-3)

    points.sort()  # put leftmost first
    ref_point = points[0]

    # can concatenate only lists
    points = points[:1] + sorted(points[1:], key=slope)
    return points

def graham_scan(points):
    def clockwise(a, b, c):
        return (b.y-a.y)*(c.x-a.x) - (b.x-a.x)*(c.y-a.y) > 0 

    convex_hull = []
    sorted_points = sort_points(points)

    for p in sorted_points:
        while len(convex_hull) > 1 and clockwise(convex_hull[-2], convex_hull[-1], p):
            convex_hull.pop()
        convex_hull.append(p)

    return convex_hull


if __name__=='__main__':
    inputs = [[0,3],[1,1],[2,2],[4,4],[0,0],[1,2],[3,1],[3,3]]
    inputs = list(map(Point, inputs))
    hull = graham_scan(inputs)
    for i, elem in enumerate(hull):
        print('({}, {})'.format(elem.x, elem.y))