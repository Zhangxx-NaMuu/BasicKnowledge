import heapq

from data_stucture.OCtree.octrees.geometry import *
from data_stucture.OCtree.octrees.inner.octree_inner import *


class Octree:
    """
    Octrees: 与点相关的数据的有效数据结构
    in 3D space.

    Usage:
        Octree((minx, maxx), (miny, maxy), (minz, maxz))
    创建一个给定边界的空八叉树
    """

    def __init__(self, bounds, tree=Tree.empty()):
        self.bounds = bounds
        self.tree = tree

    def check_bounds(self, p):
        if not point_in_box(p, self.bounds):
            raise KeyError("Point (%s, %s, %s) out of bounds" % p)

    def __len__(self):
        return len(self.tree)

    def __eq__(self, other):
        return self.bounds == other.bounds and self.tree == other.tree

    def __iter__(self):
        return iter(self.tree)

    def copy(self):
        """
        Return a copy of self.

        由于 Octree 只是纯数据结构的包装器，因此这是在常量时间内执行的。
        """
        return Octree(self.bounds, self.tree)

    def get(self, p, default=None):
        """
        Finds the data associated to the point at p
        在p处查找与该点关联的数据
        """
        if point_in_box(p, self.bounds):
            return self.tree.get(self.bounds, p, default)
        else:
            return default

    def insert(self, p, d):
        """
       如果已经有一个点，则在 p 处添加一个值为 d 的点(不同于“ update”方法)。
        """
        self.check_bounds(p)
        self.tree = self.tree.insert(self.bounds, p, d)

    def update(self, p, d):
        """
        在 p 处添加一个值为 d 的点，如果那里已经有一个点，则重写(不同于“ insert”方法)。
        """
        self.check_bounds(p)
        self.tree = self.tree.update(self.bounds, p, d)

    def remove(self, p):
        """
        移除 p 处的点; 如果没有该点，则引发 KeyError。
        """
        self.check_bounds(p)
        self.tree = self.tree.remove(self.bounds, p)

    def extend(self, g):
        "插入 g 中的所有点"
        for (p, d) in g:
            self.insert(p, d)

    def simple_union(self, other):
        """
        返回边界相同的两个八叉树的并。右边的点会覆盖左边的点
        """
        if self.bounds != other.bounds:
            raise ValueError("Bounds don't agree")
        return Octree(self.bounds, self.tree.union(other.tree, self.bounds))

    def rebound(self, newbounds):
        """
        新版本具有更改的边界，并且具有相应的重新构造的树。在新界线外丢掉所有分数。
        """
        return Octree(newbounds, self.tree.rebound(self.bounds, newbounds))

    def general_union(self, other):
        """
        返回两个任意边界八叉树的并。当两者都有点时，其中一个将覆盖另一个(两者中哪一个保留是未定义的）
        """
        x = self
        y = other
        b = union_box(x.bounds, y.bounds)
        if b != x.bounds:
            x = x.rebound(b)
        if b != y.bounds:
            y = y.rebound(b)
        return x.simple_union(y)

    def subset(self, point_fn, box_fn=None):
        """
        选择八叉树的子集。函数 point _ fn 接受坐标并返回 True 或 False。函数 box _ fn 获取坐标并返回 True
        (如果该框中的所有点都在其中)、 False (如果该框中的所有点都出局)或 Nothing (如果一些点可能在其中，另一些点可能出局)。

        如果没有给出 box _ fn，则默认在 box 的8个顶点上考虑 point _ fn (如果它们不同意，则返回 Nothing)。
        """
        if box_fn is None:
            box_fn = lambda b: agreement(point_fn(v) for v in vertices(b))

        return Octree(self.bounds,
                      self.tree.subset(self.bounds, point_fn, box_fn))

    def by_score(self, pointscore, boxscore):
        """
        Iterates through points in some kind of geometric order: for
        example, proximity to a given point.

        Returns tuples of the form (score, coords, value).

        Arguments:

        - pointscore
            A function which associates to coordinates (x, y, z) a
            value, the "score". Lower scores will be returned
            earlier. A score of None is considered infinite: that
            point will not be returned.

        - boxscore
            A function which assigns to a box the lowest possible
            score of any point in that box. Again, a score of None is
            considered infinite: we cannot be interested in any point
            in that box.

        The algorithm maintains a heap of points and boxes in order of
        how promising they are. In particular, if only the earliest
        results are needed, not much extra processing is done.
        """
        l = []
        self.tree.enqueue(l, self.bounds, pointscore, boxscore)

        while len(l) > 0:
            (score, isnode, location, stuff) = heapq.heappop(l)
            if isnode:
                for (b, t) in stuff.children(location):
                    t.enqueue(l, b, pointscore, boxscore)
            else:
                yield (score, location, stuff)

    def by_distance_from_point(self, p, epsilon=None):
        """
        Return points in order of distance from p, in the form
        (distance, coords, value).

        Takes an optional argument epsilon; if this is given, then it
        stops when the distance exceeds epsilon. This is more
        efficient than merely truncating the results.
        """
        if epsilon is None:
            p_fn = lambda q: euclidean_point_point(p, q)
            b_fn = lambda b: euclidean_point_box(p, b)
        else:
            p_fn = lambda q: bounding(euclidean_point_point(p, q), epsilon)
            b_fn = lambda b: bounding(euclidean_point_box(p, b), epsilon)

        for t in self.by_score(p_fn, b_fn):
            yield t

    def by_distance_from_point_rev(self, p):
        """
        Return points in order of distance from p, in the form
        (distance, coords, value), furthest first.
        """
        fp = lambda q: -euclidean_point_point(p, q)
        fb = lambda b: min(-euclidean_point_point(p, q) for q in vertices(b))
        for (d, c, v) in self.by_score(fp, fb):
            yield (-d, c, v)

    def nearest_to_point(self, p):
        """
        Return the nearest point to p, in the form (distance, coords,
        value).
        """
        for t in self.by_distance_from_point(p):
            return t

    def nearest_to_box(self, b):
        """
        Return the nearest point to a box, in the form (distance, coords,
        value).
        """
        for t in self.by_score(lambda p: euclidean_point_box(p, b),
                               lambda b2: euclidean_box_box(b2, b)):
            return t

    def nearest_to_box_far_corner(self, b):
        """
        Return the point which has the lowest maximum distance to a
        box, in the form (distance, coords, value).
        """
        for t in self.by_score(lambda p: euclidean_point_box_max(p, b),
                               lambda b2: euclidean_box_box_minmax(b2, b)):
            return t

    def by_proximity(self, other, epsilon=None):
        """
        Given two octrees, return points from the first which are
        close to some point from the second, in decreasing order of
        proximity.

        Yields tuples of the form (distance, coords1, coords2, data1,
        data2).

        If epsilon is given, then it does not return points further
        apart than epsilon. This can be slightly more efficient than
        simply stopping once the distances exceed epsilon.
        """

        def pointscore(p):
            t = other.nearest_to_point(p)
            if t is None:
                return None
            elif epsilon is not None and t[0] > epsilon:
                return None
            else:
                return t

        def boxscore(b):
            t = other.nearest_to_box(b)
            if t is None:
                return None
            elif epsilon is not None and t[0] > epsilon:
                return None
            else:
                return t

        for ((d, c2, v2), c1, v1) in self.by_score(pointscore, boxscore):
            yield (d, c1, c2, v1, v2)

    def by_isolation(self, other, epsilon=None):
        """
        Given two octrees, return points from the first which are far
        from every point in the second, in increasing order of
        proximity.

        Yields tuples of the form (distance, coords1, coords2, data1,
        data2).

        If epsilon is given, it does not return points that are closer
        than epsilon to the other octree. This is more efficient than
        simply truncating the output.
        """

        def pointscore(p):
            t = other.nearest_to_point(p)
            if t is None:
                return None
            (s, c, v) = t
            if epsilon is not None and s < epsilon:
                return None
            else:
                return (-s, c, v)

        def boxscore(b):
            t = other.nearest_to_box_far_corner(b)
            if t is None:
                return None
            (s, c, v) = t
            if epsilon is not None and s < epsilon:
                return None
            else:
                return (-s, c, v)

        for ((d, c2, v2), c1, v1) in self.by_score(pointscore, boxscore):
            yield (-d, c1, c2, v1, v2)

    def deform(self, point_function, bounds=None, box_function=None):
        """
        Moves all the points according to point_function, assumed to
        be a continuous function.

        It also uses box_function to compute a box bounding the image
        of a given box. If box_function is not given, it assumes that
        the image of boxes is a convex set.

        One can also give explicit bounds, which by default are
        obtained by calling box_function on the existing bounds.

        For large octrees and well-behaved functions, this should be
        significantly faster than repopulating an octree from scratch.
        """

        if box_function is None:
            box_function = lambda b: convex_box_deform(point_function, b)
        if bounds is None:
            bounds = box_function(self.bounds)
        return Octree(bounds,
                      self.tree.deform(self.bounds, bounds,
                                       point_function, box_function))

    def apply_matrix(self, matrix, bounds=None):
        """
        Moves the points according to the given matrix.

        Bounds can be given.
        """
        return self.deform(lambda p: matrix_action(matrix, p), bounds)

    def pairs_by_score(self, other, p_p_score, p_b_score,
                       b_p_score, b_b_score):
        """
        Iterates through pairs of points, one from each of two
        argument octrees.

        This is more elaborate than "by_score" above, but similar in
        many regards. In order to make it efficient, we need to
        provide four scoring functions. The first one scores two
        points, the others give the minimum possible score for a point
        and a box, a box and a point, and a box and a box
        respectively.

        If any scoring functions return None, that is treated as
        infinite: the pairs are not of interest.

        Returns a 5-tuple consisting of: the score, the two sets of
        coordinates, and the data associated to the two points.
        """
        l = []

        def enqueue2(tree1, tree2, bounds1, bounds2):
            if isinstance(tree1, Empty) or isinstance(tree2, Empty):
                pass
            elif isinstance(tree1, Singleton):
                if isinstance(tree2, Singleton):
                    s = p_p_score(tree1.coords, tree2.coords)
                    if s is not None:
                        heapq.heappush(l, (s, False, False,
                                           tree1.coords, tree2.coords,
                                           tree1.data, tree2.data))
                else:
                    s = p_b_score(tree1.coords, bounds2)
                    if s is not None:
                        heapq.heappush(l, (s, False, True, tree1.coords,
                                           bounds2, tree1.data, tree2))
            else:
                if isinstance(tree2, Singleton):
                    s = b_p_score(bounds1, tree2.coords)
                    if s is not None:
                        heapq.heappush(l, (s, True, False, bounds1,
                                           tree2.coords, tree1, tree2.data))
                else:
                    s = b_b_score(bounds1, bounds2)
                    if s is not None:
                        heapq.heappush(l, (s, True, True, bounds1,
                                           bounds2, tree1, tree2))

        enqueue2(self.tree, other.tree, self.bounds, other.bounds)

        while len(l) > 0:
            (score, isnode1, isnode2,
             loc1, loc2, stuff1, stuff2) = heapq.heappop(l)
            if isnode1:
                if isnode2:
                    for (b1, t1) in stuff1.children(loc1):
                        for (b2, t2) in stuff2.children(loc2):
                            enqueue2(t1, t2, b1, b2)
                else:
                    for (b1, t1) in stuff1.children(loc1):
                        enqueue2(t1, Tree.singleton(loc2, stuff2), b1, None)
            else:
                if isnode2:
                    for (b2, t2) in stuff2.children(loc2):
                        enqueue2(Tree.singleton(loc1, stuff1), t2, None, b2)
                else:
                    yield (score, loc1, loc2, stuff1, stuff2)

    def pairs_by_distance(self, other, epsilon):
        """
        Returns pairs within epsilon of each other, one from each
        octree. Returns them in increasing order of distance.
        """
        pp = lambda p1, p2: bounding(euclidean_point_point(p1, p2), epsilon)
        bp1 = lambda p, b: bounding(euclidean_point_box(p, b), epsilon)
        bp2 = lambda b, p: bounding(euclidean_point_box(p, b), epsilon)
        bb = lambda b1, b2: bounding(euclidean_box_box(b1, b2), epsilon)
        for t in self.pairs_by_score(other, pp, bp1, bp2, bb):
            yield t

    def pairs_generate(self, other, p_p_fn,
                       p_b_fn=None, b_p_fn=None, b_b_fn=None):
        """
        产生满足一定条件的点对。需要四个函数: p _ p _ fn，它接受 True 或 False 值来判断是否应该生成一对。其他的可以对盒子进行操作，
        并且可以返回 True (“一切都是有趣的”)、 False (“没有什么是有趣的”)或 Nothing (“也许有些东西是有趣的，有些东西不是”)。
        如果没有给出后面的函数，它们默认考虑所有的顶点
        Yields pairs of points satisfying some criterion.

        Requires four functions: p_p_fn which takes values either
        True or False to say whether a pair should be yielded. The
        others act on boxes, and may return True ("everything is of
        interest"), False ("nothing is of interest") or None ("maybe
        some things are and some things aren't").

        If the later functions are not given, they default to
        considering all vertices.
        """
        if p_b_fn is None:
            p_b_fn = lambda p, b: agreement(p_p_fn(p, q)
                                            for q in vertices(b))
        if b_p_fn is None:
            b_p_fn = lambda b, p: agreement(p_p_fn(q, p)
                                            for q in vertices(b))
        if b_b_fn is None:
            b_b_fn = lambda b1, b2: agreement(p_b_fn(p, b2)
                                              for p in vertices(b1))

        def inner(tree1, tree2, bounds1, bounds2):
            x = b_b_fn(bounds1, bounds2)
            if x is True:
                for (c1, v1) in tree1:
                    for (c2, v2) in tree2:
                        yield c1, c2, v1, v2
            elif x is False:
                pass
            elif isinstance(tree1, Empty):
                pass
            elif isinstance(tree2, Empty):
                pass
            elif isinstance(tree1, Singleton):
                c1 = tree1.coords
                v1 = tree1.data
                for (c2, v2) in tree2.subset(bounds2,
                                             lambda p: p_p_fn(c1, p),
                                             lambda b: p_b_fn(c1, b)):
                    yield c1, c2, v1, v2
            elif isinstance(tree2, Singleton):
                c2 = tree2.coords
                v2 = tree2.data
                for (c1, v1) in tree1.subset(bounds1,
                                             lambda p: p_p_fn(p, c2),
                                             lambda b: b_p_fn(b, c2)):
                    yield c1, c2, v1, v2
            else:
                for (b1, t1) in tree1.children(bounds1):
                    for (b2, t2) in tree2.children(bounds2):
                        for t in inner(t1, t2, b1, b2):
                            yield t

        for t in inner(self.tree, other.tree, self.bounds, other.bounds):
            yield t

    def pairs_nearby(self, other, epsilon):
        """
        以任意顺序在 ε 范围内生成对
        """

        def p_p_fn(p1, p2):
            # 返沪p1和p2的欧几里得距离，即两个点之间的距离
            return euclidean_point_point(p1, p2) < epsilon

        def p_b_fn(p, b):
            if euclidean_point_box(p, b) < epsilon:
                if euclidean_point_box_max(p, b) < epsilon:
                    return True
                else:
                    return None
            else:
                return False

        def b_p_fn(b, p):
            if euclidean_point_box(p, b) < epsilon:
                if euclidean_point_box_max(p, b) < epsilon:
                    return True
                else:
                    return None
            else:
                return False

        def b_b_fn(b1, b2):
            if euclidean_box_box(b1, b2) < epsilon:
                if euclidean_box_box_max(b1, b2) < epsilon:
                    return True
                else:
                    return None
            else:
                return False

        for t in self.pairs_generate(other, p_p_fn, p_b_fn, b_p_fn, b_b_fn):
            yield t


def octree_from_list(bounds, l):
    """
    从列表 l 构造八叉树。在建造之后，l 将仍然存在，但是它的顺序可能已经被改变了。
    """
    o = octree_from_list_inner(bounds, l, 0, len(l))
    return Octree(bounds, o)
