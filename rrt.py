import random
import math
from main import *


class Node:
    def __init__(self, point: Point, parent=None):
        self.point = point
        self.parent = parent


class RRT:
    def __init__(self, world: World, step_size=0.5, max_iter=5000):
        self.world = world
        self.step_size = step_size
        self.max_iter = max_iter
        self.nodes = []

    # --------------------------------------------------
    # Utilidades
    # --------------------------------------------------

    def sample(self, goal: Point, goal_bias=0.1) -> Point:
        """Amostragem com bias para o goal"""
        if random.random() < goal_bias:
            return goal

        # bounding box simples
        if self.world.boundaries:
            xs = [v.x for v in self.world.boundaries.vertices]
            ys = [v.y for v in self.world.boundaries.vertices]
            return Point(random.uniform(min(xs), max(xs)),
                         random.uniform(min(ys), max(ys)))

        return Point(random.uniform(0, 10), random.uniform(0, 10))

    def nearest(self, point: Point) -> Node:
        return min(self.nodes, key=lambda n: n.point.distance_to(point))

    def steer(self, from_p: Point, to_p: Point) -> Point:
        v = Vector.from_points(from_p, to_p)
        d = v.get_norm()
        if d < self.step_size:
            return to_p
        v = v.get_normalized()
        return v.to_point(from_p)

    def backtrack(self, node: Node):
        path = []
        while node:
            path.append(node.point)
            node = node.parent
        return path[::-1]

    # --------------------------------------------------
    # CORE
    # --------------------------------------------------

    def plan(self, start: Point, goal: Point):
        self.nodes = [Node(start)]

        for _ in range(self.max_iter):

            rand = self.sample(goal)
            nearest = self.nearest(rand)

            new_point = self.steer(nearest.point, rand)

            # -----------------------------------
            # CASO 1: caminho livre
            # -----------------------------------
            if self.world.is_free_path(nearest.point, new_point):
                new_node = Node(new_point, nearest)
                self.nodes.append(new_node)

                if new_point.distance_to(goal) < self.step_size:
                    return self.backtrack(new_node)

            # -----------------------------------
            # CASO 2: colisão → usar inteligência
            # -----------------------------------
            else:
                obs = self.world.obstacle_hit(nearest.point, new_point)
                if obs is None:
                    continue

                # -----------------------------------
                # 2.1 Se entrou dentro → sair
                # -----------------------------------
                inside = self.world.inside_obstacle(nearest.point)
                if inside:
                    exit_point = inside.get_exit_point(nearest.point)
                    if self.world.is_free_path(nearest.point, exit_point):
                        self.nodes.append(Node(exit_point, nearest))
                    continue

                # -----------------------------------
                # 2.2 Contornar usando tangentes
                # -----------------------------------
                tangents = obs.get_tangent_points(nearest.point)

                for t in tangents:
                    if not self.world.do_contain_point(t):
                        continue

                    if self.world.is_free_path(nearest.point, t):
                        self.nodes.append(Node(t, nearest))

        return None