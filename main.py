"""
Path Planning Algorithm
=======================
Implementação baseada no diagrama de classes e fluxograma fornecidos.

Classes:
    Point, Vector, SemiLine, Node, Tree,
    World, Obstacle (abstract), Circle, Quadrilateral, Stadium
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Optional

from bbsteer import integrate_control_2d, control_time, time_optimal_steer_2d_vlim


# ---------------------------------------------------------------------------
# Point
# ---------------------------------------------------------------------------

class Point:
    """Ponto 2D."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def distance_to(self, other: Point) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __repr__(self) -> str:
        return f"Point({self.x:.3f}, {self.y:.3f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return math.isclose(self.x, other.x, abs_tol=1e-9) and \
               math.isclose(self.y, other.y, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Vector
# ---------------------------------------------------------------------------

class Vector:
    """Vetor 2D com operações geométricas."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    # --- factories ---

    @staticmethod
    def from_points(origin: Point, target: Point) -> Vector:
        return Vector(target.x - origin.x, target.y - origin.y)

    # --- métodos do diagrama ---

    def get_norm(self) -> float:
        return math.hypot(self.x, self.y)

    def get_angle_with_vector(self, other: Vector) -> float:
        """Ângulo (rad) entre este vetor e *other*."""
        cos = (self.x * other.x + self.y * other.y) / \
              (self.get_norm() * other.get_norm() + 1e-12)
        return math.acos(max(-1.0, min(1.0, cos)))

    def get_ortogonal_projection(self, other: Vector) -> float:
        """Projeção escalar de *other* sobre este vetor."""
        n = self.get_norm()
        if n < 1e-12:
            return 0.0
        return (self.x * other.x + self.y * other.y) / n

    def get_normalized(self) -> Vector:
        n = self.get_norm()
        if n < 1e-12:
            return Vector(0.0, 0.0)
        return Vector(self.x / n, self.y / n)

    def get_rotated(self, angle: float) -> Vector:
        """Rotaciona o vetor em *angle* radianos (sentido anti-horário)."""
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return Vector(self.x * cos_a - self.y * sin_a,
                      self.x * sin_a + self.y * cos_a)

    # --- utilidades ---

    def to_point(self, origin: Point) -> Point:
        return Point(origin.x + self.x, origin.y + self.y)

    def __repr__(self) -> str:
        return f"Vector({self.x:.3f}, {self.y:.3f})"


# ---------------------------------------------------------------------------
# SemiLine (Semi-reta)
# ---------------------------------------------------------------------------

class SemiLine:
    """Semi-reta que parte de *start* na direção de *end*."""

    def __init__(self, start: Point, end: Point) -> None:
        self.start = start
        self.end = end

    def get_length(self) -> float:
        return self.start.distance_to(self.end)

    def get_angle_with_line(self, other: SemiLine) -> float:
        v1 = Vector.from_points(self.start, self.end)
        v2 = Vector.from_points(other.start, other.end)
        return v1.get_angle_with_vector(v2)

    def direction(self) -> Vector:
        return Vector.from_points(self.start, self.end).get_normalized()

    def __repr__(self) -> str:
        return f"SemiLine({self.start} -> {self.end})"


# ---------------------------------------------------------------------------
# Obstacle (abstract)
# ---------------------------------------------------------------------------

class Obstacle(ABC):
    """Classe base abstrata para obstáculos."""

    @abstractmethod
    def do_contain_the_point(self, point: Point) -> bool:
        """Retorna True se *point* estiver dentro do obstáculo."""

    @abstractmethod
    def is_intercepted_by(self, a: Point, b: Point) -> bool:
        """Retorna True se o segmento AB intercepta o obstáculo."""

    @abstractmethod
    def get_tangent_points(self, origin: Point) -> list[Point]:
        """Retorna os pontos tangentes ao obstáculo a partir de *origin*."""


# ---------------------------------------------------------------------------
# Circle
# ---------------------------------------------------------------------------

class Circle(Obstacle):
    """Obstáculo circular."""

    def __init__(self, center: Point, radius: float) -> None:
        self.center = center
        self.radius = radius

    # --- Obstacle interface ---

    def do_contain_the_point(self, point: Point) -> bool:
        return self.center.distance_to(point) <= self.radius

    def is_intercepted_by(self, a: Point, b: Point) -> bool:
        """Verifica se o segmento AB intercepta o círculo."""
        dx = b.x - a.x
        dy = b.y - a.y
        fx = a.x - self.center.x
        fy = a.y - self.center.y

        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            return self.do_contain_the_point(a)

        t = -(fx * dx + fy * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))

        closest_x = a.x + t * dx
        closest_y = a.y + t * dy
        dist_sq = (closest_x - self.center.x) ** 2 + \
                  (closest_y - self.center.y) ** 2

        return dist_sq <= self.radius * self.radius

    def get_tangent_points(self, origin: Point,
                           margin: float = 0.1) -> list[Point]:
        """
        Calcula os dois pontos de tangência a partir de *origin*.
        Os pontos são gerados com raio efetivo (radius + margin).

        Quando a origem está muito próxima da borda efetiva (tangent_dist < margin),
        os tangentes seriam degenerados e o segmento origin→tangente cruzaria o
        interior do círculo. Nesses casos, deslocamos a origem virtualmente para
        fora (distância mínima = eff_radius + margin) antes de calcular os tangentes,
        produzindo pontos geometricamente válidos e navegáveis.
        """
        eff_radius = self.radius + margin
        d = self.center.distance_to(origin)

        # Origem dentro ou sobre o raio efetivo — usar origem virtual deslocada
        if d < eff_radius:
            d = eff_radius

        # Se tangent_dist é muito pequeno, deslocar origem virtualmente para fora
        tangent_dist_sq = d * d - eff_radius * eff_radius
        if tangent_dist_sq < margin * margin:
            angle_out = math.atan2(origin.y - self.center.y,
                                   origin.x - self.center.x)
            virtual_d = eff_radius + margin
            virtual_origin = Point(
                self.center.x + virtual_d * math.cos(angle_out),
                self.center.y + virtual_d * math.sin(angle_out),
            )
            # Recalcular com origem virtual
            d = virtual_d
            tangent_dist_sq = d * d - eff_radius * eff_radius
            origin = virtual_origin

        alpha = math.asin(min(1.0, eff_radius / d))
        base_angle = math.atan2(self.center.y - origin.y,
                                self.center.x - origin.x)
        tangent_dist = math.sqrt(max(0.0, tangent_dist_sq))

        points = []
        for sign in (+1, -1):
            tang_angle = base_angle + sign * alpha
            tx = origin.x + tangent_dist * math.cos(tang_angle)
            ty = origin.y + tangent_dist * math.sin(tang_angle)
            points.append(Point(tx, ty))
        return points

    def get_exit_point(self, point: Point, margin: float = 0.2) -> Point:
        """Retorna o ponto mais próximo na borda do círculo (+ margem) a partir de *point*."""
        angle = math.atan2(point.y - self.center.y, point.x - self.center.x)
        r = self.radius + margin
        return Point(self.center.x + r * math.cos(angle),
                     self.center.y + r * math.sin(angle))

    def __repr__(self) -> str:
        return f"Circle(center={self.center}, radius={self.radius})"


# ---------------------------------------------------------------------------
# Quadrilateral
# ---------------------------------------------------------------------------

class Quadrilateral(Obstacle):
    """Obstáculo quadrilateral (polígono convexo de 4 vértices)."""

    def __init__(self, vertices: list[Point]) -> None:
        if len(vertices) != 4:
            raise ValueError("Quadrilateral requires exactly 4 vertices.")
        self.vertices = vertices

    # --- helpers ---

    def _cross(self, o: Point, a: Point, b: Point) -> float:
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

    def _segments(self) -> list[tuple[Point, Point]]:
        n = len(self.vertices)
        return [(self.vertices[i], self.vertices[(i + 1) % n])
                for i in range(n)]

    def _seg_intersect(self, p1: Point, p2: Point,
                       p3: Point, p4: Point) -> bool:
        """Verifica se os segmentos p1-p2 e p3-p4 se interceptam."""
        def cross2d(ax, ay, bx, by):
            return ax * by - ay * bx

        dx1, dy1 = p2.x - p1.x, p2.y - p1.y
        dx2, dy2 = p4.x - p3.x, p4.y - p3.y

        denom = cross2d(dx1, dy1, dx2, dy2)
        if abs(denom) < 1e-12:
            return False

        dx3, dy3 = p3.x - p1.x, p3.y - p1.y
        t = cross2d(dx3, dy3, dx2, dy2) / denom
        u = cross2d(dx3, dy3, dx1, dy1) / denom

        return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0

    # --- Obstacle interface ---

    def do_contain_the_point(self, point: Point,
                              tol: float = 1e-9) -> bool:
        """Ray-casting para polígono (com tolerância na borda)."""
        n = len(self.vertices)
        inside = False
        x, y = point.x, point.y
        j = n - 1
        for i in range(n):
            xi, yi = self.vertices[i].x, self.vertices[i].y
            xj, yj = self.vertices[j].x, self.vertices[j].y
            # ponto exatamente na aresta → não considera dentro
            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi - tol):
                inside = not inside
            j = i
        return inside

    def is_intercepted_by(self, a: Point, b: Point) -> bool:
        if self.do_contain_the_point(a) or self.do_contain_the_point(b):
            return True
        for p1, p2 in self._segments():
            if self._seg_intersect(a, b, p1, p2):
                return True
        return False

    def _is_directly_visible(self, origin: Point, vertex_idx: int) -> bool:
        """
        Verifica se o segmento origin → vértice[vertex_idx] não atravessa
        nenhuma aresta não-adjacente do polígono.
        Necessário para polígonos convexos onde a origem está numa posição
        que torna um vértice geometricamente 'tangente' mas não diretamente
        alcançável em linha reta (o segmento passaria pelo interior).
        """
        n = len(self.vertices)
        vi = self.vertices[vertex_idx]

        for i in range(n):
            # Ignora as duas arestas adjacentes ao vértice candidato
            if i == vertex_idx or i == (vertex_idx - 1) % n:
                continue
            a = self.vertices[i]
            b = self.vertices[(i + 1) % n]
            if self._seg_intersect(origin, vi, a, b):
                return False
        return True

    def get_tangent_points(self, origin: Point,
                           margin: float = 0.2) -> list[Point]:
        """
        Calcula os vértices da silhueta (tangentes) do polígono convexo
        visto a partir de *origin*, usando a formulação correta de silhueta:

            V_i é tangente sse cross(O, V_{i-1}, V_i) e
                                  cross(O, V_i,   V_{i+1})
            têm sinais opostos.

        Isso é mais robusto do que a formulação anterior e funciona para
        qualquer orientação/rotação do polígono e qualquer posição da origem
        (desde que fora do polígono).

        O afastamento (margin) é feito na direção da bissetriz das normais
        das arestas adjacentes, garantindo que o ponto fique fora do polígono.
        """
        tangents: list[Point] = []
        n = len(self.vertices)
        for i in range(n):
            prev_v = self.vertices[(i - 1) % n]
            curr_v = self.vertices[i]
            next_v = self.vertices[(i + 1) % n]

            # Formulação silhueta: cross do segmento prev→curr e curr→next
            # vistos a partir de origin
            c1 = self._cross(origin, prev_v, curr_v)
            c2 = self._cross(origin, curr_v, next_v)

            if (c1 >= 0) != (c2 >= 0):
                # Afasta pela bissetriz das normais das arestas adjacentes
                n1x, n1y = self._outward_normal_of_edge(prev_v, curr_v)
                n2x, n2y = self._outward_normal_of_edge(curr_v, next_v)
                bx, by = n1x + n2x, n1y + n2y
                length = math.hypot(bx, by)
                if length > 1e-9:
                    bx, by = bx / length, by / length
                else:
                    dx = curr_v.x - origin.x
                    dy = curr_v.y - origin.y
                    d = math.hypot(dx, dy)
                    bx, by = (dx / d, dy / d) if d > 1e-9 else (1.0, 0.0)

                tangents.append(Point(
                    curr_v.x + margin * bx,
                    curr_v.y + margin * by,
                ))

        return tangents if tangents else list(self.vertices)

    def _closest_point_on_edge(self, p: Point,
                               a: Point, b: Point) -> Point:
        """Projeta *p* no segmento AB e retorna o ponto mais próximo."""
        dx, dy = b.x - a.x, b.y - a.y
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-12:
            return a
        t = max(0.0, min(1.0,
                ((p.x - a.x) * dx + (p.y - a.y) * dy) / len_sq))
        return Point(a.x + t * dx, a.y + t * dy)

    def _outward_normal_of_edge(self, a: Point, b: Point) -> tuple[float, float]:
        """
        Normal unitária apontando para fora da aresta AB.
        Determinada comparando com o centróide do polígono,
        sem assumir orientação (horária ou anti-horária) dos vértices.
        """
        dx, dy = b.x - a.x, b.y - a.y
        length = math.hypot(dx, dy)
        if length < 1e-12:
            return (0.0, 0.0)
        nx, ny = -dy / length, dx / length
        cx = sum(v.x for v in self.vertices) / len(self.vertices)
        cy = sum(v.y for v in self.vertices) / len(self.vertices)
        mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2
        # Normal deve afastar-se do centróide
        if (mx + nx - cx) ** 2 + (my + ny - cy) ** 2 < \
           (mx - nx - cx) ** 2 + (my - ny - cy) ** 2:
            nx, ny = -nx, -ny
        return nx, ny

    def get_exit_point(self, point: Point, margin: float = 0.2) -> Point:
        """
        Retorna o ponto de saída mais próximo na borda do polígono:
        projeta *point* sobre cada aresta, escolhe a mais próxima e
        empurra o resultado para fora pela normal da aresta (+ margem).
        Funciona para qualquer orientação e ângulo dos vértices.
        """
        best_dist = float("inf")
        best_proj = point
        best_nx, best_ny = 0.0, 1.0

        n = len(self.vertices)
        for i in range(n):
            a = self.vertices[i]
            b = self.vertices[(i + 1) % n]
            proj = self._closest_point_on_edge(point, a, b)
            dist = point.distance_to(proj)
            if dist < best_dist:
                best_dist = dist
                best_proj = proj
                best_nx, best_ny = self._outward_normal_of_edge(a, b)

        return Point(best_proj.x + margin * best_nx,
                     best_proj.y + margin * best_ny)

    def __repr__(self) -> str:
        return f"Quadrilateral(vertices={self.vertices})"


# ---------------------------------------------------------------------------
# Stadium (retângulo com semicírculos nas extremidades)
# ---------------------------------------------------------------------------

class Stadium(Obstacle):
    """
    Obstáculo em forma de 'stadium': um retângulo com semicírculos
    nas extremidades. Definido por dois vértices centrais e um raio.
    """

    def __init__(self, vertices: list[Point], radius: float) -> None:
        if len(vertices) != 2:
            raise ValueError("Stadium requires exactly 2 vertices (endpoints).")
        self.vertices = vertices
        self.radius = radius

    def _closest_point_on_segment(self, p: Point) -> Point:
        a, b = self.vertices[0], self.vertices[1]
        dx, dy = b.x - a.x, b.y - a.y
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-12:
            return a
        t = max(0.0, min(1.0, ((p.x - a.x) * dx + (p.y - a.y) * dy) / len_sq))
        return Point(a.x + t * dx, a.y + t * dy)

    # --- Obstacle interface ---

    def do_contain_the_point(self, point: Point) -> bool:
        closest = self._closest_point_on_segment(point)
        return closest.distance_to(point) <= self.radius

    def is_intercepted_by(self, a: Point, b: Point) -> bool:
        """Amostragem ao longo do segmento AB para verificar colisão."""
        steps = 20
        for i in range(steps + 1):
            t = i / steps
            sample = Point(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))
            if self.do_contain_the_point(sample):
                return True
        return False

    def get_tangent_points(self, origin: Point,
                           margin: float = 0.2) -> list[Point]:
        """Usa os semicírculos das extremidades como círculos para obter tangentes."""
        points = []
        for vertex in self.vertices:
            cap = Circle(vertex, self.radius)
            points.extend(cap.get_tangent_points(origin, margin=margin))
        return points

    def get_exit_point(self, point: Point, margin: float = 0.2) -> Point:
        closest = self._closest_point_on_segment(point)
        angle = math.atan2(point.y - closest.y, point.x - closest.x)
        r = self.radius + margin
        return Point(closest.x + r * math.cos(angle),
                     closest.y + r * math.sin(angle))

    def __repr__(self) -> str:
        return f"Stadium(vertices={self.vertices}, radius={self.radius})"


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

class World:
    """Contém os obstáculos e os limites do espaço."""

    def __init__(self, obstacles: list[Obstacle],
                 boundaries: Optional[Quadrilateral] = None) -> None:
        self.obstacles: list[Obstacle] = obstacles
        self.boundaries: Optional[Quadrilateral] = boundaries

    def do_contain_point(self, point: Point) -> bool:
        """Retorna True se o ponto estiver dentro dos limites (se definidos)."""
        if self.boundaries is None:
            return True
        return self.boundaries.do_contain_the_point(point)

    def is_free_path(self, a: Point, b: Point) -> bool:
        """Retorna True se o segmento AB não interceptar nenhum obstáculo."""
        for obs in self.obstacles:
            if obs.is_intercepted_by(a, b):
                return False
        return True

    def obstacle_hit(self, a: Point, b: Point) -> Optional[Obstacle]:
        """Retorna o primeiro obstáculo que intercepta o segmento AB."""
        for obs in self.obstacles:
            if obs.is_intercepted_by(a, b):
                return obs
        return None

    def inside_obstacle(self, point: Point) -> Optional[Obstacle]:
        """Retorna o obstáculo que contém *point*, ou None."""
        for obs in self.obstacles:
            if obs.do_contain_the_point(point):
                return obs
        return None

    def __repr__(self) -> str:
        return f"World(obstacles={len(self.obstacles)})"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """Nó da árvore de planejamento de caminho."""

    def __init__(self, position: Point,
                 parent: Optional[Node] = None) -> None:
        self.position = position
        self.parent: Optional[Node] = parent
        self.childs: list[Node] = []
        self.explored: bool = False
        self.depth: int = 0 if parent is None else parent.depth + 1

    def generate_child(self, position: Point) -> Node:
        child = Node(position, parent=self)
        self.childs.append(child)
        return child

    def __repr__(self) -> str:
        return f"Node(pos={self.position}, depth={self.depth})"


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------

class Tree:
    """Árvore de busca de caminho."""

    def __init__(self, root: Point, goal: Point) -> None:
        self.root = root
        self.goal = goal
        self.nodes: list[Node] = [Node(root)]

    def build_path(self, node: Node) -> list[Point]:
        """Reconstrói o caminho do nó raiz até *node*."""
        path: list[Point] = []
        current: Optional[Node] = node
        while current is not None:
            path.append(current.position)
            current = current.parent
        path.reverse()
        return path

    def get_leaf_nodes(self) -> list[Node]:
        return [n for n in self.nodes if not n.explored]

    def __repr__(self) -> str:
        return f"Tree(nodes={len(self.nodes)})"


# ---------------------------------------------------------------------------
# PathPlanner  — implementação do fluxograma
# ---------------------------------------------------------------------------

class PathPlanner:
    """
    Algoritmo de planejamento de caminho baseado no fluxograma fornecido.

    Passos (ver fluxograma):
    1. Verifica se Ponto A / Ponto B estão dentro de obstáculos.
       - Se A está dentro → gera ponto de saída e o inclui na trajetória.
       - Se B está dentro → gera novo ponto final fora do obstáculo.
    2. Loop principal:
       - Há colisão no trecho atual? Não → monta árvore e retorna trajetória.
       - Sim → gera novos pontos candidatos (tangentes ao obstáculo atingido).
         - Há pontos gerados? Não → volta para o nó anterior.
         - Sim → ordena por melhor heurística e armazena no nó.
         - Algum tem trajetória livre até B? Sim → monta árvore e retorna.
         - Não → seleciona o melhor ponto e repete.
    3. Geração de novos pontos:
       - O obstáculo já gerou tangentes para esta origem? Sim → descarta.
       - Não → gera pontos tangentes ao obstáculo.
         - Há colisão entre o ponto gerado e o anterior? Sim → descarta.
         - Não → armazena como alternativa.
    """

    def __init__(self, world: World, max_iterations: int = 500) -> None:
        self.world = world
        self.max_iterations = max_iterations
        # Cache baseado em posição aproximada (grade de resolução 0.05) + id do obstáculo.
        # Evita re-gerar tangentes degenerados quando nós distintos ficam
        # na mesma vizinhança de um obstáculo.
        self._tangent_cache: set[tuple[int, int, int]] = set()
        self._cache_resolution: float = 0.05

    # ------------------------------------------------------------------
    # Ponto de entrada principal
    # ------------------------------------------------------------------

    def plan(self, point_a: Point, point_b: Point) -> list[Point]:
        """
        Executa o algoritmo de planejamento e retorna a lista de pontos
        que formam o caminho de A até B.
        """
        trajectory_prefix: list[Point] = []

        # --- Passo 1a: A está dentro de um obstáculo? ---
        obs_a = self.world.inside_obstacle(point_a)
        if obs_a is not None:
            exit_pt = self._get_exit_point(obs_a, point_a)
            trajectory_prefix.append(point_a)
            point_a = exit_pt
            trajectory_prefix.append(exit_pt)

        # --- Passo 1b: B está dentro de um obstáculo? ---
        obs_b = self.world.inside_obstacle(point_b)
        if obs_b is not None:
            point_b = self._get_exit_point(obs_b, point_b)

        # --- Caminho direto? ---
        if self.world.is_free_path(point_a, point_b):
            return trajectory_prefix + [point_a, point_b]

        # --- Inicializa a árvore ---
        tree = Tree(root=point_a, goal=point_b)
        root_node = tree.nodes[0]

        # Pilha de nós a explorar
        stack: list[Node] = [root_node]

        # Cada nó mantém uma fila de candidatos ainda não tentados
        # { node_id → [Point, ...] }
        pending_candidates: dict[int, list[Point]] = {}

        for _iteration in range(self.max_iterations):
            if not stack:
                break  # sem caminho

            current_node = stack[-1]
            origin = current_node.position

            # --- Há colisão de origin até B? ---
            obs_hit = self.world.obstacle_hit(origin, point_b)

            if obs_hit is None:
                # Caminho livre! Monta a árvore e retorna trajetória.
                path = tree.build_path(current_node) + [point_b]
                return trajectory_prefix + path

            node_key = id(current_node)

            # --- Inicializa candidatos para este nó se necessário ---
            if node_key not in pending_candidates:
                new_points = self._generate_new_points(
                    current_node, obs_hit, origin
                )
                if not new_points:
                    # Sem pontos gerados → backtrack
                    current_node.explored = True
                    stack.pop()
                    continue

                # Ordena por heurística (distância ao goal)
                new_points.sort(key=lambda p: p.distance_to(point_b))

                # Algum tem trajetória livre até B?
                for p in new_points:
                    if self.world.is_free_path(p, point_b):
                        child_node = current_node.generate_child(p)
                        tree.nodes.append(child_node)
                        path = tree.build_path(child_node) + [point_b]
                        return trajectory_prefix + path

                pending_candidates[node_key] = new_points

            # Ainda há candidatos a tentar neste nó?
            if not pending_candidates[node_key]:
                # Esgotados → backtrack
                current_node.explored = True
                stack.pop()
                continue

            # Seleciona o melhor ponto ainda não tentado
            best = pending_candidates[node_key].pop(0)

            # Verifica se o melhor ponto em si está em região livre
            if self.world.inside_obstacle(best) is not None:
                continue  # descarta e tenta o próximo

            child_node = current_node.generate_child(best)
            tree.nodes.append(child_node)
            stack.append(child_node)

        # Sem caminho encontrado
        return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_exit_point(self, obstacle: Obstacle, point: Point) -> Point:
        """Obtém o ponto de saída de um obstáculo."""
        if isinstance(obstacle, Circle):
            return obstacle.get_exit_point(point)
        if isinstance(obstacle, Quadrilateral):
            return obstacle.get_exit_point(point)
        if isinstance(obstacle, Stadium):
            return obstacle.get_exit_point(point)
        # fallback genérico
        candidates = obstacle.get_tangent_points(point)
        if candidates:
            return min(candidates, key=lambda p: p.distance_to(point))
        return point

    def _generate_new_points(self, node: Node,
                             obstacle: Obstacle,
                             origin: Point) -> list[Point]:
        """
        Implementa a sub-rotina 'Gera novos pontos' do fluxograma:

        - O obstáculo já gerou tangentes para esta posição de origem? → descarta.
        - Caso contrário, gera tangentes e filtra as que colidem com algum obstáculo.

        O cache usa posição quantizada (grade de 0.05 unidades) + id do obstáculo,
        evitando re-geração degenerada quando nós distintos ficam muito próximos
        da borda efetiva de um mesmo obstáculo.
        """
        res = self._cache_resolution
        qx = int(round(origin.x / res))
        qy = int(round(origin.y / res))
        cache_key = (qx, qy, id(obstacle))

        if cache_key in self._tangent_cache:
            return []  # já processado nesta vizinhança — descarta

        self._tangent_cache.add(cache_key)

        # Gera pontos tangentes ao obstáculo atingido
        tangent_points = obstacle.get_tangent_points(origin)

        # Filtra pontos inválidos:
        # 1. O ponto tangente não deve estar dentro do próprio obstáculo que o gerou
        # 2. O segmento origem→tangente não deve colidir com nenhum obstáculo.
        #    Para Circle, is_intercepted_by retorna False para seus próprios tangentes.
        #    Para Stadium, verifica corretamente a travessia do corpo.

        valid_points: list[Point] = []
        for pt in tangent_points:
            if obstacle.do_contain_the_point(pt):
                continue  # ainda dentro — descarta

            if self.world.obstacle_hit(origin, pt) is not None:
                continue  # algum obstáculo bloqueia o trecho — descarta

            valid_points.append(pt)

        return valid_points

# ---------------------------------------------------------------------------
# Exemplo de uso / demonstração
# ---------------------------------------------------------------------------

import random

def random_point_in_bounds(boundaries: Quadrilateral) -> Point:
    xs = [v.x for v in boundaries.vertices]
    ys = [v.y for v in boundaries.vertices]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    return Point(
        random.uniform(min_x, max_x),
        random.uniform(min_y, max_y)
    )

def generate_far_points(boundaries: Quadrilateral, margin_ratio=0.1):
    xs = [v.x for v in boundaries.vertices]
    ys = [v.y for v in boundaries.vertices]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x
    height = max_y - min_y

    margin_x = width * margin_ratio
    margin_y = height * margin_ratio

    # escolhe eixo principal (horizontal ou vertical)
    if random.random() < 0.5:
        # esquerda vs direita
        point_a = Point(
            random.uniform(min_x, min_x + margin_x),
            random.uniform(min_y, max_y)
        )
        point_b = Point(
            random.uniform(max_x - margin_x, max_x),
            random.uniform(min_y, max_y)
        )
    else:
        # baixo vs cima
        point_a = Point(
            random.uniform(min_x, max_x),
            random.uniform(min_y, min_y + margin_y)
        )
        point_b = Point(
            random.uniform(min_x, max_x),
            random.uniform(max_y - margin_y, max_y)
        )

    return point_a, point_b


def generate_random_world(
    boundaries: Quadrilateral,
    n_circles: int,
    n_quads: int,
    n_stadiums: int
):
    obstacles: list[Obstacle] = []

    xs = [v.x for v in boundaries.vertices]
    ys = [v.y for v in boundaries.vertices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x
    height = max_y - min_y

    # -------- Circles --------
    for _ in range(n_circles):
        center = random_point_in_bounds(boundaries)
        radius = random.uniform(0.02, 0.08) * min(width, height)

        obstacles.append(
            Circle(center=center, radius=radius)
        )

    # -------- Quadrilaterals --------
    for _ in range(n_quads):
        cx, cy = random_point_in_bounds(boundaries).x, random_point_in_bounds(boundaries).y

        w = random.uniform(0.05, 0.15) * width
        h = random.uniform(0.05, 0.15) * height

        quad = Quadrilateral(vertices=[
            Point(cx - w/2, cy - h/2),
            Point(cx + w/2, cy - h/2),
            Point(cx + w/2, cy + h/2),
            Point(cx - w/2, cy + h/2),
        ])

        obstacles.append(quad)

    # -------- Stadiums --------
    for _ in range(n_stadiums):
        p1 = random_point_in_bounds(boundaries)
        p2 = random_point_in_bounds(boundaries)

        radius = random.uniform(0.02, 0.06) * min(width, height)

        obstacles.append(
            Stadium(vertices=[p1, p2], radius=radius)
        )

    # -------- Points A e B --------
    point_a, point_b = generate_far_points(boundaries)

    return point_a, point_b, obstacles


def no_collision(x0, controls, world):
    x1 = integrate_control_2d(x0, controls)
    A = Point(x0[0], x0[1])
    B = Point(x1[0], x1[1])
    return world.is_free_path(A, B)

if __name__ == "__main__":
    # Cria um mundo com alguns obstáculos
    vmax = 3
    umax = [0.1, 0.1]
    umin = [-0.1, -0.1]
    vi = [0, 0]

    boundaries = Quadrilateral(vertices=[
        Point(0.0, 0.0),
        Point(12.0, 0.0),
        Point(12.0, 10.0),
        Point(0.0, 10.0),
    ])

    point_a, point_b, obstacles = generate_random_world(
        boundaries,
        n_circles=16,
        n_quads=0,
        n_stadiums=0
    )

    print(point_a, point_b)
    print(len(obstacles))

    import plot
    from bboptimizer import bb_optimizer
    world = World(obstacles=obstacles, boundaries=boundaries)
    plot.plot_world_and_path(world, [point_a, point_b])

    time_start = perf_counter()
    planner = PathPlanner(world=world, max_iterations=5000)
    path = planner.plan(point_a, point_b)


    if path:
        print("Caminho encontrado:")
        for i, p in enumerate(path):
            print(f"  [{i}] {p}")
    else:
        print("Nenhum caminho encontrado.")




    acc_path = []
    state = list([point_a.x, point_a.y, vi[0], vi[1]])
    for i in range(len(path) - 1):
        print(i)
        xg = [path[i + 1].x, path[i + 1].y, 0, 0]
        seg = time_optimal_steer_2d_vlim(state, xg, umin=umin, umax=umax, vmax=vmax)
        acc_path.extend(seg)
        state = list(integrate_control_2d(state, seg))






    # Optimiza
    optimized = bb_optimizer(
        xinit=[path[0].x, path[0].y, vi[0], vi[1]],
        controls=acc_path,
        world=world,
        vmax=3,
        collision_check_fn=no_collision,
        max_iter=500,
        patience=50,
        umax=umax,
        umin=umin
    )

    print("Tempo original:", control_time(acc_path))
    print("Tempo optimizado:", control_time(optimized))

    pos_path = [[path[0].x, path[0].y, 0, 0]]
    pos_and_acc = [[path[0].x, path[0].y], (vi[0]**2 + vi[1]**2)**0.5]
    state = [path[0].x, path[0].y, vi[0], vi[1]]
    for seg in optimized:
        state = list(integrate_control_2d(state, [seg]))  # passa [seg] em vez de seg
        pos_and_acc.append([[state[0], state[1]], (state[2]**2 + state[3]**2)**0.5])
        pos_path.append(Point(state[0], state[1]))
    pos_path[0] = Point(pos_path[0][0], pos_path[0][1])

    acc_pos_path = [[path[0].x, path[0].y, vi[0], vi[1]]]
    state = [path[0].x, path[0].y, vi[0], vi[1]]
    for seg in acc_path:
        state = list(integrate_control_2d(state, [seg]))
        acc_pos_path.append(Point(state[0], state[1]))
    acc_pos_path[0] = Point(acc_pos_path[0][0], acc_pos_path[0][1])


    time_end = perf_counter()

    print("tempo:", time_end - time_start)

    print(pos_and_acc)

    plot.plot_world_and_path(world, path)
    plot.plot_world_and_path(world, acc_pos_path)
    plot.plot_world_and_path(world, pos_path)

    xinit = [point_a.x, point_a.y, vi[0], vi[1]]
    xgoal = [point_b.x, point_b.y, 0, 0]

    xfinal = integrate_control_2d(xinit, optimized)

    print("inicial esperado:", xinit)
    print("final esperado:  ", xgoal)
    print("final obtido:    ", xfinal)
    print("erro posição:    ", abs(xfinal[0] - xgoal[0]), abs(xfinal[1] - xgoal[1]))
    print("erro velocidade: ", abs(xfinal[2] - xgoal[2]), abs(xfinal[3] - xgoal[3]))