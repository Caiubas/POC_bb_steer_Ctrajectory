from main import *
from new_bboptimizer import AccelLimits, Steer2D, BangBangOptimizer, State2D, PhaseState, ControlSegment2D
import plot
from rrt import RRT

def new_no_collision(x0: State2D, controls: ControlSegment2D, world):
    x1 = controls.integrate(x0)
    A = Point(x0.x.q, x0.y.q)
    B = Point(x1.x.q, x1.y.q)
    return world.is_free_path(A, B)

def find_null_path():
    vmax = 3
    umax = Vector(0.1, 0.1)
    umin = Vector(-0.1, -0.1)
    vi = Vector(0, 0)

    boundaries = Quadrilateral(vertices=[
        Point(0.0, 0.0),
        Point(12.0, 0.0),
        Point(12.0, 10.0),
        Point(0.0, 10.0),
    ])

    path = [[0,0], [10, 10]]
    count = 0
    while path:
        count += 1
        print(count)
        point_a, point_b, obstacles = generate_random_world(
            boundaries,
            n_circles=16,
            n_quads=0,
            n_stadiums=0
        )

        print("obstacles=", obstacles)
        print("point_a=", point_a)
        print("point_b=", point_b)

        print(point_a, point_b)
        print(len(obstacles))

        world = World(obstacles=obstacles, boundaries=boundaries)

        planner = PathPlanner(world=world, max_iterations=5000)
        path = planner.plan(point_a, point_b)

    print("path", path)
    path = [point_a, point_b]
    plot.plot_world_and_path(world, path)


if __name__ == "__main_":
    find_null_path()

if __name__ == "__main__":
    # Cria um mundo com alguns obstáculos
    vmax = 3
    umax = Vector(0.1, 0.1)
    umin = Vector(-0.1, -0.1)
    vi = Vector(0, 0)

    boundaries = Quadrilateral(vertices=[
        Point(0.0, 0.0),
        Point(12.0, 0.0),
        Point(12.0, 10.0),
        Point(0.0, 10.0),
    ])

    point_a, point_b, obstacles = generate_random_world(
        boundaries,
        n_circles=32,
        n_quads=4,
        n_stadiums=2
    )


    print("obstacles=",obstacles)
    print("point_a=",point_a)
    print("point_b=",point_b)

    print(point_a, point_b)
    print(len(obstacles))

    world = World(obstacles=obstacles, boundaries=boundaries)
    plot.plot_world_and_path(world, [point_a, point_b])

    planner = PathPlanner(world=world, max_iterations=5000)
    path = planner.plan(point_a, point_b)
    plot.plot_world_and_path(world, path)

    path = []
    if not path:
        rrt = RRT(world)
        path = rrt.plan(point_a, point_b)

    if path:
        print("Caminho encontrado:")
        for i, p in enumerate(path):
            print(f"  [{i}] {p}")
    else:
        print("Nenhum caminho encontrado.")

    if len(path) > 0:
        x0 = State2D(PhaseState(path[0].x, vi.x), PhaseState(path[0].y, vi.y))
        limits = AccelLimits(umin, umax, vmax=3.0)
        steer = Steer2D(limits)
        seq = steer.steer_list(path, vi)  # ControlSequence
        opt = BangBangOptimizer(steer, new_no_collision, world)
        result = opt.optimize(x0, seq) # ControlSequence

        rebuilt_path = seq.integrate_list(x0)
        plot.plot_world_and_path(world, [Point(p.x.q, p.y.q) for p in rebuilt_path])


        optmized_path = result.integrate_list(x0)
        plot.plot_world_and_path(world, [Point(p.x.q, p.y.q) for p in optmized_path])

