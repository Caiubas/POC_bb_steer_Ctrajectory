from main import *
from new_bboptimizer import AccelLimits, Steer2D, BangBangOptimizer, State2D, PhaseState, ControlSegment2D
import plot

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
        n_circles=16,
        n_quads=0,
        n_stadiums=0
    )
    obstacles = [Circle(center=Point(4.780, 7.087), radius=0.7122819151457056),
                 Circle(center=Point(0.924, 3.453), radius=0.4309845260556222),
                 Circle(center=Point(0.817, 4.210), radius=0.3956059387153557),
                 Circle(center=Point(4.550, 1.723), radius=0.6367514853961431),
                 Circle(center=Point(1.639, 5.912), radius=0.6670616384769612),
                 Circle(center=Point(5.741, 0.898), radius=0.735139790355845),
                 Circle(center=Point(2.320, 6.653), radius=0.7746661803729162),
                 Circle(center=Point(10.719, 2.534), radius=0.7713802573752532),
                 Circle(center=Point(2.366, 8.363), radius=0.40761088424835956),
                 Circle(center=Point(2.231, 1.364), radius=0.31515878079948034),
                 Circle(center=Point(6.584, 3.075), radius=0.43133037649949135),
                 Circle(center=Point(5.439, 5.046), radius=0.7066478940041843),
                 Circle(center=Point(1.140, 5.818), radius=0.780481186767859),
                 Circle(center=Point(5.285, 2.724), radius=0.7884462013081022),
                 Circle(center=Point(1.976, 0.762), radius=0.7106565743351528),
                 Circle(center=Point(4.081, 2.656), radius=0.625537327712281)]
    point_a = Point(8.396, 0.045)
    point_b = Point(1.087, 9.189)

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

