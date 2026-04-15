from time import perf_counter
import statistics as stats

import numpy as np

import plot
from bboptimizer import bb_optimizer
from bbsteer import integrate_control_2d, time_optimal_steer_2d_vlim
from main import no_collision, World, generate_random_world, Point, Quadrilateral, PathPlanner

import matplotlib.pyplot as plt

def plot_benchmark_results(times):
    """
    times: dict retornado pelo benchmark_pipeline
    """

    # Remove listas vazias ou inconsistentes
    clean_times = {k: v for k, v in times.items() if len(v) > 0}

    if not clean_times:
        print("Sem dados válidos para plotar.")
        return

    # -------------------------------
    # BOXPLOT (distribuição)
    # -------------------------------
    plt.figure()
    plt.boxplot(clean_times.values(), labels=clean_times.keys())
    plt.title("Distribuição dos Tempos por Etapa")
    plt.ylabel("Tempo (s)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # MÉDIA POR ETAPA
    # -------------------------------
    means = [sum(v) / len(v) for v in clean_times.values()]

    plt.figure()
    plt.bar(clean_times.keys(), means)
    plt.title("Tempo Médio por Etapa")
    plt.ylabel("Tempo (s)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # HISTOGRAMA (TOTAL)
    # -------------------------------
    if "total" in clean_times:
        plt.figure()
        plt.hist(clean_times["total"], bins=20)
        plt.title("Distribuição do Tempo Total")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.show()

def benchmark_pipeline(N):
    vmax = 3
    umax = [0.1, 0.1]
    umin = [-0.1, -0.1]
    vi = [0, 0]

    boundaries = Quadrilateral(vertices=[
        Point(0.0, 0.0),
        Point(24.0, 0.0),
        Point(24.0, 20.0),
        Point(0.0, 20.0),
    ])

    worlds = []
    for _ in range(N):
        point_a, point_b, obstacles = generate_random_world(
            boundaries,
            n_circles=32,
            n_quads=10,
            n_stadiums=3
        )
        world = World(obstacles=obstacles, boundaries=boundaries)
        worlds.append((point_a, point_b, world))

    times = {
        "planner": [],
        "steer": [],
        "optimizer": [],
        "integration": [],
        "total": []
    }

    results = []

    for i in range(N):
        point_a, point_b, world = worlds[i]

        t_total_start = perf_counter()

        # -------------------------------
        # PLANNER
        # -------------------------------
        t0 = perf_counter()
        planner = PathPlanner(world=world, max_iterations=5000)
        path = planner.plan(point_a, point_b)
        t1 = perf_counter()

        if not path:
            continue

        times["planner"].append(t1 - t0)

        # -------------------------------
        # STEER
        # -------------------------------
        t0 = perf_counter()
        acc_path = []
        state = [point_a.x, point_a.y, vi[0], vi[1]]

        for j in range(len(path) - 1):
            xg = [path[j + 1].x, path[j + 1].y, 0, 0]
            seg = time_optimal_steer_2d_vlim(
                state, xg,
                umin=umin, umax=umax, vmax=vmax
            )
            acc_path.extend(seg)
            state = list(integrate_control_2d(state, seg))
        t1 = perf_counter()

        times["steer"].append(t1 - t0)

        # -------------------------------
        # OTIMIZAÇÃO
        # -------------------------------
        t0 = perf_counter()
        optimized = bb_optimizer(
            xinit=[path[0].x, path[0].y, vi[0], vi[1]],
            controls=acc_path,
            world=world,
            vmax=vmax,
            collision_check_fn=no_collision,
            max_iter=500,
            patience=50,
            umax=umax,
            umin=umin
        )
        t1 = perf_counter()

        times["optimizer"].append(t1 - t0)

        # -------------------------------
        # INTEGRAÇÃO DO OTIMIZADO
        # -------------------------------
        t0 = perf_counter()
        state = [path[0].x, path[0].y, vi[0], vi[1]]
        optimized_path = [Point(state[0], state[1])]

        for seg in optimized:
            state = list(integrate_control_2d(state, [seg]))
            optimized_path.append(Point(state[0], state[1]))
        t1 = perf_counter()

        times["integration"].append(t1 - t0)

        t_total_end = perf_counter()
        times["total"].append(t_total_end - t_total_start)

        # -------------------------------
        # SALVA RESULTADO COMPLETO
        # -------------------------------
        results.append({
            "world": world,
            "path": path,
            "optimized_path": optimized_path
        })

    return times, results

def plot_paths_comparison(results, max_plots=None):

    if not results:
        print("Sem resultados para plotar.")
        return

    plotted = 0

    for i, data in enumerate(results):
        if max_plots is not None and plotted >= max_plots:
            break

        world = data["world"]
        path = data["path"]
        opt_path = data["optimized_path"]

        fig, ax = plt.subplots()

        # ---------------------------
        # Boundaries
        # ---------------------------
        if world.boundaries:
            verts = world.boundaries.vertices
            xs = [p.x for p in verts] + [verts[0].x]
            ys = [p.y for p in verts] + [verts[0].y]
            ax.plot(xs, ys, linestyle='--')

        # ---------------------------
        # Obstáculos
        # ---------------------------
        for obs in world.obstacles:

            if obs.__class__.__name__ == "Circle":
                circle = plt.Circle((obs.center.x, obs.center.y),
                                    obs.radius,
                                    fill=False)
                ax.add_patch(circle)

            elif obs.__class__.__name__ == "Quadrilateral":
                verts = obs.vertices
                xs = [p.x for p in verts] + [verts[0].x]
                ys = [p.y for p in verts] + [verts[0].y]
                ax.plot(xs, ys)

                # --- Stadium ---
            elif obs.__class__.__name__ == "Stadium":
                p1, p2 = obs.vertices
                r = obs.radius

                # vetor direção
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                length = np.hypot(dx, dy)

                if length == 0:
                    continue

                ux, uy = dx / length, dy / length
                # vetor perpendicular
                px, py = -uy, ux

                # retângulo central
                corners = [
                    (p1.x + px * r, p1.y + py * r),
                    (p2.x + px * r, p2.y + py * r),
                    (p2.x - px * r, p2.y - py * r),
                    (p1.x - px * r, p1.y - py * r),
                    (p1.x + px * r, p1.y + py * r),
                ]
                xs, ys = zip(*corners)
                ax.plot(xs, ys)

                # semicirculos
                theta = np.linspace(0, np.pi, 50)

                # lado p1
                angle = np.arctan2(uy, ux)
                x1 = p1.x + r * np.cos(theta + angle + np.pi / 2)
                y1 = p1.y + r * np.sin(theta + angle + np.pi / 2)
                ax.plot(x1, y1)

                # lado p2
                x2 = p2.x + r * np.cos(theta + angle - np.pi / 2)
                y2 = p2.y + r * np.sin(theta + angle - np.pi / 2)
                ax.plot(x2, y2)

        # ---------------------------
        # PATH ORIGINAL (planner)
        # ---------------------------
        if path:
            xs = [p.x for p in path]
            ys = [p.y for p in path]
            ax.plot(xs, ys, marker='o', linestyle='--')

        # ---------------------------
        # PATH OTIMIZADO
        # ---------------------------
        if opt_path:
            xs = [p.x for p in opt_path]
            ys = [p.y for p in opt_path]
            ax.plot(xs, ys, marker='o')

        # start/goal
        ax.scatter(path[0].x, path[0].y, s=100)
        ax.scatter(path[-1].x, path[-1].y, s=100)

        ax.set_title(f"Execução {i} | path={len(path)} pts | opt={len(opt_path)} pts")
        ax.set_aspect('equal', adjustable='box')
        ax.grid()

        plt.show()

        plotted += 1

    print(f"\nPlots gerados: {plotted}")

if __name__ == "__main__":
    times, results = benchmark_pipeline(1000)
    print(len(times["planner"]))
    print(np.mean(times["planner"]))
    print(times)
    plot_benchmark_results(times)
    plot_paths_comparison(results)