import matplotlib.pyplot as plt
import numpy as np
from main import *

def plot_world_and_path(world, path=None):
    fig, ax = plt.subplots()

    # ---------------------------
    # Boundaries
    # ---------------------------
    if world.boundaries is not None:
        verts = world.boundaries.vertices
        xs = [p.x for p in verts] + [verts[0].x]
        ys = [p.y for p in verts] + [verts[0].y]
        ax.plot(xs, ys, linestyle='--')

    # ---------------------------
    # Obstáculos
    # ---------------------------
    for obs in world.obstacles:

        # --- Circle ---
        if obs.__class__.__name__ == "Circle":
            circle = plt.Circle((obs.center.x, obs.center.y),
                                obs.radius,
                                fill=False)
            ax.add_patch(circle)

        # --- Quadrilateral ---
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
            x1 = p1.x + r * np.cos(theta + angle + np.pi/2)
            y1 = p1.y + r * np.sin(theta + angle + np.pi/2)
            ax.plot(x1, y1)

            # lado p2
            x2 = p2.x + r * np.cos(theta + angle - np.pi/2)
            y2 = p2.y + r * np.sin(theta + angle - np.pi/2)
            ax.plot(x2, y2)

    # ---------------------------
    # Trajetória
    # ---------------------------
    if path and len(path) > 0:
        xs = [p.x for p in path]
        ys = [p.y for p in path]

        ax.plot(xs, ys, marker='o')

        # start e goal
        ax.scatter(xs[0], ys[0], s=100)
        ax.scatter(xs[-1], ys[-1], s=100)

    # ---------------------------
    # Configurações finais
    # ---------------------------
    ax.set_aspect('equal', adjustable='box')
    ax.grid()

    plt.show()