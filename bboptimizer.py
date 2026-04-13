import random
from math import sqrt, fabs
from bbsteer import *

def check_segment_collision(x0, controls, collision_check_fn, world, dt=0.05):
    """Verifica colisão ao longo da trajectória parabolica em passos de dt."""
    state = list(x0)
    for seg in controls:
        u, seg_t = seg[0], seg[1]
        elapsed = 0.0
        while elapsed < seg_t:
            step = min(dt, seg_t - elapsed)
            new_state = [
                state[0] + state[2]*step + 0.5*u[0]*step**2,
                state[1] + state[3]*step + 0.5*u[1]*step**2,
                state[2] + u[0]*step,
                state[3] + u[1]*step,
            ]
            if not collision_check_fn([state[0], state[1], state[2], state[3]],
                                       [[u, step]], world):
                return False
            state = new_state
            elapsed += step
    return True

def halton(index, base):
    """Gera o i-ésimo elemento da sequência de Halton numa dada base."""
    result = 0.0
    f = 1.0
    i = index
    while i > 0:
        f /= base
        result += f * (i % base)
        i //= base
    return result

def bb_optimizer(xinit, controls, world, collision_check_fn, vmax, umin=(-0.2, -0.2), umax=(0.2, 0.2), max_iter=500, min_improvement=0.1, patience=50):
    best = [c[:] for c in controls]
    no_improve = 0
    halton_index = 1

    for iteration in range(max_iter):
        tf = control_time(best)
        if tf < 1e-9:
            break

        h1 = halton(halton_index, 2)
        h2 = halton(halton_index, 3)
        halton_index += 1

        t1 = h1 * tf
        t2 = h2 * tf

        if t1 > t2:
            if h1 < 0.5:
                t1 = 0.0
            else:
                t2 = tf

        if t1 >= t2 - time_epsilon:
            continue

        seg_before, seg_mid, seg_after, x_t1, x_t2 = split_controls(xinit, best, t1, t2)

        if seg_mid is None:
            continue

        # usa os umin/umax correctos e os estados exactos do segmento
        new_mid = time_optimal_steer_2d_vlim(x_t1, x_t2, umin, umax, vmax=vmax)

        new_mid_time = control_time(new_mid)
        old_mid_time = control_time(seg_mid)

        if new_mid_time >= old_mid_time - 1e-6:
            continue

        candidate = seg_before + new_mid + seg_after

        if not collision_check_fn(x_t1, new_mid, world):
            continue

        if not check_segment_collision(x_t1, new_mid, collision_check_fn, dt=0.05, world=world):
            continue

        improvement = old_mid_time - new_mid_time
        best = candidate
        if improvement > min_improvement:
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best


def split_controls(xinit, controls, t1, t2):
    n = int(0.5 * len(xinit))

    def integrate_step(state, seg):
        s = list(state)
        u, dt = seg[0], seg[1]
        for i in range(n):
            s[i] = state[i] + state[n+i]*dt + 0.5*u[i]*dt**2
            s[n+i] = state[n+i] + u[i]*dt
        return s

    def split_at(segs, state, t_split):
        before = []
        after = []
        t = 0.0
        s = list(state)
        for idx, seg in enumerate(segs):
            u, dt = seg[0], seg[1]
            if t + dt <= t_split + 1e-9:
                before.append(seg)
                s = integrate_step(s, seg)
                t += dt
            else:
                dt1 = t_split - t
                dt2 = dt - dt1
                if dt1 > 1e-9:
                    before.append([u, dt1])
                    s = integrate_step(s, [u, dt1])
                if dt2 > 1e-9:
                    after.append([u, dt2])
                # usa idx em vez de segs.index(seg) para evitar erros com duplicados
                after += segs[idx+1:]
                return before, after, s
        return before, after, s

    seg_before, rest, x_t1 = split_at(controls, xinit, t1)
    t2_local = t2 - t1
    seg_mid, seg_after, x_t2 = split_at(rest, x_t1, t2_local)

    if control_time(seg_mid) < 1e-9:
        return seg_before, None, seg_after, x_t1, x_t2

    return seg_before, seg_mid, seg_after, x_t1, x_t2