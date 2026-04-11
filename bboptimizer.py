import random
from math import sqrt, fabs
from bbsteer import *

# Cola com o código original do bang_bang que já tens

def bb_optimizer(xinit, controls, world, collision_check_fn, max_iter=500, min_improvement=0.1, patience=50):
    """
    Bang-bang trajectory optimizer (Secção IV-B do artigo LaValle et al. 2023).

    Args:
        xinit:              Estado inicial [x, y, vx, vy]
        controls:           Lista de segmentos [[u, t], ...] com u = [ux, uy]
        collision_check_fn: Função(xinit, controls) -> bool, True se livre de colisões
        max_iter:           Número máximo de iterações
        min_improvement:    Melhoria mínima (segundos) para contar como progresso
        patience:           Iterações sem melhoria antes de terminar

    Returns:
        controls optimizados
    """
    umin = [-1.0, -1.0]
    umax = [1.0, 1.0]

    best = [c[:] for c in controls]
    no_improve = 0

    for iteration in range(max_iter):
        tf = control_time(best)
        if tf < 1e-9:
            break

        # Escolhe t1 e t2 aleatoriamente (estratégia do artigo)
        t1 = random.uniform(0, tf)
        t2 = random.uniform(0, tf)
        if t1 > t2:
            # Foca nas extremidades (coin flip do artigo)
            if random.random() < 0.5:
                t1 = 0.0
            else:
                t2 = tf
        if t1 >= t2:
            continue

        # Divide a trajectória em 3 partes: [0,t1], [t1,t2], [t2,tf]
        seg_before, seg_mid, seg_after, x_t1, x_t2 = split_controls(xinit, best, t1, t2)

        if seg_mid is None:
            continue

        # Substitui segmento intermédio por bang-bang time-optimal
        new_mid = time_optimal_steer_2d_vlim(x_t1, x_t2,
                                        tuple(umin), tuple(umax))

        new_mid_time = control_time(new_mid)
        old_mid_time = control_time(seg_mid)

        if new_mid_time >= old_mid_time - 1e-6:
            continue  # Sem melhoria

        # Monta nova trajectória candidata
        candidate = seg_before + new_mid + seg_after

        # Verifica colisões no novo segmento
        if not collision_check_fn(x_t1, new_mid, world):
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