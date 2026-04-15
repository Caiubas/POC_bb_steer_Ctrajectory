from __future__ import annotations
import copy
from dataclasses import dataclass


import numpy as np

from main import Vector, World

from math import fabs, sqrt


class AccelLimits:
    def __init__(self, acceleration_min: Vector, acceleration_max: Vector, vmax: float):
        self.acceleration_min = acceleration_min
        self.acceleration_max = acceleration_max
        self.vmax = vmax

    def get_per_axis_vlim(self):
        return self.vmax/np.sqrt(2)

@dataclass
class SplitResult:
    before: ControlSequence2D
    mid: ControlSequence2D | None
    after: ControlSequence2D
    x_t1: State2D
    x_t2: State2D

class State2D:
    x: PhaseState  # (q=0, v=0)
    y: PhaseState  # (q=0, v=0)

    def __init__(self, x: PhaseState, y: PhaseState):
        self.x = x
        self.y = y

    @classmethod
    def from_list(cls, s: list) -> "State2D":
        return cls(PhaseState(s[0], s[2]), PhaseState(s[1], s[3]))

    def to_list(self) -> list:
        return [self.x.q, self.y.q, self.x.v, self.y.v]

    """def integrate(self, control: ControlSegment2D) -> State2D:
        xo = self.x.q + self.x.v * control.duration + 0.5 * control.accel.x * control.duration ** 2
        xdoto = self.x.v + control.accel.x * control.duration
        yo = self.y.q + self.y.v * control.duration + 0.5 * control.accel.y * control.duration ** 2
        ydoto = self.y.v + control.accel.y * control.duration
        return State2D(PhaseState(xo, xdoto), PhaseState(yo, ydoto))"""

class ControlSegment2D:
    def __init__(self, accel: Vector, duration: float):
        self.accel = accel
        self.duration = duration

    def integrate(self, x0: State2D) -> State2D:
        xo = x0.x.q + x0.x.v * self.duration + 0.5 * self.accel.x * self.duration ** 2
        xdoto = x0.x.v + self.accel.x * self.duration
        yo = x0.y.q + x0.y.v * self.duration + 0.5 * self.accel.y * self.duration ** 2
        ydoto = x0.y.v + self.accel.y * self.duration
        return State2D(PhaseState(xo, xdoto), PhaseState(yo, ydoto))


class ControlSegment1D:
    def __init__(self, accel: float, duration: float):
        self.accel = accel
        self.duration = duration

class PhaseState:
    def __init__(self, q: float, v: float):
        self.q = q
        self.v = v

    def integrate(self, control: ControlSegment1D) -> PhaseState:
        xo = self.q + self.v * control.duration + 0.5 * control.accel * control.duration ** 2
        xdoto = self.v + control.accel * control.duration
        return PhaseState(xo, xdoto)

class ControlSequence1D:
    def __init__(self, segments: list[ControlSegment1D]):
        self.segments = segments

    def total_time(self) -> float:
        t = 0.0
        for c in self.segments:
            t += c.duration
        return t

class ControlSequence2D:
    def __init__(self, segments: list[ControlSegment2D]):
        self.segments = segments

    def total_time(self) -> float:
        t = 0.0
        for c in self.segments:
            t += c.duration
        return t

    def integrate(self, x0: State2D) -> State2D:
        x1 = x0
        for c in self.segments:
            x1 = c.integrate(x1)
        return x1

    def integrate_list(self, x0: State2D) -> list[State2D]:
        state = copy.deepcopy(x0)
        path = [state]
        for c in self.segments:
            state = c.integrate(state)
            path.append(state)
        return path

    def split(self, x0: State2D, t1, t2) -> SplitResult:

        def split_at(segs: ControlSequence2D, state: State2D, t_split: float) -> tuple[ControlSequence2D, ControlSequence2D, State2D]:
            before = []
            after = []
            s = state
            t = 0.0
            for idx, seg in enumerate(segs):
                u, dt = seg.accel, seg.duration
                if t + dt <= t_split + 1e-9:
                    before.append(seg)
                    s = seg.integrate(s)
                    t += dt
                else:
                    dt1 = t_split - t
                    dt2 = dt - dt1
                    if dt1 > 1e-9:
                        before.append(ControlSegment2D(seg.accel, dt1))
                        s = ControlSegment2D(seg.accel, dt1).integrate(s)
                    if dt2 > 1e-9:
                        after.append(ControlSegment2D(seg.accel, dt2))
                    # usa idx em vez de segs.index(seg) para evitar erros com duplicados
                    after += segs.segments[idx + 1:]
                    return ControlSequence2D(before), ControlSequence2D(after), s
            return ControlSequence2D(before), ControlSequence2D(after), s

        seq_before, rest, x_t1 = split_at(self, x0, t1)
        t2_local = t2 - t1
        seq_mid, seq_after, x_t2 = split_at(rest, x_t1, t2_local)

        return SplitResult(
            before=seq_before,
            mid=seq_mid if seq_mid.total_time() >= 1e-9 else None,
            after=seq_after,
            x_t1=x_t1,
            x_t2=x_t2,
        )

    def replace(self, x0: State2D, t1: float, t2: float, new_seq: ControlSequence2D) -> ControlSequence2D:
        result = self.split(x0, t1, t2)
        combined = result.before.segments + new_seq.segments + result.after.segments
        return ControlSequence2D(combined)

    def merge(self, other: ControlSequence2D) -> ControlSequence2D:
        return ControlSequence2D(self.segments + other.segments)

    def collision_free(self, x0: State2D, fn, world):
        state = x0
        for seg in self.segments:
            if not fn(state, seg, world):
                return False
            state = seg.integrate(state)
        return True

    def __iter__(self):
        return iter(self.segments)

    def __len__(self):
        return len(self.segments)


class ScalarBangBang:
    def __init__(self, limits: AccelLimits):
        self.limits = limits
        self.time_epsilon = 0.0000001
        self.float_epsilon = 1.0E-200

    def optimal_no_vlim(self, x0: PhaseState, x1: PhaseState) -> ControlSequence1D:
        if x0.q == x1.q and x0.v == x1.v:
            return ControlSequence1D([])

        invmin = 1 / self.limits.acceleration_min.x
        invmax = 1 / self.limits.acceleration_max.x
        c1 = x0.q - x1.q - 0.5 * (invmin * x0.v * x0.v - invmax * x1.v * x1.v)
        a1 = 0.5 * (invmin - invmax)
        s1 = -4.0 * a1 * c1

        c2 = x0.q - x1.q - 0.5 * (invmax * x0.v * x0.v - invmin * x1.v * x1.v)
        s2 = 4.0 * a1 * c2  # Saving computation by noting that a2 = -a1

        t1 = t1b = 1.0E20
        t2 = t2b = 1.0E20
        u1 = u1b = self.limits.acceleration_min.x
        u2 = u2b = self.limits.acceleration_max.x

        if s1 >= 0:
            xdot = sqrt(s1) / (2.0 * a1)
            t1 = invmin * (xdot - x0.v)
            t2 = invmax * (x1.v - xdot)
            u1 = self.limits.acceleration_min.x
            u2 = self.limits.acceleration_max.x

        if s2 >= 0:
            xdot = -sqrt(s2) / (2.0 * a1)
            t1b = invmax * (xdot - x0.v)
            t2b = invmin * (x1.v - xdot)
            u1b = self.limits.acceleration_max.x
            u2b = self.limits.acceleration_min.x

        if fabs(t1) < self.time_epsilon:
            t1 = 0.0
        if fabs(t2) < self.time_epsilon:
            t2 = 0.0
        if fabs(t1b) < self.time_epsilon:
            t1b = 0.0
        if fabs(t2b) < self.time_epsilon:
            t2b = 0.0

        if (t1b + t2b < t1 + t2 and t1b >= 0.0 and t2b >= 0.0) or t1 < 0.0 or t2 < 0.0:
            t1 = t1b
            t2 = t2b
            u1 = u1b
            u2 = u2b

        # No need to include zero-time control segments
        if t1 == 0.0:
            return ControlSequence1D([ControlSegment1D(u2, t2)])
        if t2 == 0.0:
            return ControlSequence1D([ControlSegment1D(u1, t1)])

        return ControlSequence1D([ControlSegment1D(u1, t1), ControlSegment1D(u2, t2)])

    def optimal(self, x0: PhaseState, x1: PhaseState) -> ControlSequence1D:    # Tenta solução bang-bang normal
        c = self.optimal_no_vlim(x0, x1)

        if c.segments == []:
            return c

        # Calcula velocidade de pico (ocorre no fim do primeiro bang)
        if len(c.segments) == 1:
            v_peak = x0.v + c.segments[0].accel * c.segments[0].duration ##TODO verificar esse if else bizarro
        else:
            v_peak = x0.v + c.segments[0].accel * c.segments[0].duration

        # Se não viola os limites, devolve solução normal
        if -self.limits.get_per_axis_vlim() - self.time_epsilon <= v_peak <= self.limits.get_per_axis_vlim() + self.time_epsilon:
            return c

        # Viola — constrói perfil trapezoidal
        if v_peak > self.limits.get_per_axis_vlim():
            v_cruise = self.limits.get_per_axis_vlim()
            u_acc = self.limits.acceleration_max.x
            u_dec = self.limits.acceleration_min.x
        else:
            v_cruise = -self.limits.get_per_axis_vlim()
            u_acc = self.limits.acceleration_min.x
            u_dec = self.limits.acceleration_max.x

        if fabs(u_acc) < self.float_epsilon or fabs(u_dec) < self.float_epsilon:
            return c

        # Tempo para atingir v_cruise a partir de iv
        t_acc = (v_cruise - x0.v) / u_acc
        if t_acc < -self.time_epsilon:
            return c
        t_acc = max(0.0, t_acc)

        # Tempo para desacelerar de v_cruise até gv
        t_dec = (x1.v - v_cruise) / u_dec
        if t_dec < -self.time_epsilon:
            return c
        t_dec = max(0.0, t_dec)

        # Posição no fim da aceleração
        x_end_acc = x0.q + x0.v * t_acc + 0.5 * u_acc * t_acc ** 2

        # Posição no início da desaceleração (calculada de trás para a frente)
        x_start_dec = x1.q - v_cruise * t_dec - 0.5 * u_dec * t_dec ** 2

        # Distância a percorrer a velocidade constante
        x_cruise_dist = x_start_dec - x_end_acc

        if fabs(v_cruise) < self.float_epsilon:
            return c

        t_cruise = x_cruise_dist / v_cruise

        if t_cruise < -self.time_epsilon:
            # Não há espaço para cruzeiro — vmax não é atingível neste segmento
            return c

        t_cruise = max(0.0, t_cruise)

        result = []
        if t_acc > self.time_epsilon:
            result.append(ControlSegment1D(u_acc, t_acc))
        if t_cruise > self.time_epsilon:
            result.append(ControlSegment1D(0.0, t_cruise))
        if t_dec > self.time_epsilon:
            result.append(ControlSegment1D(u_dec, t_dec))

        # Verifica que a posição final está correcta
        x_final = x_end_acc + v_cruise * t_cruise + x1.q - x_start_dec
        if fabs(x_final - x1.q) > 0.001:
            return self.optimal_no_vlim(x0, x1)

        return ControlSequence1D(result) if result else c

    def scaled_bb_no_vlim(self,x0: PhaseState, x1: PhaseState, tf: float) -> ControlSequence1D:
        co = self.optimal_no_vlim(x0, x1)
        tfo = co.total_time()

        if tfo > tf:
            print("Error: Requested final time is less than time optimal")
            return ControlSequence1D([])

        # Recently added divide by zero tests. Maybe it's beter to just test whether co is a singleton
        u1 = self.limits.acceleration_max.x
        num = x1.q - x0.q - (x0.v + x1.v) * tf * 0.5
        den = (x0.v - x1.v) * 0.5 + tf * u1 * 0.5

        if fabs(den) < self.float_epsilon:
            return ControlSequence1D([])
        t1 = num / den

        if t1 < 0:
            u1 = self.limits.acceleration_min.x
            den = (x0.v - x1.v) * 0.5 + tf * u1 * 0.5
            if fabs(den) < self.float_epsilon:
                return ControlSequence1D([])
            t1 = num / den

        tden = (tf - t1)
        if fabs(tden) < self.float_epsilon:
            return []
        u2 = ((x1.v - x0.v) - u1 * t1) / tden

        if u2 > self.limits.acceleration_max.x + self.time_epsilon or u2 < self.limits.acceleration_min.x - self.time_epsilon:
            # print("Failure: Not stretchable from time",tfo,"to",tf,"  u2",u2)
            return ControlSequence1D([])

        return ControlSequence1D([ControlSegment1D(u1, t1), ControlSegment1D(u2, tf - t1)])


    def scaled_bb(self,x0: PhaseState, x1: PhaseState, tf: float) -> ControlSequence1D:
        # Tenta sempre a quadrática primeiro — é o caso geral
        # u_acc/u_dec dependem da direcção do movimento
        for u_acc, u_dec in [(self.limits.acceleration_max.x, self.limits.acceleration_min.x), (self.limits.acceleration_min.x, self.limits.acceleration_max.x)]:
            if fabs(u_acc) < self.float_epsilon or fabs(u_dec) < self.float_epsilon:
                continue

            a = 1.0 / u_acc
            b = 1.0 / u_dec

            A_coef = 0.5 * (b - a)
            B_coef = tf + x0.v * a - x1.v * b
            C_coef = x0.q - 0.5 * x0.v ** 2 * a + 0.5 * x1.v ** 2 * b - x1.q

            if fabs(A_coef) < self.float_epsilon:
                if fabs(B_coef) < self.float_epsilon:
                    continue
                vc_candidates = [-C_coef / B_coef]
            else:
                disc = B_coef ** 2 - 4 * A_coef * C_coef
                if disc < 0:
                    continue
                sq = sqrt(disc)
                vc_candidates = [(-B_coef + sq) / (2 * A_coef),
                                 (-B_coef - sq) / (2 * A_coef)]

            for vc in vc_candidates:
                if vc < -self.limits.get_per_axis_vlim() - self.time_epsilon or vc > self.limits.get_per_axis_vlim() + self.time_epsilon:
                    continue
                t_acc_v = (vc - x0.v) * a
                t_dec_v = (x1.v - vc) * b
                t_cr_v = tf - t_acc_v - t_dec_v
                if t_acc_v < -self.time_epsilon or t_dec_v < -self.time_epsilon or t_cr_v < -self.time_epsilon:
                    continue
                t_acc_v = max(0.0, t_acc_v)
                t_dec_v = max(0.0, t_dec_v)
                t_cr_v = max(0.0, t_cr_v)
                segs = []
                if t_acc_v > self.time_epsilon:
                    segs.append(ControlSegment1D(u_acc, t_acc_v))
                if t_cr_v > self.time_epsilon:
                    segs.append(ControlSegment1D(0.0, t_cr_v))
                if t_dec_v > self.time_epsilon:
                    segs.append(ControlSegment1D(u_dec, t_dec_v))
                if segs:
                    return ControlSequence1D(segs)

        return self.scaled_bb_no_vlim(x0, x1, tf)

    def hard_stop_bb(self,x0: PhaseState, x1: PhaseState) -> ControlSequence1D:
        # Hard stopping time
        if x0.v > 0:
            ux = self.limits.acceleration_min.x
            s = -x0.v / ux
            sx = x0.q + x0.v * s + 0.5 * ux * s * s
        else:
            ux = self.limits.acceleration_max.x
            s = -x0.v / ux
            sx = x0.q + x0.v * s + 0.5 * ux * s * s
        d = self.optimal_no_vlim(PhaseState(sx, 0.0), x1)

        c = []

        # Make the hard stop (if needed)
        if s > 0.0:
            c.append(ControlSegment1D(ux, s))

        c += d.segments
        return ControlSequence1D(c)

    def hard_stop_wait_bb(self,x0: PhaseState, x1: PhaseState, tf) -> ControlSequence1D:
        c = self.hard_stop_bb(x0, x1)
        tt = c.total_time()

        if tt > tf:
            # print("No enough time for hard stop")
            return ControlSequence1D([])

        # Make a wait (if needed)
        if tf > tt:
            if x0.v == 0.0:
                c.segments.insert(0, ControlSegment1D(0.0, tf - tt))
            else:
                c.segments.insert(1, ControlSegment1D(0.0, tf - tt))

        return c

class Steer2D:

    def __init__(self, limits: AccelLimits):
        self.limits = limits
        self.x_axis = ScalarBangBang(limits)
        self.y_axis = ScalarBangBang(limits)
        self.time_epsilon = 0.0000001
        self.float_epsilon = 1.0E-200

    def merge_axes(self, c1: ControlSequence1D, c2: ControlSequence1D) -> ControlSequence2D:
        """Combina dois ControlSequence de eixos diferentes com durações potencialmente distintas."""
        segments = []
        t = 0.0
        tt = c1.total_time()  # devem ter o mesmo tempo total

        i, j = 0, 0
        t1_acc, t2_acc = 0.0, 0.0

        ts1 = []
        t = 0.0
        for seg in c1.segments:
            t += seg.duration
            ts1.append(t)

        ts2 = []
        t = 0.0
        for seg in c2.segments:
            t += seg.duration
            ts2.append(t)

        t = 0.0
        while i < len(c1.segments) and j < len(c2.segments):
            next1 = ts1[i]
            next2 = ts2[j]
            dt = min(next1, next2) - t
            seg = ControlSegment2D(
                accel=Vector(c1.segments[i].accel, c2.segments[j].accel),
                duration=dt
            )
            segments.append(seg)
            t += dt
            if abs(t - next1) < 1e-9:
                i += 1
            if abs(t - next2) < 1e-9:
                j += 1

        return ControlSequence2D(segments)

    def steer_no_vlim(self, x0: PhaseState, x1: PhaseState) -> ControlSequence2D:
        pass

    def steer(self, x0: State2D, x1: State2D) -> ControlSequence2D:
        c1 = self.x_axis.optimal(x0.x, x1.x)
        c2 = self.y_axis.optimal(x0.y, x1.y)

        t1 = c1.total_time()
        t2 = c2.total_time()

        # O resto é igual ao time_optimal_steer_2d original
        if fabs(t1 - t2) > self.time_epsilon:
            if t1 < t2:
                c1 = self.x_axis.scaled_bb(x0.x, x1.x, t2)
                if c1.segments == []:
                    c1 = self.x_axis.hard_stop_bb(x0.x, x1.x)#bang_bang_hard_stop(xinit[0], xinit[2], xgoal[0], xgoal[2], umin[0], umax[0])
                    tt1 = c1.total_time()
                    if tt1 > t2:
                        c2 = self.y_axis.scaled_bb(x0.y, x1.y, tt1)#bang_bang_scaled_vlim(xinit[1], xinit[3], xgoal[1], xgoal[3], tt1, umin[1], umax[1],
                              #                     vmax=vmax_per_axis, vmin=vmin_per_axis)
                        if c2.segments == []:
                            c2 = self.y_axis.hard_stop_bb(x0.y, x1.y)#bang_bang_hard_stop(xinit[1], xinit[3], xgoal[1], xgoal[3], umin[1], umax[1])
                            tt2 = c2.total_time()
                            if tt2 < tt1:
                                c2 = self.y_axis.hard_stop_wait_bb(x0.y, x1.y, tt1)#bang_bang_hard_stop_wait(xinit[1], xinit[3], xgoal[1], xgoal[3], tt1, umin[1],
                                     #                         umax[1])
                            else:
                                c1 = self.x_axis.hard_stop_wait_bb(x0.x, x1.x, tt2)#bang_bang_hard_stop_wait(xinit[0], xinit[2], xgoal[0], xgoal[2], tt2, umin[0],
                                     #                         umax[0])
                    else:
                        c1 = self.x_axis.hard_stop_wait_bb(x0.x, x1.x, t2)#bang_bang_hard_stop_wait(xinit[0], xinit[2], xgoal[0], xgoal[2], t2, umin[0], umax[0])
            else:
                c2 = self.y_axis.scaled_bb(x0.y, x1.y, t1)#bang_bang_scaled_vlim(xinit[1], xinit[3], xgoal[1], xgoal[3], t1, umin[1], umax[1],
                     #                      vmax=vmax_per_axis, vmin=vmin_per_axis)
                if c2.segments == []:
                    c2 = self.y_axis.hard_stop_bb(x0.y, x1.y)#bang_bang_hard_stop(xinit[1], xinit[3], xgoal[1], xgoal[3], umin[1], umax[1])
                    tt2 = c2.total_time()
                    if tt2 > t1:
                        c1 = self.x_axis.scaled_bb(x0.x, x1.x, tt2)#bang_bang_scaled_vlim(xinit[0], xinit[2], xgoal[0], xgoal[2], tt2, umin[0], umax[0],
                             #                      vmax=vmax_per_axis, vmin=vmin_per_axis)
                        if c1.segments == []:
                            c1 = self.x_axis.hard_stop_bb(x0.x, x1.x)#bang_bang_hard_stop(xinit[0], xinit[2], xgoal[0], xgoal[2], umin[0], umax[0])
                            tt1 = c1.total_time()
                            if tt1 < tt2:
                                c1 = self.x_axis.hard_stop_wait_bb(x0.x, x1.x, tt2)#bang_bang_hard_stop_wait(xinit[0], xinit[2], xgoal[0], xgoal[2], tt2, umin[0],
                                     #                         umax[0])
                            else:
                                c2 = self.y_axis.hard_stop_wait_bb(x0.y, x1.y, tt1)#bang_bang_hard_stop_wait(xinit[1], xinit[3], xgoal[1], xgoal[3], tt1, umin[1],
                                     #                         umax[1])
                    else:
                        c2 = self.y_axis.hard_stop_wait_bb(x0.y, x1.y, t1)#bang_bang_hard_stop_wait(xinit[1], xinit[3], xgoal[1], xgoal[3], t1, umin[1], umax[1])

        if not c1 or not c2:
            return ControlSequence2D([])

        c = self.merge_axes(c1, c2)
        return c

    def steer_list(self, path: list, v0: Vector) -> ControlSequence2D:
        if len(path) == 0:
            raise RuntimeError('Path is empty')
        seq = self.steer(State2D(PhaseState(path[0].x, v0.x), PhaseState(path[0].y, v0.y)),
                   State2D(PhaseState(path[1].x, 0), PhaseState(path[1].y, 0)))
        for i in range(1, len(path) - 1):
            seq = seq.merge(self.steer(State2D(PhaseState(path[i].x, 0), PhaseState(path[i].y, 0)), State2D(PhaseState(path[i + 1].x, 0), PhaseState(path[i + 1].y, 0))))

        return seq

class BangBangOptimizer:
    def __init__(self, steer: Steer2D, collision_fn, world: World, max_iter = 500, patience = 50, min_improvement = 0.1, time_epsilon = 1e-6, float_epsilon = 1e-6):
        self.collision_fn = collision_fn
        self.world = world
        self.max_iter = max_iter
        self.patience = patience
        self.min_improvement = min_improvement
        self.time_epsilon = time_epsilon
        self.float_epsilon = float_epsilon
        self.steer = steer

    def _halton(self, index, base):
        """Gera o i-ésimo elemento da sequência de Halton numa dada base."""
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    def optimize(self, x0: State2D, controls: ControlSequence2D) -> ControlSequence2D:
        best = copy.deepcopy(controls)
        no_improve = 0
        halton_index = 1

        for iteration in range(self.max_iter):
            tf = best.total_time()
            if tf < 1e-9:
                break

            h1 = self._halton(halton_index, 2)
            h2 = self._halton(halton_index, 3)
            halton_index += 1

            t1 = h1 * tf
            t2 = h2 * tf

            if t1 > t2:
                if h1 < 0.5:
                    t1 = 0.0
                else:
                    t2 = tf

            if t1 >= t2 - self.time_epsilon:
                continue

            split_result = best.split(x0, t1, t2)
            seg_before = split_result.before
            seg_after = split_result.after
            seg_mid = split_result.mid
            x_t1, x_t2 = split_result.x_t1, split_result.x_t2

            if seg_mid is None:
                continue

            # usa os umin/umax correctos e os estados exactos do segmento
            new_mid = self.steer.steer(x_t1, x_t2)

            new_mid_time = new_mid.total_time()
            old_mid_time = seg_mid.total_time()

            if new_mid_time >= old_mid_time - self.time_epsilon:
                continue

            candidate = seg_before.merge(new_mid.merge(seg_after))

            if not new_mid.collision_free(x_t1, self.collision_fn, world=self.world): #colisao ponto a ponto
                continue

            improvement = old_mid_time - new_mid_time
            best = candidate
            if improvement > self.min_improvement:
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                break

        return best


