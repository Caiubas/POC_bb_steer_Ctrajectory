from dataclasses import dataclass
from __future__ import annotations

import numpy as np

from bboptimizer import split_controls
from main import Vector

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

@dataclass
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

    def integrate(self, control: ControlSegment2D) -> State2D:
        xo = self.x.q + self.x.v * control.duration + 0.5 * control.accel.x * control.duration ** 2
        xdoto = self.x.v + control.accel.x * control.duration
        yo = self.y.q + self.y.v * control.duration + 0.5 * control.accel.y * control.duration ** 2
        ydoto = self.y.v + control.accel.y * control.duration
        return State2D(PhaseState(xo, xdoto), PhaseState(yo, ydoto))

class ControlSegment2D:
    def __init__(self, accel: Vector, duration: float):
        self.accel = accel
        self.duration = duration

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

class ControlSequence2D:
    def __init__(self, segments: list[ControlSegment2D]):
        self.segments = segments

    def total_time(self) -> float:
        t = 0.0
        for c in self.segments:
            t += c.duration
        return t

    def integrate(self, x0: State2D) -> PhaseState:
        x1 = x0
        for c in self.segments:
            x1 = x1.integrate(c)
        return x1

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
                    s = s.integrate(seg)
                    t += dt
                else:
                    dt1 = t_split - t
                    dt2 = dt - dt1
                    if dt1 > 1e-9:
                        before.append(ControlSegment2D(seg.accel, dt1))
                        s = s.integrate(ControlSegment2D(seg.accel, dt1))
                    if dt2 > 1e-9:
                        after.append(ControlSegment2D(seg.accel, dt2))
                    # usa idx em vez de segs.index(seg) para evitar erros com duplicados
                    after += segs[idx + 1:]
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
            new_state = State2D(
                PhaseState(state.x.q + state.x.v * seg.duration + 0.5 * seg.accel.x * seg.duration ** 2,
                state.x.v + seg.accel.x * seg.duration),
                PhaseState(state.y.q + state.y.v * seg.duration + 0.5 * seg.accel.y * seg.duration ** 2,
                state.y.v + seg.accel.y * seg.duration)
            )
            if not fn(new_state, [seg], world):
                return False
            state = new_state

        if not fn(x0, self.segments, world):
            return False
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
            v_cruise = self.limits.get_per_axis_vlim()
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


    def scaled_bb(self,x0: PhaseState, x1: PhaseState, tf: float) -> ControlSequence1D:
        pass

    def hard_stop_bb(self,x0: PhaseState, x1: PhaseState, tf: float) -> ControlSequence1D:
        pass
