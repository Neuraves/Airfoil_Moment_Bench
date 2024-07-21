# moment_bench.py

# Wonhee LEE
# 2024 JUL 20 (SAT)

"""
flight environment for a rectangular flat plate airfoil fixed at the center.

- key assumptions
    - incompressible flow
    - no lateral {forces, moments, winds}
    - no sideslip angle (beta = 0)
    - ignore center of mass change due to control surface deflection
    - ignore moment of inertia change due to control surface deflection

- key notations
    - L, D : lift, drag
    - Mm : pitching moment
"""

# reference:


import math
import numpy as np
import matplotlib.pyplot as plt


class Airfoil:
    def __init__(self, C_l0, c, s, mass, moi_yy, l_cm, flap):
        """

        :param mass: total mass
        :param moi_yy: moment of inertia in yy
        :param l_cm: chord-wise distance from the pivot point (center) to the center of mass.
        """
        self.C_l0 = C_l0  # zero-lift lift coefficient
        self.c = c  # chord
        self.s = s  # span
        self.ar = s / c  # aspect ratio
        self.area = self.c * self.s
        self.C_l_alpha = 2 * math.pi  # airfoil lift curve slope

        self.mass = mass
        self.moi_yy = moi_yy
        self.l_cm = l_cm

        self.flap = flap


class Control_Surface:
    def __init__(self, c, s):
        self.c = c  # chord
        self.s = s  # span
        self.ar = s / c  # aspect ratio
        self.area = c * s
        self.tau = None  # control surface effectiveness


class Air:
    def __init__(self, density=1.225):
        self.density = density


class Initial_Condition:
    def __init__(self, v_a=0, alpha=0, u_f=0):
        self.v_a = v_a  # air velocity
        self.alpha = alpha  # angle of attack
        self.u_f = u_f  # flap deflection angle


class Dynamics:
    def __init__(self, airfoil, air, init_cond, flap_limits, g=9.81):
        self.airfoil = airfoil
        self.airfoil.flap.tau = control_surface_effectiveness(airfoil, airfoil.flap)
        self.air = air
        self.init_cond = init_cond
        self.v_a = init_cond.v_a
        self.alpha = init_cond.alpha
        self.u_f = init_cond.u_f
        self.g = g  # gravitational acceleration
        self.W = airfoil.mass * g
        self.flap_limits = flap_limits  # flap deflection limits (lower bound, upper bound)

        self.L = 0
        self.Mm = 0

    def compute_forces_and_moments(self, airfoil, air, u_f):
        # air stream
        q = 0.5 * air.density * np.linalg.norm(self.v_a) ** 2

        # compute change in lift due to control inputs
        u_f = self.limit_control_input(u_f, self.flap_limits)
        delta_L = q * airfoil.area * airfoil.C_l_alpha * airfoil.flap.tau * u_f
        # print(f"u_f: {math.degrees(u_f):.4F} [deg], delta_L: {delta_L:.4F} [N]")

        # update lift
        """includes lift change due to wind"""
        L = q * airfoil.area * (airfoil.C_l0 + airfoil.C_l_alpha * self.alpha) + delta_L
        # print(f"L: {L:.4F} [N], W: {self.W:.4F} [N]")

        # moment
        arm_ac = airfoil.c / 4 * np.cos(self.alpha)
        arm_cm = airfoil.l_cm * np.cos(self.alpha)
        Mm = arm_ac * L - arm_cm * self.W

        self.L = L
        self.Mm = Mm
        return L, Mm

    def limit_control_input(self, u, limits):
        if u < limits[0]:
            u = limits[0]
        elif u > limits[1]:
            u = limits[1]

        return u

    def reset(self):
        self.v_a = self.init_cond.v_a
        self.alpha = self.init_cond.alpha
        self.u_f = self.init_cond.u_f
        self.L = 0
        self.Mm = 0

    def render(self):
        plt.figure()

        # pivot
        plt.scatter(0, 0, s=25, c='k', marker='o', alpha=0.5)

        # plot airfoil
        body = np.array([[-self.airfoil.c / 2, self.airfoil.c / 2 - self.airfoil.flap.c],
                         [0, 0]])
        plt.xlim(math.floor(body[0, 0]), math.ceil(body[0, 1]))
        plt.ylim(math.floor(-self.airfoil.c / 2), math.ceil(self.airfoil.c / 2))
        body = R(-self.alpha) @ body
        plt.plot(body[0, :], body[1, :], 'k', linewidth=2)

        # plot flap
        flap_ghost = np.array([[self.airfoil.c / 2 - self.airfoil.flap.c, self.airfoil.c / 2],
                               [0, 0]])
        flap_ghost = R(-self.alpha) @ flap_ghost
        plt.plot(flap_ghost[0, :], flap_ghost[1, :], 'k--', linewidth=0.5)
        flap_start = flap_ghost[:, 0].reshape(-1, 1)
        flap_end = np.array([self.airfoil.flap.c, 0]).reshape(-1, 1)
        flap_end = R(-self.u_f) @ R(-self.alpha) @ flap_end + flap_start
        flap = np.hstack((flap_start, flap_end))
        plt.plot(flap[0, :], flap[1, :], color=plt.cm.tab10(1), linewidth=2)

        # plot forces and moments
        buffer = 0.02
        plt.arrow(body[0, 0] / 2, body[1, 0] / 2 + buffer, 0, self.L, width=0.005,
                  color=plt.cm.tab10(0))
        plt.text(-0.25, -0.25, f"Moment about Pivot: {self.Mm:.4F}")

        plt.grid()
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')

        plt.show()

    def initialize_figure(self, ax):
        # pivot
        plt.scatter(0, 0, s=25, c='k', marker='o', alpha=0.5)

        # airfoil
        body = np.array([[-self.airfoil.c / 2, self.airfoil.c / 2 - self.airfoil.flap.c],
                         [0, 0]])
        body = R(-self.alpha) @ body

        # flap
        flap_ghost = np.array([[self.airfoil.c / 2 - self.airfoil.flap.c, self.airfoil.c / 2],
                               [0, 0]])
        flap_ghost = R(-self.alpha) @ flap_ghost
        chord_line, = ax.plot(flap_ghost[0, :], flap_ghost[1, :], 'k--', linewidth=0.5)
        flap_start = flap_ghost[:, 0].reshape(-1, 1)
        flap_end = np.array([self.airfoil.flap.c, 0]).reshape(-1, 1)
        flap_end = R(-self.u_f) @ R(-self.alpha) @ flap_end + flap_start
        flap = np.hstack((flap_start, flap_end))
        flap_line, = ax.plot(flap[0, :], flap[1, :], color=plt.cm.tab10(1), linewidth=2)

        # forces and moments
        buffer = 0.02
        L_arrow = ax.arrow(body[0, 0] / 2, body[1, 0] / 2 + buffer, 0, self.L, width=0.005,
                           color=plt.cm.tab10(0))
        Mm_text = ax.text(0, -0.075, f"M: {self.Mm:.4F} [Nm]")

        body_line, = ax.plot(body[0, :], body[1, :], 'k', linewidth=2)

        return body_line, flap_line, chord_line, L_arrow, Mm_text

    def animate(self, i, assets, trajectories):
        ts = trajectories[0]
        state_trajectory = trajectories[1]
        control_trajectory = trajectories[2]
        fm_trajectory = trajectories[3]

        body_line, flap_line, chord_line, L_arrow, Mm_text = assets

        # plot airfoil
        body = np.array([[-self.airfoil.c / 2, self.airfoil.c / 2 - self.airfoil.flap.c],
                         [0, 0]])
        body = R(-self.alpha) @ body
        body_line.set_data(body[0, :], body[1, :])

        # plot flap
        flap_ghost = np.array([[self.airfoil.c / 2 - self.airfoil.flap.c, self.airfoil.c / 2],
                               [0, 0]])
        flap_ghost = R(-self.alpha) @ flap_ghost
        chord_line.set_data(flap_ghost[0, :], flap_ghost[1, :])
        flap_start = flap_ghost[:, 0].reshape(-1, 1)
        flap_end = np.array([self.airfoil.flap.c, 0]).reshape(-1, 1)
        flap_end = R(-control_trajectory[i]) @ R(-self.alpha) @ flap_end + flap_start
        flap = np.hstack((flap_start, flap_end))
        flap_line.set_data(flap[0, :], flap[1, :])

        # plot forces and moments
        buffer = 0.02
        scale = 1
        L_arrow.set_data(x=body[0, 0] / 2, y=body[1, 0] / 2 + buffer, dx=0, dy=scale * fm_trajectory[i, 0])
        Mm_text.set_text(f"M: {state_trajectory[i]:.4F} [Nm]")


def control_surface_effectiveness(airfoil, control_surface):
    """
    doi=10.30958/ajte.5-3-2
    """
    return math.sqrt(0.914 * control_surface.area / airfoil.area)


def R(alpha):
    return np.array([[np.cos(alpha), -np.sin(alpha)],
                     [np.sin(alpha), np.cos(alpha)]])


class Environment:
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.episode_step = 0

    def step(self, action):
        L, Mm = self.dynamics.compute_forces_and_moments(self.dynamics.airfoil, self.dynamics.air, u_f=action)

        reward = 0
        terminated = False
        truncated = False
        if abs(Mm) < 1E-3:
            reward = 1

        if self.episode_step > 1000:
            truncated = True

        observation = Mm

        self.episode_step += 1

        info = dict()
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.dynamics.reset()
        self.episode_step = 0

    def render(self):
        self.dynamics.render()
