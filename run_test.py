# run_test.py

# Wonhee LEE
# 2024 JUL 20 (SAT)

"""
simulate the airfoil moment bench test.
"""

# reference:


import math
import numpy as np
import matplotlib.pyplot as plt
from simple_pid import PID

import moment_bench


# design airfoil
C_l0 = 0  # symmetric aifoil (flat plate)
c = 0.2  # [m] chord
s = 1  # [m] span
mass = 15  # [kg]
moi_yy = mass * c ** 2 / 12  # [kg*m^2]
l_cm = c / 7
flap = moment_bench.Control_Surface(c=c / 3, s=0.65 * s)

airfoil = moment_bench.Airfoil(C_l0, c, s, mass, moi_yy, l_cm, flap)

# define dynamics
air = moment_bench.Air(density=1.225)

air_speed = 34  # [m/s] wind tunnel air speed
init_cond = moment_bench.Initial_Condition(v_a=air_speed, alpha=math.radians(5), u_f=0)

flap_limits = (math.radians(-20), math.radians(20))

dynamics = moment_bench.Dynamics(airfoil, air, init_cond, flap_limits, g=9.81)

# set up environment
env = moment_bench.Environment(dynamics)
env.render()

# control
controller = PID(0.1, 0.005, 0.0005, setpoint=0)

max_num_steps = 3

Mm, _, _, _, _ = env.step(action=env.dynamics.init_cond.u_f)
print(f"initial pitching moment about the pivot: {Mm:.4F} [Nm]")

for i in range(max_num_steps):
    u_f = controller(Mm)
    Mm, _, _, _, _ = env.step(u_f)

    print(math.degrees(u_f), Mm)
