import numpy as np
from scipy import constants as C
import pandas as pd

def Universe_R_DataFrame(H_0, Wm, Wr, Wv):
    H_0 = H_0 * 60 * 60 * 24 * 365 * 1e6 / C.parsec  # Conversion of H_0
    a = 2E-5
    R_0 = 2e-3
    R_0p = R_0
    WT = Wm + Wr + Wv
    Wk = 1.0 - WT

    def f(x):
        return x / np.sqrt(Wr + Wm * x + Wk * x**2 + Wv * x**4)

    def Rp(x):
        return H_0 * np.sqrt(Wr / x**2 + Wm / x + Wk + Wv * x**2)

    def dis(x):
        return Wr / x**2 + Wm / x + Wk + Wv * x**2

    if Wk < 0:
        print("Curvature sign 1, positive curvature (spherical, closed space)")
    elif Wk == 0:
        print("Curvature sign 0, zero curvature (flat, open space)")
    else:
        print("Curvature sign -1, negative curvature (hyperbolic, open space)")

    # Initial conditions
    dx2 = 1E-7  # 1,000 sub-intervals for integration
    o_vals = np.arange(a, R_0 + a, 2 * dx2)
    sumaf = (dx2 / 3.0) * np.sum(f(o_vals) + 4 * f(o_vals + dx2) + f(o_vals + 2 * dx2)) 
    t_0 = sumaf / H_0
    tf = [t_0]
    Rf = [R_0]

    # Now solving the equation using Euler's method
    t_f = 40.0  # Final time in Gyears
    dx = 1E-3  # 1,000 points in the integral
    dot_R_0 = Rp(R_0)
    t_vals = np.arange(t_0, t_f, dx)
    tdis = t_f

    for t in t_vals:
        if t < tdis:
            R_0 = Rp(R_0) * dx + R_0
            if dis(R_0) <= 2e-7:
                tdis = t
        else:
            R_0 = -Rp(R_0) * dx + R_0
            if R_0 <= R_0p:
                tdis = t_f
        Rf.append(R_0)
        tf.append(t)

    DataF_RU = pd.DataFrame({"time": tf, "radi": Rf})
    return DataF_RU

def Universe_Age(H_0, Wm, Wr, Wv):
    H_0 = H_0 * 60 * 60 * 24 * 365 * 1e6 / C.parsec  # Conversion of H_0
    a = 2E-3
    WT = Wm + Wr + Wv
    Wk = 1.0 - WT

    def f(x):
        return x / np.sqrt(Wr + Wm * x + Wk * x**2 + Wv * x**4)

    if Wk < 0:
        print("Curvature sign 1, positive curvature (spherical, closed space)")
    elif Wk == 0:
        print("Curvature sign 0, zero curvature (flat, open space)")
    else:
        print("Curvature sign -1, negative curvature (hyperbolic, open space)")

    # Values at present
    dx2_p = 1E-3  # 1,000 sub-intervals for integration
    o_vals = np.arange(a, 1, 2 * dx2_p)
    sumaf_p = (dx2_p / 3.0) * np.sum(f(o_vals) + 4 * f(o_vals + dx2_p) + f(o_vals + 2 * dx2_p))

    t_p = sumaf_p / H_0
    return [t_p, 1]
