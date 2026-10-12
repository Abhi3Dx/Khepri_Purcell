"""
Quick polarization/orientation check for the inverse PhC near 800 meV.

Checks exact M-point P(q) for x- and y-oriented dipoles at:
794, 796, 798, 800 meV.

This is a fast diagnostic only; no BZ integration is performed.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from khepri.crystal import Crystal
from khepri.draw import Drawing
from khepri.dipole import dipole_power_q

PW = (9, 9)
PIX = (384, 384)

ENERGIES_MEV = [794.0, 796.0, 798.0, 800.0]

A_NM = 445.0
H_NM = 230.0
R_OVER_A = 0.405

N_SI = 3.48
N_SIO2 = 1.45

EPS_AIR = 1.0
EPS_SI = N_SI**2
EPS_SUB = N_SIO2**2 - 0.001j

HC_EV_NM = 1239.841984
H = H_NM / A_NM

a1 = np.array([np.sqrt(3.0)/2.0, +0.5])
a2 = np.array([np.sqrt(3.0)/2.0, -0.5])
lattice = np.array([a1, a2])

B = (a1 + a2)/2.0
DIPOLE_POSITION = (float(B[0]), float(B[1]), H/2.0)

def reciprocal_vectors(a1, a2):
    det = a1[0]*a2[1] - a1[1]*a2[0]
    b1 = (2*np.pi/det) * np.array([a2[1], -a2[0]])
    b2 = (2*np.pi/det) * np.array([-a1[1], a1[0]])
    return b1, b2

b1, b2 = reciprocal_vectors(a1, a2)

M_POINTS = {
    "M1": -0.5*b1,
    "M2": -0.5*b2,
    "M3": -0.5*(b1+b2),
}

POLARIZATIONS = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
}

def build_crystal():
    c = Crystal(
        pw=PW,
        lattice=lattice,
        lattice_pitch=1.0,
        epsi=EPS_AIR,
        epse=EPS_SUB,
    )

    d = Drawing(PIX, EPS_SI, c.lattice)
    d.disc((0.0, 0.0), R_OVER_A, EPS_AIR)
    pattern = d.canvas()

    c.add_layer_pixmap("L", pattern, H/2.0)
    c.add_layer_pixmap("R", pattern, H/2.0)

    c.set_device(["L", "R"], fields_mask=[True, False])
    return c

def wavelength_over_a(energy_mev):
    wavelength_nm = HC_EV_NM / (energy_mev/1000.0)
    return wavelength_nm / A_NM

def solve_M(crystal, energy_mev, q, current):
    result = dipole_power_q(
        crystal=crystal,
        source_after="L",
        wavelength=wavelength_over_a(energy_mev),
        kp=(float(q[0]), float(q[1])),
        position=DIPOLE_POSITION,
        current=current,
        backend="dense",
        return_details=False,
    )
    return float(result["P_total"])

def main():
    print("="*86)
    print("M-POINT POLARIZATION CHECK NEAR 800 meV")
    print("="*86)
    print(f"PW                = {PW}")
    print(f"B/a               = {B}")
    print(f"z/a               = {H/2:.9f}")
    print(f"substrate epsilon = {EPS_SUB}\n")

    print("M points:")
    for name, q in M_POINTS.items():
        print(f"  {name}: q = ({q[0]:+.9f}, {q[1]:+.9f})")

    crystal = build_crystal()

    all_results = {}

    print()
    print(
        f"{'E (meV)':>8}  {'M':>3}  {'P_x(M)':>14}  "
        f"{'P_y(M)':>14}  {'y/x':>12}  {'stronger':>8}"
    )
    print("-"*86)

    for energy in ENERGIES_MEV:
        all_results[energy] = {}

        for mname, q in M_POINTS.items():
            px = solve_M(crystal, energy, q, POLARIZATIONS["x"])
            py = solve_M(crystal, energy, q, POLARIZATIONS["y"])

            ratio = py / max(px, 1e-300)
            stronger = "y" if py > px else "x"

            all_results[energy][mname] = {
                "Px": px,
                "Py": py,
                "ratio_y_over_x": ratio,
            }

            print(
                f"{energy:8.1f}  {mname:>3}  "
                f"{px:14.6e}  {py:14.6e}  "
                f"{ratio:12.4e}  {stronger:>8}"
            )

    best_x = None
    best_y = None

    for energy, data in all_results.items():
        for mname, vals in data.items():
            rx = (vals["Px"], energy, mname)
            ry = (vals["Py"], energy, mname)

            if best_x is None or rx[0] > best_x[0]:
                best_x = rx
            if best_y is None or ry[0] > best_y[0]:
                best_y = ry

    print("\n" + "="*86)
    print("SUMMARY")
    print("="*86)

    print(
        f"Largest exact-M x response:\n"
        f"  P = {best_x[0]:.8e}\n"
        f"  E = {best_x[1]:.1f} meV\n"
        f"  M = {best_x[2]}"
    )

    print(
        f"\nLargest exact-M y response:\n"
        f"  P = {best_y[0]:.8e}\n"
        f"  E = {best_y[1]:.1f} meV\n"
        f"  M = {best_y[2]}"
    )

    if best_y[0] > 10.0*best_x[0]:
        print(
            "\nDIAGNOSTIC:\n"
            "The y-oriented dipole couples much more strongly than x at M.\n"
            "That strongly suggests our numerical axes are rotated relative\n"
            "to the paper's x/y convention."
        )
    elif best_x[0] > 10.0*best_y[0]:
        print(
            "\nDIAGNOSTIC:\n"
            "The x-oriented dipole couples much more strongly than y at M.\n"
            "That is consistent with the current polarization assignment."
        )
    else:
        print(
            "\nDIAGNOSTIC:\n"
            "Neither polarization dominates by more than one order of magnitude.\n"
            "If the expected 800-meV M resonance is still absent, next verify\n"
            "the exact B-point coordinate and material constants."
        )

if __name__ == "__main__":
    main()
