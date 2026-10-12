"""
Validation tests for z-oriented electric-dipole support in Khepri.

Run from repository root:

    python examples/test_dipole_jz.py

Checks
------
1. Jz=0 reproduces the old Jx/Jy jump exactly.
2. Vacuum Jz source vector has Delta Ex=Kx*Jz and Delta Ey=Ky*Jz.
3. |Jz|^2 scaling.
4. q -> -q reciprocity.
5. Lateral translation invariance in homogeneous vacuum.
6. Homogeneous-vacuum BZ integration gives the same total power for
   x-, y-, and z-oriented dipoles and agrees with k0^2/(6*pi).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from khepri.crystal import Crystal
from khepri.dipole import (
    dipole_source_jump_xy,
    dipole_source_jump_xyz,
    dipole_power_q,
    homogeneous_reference_power,
)

PW = (1, 1)
A = 1.0
WAVELENGTH = 2.0
NQ = 101

TOL_BASIC = 1e-9
TOL_INTEGRATED = 0.03


def build_vacuum_crystal():
    c = Crystal(
        pw=PW,
        lattice="square",
        lattice_pitch=A,
        epsi=1.0,
        epse=1.0,
    )

    c.add_layer_uniform(
        "minus_z_half",
        epsilon=1.0,
        depth=0.5,
    )
    c.add_layer_uniform(
        "plus_z_half",
        epsilon=1.0,
        depth=0.5,
    )

    c.set_device(
        ["minus_z_half", "plus_z_half"],
        fields_mask=[True, False],
    )
    return c


def solve_power(
    crystal,
    qx,
    qy,
    current,
    position=(0.0, 0.0, 0.5),
):
    return dipole_power_q(
        crystal=crystal,
        source_after="minus_z_half",
        wavelength=WAVELENGTH,
        kp=(float(qx), float(qy)),
        position=position,
        current=current,
        backend="dense",
        return_details=False,
    )["P_total"]


def relative_difference(a, b):
    return abs(a-b) / max(abs(a), abs(b), 1e-30)


def test_xy_reduction():
    Jx = np.array([1.0 + 0.2j, -0.3 + 0.4j], dtype=complex)
    Jy = np.array([0.7 - 0.1j, 0.2 + 0.9j], dtype=complex)
    zeros = np.zeros_like(Jx)

    old = dipole_source_jump_xy(Jx, Jy)
    new = dipole_source_jump_xyz(
        Jx=Jx,
        Jy=Jy,
        Jz=zeros,
        Kx=zeros,
        Ky=zeros,
        inverse_epsilon_operator=1.0,
    )

    expected = np.concatenate(
        [zeros, zeros, -1j*Jy, +1j*Jx]
    )

    assert np.allclose(old, expected, atol=1e-14, rtol=1e-14)
    assert np.allclose(new, expected, atol=1e-14, rtol=1e-14)
    print("PASS 1: Jx/Jy source is unchanged.")


def test_jz_source_vector():
    Jz = np.array([1.0+0.0j, 0.5-0.2j], dtype=complex)
    zeros = np.zeros_like(Jz)
    Kx = np.array([0.2, -0.4], dtype=complex)
    Ky = np.array([0.3, 0.1], dtype=complex)

    b = dipole_source_jump_xyz(
        Jx=zeros,
        Jy=zeros,
        Jz=Jz,
        Kx=Kx,
        Ky=Ky,
        inverse_epsilon_operator=1.0,
    )

    expected = np.concatenate(
        [Kx*Jz, Ky*Jz, zeros, zeros]
    )

    assert np.allclose(b, expected, atol=1e-14, rtol=1e-14)
    print("PASS 2: vacuum Jz source vector is correct.")


def test_basic_physics():
    c = build_vacuum_crystal()
    k0 = 2*np.pi/WAVELENGTH
    qx = 0.27*k0
    qy = 0.19*k0

    p1 = solve_power(c, qx, qy, current=(0.0, 0.0, 1.0))
    p2 = solve_power(c, qx, qy, current=(0.0, 0.0, 2.0))
    scaling = p2/p1
    assert abs(scaling - 4.0) < 1e-8
    print(f"PASS 3: |Jz|^2 scaling. P(2J)/P(J)={scaling:.12f}")

    p_minus = solve_power(c, -qx, -qy, current=(0.0, 0.0, 1.0))
    reciprocity = relative_difference(p1, p_minus)
    assert reciprocity < TOL_BASIC
    print(
        "PASS 4: q -> -q reciprocity. "
        f"relative difference={reciprocity:.3e}"
    )

    p_shifted = solve_power(
        c,
        qx,
        qy,
        current=(0.0, 0.0, 1.0),
        position=(0.217, -0.163, 0.5),
    )
    translation = relative_difference(p1, p_shifted)
    assert translation < TOL_BASIC
    print(
        "PASS 5: homogeneous lateral translation invariance. "
        f"relative difference={translation:.3e}"
    )


def integrated_vacuum_power(current):
    c = build_vacuum_crystal()

    qmin = -np.pi/A
    qmax = +np.pi/A
    dq = (qmax-qmin)/NQ
    q = qmin + (np.arange(NQ)+0.5)*dq

    total = 0.0
    for iy, qy in enumerate(q):
        for qx in q:
            total += solve_power(c, qx, qy, current=current)

        if (iy+1) % 20 == 0 or iy == NQ-1:
            print(f"    rows: {iy+1}/{NQ}")

    return total*dq*dq/(2*np.pi)**2


def test_homogeneous_isotropy():
    print()
    print(
        "Running homogeneous-vacuum orientation test "
        f"with NQ={NQ} ..."
    )

    Px = integrated_vacuum_power((1.0, 0.0, 0.0))
    Py = integrated_vacuum_power((0.0, 1.0, 0.0))
    Pz = integrated_vacuum_power((0.0, 0.0, 1.0))

    Pref = homogeneous_reference_power(
        wavelength=WAVELENGTH,
        refractive_index=1.0,
        current=(1.0, 0.0, 0.0),
    )

    print()
    print("Integrated powers:")
    print(f"  Px   = {Px:.12e}")
    print(f"  Py   = {Py:.12e}")
    print(f"  Pz   = {Pz:.12e}")
    print(f"  Pref = {Pref:.12e}")

    errors = {
        "Px": abs(Px-Pref)/Pref,
        "Py": abs(Py-Pref)/Pref,
        "Pz": abs(Pz-Pref)/Pref,
    }

    print("\nRelative errors vs exact homogeneous reference:")
    for key, value in errors.items():
        print(f"  {key}: {100*value:.4f}%")

    assert all(err < TOL_INTEGRATED for err in errors.values())

    xyz_spread = (
        max(Px, Py, Pz)-min(Px, Py, Pz)
    ) / ((Px+Py+Pz)/3)

    print(f"\nOrientation spread = {100*xyz_spread:.4f}%")
    assert xyz_spread < TOL_INTEGRATED

    print(
        "\nPASS 6: homogeneous integrated power is isotropic "
        "for x, y, and z dipoles."
    )


def main():
    print("="*78)
    print("KHEPRI Jz DIPOLE VALIDATION")
    print("="*78)

    test_xy_reduction()
    test_jz_source_vector()
    test_basic_physics()
    test_homogeneous_isotropy()

    print()
    print("="*78)
    print("ALL Jz VALIDATION TESTS PASSED")
    print("="*78)


if __name__ == "__main__":
    main()
