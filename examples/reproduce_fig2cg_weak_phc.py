"""
Rigorous reproduction attempt for Fig. 2(c) and Fig. 2(g) of

    S. A. Dyakov et al.,
    Phys. Rev. B 108, 155416 (2023)

Weak triangular photonic-crystal lattice at hbar*omega = 850 meV.

Panels
------
(c) vertical z-oriented dipole at point A
(g) horizontal dipole, defined as (x + y)/2, at point A

Paper geometry
--------------
a = 625 nm
h = 218 nm
r = 0.25 a
z0 = 80 nm measured from the Si/SiO2 interface

Weak lattice
------------
eps_pore = 1.001 * eps_Si

The slab is on a SiO2 substrate and has air above it.

Important implementation choices
--------------------------------
1. The triangular lattice is represented by

       a1 = a * (sqrt(3)/2, +1/2)
       a2 = a * (sqrt(3)/2, -1/2)

   with a normalized to 1 inside Khepri.

2. The circular pore is represented ANALYTICALLY, not by a pixmap.
   This avoids rasterization error in the Fourier coefficients.

3. Point A is chosen as the center of the triangular interstitial region

       A = (a1 + a2)/3 = (sqrt(3)/3, 0)

   which is equidistant from the three nearest pores.

4. The source plane splits the patterned Si layer into two identical
   patterned pieces:

       air
        |
       patterned Si, thickness h-z0 = 138 nm
        |
       dipole plane
        |
       patterned Si, thickness z0 = 80 nm
        |
       SiO2

5. Khepri uses dense RCWA only.

6. The paper reports 85 Fourier harmonics. Khepri uses rectangular
   harmonic grids, so (9,9) = 81 harmonics is the closest natural
   truncation. An optional convergence check compares 49, 81, and
   121 harmonics at several q points.

7. The small substrate absorption used for total-Purcell calculations
   is represented in Khepri's time convention as

       eps_SiO2 = n_SiO2^2 - 0.001j

8. The plotted quantity is the dimensionless Purcell-emissivity density

       rho_F(sx,sy)
         = k0^2/(2*pi)^2 * P_Khepri(qx,qy) / P_ref,Khepri

   where

       sx = qx/k0
       sy = qy/k0

   and P_ref is the homogeneous bulk-Si reference power.

9. Only points inside the first Brillouin zone are calculated.

10. The two panels use independent logarithmic color scales, matching
    the statement in the paper that the Fig. 2 panel scales differ.

Run from repository root
------------------------
    python examples/reproduce_fig2cg_weak_phc.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext

from khepri.crystal import Crystal
from khepri.alternative import free_space_eigenmodes
from khepri.fields import layer_eigenbasis_matrix
from khepri.dipole import (
    dipole_fourier_current,
    dipole_source_jump_xyz,
    solve_dipole_source_plane,
    dipole_source_plane_fields,
    dipole_total_nearfield_power,
    homogeneous_reference_power,
)


# ============================================================
# PAPER PARAMETERS
# ============================================================

ENERGY_MEV = 850.0
HC_EV_NM = 1239.841984

A_NM = 625.0
H_NM = 218.0
R_OVER_A = 0.25
Z0_FROM_SUBSTRATE_NM = 80.0

WAVELENGTH_NM = HC_EV_NM / (ENERGY_MEV / 1000.0)

# Khepri normalized units: a = 1
WAVELENGTH = WAVELENGTH_NM / A_NM
K0 = 2.0 * np.pi / WAVELENGTH

H = H_NM / A_NM
DEPTH_AIR_TO_SOURCE = (H_NM - Z0_FROM_SUBSTRATE_NM) / A_NM
DEPTH_SOURCE_TO_SUBSTRATE = Z0_FROM_SUBSTRATE_NM / A_NM


# ============================================================
# MATERIALS
# ============================================================

# Working optical constants near lambda ~ 1.459 um.
# The PRB paper does not explicitly tabulate the exact n-values
# in the main text. Change these if the exact dispersion model
# used by the authors is identified.
N_SI = 3.48
N_SIO2 = 1.45

EPS_SI = N_SI**2
EPS_PORE_WEAK = 1.001 * EPS_SI

EPS_AIR = 1.0

# Passive-loss sign for Khepri's exp(-i*kz*z) convention.
EPS_SIO2 = N_SIO2**2 - 0.001j


# ============================================================
# LATTICE
# ============================================================

LATTICE = np.array(
    [
        [np.sqrt(3.0) / 2.0, +0.5],
        [np.sqrt(3.0) / 2.0, -0.5],
    ],
    dtype=float,
)

A1 = LATTICE[0]
A2 = LATTICE[1]

# High-symmetry interstitial point A.
POINT_A = (A1 + A2) / 3.0

SOURCE_POSITION = (
    float(POINT_A[0]),
    float(POINT_A[1]),
    float(DEPTH_AIR_TO_SOURCE),
)


# ============================================================
# NUMERICAL SETTINGS
# ============================================================

# 81 harmonics: nearest rectangular Khepri truncation to paper's 85.
PW = (9, 9)

# Midpoint Cartesian sampling over the bounding box of the hexagonal FBZ.
# Only points inside the FBZ are actually solved.
NQ = 61

# Show at most this many decades below each panel maximum.
# This affects visualization only; saved data are untouched.
LOG_DECADES = 6.0

# Optional harmonic convergence check before the full map.
RUN_PW_CONVERGENCE_CHECK = True
PW_CONVERGENCE_LIST = [
    (7, 7),     # 49 harmonics
    (9, 9),     # 81 harmonics
    (11, 11),   # 121 harmonics
]

OUTPUT_DIR = Path("fig2_weak_phc_output")

SOURCE_AFTER = "weak_air_side"


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def weak_pore_islands():
    """
    Exact analytical description of one circular pore per primitive cell.

    The pore is centered at a triangular-lattice point (the origin).
    Periodic repetition is handled by RCWA.
    """

    return [
        {
            "type": "disc",
            "params": [
                0.0,
                0.0,
                R_OVER_A,
            ],
            "epsilon": EPS_PORE_WEAK,
        }
    ]


def build_weak_crystal(pw=PW):
    """
    Build air / weak-PhC slab / SiO2 with the dipole plane embedded
    inside the patterned slab.
    """

    crystal = Crystal(
        pw=pw,
        lattice=LATTICE,
        lattice_pitch=1.0,
        epsi=EPS_AIR,
        epse=EPS_SIO2,
    )

    islands = weak_pore_islands()

    crystal.add_layer_analytical(
        "weak_air_side",
        epsilon=islands,
        epsilon_host=EPS_SI,
        depth=DEPTH_AIR_TO_SOURCE,
    )

    crystal.add_layer_analytical(
        "weak_substrate_side",
        epsilon=islands,
        epsilon_host=EPS_SI,
        depth=DEPTH_SOURCE_TO_SUBSTRATE,
    )

    # The source-adjacent layer must retain its eigenspace and IC
    # because Jz uses the inverse-epsilon operator.
    crystal.set_device(
        [
            "weak_air_side",
            "weak_substrate_side",
        ],
        fields_mask=[
            True,
            False,
        ],
    )

    return crystal


# ============================================================
# FIRST BRILLOUIN ZONE
# ============================================================

def reciprocal_nearest_vectors(crystal):
    """
    Return the six shortest reciprocal lattice vectors that define
    the Wigner-Seitz first Brillouin zone.
    """

    b1 = np.asarray(
        crystal.expansion.reciprocal[0],
        dtype=float,
    )

    b2 = np.asarray(
        crystal.expansion.reciprocal[1],
        dtype=float,
    )

    candidates = [
        b1,
        b2,
        b1 + b2,
    ]

    g6 = []

    for g in candidates:
        g6.append(g)
        g6.append(-g)

    return np.asarray(g6)


def inside_first_bz(qx, qy, g6, tol=1e-10):
    """
    Wigner-Seitz condition:

        q . G <= |G|^2 / 2

    for the six nearest reciprocal lattice vectors.
    """

    q = np.array(
        [qx, qy],
        dtype=float,
    )

    for g in g6:
        if (
            np.dot(q, g)
            >
            0.5 * np.dot(g, g) + tol
        ):
            return False

    return True


def first_bz_vertices(g6):
    """
    Construct the FBZ vertices by intersecting the six Bragg planes.
    """

    vertices = []

    for i in range(len(g6)):
        for j in range(i + 1, len(g6)):

            A = np.vstack(
                [
                    g6[i],
                    g6[j],
                ]
            )

            if abs(np.linalg.det(A)) < 1e-12:
                continue

            rhs = np.array(
                [
                    0.5 * np.dot(g6[i], g6[i]),
                    0.5 * np.dot(g6[j], g6[j]),
                ]
            )

            q = np.linalg.solve(
                A,
                rhs,
            )

            if all(
                np.dot(q, g)
                <=
                0.5 * np.dot(g, g) + 1e-9
                for g in g6
            ):
                vertices.append(q)

    # Remove duplicates.
    unique = []

    for q in vertices:
        if not any(
            np.linalg.norm(q - u) < 1e-8
            for u in unique
        ):
            unique.append(q)

    unique = np.asarray(unique)

    angles = np.arctan2(
        unique[:, 1],
        unique[:, 0],
    )

    unique = unique[
        np.argsort(angles)
    ]

    return unique


def build_q_grid(crystal):
    """
    Midpoint grid in (qx/k0, qy/k0), masked to the FBZ.
    """

    g6 = reciprocal_nearest_vectors(
        crystal
    )

    vertices_q = first_bz_vertices(
        g6
    )

    vertices_s = vertices_q / K0

    smax = (
        1.03
        *
        np.max(
            np.abs(vertices_s)
        )
    )

    ds = 2.0 * smax / NQ

    axis = (
        -smax
        +
        (np.arange(NQ) + 0.5) * ds
    )

    SX, SY = np.meshgrid(
        axis,
        axis,
        indexing="xy",
    )

    mask = np.zeros(
        SX.shape,
        dtype=bool,
    )

    for iy in range(NQ):
        for ix in range(NQ):

            qx = K0 * SX[iy, ix]
            qy = K0 * SY[iy, ix]

            mask[iy, ix] = inside_first_bz(
                qx,
                qy,
                g6,
            )

    return (
        axis,
        ds,
        SX,
        SY,
        mask,
        vertices_s,
        g6,
    )


# ============================================================
# ONE RCWA SOLVE -> ALL THREE DIPOLE ORIENTATIONS
# ============================================================

def source_power_from_solved_q(
    crystal,
    kp,
    current,
    R0,
    Sl,
    Sr,
    Kx,
    Ky,
    inverse_epsilon_operator,
):
    """
    Evaluate one dipole orientation using an already-solved passive RCWA
    structure. This avoids repeating the expensive patterned-layer solve
    for Jx, Jy, and Jz at the same Bloch q.
    """

    Jx, Jy, Jz = dipole_fourier_current(
        expansion=crystal.expansion,
        kp=kp,
        position=SOURCE_POSITION,
        current=current,
    )

    b = dipole_source_jump_xyz(
        Jx=Jx,
        Jy=Jy,
        Jz=Jz,
        Kx=Kx,
        Ky=Ky,
        inverse_epsilon_operator=inverse_epsilon_operator,
    )

    c_plus, c_minus = solve_dipole_source_plane(
        R0=R0,
        Sl=Sl,
        Sr=Sr,
        b=b,
    )

    psi_minus_z, psi_plus_z = dipole_source_plane_fields(
        R0=R0,
        Sl=Sl,
        Sr=Sr,
        c_plus=c_plus,
        c_minus=c_minus,
    )

    P_minus_z, P_plus_z, P_total = dipole_total_nearfield_power(
        psi_minus_z,
        psi_plus_z,
    )

    return float(
        np.real(P_total)
    )


def solve_q_all_orientations(
    crystal,
    qx,
    qy,
):
    """
    Dense RCWA solve once at q, then evaluate Jx, Jy, Jz from the same
    passive scattering solution.
    """

    kp = (
        float(qx),
        float(qy),
    )

    # Dummy ordinary plane-wave source required only to make Crystal.solve()
    # construct the passive RCWA scattering matrices.
    crystal.set_source(
        wavelength=WAVELENGTH,
        te=1.0,
        tm=0.0,
        kp=kp,
    )

    crystal.solve(
        backend="dense"
    )

    source_index = crystal.global_stacking.index(
        SOURCE_AFTER
    )

    Sl = crystal.stacking_matrices[
        source_index
    ]

    Sr = crystal.stacking_reverse_matrices[
        source_index
    ]

    if Sl is None or Sr is None:
        raise RuntimeError(
            "Source-plane partial scattering matrices were not retained."
        )

    Kx, Ky, _ = crystal.expansion.k_vectors(
        kp,
        WAVELENGTH,
    )

    W0, V0 = free_space_eigenmodes(
        Kx,
        Ky,
    )

    R0 = layer_eigenbasis_matrix(
        W0,
        V0,
    )

    inverse_epsilon_operator = crystal.layers[
        SOURCE_AFTER
    ].IC

    if inverse_epsilon_operator is None:
        raise RuntimeError(
            "The source-layer inverse epsilon operator IC is missing. "
            "Check fields_mask and the Jz implementation."
        )

    Px = source_power_from_solved_q(
        crystal=crystal,
        kp=kp,
        current=(1.0, 0.0, 0.0),
        R0=R0,
        Sl=Sl,
        Sr=Sr,
        Kx=Kx,
        Ky=Ky,
        inverse_epsilon_operator=inverse_epsilon_operator,
    )

    Py = source_power_from_solved_q(
        crystal=crystal,
        kp=kp,
        current=(0.0, 1.0, 0.0),
        R0=R0,
        Sl=Sl,
        Sr=Sr,
        Kx=Kx,
        Ky=Ky,
        inverse_epsilon_operator=inverse_epsilon_operator,
    )

    Pz = source_power_from_solved_q(
        crystal=crystal,
        kp=kp,
        current=(0.0, 0.0, 1.0),
        R0=R0,
        Sl=Sl,
        Sr=Sr,
        Kx=Kx,
        Ky=Ky,
        inverse_epsilon_operator=inverse_epsilon_operator,
    )

    return (
        Px,
        Py,
        Pz,
    )


# ============================================================
# NORMALIZATION
# ============================================================

def reference_power(current):
    return float(
        homogeneous_reference_power(
            wavelength=WAVELENGTH,
            refractive_index=N_SI,
            current=current,
        )
    )


PREF_X = reference_power(
    (1.0, 0.0, 0.0)
)

PREF_Y = reference_power(
    (0.0, 1.0, 0.0)
)

PREF_Z = reference_power(
    (0.0, 0.0, 1.0)
)


def normalized_density(
    power,
    pref,
):
    return (
        K0**2
        /
        (2.0 * np.pi)**2
        *
        power
        /
        pref
    )


# ============================================================
# OPTIONAL PW CONVERGENCE CHECK
# ============================================================

def convergence_q_points(reference_crystal):
    """
    A few non-boundary points inside the FBZ.

    These are diagnostics, not formal convergence guarantees.
    """

    g6 = reciprocal_nearest_vectors(
        reference_crystal
    )

    vertices = first_bz_vertices(
        g6
    )

    # Pick one K vertex and an M-edge center.
    K = vertices[
        np.argmax(
            vertices[:, 0]
            +
            0.3 * vertices[:, 1]
        )
    ]

    # M on the +x Bragg plane.
    bsum = (
        reference_crystal.expansion.reciprocal[0]
        +
        reference_crystal.expansion.reciprocal[1]
    )

    M = 0.5 * np.asarray(
        bsum,
        dtype=float,
    )

    b1 = np.asarray(
        reference_crystal.expansion.reciprocal[0],
        dtype=float,
    )

    b2 = np.asarray(
        reference_crystal.expansion.reciprocal[1],
        dtype=float,
    )

    generic = (
        0.13 * b1
        +
        0.09 * b2
    )

    return {
        "0.37 Gamma-K": 0.37 * K,
        "0.43 Gamma-M": 0.43 * M,
        "generic": generic,
    }


def run_pw_convergence_check():
    print()
    print("=" * 78)
    print("PLANE-WAVE CONVERGENCE SPOT CHECK")
    print("=" * 78)

    reference_crystal = build_weak_crystal(
        pw=(3, 3)
    )

    qpoints = convergence_q_points(
        reference_crystal
    )

    all_results = {}

    for pw in PW_CONVERGENCE_LIST:

        print()
        print(
            f"PW = {pw}  ->  {pw[0]*pw[1]} harmonics"
        )

        crystal = build_weak_crystal(
            pw=pw
        )

        results_pw = {}

        for name, q in qpoints.items():

            Px, Py, Pz = solve_q_all_orientations(
                crystal,
                q[0],
                q[1],
            )

            rho_h = 0.5 * (
                normalized_density(
                    Px,
                    PREF_X,
                )
                +
                normalized_density(
                    Py,
                    PREF_Y,
                )
            )

            rho_z = normalized_density(
                Pz,
                PREF_Z,
            )

            results_pw[name] = (
                rho_z,
                rho_h,
            )

            print(
                f"  {name:16s}  "
                f"rho_z={rho_z: .8e}  "
                f"rho_h={rho_h: .8e}"
            )

        all_results[pw] = results_pw

    print()
    print("Relative change 81 -> 121 harmonics:")

    r81 = all_results[
        (9, 9)
    ]

    r121 = all_results[
        (11, 11)
    ]

    for name in qpoints:

        z81, h81 = r81[name]
        z121, h121 = r121[name]

        dz = abs(
            z121 - z81
        ) / max(
            abs(z121),
            1e-30,
        )

        dh = abs(
            h121 - h81
        ) / max(
            abs(h121),
            1e-30,
        )

        print(
            f"  {name:16s}  "
            f"z={100*dz:.4f}%  "
            f"h={100*dh:.4f}%"
        )

    print(
        "\nDo not expect tiny relative changes if one of these diagnostic "
        "points happens to fall on a narrow resonance. The full-map "
        "comparison is the final convergence test."
    )


# ============================================================
# FULL FBZ MAP
# ============================================================

def calculate_fbz_maps(
    crystal,
    axis,
    SX,
    SY,
    mask,
):
    rho_z = np.full(
        SX.shape,
        np.nan,
        dtype=float,
    )

    rho_h = np.full(
        SX.shape,
        np.nan,
        dtype=float,
    )

    rho_x = np.full(
        SX.shape,
        np.nan,
        dtype=float,
    )

    rho_y = np.full(
        SX.shape,
        np.nan,
        dtype=float,
    )

    inside_indices = np.argwhere(
        mask
    )

    n_inside = len(
        inside_indices
    )

    print()
    print(
        f"FBZ points to solve = {n_inside} "
        f"out of {NQ*NQ} rectangular-grid points"
    )

    t0 = perf_counter()

    for counter, (iy, ix) in enumerate(
        inside_indices,
        start=1,
    ):

        qx = K0 * SX[
            iy,
            ix,
        ]

        qy = K0 * SY[
            iy,
            ix,
        ]

        Px, Py, Pz = solve_q_all_orientations(
            crystal,
            qx,
            qy,
        )

        rx = normalized_density(
            Px,
            PREF_X,
        )

        ry = normalized_density(
            Py,
            PREF_Y,
        )

        rz = normalized_density(
            Pz,
            PREF_Z,
        )

        rh = 0.5 * (
            rx
            +
            ry
        )

        rho_x[
            iy,
            ix,
        ] = rx

        rho_y[
            iy,
            ix,
        ] = ry

        rho_z[
            iy,
            ix,
        ] = rz

        rho_h[
            iy,
            ix,
        ] = rh

        if (
            counter % 50 == 0
            or
            counter == n_inside
        ):
            elapsed = perf_counter() - t0
            rate = counter / max(
                elapsed,
                1e-12,
            )

            remaining = (
                n_inside - counter
            ) / max(
                rate,
                1e-12,
            )

            print(
                f"{counter:5d}/{n_inside:5d}  "
                f"elapsed={elapsed/60.0:7.2f} min  "
                f"ETA={remaining/60.0:7.2f} min"
            )

    return (
        rho_x,
        rho_y,
        rho_z,
        rho_h,
    )


# ============================================================
# SYMMETRY DIAGNOSTIC
# ============================================================

def c6_symmetry_spot_check(crystal):
    """
    For point A and the horizontal orientation average, rho(q) should
    be C6-symmetric up to numerical/truncation errors.
    """

    radius_s = 0.75

    zvals = []
    hvals = []

    print()
    print("=" * 78)
    print("C6 SYMMETRY SPOT CHECK")
    print("=" * 78)

    for m in range(6):

        theta = m * np.pi / 3.0

        qx = (
            K0
            *
            radius_s
            *
            np.cos(theta)
        )

        qy = (
            K0
            *
            radius_s
            *
            np.sin(theta)
        )

        Px, Py, Pz = solve_q_all_orientations(
            crystal,
            qx,
            qy,
        )

        rz = normalized_density(
            Pz,
            PREF_Z,
        )

        rh = 0.5 * (
            normalized_density(
                Px,
                PREF_X,
            )
            +
            normalized_density(
                Py,
                PREF_Y,
            )
        )

        zvals.append(
            rz
        )

        hvals.append(
            rh
        )

        print(
            f"theta={60*m:3d} deg  "
            f"rho_z={rz:.8e}  "
            f"rho_h={rh:.8e}"
        )

    zvals = np.asarray(
        zvals
    )

    hvals = np.asarray(
        hvals
    )

    zspread = (
        np.max(zvals)
        -
        np.min(zvals)
    ) / max(
        np.mean(np.abs(zvals)),
        1e-30,
    )

    hspread = (
        np.max(hvals)
        -
        np.min(hvals)
    ) / max(
        np.mean(np.abs(hvals)),
        1e-30,
    )

    print()
    print(
        f"C6 relative spread, z          = {100*zspread:.6f}%"
    )

    print(
        f"C6 relative spread, horizontal = {100*hspread:.6f}%"
    )


# ============================================================
# INTEGRATION DIAGNOSTIC
# ============================================================

def coarse_fbz_integral(
    rho,
    mask,
    ds,
):
    """
    Midpoint estimate of integral rho dsx dsy over the FBZ.

    Narrow resonances may require much finer sampling, so this value
    is explicitly reported only as a coarse-grid estimate.
    """

    return float(
        np.nansum(
            np.where(
                mask,
                rho,
                np.nan,
            )
        )
        *
        ds
        *
        ds
    )


# ============================================================
# PLOTTING
# ============================================================

def log_limits(data):
    finite_positive = np.asarray(data)[
        np.isfinite(data)
        &
        (np.asarray(data) > 0.0)
    ]

    if finite_positive.size == 0:
        raise RuntimeError(
            "No positive values are available for log plotting."
        )

    vmax = float(
        np.max(
            finite_positive
        )
    )

    vmin = max(
        float(
            np.min(
                finite_positive
            )
        ),
        vmax * 10.0**(-LOG_DECADES),
    )

    return (
        vmin,
        vmax,
    )


def draw_bz_boundary(
    ax,
    vertices_s,
):
    closed = np.vstack(
        [
            vertices_s,
            vertices_s[0],
        ]
    )

    # Black underlay + white top line remains visible on both dark and
    # bright parts of the hot map.
    ax.plot(
        closed[:, 0],
        closed[:, 1],
        linewidth=2.6,
        color="black",
        zorder=5,
    )

    ax.plot(
        closed[:, 0],
        closed[:, 1],
        linewidth=1.2,
        color="white",
        zorder=6,
    )


def plot_maps(
    axis,
    rho_z,
    rho_h,
    vertices_s,
):
    ds = axis[1] - axis[0]

    extent = [
        axis[0] - 0.5 * ds,
        axis[-1] + 0.5 * ds,
        axis[0] - 0.5 * ds,
        axis[-1] + 0.5 * ds,
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 5.0),
        constrained_layout=True,
    )

    panels = [
        (
            axes[0],
            rho_z,
            "(c) weak PhC — vertical dipole",
        ),
        (
            axes[1],
            rho_h,
            "(g) weak PhC — horizontal average",
        ),
    ]

    for ax, data, title in panels:

        vmin, vmax = log_limits(
            data
        )

        masked = np.ma.masked_invalid(
            data
        )

        masked = np.ma.masked_less_equal(
            masked,
            0.0,
        )

        im = ax.imshow(
            masked,
            origin="lower",
            extent=extent,
            aspect="equal",
            interpolation="nearest",
            cmap="hot",
            norm=LogNorm(
                vmin=vmin,
                vmax=vmax,
            ),
        )

        draw_bz_boundary(
            ax,
            vertices_s,
        )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            r"$q_x/k_0$"
        )

        ax.set_ylabel(
            r"$q_y/k_0$"
        )

        ax.set_aspect(
            "equal"
        )

        cbar = fig.colorbar(
            im,
            ax=ax,
            shrink=0.88,
            format=LogFormatterMathtext(),
        )

        cbar.set_label(
            r"$\rho_F(q_x/k_0,q_y/k_0)$"
        )

        print(
            f"{title}: "
            f"log range [{vmin:.6e}, {vmax:.6e}]"
        )

    fig.suptitle(
        "Dyakov et al. Fig. 2(c,g) weak triangular lattice — 850 meV"
    )

    return fig


# ============================================================
# GEOMETRY DIAGNOSTICS
# ============================================================

def print_geometry_diagnostics(
    crystal,
    vertices_s,
):
    sigma = abs(
        np.linalg.det(
            LATTICE
        )
    )

    fill_fraction = (
        np.pi * R_OVER_A**2
        /
        sigma
    )

    print("=" * 78)
    print("FIG. 2(c,g): WEAK TRIANGULAR PHOTONIC CRYSTAL")
    print("=" * 78)

    print(
        f"energy                      = {ENERGY_MEV:.3f} meV"
    )

    print(
        f"wavelength                  = {WAVELENGTH_NM:.6f} nm"
    )

    print(
        f"wavelength/a                = {WAVELENGTH:.12f}"
    )

    print(
        f"k0                          = {K0:.12f}  [1/a]"
    )

    print()
    print(
        f"a                           = {A_NM:.3f} nm"
    )

    print(
        f"h                           = {H_NM:.3f} nm"
    )

    print(
        f"r/a                         = {R_OVER_A:.6f}"
    )

    print(
        f"z0 from substrate           = {Z0_FROM_SUBSTRATE_NM:.3f} nm"
    )

    print(
        f"air-side patterned depth    = "
        f"{H_NM-Z0_FROM_SUBSTRATE_NM:.3f} nm"
    )

    print(
        f"substrate-side depth         = "
        f"{Z0_FROM_SUBSTRATE_NM:.3f} nm"
    )

    print()
    print(
        f"eps_Si                      = {EPS_SI}"
    )

    print(
        f"eps_pore weak               = {EPS_PORE_WEAK}"
    )

    print(
        f"eps_pore / eps_Si           = {EPS_PORE_WEAK/EPS_SI}"
    )

    print(
        f"eps_SiO2                    = {EPS_SIO2}"
    )

    print()
    print(
        f"a1                          = {A1}"
    )

    print(
        f"a2                          = {A2}"
    )

    print(
        f"unit-cell area/a^2          = {sigma:.12f}"
    )

    print(
        f"pore area fraction          = {fill_fraction:.12f}"
    )

    print(
        f"point A / a                 = {POINT_A}"
    )

    d0 = np.linalg.norm(
        POINT_A
    )

    d1 = np.linalg.norm(
        POINT_A - A1
    )

    d2 = np.linalg.norm(
        POINT_A - A2
    )

    print(
        "distance A to three nearest pore centers = "
        f"{d0:.12f}, {d1:.12f}, {d2:.12f}"
    )

    print()
    print(
        f"PW                          = {PW}"
    )

    print(
        f"number of harmonics         = {PW[0]*PW[1]}"
    )

    print(
        f"paper harmonic count        = 85"
    )

    print(
        f"NQ rectangular grid         = {NQ}"
    )

    print()
    print("FBZ vertices in units q/k0:")

    for v in vertices_s:
        print(
            f"  ({v[0]:+.8f}, {v[1]:+.8f})"
        )

    print()
    print(
        f"bulk-Si reference P         = {PREF_Z:.12e}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if RUN_PW_CONVERGENCE_CHECK:
        run_pw_convergence_check()

    crystal = build_weak_crystal(
        pw=PW
    )

    (
        axis,
        ds,
        SX,
        SY,
        mask,
        vertices_s,
        g6,
    ) = build_q_grid(
        crystal
    )

    print()
    print_geometry_diagnostics(
        crystal,
        vertices_s,
    )

    # Independent symmetry test before the full map.
    c6_symmetry_spot_check(
        crystal
    )

    print()
    print("=" * 78)
    print("FULL FIRST-BRILLOUIN-ZONE MAP")
    print("=" * 78)

    (
        rho_x,
        rho_y,
        rho_z,
        rho_h,
    ) = calculate_fbz_maps(
        crystal=crystal,
        axis=axis,
        SX=SX,
        SY=SY,
        mask=mask,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    Fz_coarse = coarse_fbz_integral(
        rho_z,
        mask,
        ds,
    )

    Fh_coarse = coarse_fbz_integral(
        rho_h,
        mask,
        ds,
    )

    print()
    print("=" * 78)
    print("MAP DIAGNOSTICS")
    print("=" * 78)

    print(
        f"min rho_z inside FBZ        = "
        f"{np.nanmin(rho_z):.12e}"
    )

    print(
        f"max rho_z inside FBZ        = "
        f"{np.nanmax(rho_z):.12e}"
    )

    print(
        f"min rho_h inside FBZ        = "
        f"{np.nanmin(rho_h):.12e}"
    )

    print(
        f"max rho_h inside FBZ        = "
        f"{np.nanmax(rho_h):.12e}"
    )

    print()
    print(
        f"coarse-grid Fp(z) estimate  = {Fz_coarse:.10f}"
    )

    print(
        f"coarse-grid Fp(h) estimate  = {Fh_coarse:.10f}"
    )

    print(
        "These are midpoint-grid estimates. Narrow resonances may require "
        "a finer q grid for converged integrated Purcell factors."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    npz_path = (
        OUTPUT_DIR
        /
        "fig2cg_weak_phc_850meV.npz"
    )

    np.savez_compressed(
        npz_path,
        energy_meV=ENERGY_MEV,
        wavelength_nm=WAVELENGTH_NM,
        wavelength_over_a=WAVELENGTH,
        k0=K0,
        a_nm=A_NM,
        h_nm=H_NM,
        r_over_a=R_OVER_A,
        z0_from_substrate_nm=Z0_FROM_SUBSTRATE_NM,
        n_si=N_SI,
        n_sio2=N_SIO2,
        eps_si=EPS_SI,
        eps_pore_weak=EPS_PORE_WEAK,
        eps_sio2=EPS_SIO2,
        lattice=LATTICE,
        point_A=POINT_A,
        pw=np.array(PW),
        axis_q_over_k0=axis,
        ds=ds,
        SX=SX,
        SY=SY,
        fbz_mask=mask,
        fbz_vertices_q_over_k0=vertices_s,
        rho_x=rho_x,
        rho_y=rho_y,
        rho_z=rho_z,
        rho_horizontal=rho_h,
        Fp_z_coarse=Fz_coarse,
        Fp_horizontal_coarse=Fh_coarse,
    )

    fig = plot_maps(
        axis=axis,
        rho_z=rho_z,
        rho_h=rho_h,
        vertices_s=vertices_s,
    )

    png_path = (
        OUTPUT_DIR
        /
        "fig2cg_weak_phc_log_hot.png"
    )

    pdf_path = (
        OUTPUT_DIR
        /
        "fig2cg_weak_phc_log_hot.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    print()
    print("=" * 78)
    print("OUTPUT")
    print("=" * 78)

    print(
        f"NPZ : {npz_path}"
    )

    print(
        f"PNG : {png_path}"
    )

    print(
        f"PDF : {pdf_path}"
    )

    plt.show()


if __name__ == "__main__":
    main()
