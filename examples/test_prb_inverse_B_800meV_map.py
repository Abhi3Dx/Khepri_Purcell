"""
PRB 108, 155416 - inverse triangular photonic crystal
800 meV, B-point, x-oriented dipole: first patterned-PhC benchmark.

Purpose
-------
This script does NOT yet calculate the final Purcell factor.

Its job is to reproduce the qualitative near-field-emissivity map of
Dyakov et al. Fig. 3(a) for the inverse photonic-crystal slab:

    air
    |
    Si slab with triangular lattice of air holes
    |  x-oriented dipole at real-space B point, z=h/2
    Si slab with triangular lattice of air holes
    |
    weakly lossy SiO2 substrate

The script:
  1. builds the paper geometry,
  2. uses the real-space B point inferred from Fig. 1 high-symmetry geometry,
  3. checks Gamma, M and K,
  4. samples P_total(qx,qy) inside the true hexagonal first Brillouin zone,
  5. saves the raw map and a normalized log-scale plot.

IMPORTANT
---------
The paper states that it uses 85 Fourier harmonics. Khepri's present
rectangular truncation does not reproduce that exact harmonic set.
The default below uses pw=(7,7) for a reasonably fast first map.
Set FAST_MODE=False to use pw=(9,9)=81 retained rectangular harmonics,
which is numerically much closer to the paper's 85-harmonic calculation.

This is a diagnostic map. Do not integrate this coarse map to claim a
converged Purcell factor; the planar benchmark already showed that narrow
q-space resonances require resonance-aware integration.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from khepri.crystal import Crystal
from khepri.draw import Drawing
from khepri.dipole import dipole_power_q


# ============================================================================
# 1. USER SWITCH
# ============================================================================

FAST_MODE = True

if FAST_MODE:
    pw = (7, 7)
    Nq = 31
    pixmap_resolution = (384, 384)
else:
    # 81 rectangular harmonics, close to the paper's stated 85 harmonics.
    pw = (9, 9)
    Nq = 51
    pixmap_resolution = (512, 512)

backend = "dense"


# ============================================================================
# 2. PAPER PARAMETERS
# ============================================================================

a_nm = 445.0
h_nm = 230.0
r_over_a = 0.405

energy_eV = 0.800
hc_eV_nm = 1239.841984
wavelength_nm = hc_eV_nm / energy_eV

# Normalize lengths to lattice period a=1.
a = 1.0
h = h_nm / a_nm
half_h = 0.5 * h
r = r_over_a
wavelength = wavelength_nm / a_nm

# Approximate indices around 1550 nm.
# These are the same starting values used in the validated planar benchmark.
n_si = 3.48
n_sio2 = 1.45

eps_air = 1.0
eps_si = n_si**2
eps_sio2 = n_sio2**2

# The paper adds +0.001i in its harmonic convention for total Purcell-factor
# integration. In the corrected Khepri convention, the same passive loss is:
eps_substrate = eps_sio2 - 0.001j

current = (1.0, 0.0, 0.0)  # x dipole


# ============================================================================
# 3. TRIANGULAR REAL-SPACE LATTICE AND A/B/C POSITIONS
# ============================================================================

# This exactly matches Crystal(lattice="hexagonal") in Khepri:
#
#     a1 = (sqrt(3)/2, +1/2) a
#     a2 = (sqrt(3)/2, -1/2) a
#
# Hole centers are at integer combinations of a1 and a2.

a1 = np.array(
    [
        np.sqrt(3.0) / 2.0,
        +0.5,
    ],
    dtype=float,
)

a2 = np.array(
    [
        np.sqrt(3.0) / 2.0,
        -0.5,
    ],
    dtype=float,
)

# The main paper labels A, B and C graphically rather than printing numerical
# coordinates. With the Khepri lattice orientation above, Fig. 1 is naturally
# reproduced by the standard high-symmetry positions of the triangular void
# between the three pore centers 0, a1 and a2:
#
#   A = triangle centroid
#   B = midpoint of the edge connecting a1 and a2
#   C = midpoint of the upper edge connecting 0 and a1
#
# This puts A and B on the horizontal x-axis and C above A, matching Fig. 1(c).
#
# Fractional coordinates in (a1,a2):
#   A = (1/3, 1/3)
#   B = (1/2, 1/2)
#   C = (1/2, 0)

rho_A = (a1 + a2) / 3.0
rho_B = (a1 + a2) / 2.0
rho_C = a1 / 2.0

position = (
    float(rho_B[0]),
    float(rho_B[1]),
    half_h,
)


# ============================================================================
# 4. BASIC REAL-SPACE GEOMETRY CHECKS
# ============================================================================

def nearest_pore_distance(rho):
    """Distance from rho to nearby triangular-lattice pore centers."""
    centers = []

    for m in range(-2, 3):
        for n in range(-2, 3):
            centers.append(
                m * a1
                +
                n * a2
            )

    return min(
        np.linalg.norm(
            rho - center
        )
        for center in centers
    )


for label, rho in (
    ("A", rho_A),
    ("B", rho_B),
    ("C", rho_C),
):
    distance = nearest_pore_distance(rho)

    if distance <= r:
        raise RuntimeError(
            f"Point {label} lies inside an air pore: "
            f"nearest distance={distance:.12f}, r={r:.12f}"
        )


# ============================================================================
# 5. BUILD THE INVERSE PHOTONIC CRYSTAL
# ============================================================================

crystal = Crystal(
    pw=pw,
    lattice="hexagonal",
    lattice_pitch=a,
    epsi=eps_air,          # left / incidence / air
    epse=eps_substrate,    # right / emergence / silica
)

drawing = Drawing(
    pixmap_resolution,
    eps_si,
    crystal.lattice,
)

# One air hole per triangular primitive cell.
drawing.disc(
    (0.0, 0.0),
    r,
    eps_air,
)

epsilon_pattern = drawing.canvas()

# Split the patterned slab exactly at the source plane.
crystal.add_layer_pixmap(
    "phc_left",
    epsilon_pattern,
    half_h,
)

crystal.add_layer_pixmap(
    "phc_right",
    epsilon_pattern,
    half_h,
)

crystal.set_device(
    [
        "phc_left",
        "phc_right",
    ],
    fields_mask=[
        True,
        False,
    ],
)


# ============================================================================
# 6. RECIPROCAL LATTICE / TRUE HEXAGONAL FIRST BRILLOUIN ZONE
# ============================================================================

def reciprocal_vectors(a1, a2):
    det = (
        a1[0] * a2[1]
        -
        a1[1] * a2[0]
    )

    b1 = (
        2.0
        *
        np.pi
        /
        det
        *
        np.array(
            [
                a2[1],
                -a2[0],
            ]
        )
    )

    b2 = (
        2.0
        *
        np.pi
        /
        det
        *
        np.array(
            [
                -a1[1],
                a1[0],
            ]
        )
    )

    return b1, b2


b1, b2 = reciprocal_vectors(
    a1,
    a2,
)

# Six nearest reciprocal-lattice vectors.
nearest_G = np.array(
    [
        +b1,
        -b1,
        +b2,
        -b2,
        +(b1 + b2),
        -(b1 + b2),
    ],
    dtype=float,
)


def inside_first_bz(qx, qy, tol=1e-12):
    """
    Wigner-Seitz condition:
        |q| <= |q-G|
    for each nearest reciprocal-lattice vector G.

    Equivalent to:
        q.G <= |G|^2/2.
    """
    q = np.array(
        [
            qx,
            qy,
        ],
        dtype=float,
    )

    for G in nearest_G:
        if (
            np.dot(q, G)
            >
            0.5
            *
            np.dot(G, G)
            +
            tol
        ):
            return False

    return True


# With Khepri's orientation, convenient high-symmetry representatives are:
Gamma = np.array([0.0, 0.0])

# M: center of the right vertical edge.
M = 0.5 * (b1 + b2)

# K: upper-right hexagon corner.
K = np.array(
    [
        2.0 * np.pi / np.sqrt(3.0),
        2.0 * np.pi / 3.0,
    ],
    dtype=float,
)

# Regular-hexagon vertices for plotting.
bz_vertices = np.array(
    [
        [0.0, +4.0 * np.pi / 3.0],
        [+2.0 * np.pi / np.sqrt(3.0), +2.0 * np.pi / 3.0],
        [+2.0 * np.pi / np.sqrt(3.0), -2.0 * np.pi / 3.0],
        [0.0, -4.0 * np.pi / 3.0],
        [-2.0 * np.pi / np.sqrt(3.0), -2.0 * np.pi / 3.0],
        [-2.0 * np.pi / np.sqrt(3.0), +2.0 * np.pi / 3.0],
    ],
    dtype=float,
)

bz_vertices_closed = np.vstack(
    [
        bz_vertices,
        bz_vertices[0],
    ]
)


# ============================================================================
# 7. PRINT SETUP
# ============================================================================

print("=" * 88)
print("PRB INVERSE TRIANGULAR PhC - 800 meV - B POINT - x DIPOLE")
print("=" * 88)

print("\nPHYSICAL PARAMETERS")
print(f"a                                  = {a_nm:.6f} nm")
print(f"h                                  = {h_nm:.6f} nm")
print(f"r                                  = {r_over_a * a_nm:.6f} nm")
print(f"energy                             = {energy_eV * 1000.0:.3f} meV")
print(f"vacuum wavelength                  = {wavelength_nm:.9f} nm")

print("\nNORMALIZED PARAMETERS")
print(f"h/a                                = {h:.12f}")
print(f"r/a                                = {r:.12f}")
print(f"lambda/a                           = {wavelength:.12f}")

print("\nMATERIALS")
print(f"eps_air                            = {eps_air}")
print(f"eps_Si                             = {eps_si}")
print(f"eps_substrate                      = {eps_substrate}")

print("\nNUMERICS")
print(f"FAST_MODE                          = {FAST_MODE}")
print(f"pw                                 = {pw}")
print(f"number of retained harmonics       = {pw[0] * pw[1]}")
print(f"q bounding grid                    = {Nq} x {Nq}")
print(f"pixmap resolution                  = {pixmap_resolution}")
print(f"backend                            = {backend}")

print("\nREAL-SPACE HIGH-SYMMETRY POSITIONS")
print(f"A/a                                = {rho_A}")
print(f"B/a                                = {rho_B}")
print(f"C/a                                = {rho_C}")
print(f"selected dipole                    = B, x-oriented")
print(f"dipole position                    = {position}")
print(f"nearest pore distance at B         = {nearest_pore_distance(rho_B):.12f}")
print(f"air-hole radius                    = {r:.12f}")

print("\nRECIPROCAL LATTICE")
print(f"b1                                 = {b1}")
print(f"b2                                 = {b2}")
print(f"Gamma                              = {Gamma}")
print(f"M                                  = {M}")
print(f"K                                  = {K}")


# ============================================================================
# 8. SINGLE-q SANITY CHECKS
# ============================================================================

print("\n" + "=" * 88)
print("STEP 1 - GAMMA / M / K SANITY CHECK")
print("=" * 88)

for name, q in (
    ("Gamma", Gamma),
    ("M", M),
    ("K", K),
):
    result = dipole_power_q(
        crystal=crystal,
        source_after="phc_left",
        wavelength=wavelength,
        kp=(
            float(q[0]),
            float(q[1]),
        ),
        position=position,
        current=current,
        backend=backend,
        return_details=True,
    )

    print(
        f"{name:5s}: "
        f"P_left={result['P_left']:+.8e}  "
        f"P_right={result['P_right']:+.8e}  "
        f"P_total={result['P_total']:+.8e}  "
        f"jump_err={result['jump_relative_error']:.3e}"
    )

    if not np.isfinite(
        result["P_total"]
    ):
        raise RuntimeError(
            f"{name} power is not finite."
        )

    if (
        result["jump_relative_error"]
        >
        1e-8
    ):
        raise RuntimeError(
            f"{name} source jump failed."
        )

    if (
        result["P_total"]
        <
        -1e-8
    ):
        raise RuntimeError(
            f"{name} has significant negative emitted power."
        )


# ============================================================================
# 9. HEXAGONAL FIRST-BZ MAP
# ============================================================================

print("\n" + "=" * 88)
print("STEP 2 - FIRST-BRILLOUIN-ZONE NEAR-FIELD POWER MAP")
print("=" * 88)

qx_max = (
    2.0
    *
    np.pi
    /
    np.sqrt(3.0)
)

qy_max = (
    4.0
    *
    np.pi
    /
    3.0
)

qx_values = np.linspace(
    -qx_max,
    +qx_max,
    Nq,
)

qy_values = np.linspace(
    -qy_max,
    +qy_max,
    Nq,
)

QX, QY = np.meshgrid(
    qx_values,
    qy_values,
    indexing="xy",
)

P = np.full(
    QX.shape,
    np.nan,
    dtype=float,
)

mask = np.zeros(
    QX.shape,
    dtype=bool,
)

points = []

for iy in range(Nq):
    for ix in range(Nq):
        qx = float(
            QX[iy, ix]
        )

        qy = float(
            QY[iy, ix]
        )

        if not inside_first_bz(
            qx,
            qy,
        ):
            continue

        mask[iy, ix] = True

        points.append(
            (
                iy,
                ix,
                qx,
                qy,
            )
        )

print(
    f"Evaluating {len(points)} q points "
    f"inside the hexagonal FBZ..."
)

for ipoint, (
    iy,
    ix,
    qx,
    qy,
) in enumerate(
    points,
    start=1,
):
    result = dipole_power_q(
        crystal=crystal,
        source_after="phc_left",
        wavelength=wavelength,
        kp=(
            qx,
            qy,
        ),
        position=position,
        current=current,
        backend=backend,
        return_details=False,
    )

    P[iy, ix] = result[
        "P_total"
    ]

    if (
        ipoint == 1
        or ipoint == len(points)
        or ipoint % max(
            1,
            len(points) // 20,
        ) == 0
    ):
        print(
            f"  {ipoint:5d}/{len(points):5d}  "
            f"q=({qx:+.5f},{qy:+.5f})  "
            f"P={P[iy, ix]:.6e}"
        )


valid_P = P[
    mask
]

if not np.all(
    np.isfinite(
        valid_P
    )
):
    raise RuntimeError(
        "FBZ map contains NaN or infinity."
    )

print("\nMAP DIAGNOSTICS")
print(f"min(P_total)                       = {np.min(valid_P):.12e}")
print(f"max(P_total)                       = {np.max(valid_P):.12e}")
print(f"mean(P_total)                      = {np.mean(valid_P):.12e}")
print(f"all sampled powers nonnegative     = {np.min(valid_P) >= -1e-8}")


# ============================================================================
# 10. REPORT STRONGEST SAMPLED q POINTS
# ============================================================================

flat_indices = np.argwhere(
    mask
)

values = np.array(
    [
        P[iy, ix]
        for iy, ix in flat_indices
    ],
    dtype=float,
)

order = np.argsort(
    values
)[::-1]

print("\nSTRONGEST SAMPLED q POINTS")

M_points = 0.5 * nearest_G

for rank in range(
    min(
        10,
        len(order),
    )
):
    iy, ix = flat_indices[
        order[rank]
    ]

    q = np.array(
        [
            QX[iy, ix],
            QY[iy, ix],
        ]
    )

    value = P[
        iy,
        ix,
    ]

    dM = min(
        np.linalg.norm(
            q - m
        )
        for m in M_points
    )

    print(
        f"{rank + 1:2d}: "
        f"q=({q[0]:+.8f},{q[1]:+.8f})  "
        f"P={value:.8e}  "
        f"distance to nearest M={dM:.4e}  "
        f"|q|={np.linalg.norm(q):.6f}"
    )


# ============================================================================
# 11. SAVE RAW DATA
# ============================================================================

output_npz = (
    "prb_inverse_B_x_800meV_fbz_map.npz"
)

np.savez(
    output_npz,
    P_total=P,
    mask=mask,
    qx=QX,
    qy=QY,
    qx_values=qx_values,
    qy_values=qy_values,
    a1=a1,
    a2=a2,
    b1=b1,
    b2=b2,
    bz_vertices=bz_vertices,
    rho_A=rho_A,
    rho_B=rho_B,
    rho_C=rho_C,
    position=np.asarray(
        position
    ),
    current=np.asarray(
        current
    ),
    wavelength=wavelength,
    wavelength_nm=wavelength_nm,
    energy_eV=energy_eV,
    pw=np.asarray(
        pw
    ),
)

print(
    f"\nSaved raw map: "
    f"{os.path.abspath(output_npz)}"
)


# ============================================================================
# 12. PLOT
# ============================================================================

# For shape comparison, normalize to the maximum and plot log10.
# The floor is visualization-only and never modifies raw data.

positive = valid_P[
    valid_P > 0
]

if len(positive) == 0:
    raise RuntimeError(
        "No positive q-resolved power values were found."
    )

Pmax = float(
    np.max(
        positive
    )
)

visual_floor = (
    Pmax
    *
    1e-6
)

P_visual = np.full_like(
    P,
    np.nan,
    dtype=float,
)

P_visual[
    mask
] = np.log10(
    np.maximum(
        P[mask],
        visual_floor,
    )
    /
    Pmax
)

# Plot q in units of 2*pi/a.
scale_q = (
    2.0
    *
    np.pi
)

fig, ax = plt.subplots(
    figsize=(
        7.0,
        6.5,
    )
)

pcm = ax.pcolormesh(
    QX / scale_q,
    QY / scale_q,
    P_visual,
    shading="auto",
)

cbar = fig.colorbar(
    pcm,
    ax=ax,
)

cbar.set_label(
    r"$\log_{10}[P(\mathbf{q})/P_{\max}]$"
)

ax.plot(
    bz_vertices_closed[:, 0] / scale_q,
    bz_vertices_closed[:, 1] / scale_q,
    linewidth=1.5,
)

ax.scatter(
    [Gamma[0] / scale_q],
    [Gamma[1] / scale_q],
    s=30,
)

ax.text(
    Gamma[0] / scale_q,
    Gamma[1] / scale_q,
    r"$\Gamma$",
    ha="left",
    va="bottom",
)

ax.scatter(
    [M[0] / scale_q],
    [M[1] / scale_q],
    s=30,
)

ax.text(
    M[0] / scale_q,
    M[1] / scale_q,
    "M",
    ha="left",
    va="bottom",
)

ax.scatter(
    [K[0] / scale_q],
    [K[1] / scale_q],
    s=30,
)

ax.text(
    K[0] / scale_q,
    K[1] / scale_q,
    "K",
    ha="left",
    va="bottom",
)

ax.set_aspect(
    "equal",
    adjustable="box",
)

ax.set_xlabel(
    r"$q_x a/(2\pi)$"
)

ax.set_ylabel(
    r"$q_y a/(2\pi)$"
)

ax.set_title(
    "Inverse triangular PhC, 800 meV\n"
    "B-point x dipole - near-field power"
)

fig.tight_layout()

output_png = (
    "prb_inverse_B_x_800meV_fbz_map.png"
)

fig.savefig(
    output_png,
    dpi=220,
    bbox_inches="tight",
)

plt.close(
    fig
)

print(
    f"Saved figure: "
    f"{os.path.abspath(output_png)}"
)

print("\n" + "=" * 88)
print("DONE")
print("=" * 88)
print(
    "Next: compare the q-space hot spots with Fig. 3(a). "
    "If the map is qualitatively correct, repeat with FAST_MODE=False "
    "and then build the resonance-aware 2D Purcell integration."
)
