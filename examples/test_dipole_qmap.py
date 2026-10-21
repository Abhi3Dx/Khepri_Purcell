import numpy as np
import matplotlib.pyplot as plt

from khepri.crystal import Crystal
from khepri.dipole import dipole_power_map


# ============================================================
# PARAMETERS
# ============================================================

pw = (
    3,
    3,
)

a = 1.0

wavelength = 2.0

epsilon_air = 1.0

epsilon_slab = 2.25

half_thickness = 0.5


# ============================================================
# TOLERANCES
# ============================================================

TOL_GAMMA = 1e-10

TOL_LR = 1e-9

TOL_SYMMETRY = 1e-8

TOL_NEGATIVE_POWER = 1e-10


# ============================================================
# q GRID
# ============================================================
#
# Use midpoint sampling:
#
#   q_i = q_min + (i + 1/2) dq
#
# Advantages:
#
#   1. avoids double-counting equivalent BZ boundaries
#   2. avoids evaluating exactly at +/- pi/a
#   3. with odd Nq, Gamma = (0,0) is included exactly
# ============================================================

Nq = 11

qmin = (
    -np.pi
    /
    a
)

qmax = (
    +np.pi
    /
    a
)


dq = (
    qmax
    -
    qmin
) / Nq


qx = (
    qmin
    +
    (
        np.arange(Nq)
        +
        0.5
    )
    *
    dq
)


qy = qx.copy()


print("\n")
print("=" * 60)
print("q-SPACE GRID")
print("=" * 60)


print(
    "\nNq =",
    Nq,
)

print(
    "dq =",
    dq,
)

print(
    "qx min/max =",
    qx[0],
    qx[-1],
)

print(
    "qy min/max =",
    qy[0],
    qy[-1],
)


# ------------------------------------------------------------
# Gamma point should be present for odd Nq
# ------------------------------------------------------------

center = (
    Nq
    //
    2
)


assert np.isclose(
    qx[center],
    0.0,
)


assert np.isclose(
    qy[center],
    0.0,
)


print(
    "\n[PASS] midpoint grid contains q=0"
)


# ============================================================
# BUILD DIELECTRIC SLAB
# ============================================================

cl = Crystal(
    pw=pw,
    lattice="square",
    lattice_pitch=a,
    epsi=epsilon_air,
    epse=epsilon_air,
)


cl.add_layer_uniform(
    "left_half",
    epsilon=epsilon_slab,
    depth=half_thickness,
)


cl.add_layer_uniform(
    "right_half",
    epsilon=epsilon_slab,
    depth=half_thickness,
)


cl.set_device(
    [
        "left_half",
        "right_half",
    ],
    fields_mask=[
        True,
        False,
    ],
)


# ============================================================
# CALCULATE q MAP
# ============================================================

print("\n")
print("=" * 60)
print("CALCULATING q-SPACE DIPOLE POWER")
print("=" * 60)


result = dipole_power_map(
    crystal=cl,

    source_after="left_half",

    wavelength=wavelength,

    qx_values=qx,

    qy_values=qy,

    position=(
        0.0,
        0.0,
        half_thickness,
    ),

    current=(
        1.0,
        0.0,
        0.0,
    ),

    backend="dense",

    verbose=True,
)


P = result[
    "P_total"
]

P_left = result[
    "P_left"
]

P_right = result[
    "P_right"
]


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n")
print("=" * 60)
print("BASIC CHECKS")
print("=" * 60)


assert P.shape == (
    Nq,
    Nq,
)


assert P_left.shape == (
    Nq,
    Nq,
)


assert P_right.shape == (
    Nq,
    Nq,
)


assert np.all(
    np.isfinite(P)
)


assert np.all(
    np.isfinite(P_left)
)


assert np.all(
    np.isfinite(P_right)
)


print(
    "\n[PASS] power maps have correct shape"
)

print(
    "[PASS] all power values are finite"
)


# ============================================================
# GAMMA-POINT CHECK
# ============================================================

P_gamma = P[
    center,
    center
]


expected_gamma = (
    0.3076923076923077
)


gamma_error = abs(
    P_gamma
    -
    expected_gamma
) / expected_gamma


print("\n")
print("=" * 60)
print("GAMMA-POINT CHECK")
print("=" * 60)


print(
    "\nP(Gamma) =",
    P_gamma,
)

print(
    "Expected =",
    expected_gamma,
)

print(
    "Relative error =",
    gamma_error,
)


assert gamma_error < TOL_GAMMA


print(
    "[PASS] Gamma-point power matches previous test"
)


# ============================================================
# PASSIVITY CHECK
# ============================================================
#
# For this passive lossless structure, P(q) should not be
# substantially negative.
#
# Allow only tiny floating-point negative values.
# ============================================================

print("\n")
print("=" * 60)
print("PASSIVITY CHECK")
print("=" * 60)


P_scale = np.max(
    np.abs(P)
)


minimum_power = np.min(
    P
)


print(
    "\nMaximum |P(q)| =",
    P_scale,
)


print(
    "Minimum P(q) =",
    minimum_power,
)


negative_tolerance = (
    TOL_NEGATIVE_POWER
    *
    max(
        P_scale,
        1.0,
    )
)


assert minimum_power > -negative_tolerance, (
    "\nFAILED:\n"
    "Substantial negative emitted power was found."
)


print(
    "[PASS] q-resolved power is non-negative "
    "within numerical precision"
)


# ============================================================
# LEFT / RIGHT EMISSION SYMMETRY
# ============================================================
#
# The dipole lies at the center of a mirror-symmetric slab.
#
# Therefore:
#
#       P_left(q) = P_right(q)
#
# Here the powers are not near zero across most of the map,
# so a pointwise relative check is fine.
# ============================================================

lr_denominator = np.maximum(
    np.maximum(
        np.abs(P_left),
        np.abs(P_right),
    ),
    1e-30,
)


lr_relative_map = (
    np.abs(
        P_left
        -
        P_right
    )
    /
    lr_denominator
)


lr_error = np.max(
    lr_relative_map
)


lr_worst_index = np.unravel_index(
    np.argmax(
        lr_relative_map
    ),
    P.shape,
)


print("\n")
print("=" * 60)
print("LEFT / RIGHT EMISSION SYMMETRY")
print("=" * 60)


print(
    "\nMaximum left/right relative error =",
    lr_error,
)


print(
    "Worst point:"
)

print(
    "qx =",
    qx[
        lr_worst_index[1]
    ],
)

print(
    "qy =",
    qy[
        lr_worst_index[0]
    ],
)

print(
    "P_left =",
    P_left[
        lr_worst_index
    ],
)

print(
    "P_right =",
    P_right[
        lr_worst_index
    ],
)


assert lr_error < TOL_LR


print(
    "[PASS] left/right emission symmetry over entire BZ"
)


# ============================================================
# RECIPROCITY / INVERSION SYMMETRY
# ============================================================
#
# For this reciprocal homogeneous system:
#
#       P(qx,qy) = P(-qx,-qy)
#
# IMPORTANT:
#
# Do NOT use a pointwise relative error here.
#
# At locations where both powers are ~0, tiny absolute
# floating-point differences can produce enormous relative
# errors.
#
# Instead:
#
#       error =
#       max |P(q)-P(-q)| / max |P(q)|
# ============================================================

P_inverted = np.flip(
    P,
    axis=(
        0,
        1,
    ),
)


inversion_difference = np.abs(
    P
    -
    P_inverted
)


inversion_absolute_error = np.max(
    inversion_difference
)


inversion_error = (
    inversion_absolute_error
    /
    max(
        P_scale,
        1e-30,
    )
)


worst_index = np.unravel_index(
    np.argmax(
        inversion_difference
    ),
    P.shape,
)


iy_worst, ix_worst = worst_index


iy_opposite = (
    Nq
    -
    1
    -
    iy_worst
)


ix_opposite = (
    Nq
    -
    1
    -
    ix_worst
)


print("\n")
print("=" * 60)
print("INVERSION SYMMETRY DIAGNOSTIC")
print("=" * 60)


print(
    "\nMaximum power scale =",
    P_scale,
)


print(
    "Maximum absolute inversion difference =",
    inversion_absolute_error,
)


print(
    "Normalized global inversion error =",
    inversion_error,
)


print(
    "\nWorst point:"
)


print(
    "qx =",
    qx[
        ix_worst
    ],
)

print(
    "qy =",
    qy[
        iy_worst
    ],
)

print(
    "P(q) =",
    P[
        iy_worst,
        ix_worst,
    ],
)


print(
    "\nOpposite point:"
)


print(
    "qx =",
    qx[
        ix_opposite
    ],
)

print(
    "qy =",
    qy[
        iy_opposite
    ],
)

print(
    "P(-q) =",
    P[
        iy_opposite,
        ix_opposite,
    ],
)


print(
    "\nAbsolute difference =",
    inversion_difference[
        iy_worst,
        ix_worst,
    ],
)


assert inversion_error < TOL_SYMMETRY, (
    "\nFAILED:\n"
    "The q-space power map does not satisfy inversion "
    "symmetry within numerical precision."
)


print(
    "\n[PASS] P(q) = P(-q)"
)


# ============================================================
# qy MIRROR SYMMETRY
# ============================================================
#
# For an x-oriented dipole in this homogeneous isotropic slab:
#
#       P(qx,qy) = P(qx,-qy)
#
# Again use a GLOBAL normalization rather than a pointwise
# relative error.
# ============================================================

P_qy_mirror = np.flip(
    P,
    axis=0,
)


qy_difference = np.abs(
    P
    -
    P_qy_mirror
)


qy_absolute_error = np.max(
    qy_difference
)


qy_symmetry_error = (
    qy_absolute_error
    /
    max(
        P_scale,
        1e-30,
    )
)


qy_worst_index = np.unravel_index(
    np.argmax(
        qy_difference
    ),
    P.shape,
)


print("\n")
print("=" * 60)
print("qy MIRROR SYMMETRY")
print("=" * 60)


print(
    "\nMaximum absolute qy-mirror difference =",
    qy_absolute_error,
)


print(
    "Normalized qy-mirror error =",
    qy_symmetry_error,
)


print(
    "\nWorst qy-mirror point:"
)


print(
    "qx =",
    qx[
        qy_worst_index[1]
    ],
)

print(
    "qy =",
    qy[
        qy_worst_index[0]
    ],
)


assert qy_symmetry_error < TOL_SYMMETRY, (
    "\nFAILED:\n"
    "The power map does not satisfy qy mirror symmetry."
)


print(
    "[PASS] x-dipole has qy mirror symmetry"
)


# ============================================================
# OPTIONAL qx MIRROR SYMMETRY
# ============================================================
#
# For the present homogeneous isotropic slab, the x-dipole
# power should also be symmetric under:
#
#       qx -> -qx
#
# We test this independently.
# ============================================================

P_qx_mirror = np.flip(
    P,
    axis=1,
)


qx_difference = np.abs(
    P
    -
    P_qx_mirror
)


qx_absolute_error = np.max(
    qx_difference
)


qx_symmetry_error = (
    qx_absolute_error
    /
    max(
        P_scale,
        1e-30,
    )
)


print("\n")
print("=" * 60)
print("qx MIRROR SYMMETRY")
print("=" * 60)


print(
    "\nMaximum absolute qx-mirror difference =",
    qx_absolute_error,
)


print(
    "Normalized qx-mirror error =",
    qx_symmetry_error,
)


assert qx_symmetry_error < TOL_SYMMETRY, (
    "\nFAILED:\n"
    "The power map does not satisfy qx mirror symmetry."
)


print(
    "[PASS] x-dipole has qx mirror symmetry"
)


# ============================================================
# MAP STATISTICS
# ============================================================

print("\n")
print("=" * 60)
print("MAP STATISTICS")
print("=" * 60)


P_min = np.min(
    P
)


P_max = np.max(
    P
)


min_index = np.unravel_index(
    np.argmin(P),
    P.shape,
)


max_index = np.unravel_index(
    np.argmax(P),
    P.shape,
)


print(
    "\nMinimum P(q) =",
    P_min,
)


print(
    "located at:"
)

print(
    "qx =",
    qx[
        min_index[1]
    ],
)

print(
    "qy =",
    qy[
        min_index[0]
    ],
)


print(
    "\nMaximum P(q) =",
    P_max,
)


print(
    "located at:"
)

print(
    "qx =",
    qx[
        max_index[1]
    ],
)

print(
    "qy =",
    qy[
        max_index[0]
    ],
)


# ============================================================
# SIMPLE FIRST-BZ INTEGRAL
# ============================================================
#
# Momentum-space integral:
#
#        integral_FBz P(q) d²q / (2*pi)²
#
# With midpoint quadrature:
#
#        sum_ij P_ij dq² / (2*pi)²
#
# IMPORTANT:
#
# This is currently only a Khepri-normalized integrated
# emitted-power quantity.
#
# Do NOT call this the Purcell factor yet.
# ============================================================

BZ_weight = (
    dq
    *
    dq
    /
    (
        2.0
        *
        np.pi
    )**2
)


P_integrated = (
    np.sum(
        P
    )
    *
    BZ_weight
)


P_left_integrated = (
    np.sum(
        P_left
    )
    *
    BZ_weight
)


P_right_integrated = (
    np.sum(
        P_right
    )
    *
    BZ_weight
)


print("\n")
print("=" * 60)
print("FIRST BRILLOUIN-ZONE INTEGRAL")
print("=" * 60)


print(
    "\nIntegral P_total(q) d²q/(2π)² =",
    P_integrated,
)


print(
    "Integral P_left(q)  d²q/(2π)² =",
    P_left_integrated,
)


print(
    "Integral P_right(q) d²q/(2π)² =",
    P_right_integrated,
)


assert np.isfinite(
    P_integrated
)


assert P_integrated > 0.0


integral_sum_error = abs(
    P_integrated
    -
    (
        P_left_integrated
        +
        P_right_integrated
    )
) / max(
    abs(P_integrated),
    1e-30,
)


print(
    "\nIntegral sum consistency error =",
    integral_sum_error,
)


assert integral_sum_error < 1e-12


print(
    "[PASS] BZ integral is finite and positive"
)

print(
    "[PASS] integrated left + right powers equal integrated total"
)


# ============================================================
# LEFT / RIGHT INTEGRATED SYMMETRY
# ============================================================

integrated_lr_error = abs(
    P_left_integrated
    -
    P_right_integrated
) / max(
    abs(P_left_integrated),
    abs(P_right_integrated),
    1e-30,
)


print(
    "\nIntegrated left/right symmetry error =",
    integrated_lr_error,
)


assert integrated_lr_error < TOL_LR


print(
    "[PASS] integrated emission is left/right symmetric"
)


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        6,
        5,
    )
)


im = ax.imshow(
    P,

    origin="lower",

    extent=[
        qmin,
        qmax,
        qmin,
        qmax,
    ],

    aspect="equal",
)


cbar = fig.colorbar(
    im,
    ax=ax,
)


cbar.set_label(
    r"$P(q_x,q_y)$"
)


ax.set_xlabel(
    r"$q_x$"
)


ax.set_ylabel(
    r"$q_y$"
)


ax.set_title(
    "Near-field dipole emissivity"
)


fig.tight_layout()


plt.show()


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 60)
print("q-SPACE DIPOLE TEST PASSED")
print("=" * 60)


print(
    "\nSummary:"
)


print(
    f"P(Gamma)                     = "
    f"{P_gamma:.12g}"
)


print(
    f"Max |P(q)|                   = "
    f"{P_scale:.12g}"
)


print(
    f"Inversion symmetry error     = "
    f"{inversion_error:.3e}"
)


print(
    f"qy mirror symmetry error     = "
    f"{qy_symmetry_error:.3e}"
)


print(
    f"qx mirror symmetry error     = "
    f"{qx_symmetry_error:.3e}"
)


print(
    f"Left/right symmetry error    = "
    f"{lr_error:.3e}"
)


print(
    f"BZ-integrated power          = "
    f"{P_integrated:.12g}"
)


print(
    "\nThe single-q dipole solver works over the "
    "entire first Brillouin zone for this validation system."
)


print(
    "The integrated quantity is not yet called a Purcell "
    "factor; reference normalization is still required."
)