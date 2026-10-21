import numpy as np

from khepri.crystal import Crystal
from khepri.dipole import dipole_power_q


# ============================================================
# PARAMETERS
# ============================================================

pw = (3, 3)

a = 1.0

wavelength = 2.0

epsilon_air = 1.0

epsilon_slab = 2.25

half_thickness = 0.5


# ============================================================
# BUILD SAME VALIDATED DIELECTRIC SLAB
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
# ONE q POINT
# ============================================================

result = dipole_power_q(
    crystal=cl,

    source_after="left_half",

    wavelength=wavelength,

    kp=(
        0.0,
        0.0,
    ),

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

    return_details=True,
)


# ============================================================
# DISPLAY
# ============================================================

print("\n")
print("=" * 60)
print("SINGLE-q DIPOLE POWER")
print("=" * 60)


print(
    "\nq =",
    result["kp"]
)


print(
    "wavelength =",
    result["wavelength"]
)


print(
    "\nP_left  =",
    result["P_left"]
)

print(
    "P_right =",
    result["P_right"]
)

print(
    "P_total =",
    result["P_total"]
)


print(
    "\nJump relative error =",
    result["jump_relative_error"]
)


# ============================================================
# SELF CHECKS
# ============================================================

expected_power = (
    0.3076923076923077
)


# ------------------------------------------------------------
# Positive emitted power
# ------------------------------------------------------------

assert result["P_left"] > 0.0

assert result["P_right"] > 0.0

assert result["P_total"] > 0.0


print(
    "\n[PASS] emitted power is positive"
)


# ------------------------------------------------------------
# Symmetry
# ------------------------------------------------------------

symmetry_error = abs(
    result["P_left"]
    -
    result["P_right"]
) / max(
    result["P_left"],
    result["P_right"],
    1e-30,
)


print(
    "Left/right symmetry error =",
    symmetry_error
)


assert symmetry_error < 1e-10


print(
    "[PASS] symmetric slab emits equally left and right"
)


# ------------------------------------------------------------
# Maxwell jump
# ------------------------------------------------------------

assert (
    result["jump_relative_error"]
    <
    1e-10
)


print(
    "[PASS] Maxwell source jump"
)


# ------------------------------------------------------------
# Compare against already validated reference value
# ------------------------------------------------------------

power_error = abs(
    result["P_total"]
    -
    expected_power
) / expected_power


print(
    "Reference-power error =",
    power_error
)


assert power_error < 1e-10


print(
    "[PASS] single-q API reproduces validated slab result"
)


# ============================================================
# SUCCESS
# ============================================================

print("\n")
print("=" * 60)
print("SINGLE-q DIPOLE POWER TEST PASSED")
print("=" * 60)