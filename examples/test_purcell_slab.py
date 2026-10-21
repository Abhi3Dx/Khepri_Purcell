import numpy as np

from khepri.crystal import Crystal

from khepri.dipole import (
    dipole_power_map,
    purcell_factor_from_map,
)


# ============================================================
# PARAMETERS
# ============================================================

a = 1.0

pw = (
    3,
    3,
)

wavelength = 2.0

epsilon_air = 1.0

epsilon_slab = 2.25

n_reference = 1.5

half_thickness = 0.5


current = (
    1.0,
    0.0,
    0.0,
)


position = (
    0.0,
    0.0,
    half_thickness,
)


# ============================================================
# q GRID
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


# ============================================================
# BUILD SLAB
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
# q-SPACE POWER MAP
# ============================================================

print("\n")
print("=" * 60)
print("CALCULATING SLAB q-SPACE POWER")
print("=" * 60)


result_map = dipole_power_map(
    crystal=cl,

    source_after="left_half",

    wavelength=wavelength,

    qx_values=qx,

    qy_values=qy,

    position=position,

    current=current,

    backend="dense",

    verbose=False,
)


P = result_map[
    "P_total"
]


# ============================================================
# PURCELL FACTOR
# ============================================================

result = purcell_factor_from_map(
    power_map=P,

    qx_values=qx,

    qy_values=qy,

    wavelength=wavelength,

    reference_index=n_reference,

    current=current,
)


P_structure = result[
    "P_structure"
]


P_reference = result[
    "P_reference"
]


Fp = result[
    "F_purcell"
]


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("PURCELL RESULT")
print("=" * 60)


print(
    "\nP_structure =",
    P_structure
)


print(
    "P_reference =",
    P_reference
)


print(
    "F_purcell =",
    Fp
)


# ============================================================
# SELF CHECKS
# ============================================================

# Previous q-map test gave this value for Nq=11.

expected_structure = (
    0.21244532841628874
)


structure_error = abs(
    P_structure
    -
    expected_structure
) / expected_structure


print(
    "\nStructure-power error =",
    structure_error
)


assert structure_error < 1e-10


print(
    "[PASS] BZ integration reproduces previous result"
)


# ------------------------------------------------------------
# Homogeneous reference should be pi/4
# ------------------------------------------------------------

expected_reference = (
    np.pi
    /
    4.0
)


reference_error = abs(
    P_reference
    -
    expected_reference
) / expected_reference


print(
    "Reference-power error =",
    reference_error
)


assert reference_error < 1e-12


print(
    "[PASS] homogeneous reference is correct"
)


# ------------------------------------------------------------
# Purcell factor must equal ratio explicitly
# ------------------------------------------------------------

expected_Fp = (
    P_structure
    /
    P_reference
)


assert np.isclose(
    Fp,
    expected_Fp,
    rtol=1e-14,
    atol=1e-14,
)


print(
    "[PASS] Purcell factor is structure/reference ratio"
)


assert np.isfinite(
    Fp
)


assert Fp > 0.0


print(
    "[PASS] Purcell factor is finite and positive"
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 60)
print("PURCELL FACTOR TEST PASSED")
print("=" * 60)


print(
    "\nIMPORTANT:"
)


print(
    "This Nq=11 value is only a coarse Purcell estimate."
)


print(
    "The next step is q-grid convergence before applying "
    "the solver to a photonic crystal."
)