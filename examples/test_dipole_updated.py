import numpy as np

from khepri.crystal import Crystal

from khepri.dipole import (
    dipole_power_map,
    homogeneous_reference_power_xy,
    integrate_dipole_power_map,
    purcell_factor_from_map,
    reciprocal_vectors_from_unitcell,
    reciprocal_unitcell_grid,
    dipole_power_map_unitcell,
    integrate_dipole_power_unitcell,
    purcell_factor_from_unitcell_map,
    purcell_factor_shift_averaged_unitcell,
    purcell_factor_shift_averaged,
)


# ============================================================
# TEST SETTINGS
# ============================================================

A = 1.0
WAVELENGTH = 2.0
EPSILON_AIR = 1.0
EPSILON_SLAB = 2.25
N_REFERENCE = 1.5
HALF_THICKNESS = 0.5
PW = (3, 3)
CURRENT = (1.0, 0.0, 0.0)
POSITION = (0.0, 0.0, HALF_THICKNESS)

# This is the already validated Nq=11 square-slab result.
EXPECTED_P_STRUCTURE = 0.21244532841628874
EXPECTED_P_REFERENCE = np.pi / 4.0
EXPECTED_FP = EXPECTED_P_STRUCTURE / EXPECTED_P_REFERENCE

# Old/new map comparison grid.
NQ_REGRESSION = 11

# Small grid used only to test the new shift-averaging wrappers.
# Keeping this small makes the full regression test reasonably fast.
NQ_SHIFT_TEST = 5
SHIFT_TEST = (0.23, 0.17)


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print("\n")
    print("=" * 78)
    print(title)
    print("=" * 78)


def relative_error(value, reference):
    if reference == 0:
        return abs(value - reference)
    return abs(value - reference) / abs(reference)


def build_square_slab():
    """Build the same laterally uniform slab used in prior validation."""

    cl = Crystal(
        pw=PW,
        lattice="square",
        lattice_pitch=A,
        epsi=EPSILON_AIR,
        epse=EPSILON_AIR,
    )

    cl.add_layer_uniform(
        "left_half",
        epsilon=EPSILON_SLAB,
        depth=HALF_THICKNESS,
    )

    cl.add_layer_uniform(
        "right_half",
        epsilon=EPSILON_SLAB,
        depth=HALF_THICKNESS,
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

    return cl


# ============================================================
# 1. CHECK THAT THE UPDATED API EXISTS
# ============================================================

section("1. UPDATED DIPOLE API")

print("Imported all old and new dipole functions successfully.")
print("[PASS] updated dipole.py imports")


# ============================================================
# 2. SQUARE RECIPROCAL LATTICE
# ============================================================

section("2. SQUARE RECIPROCAL-LATTICE TEST")

square_a1 = np.array([1.0, 0.0])
square_a2 = np.array([0.0, 1.0])

b1, b2, area_real, area_recip = reciprocal_vectors_from_unitcell(
    square_a1,
    square_a2,
)

expected_b1 = np.array([2.0 * np.pi, 0.0])
expected_b2 = np.array([0.0, 2.0 * np.pi])
expected_area_real = 1.0
expected_area_recip = (2.0 * np.pi) ** 2

print("b1 =", b1)
print("b2 =", b2)
print("A_real =", area_real)
print("A_recip =", area_recip)

assert np.allclose(b1, expected_b1, rtol=1e-12, atol=1e-12)
assert np.allclose(b2, expected_b2, rtol=1e-12, atol=1e-12)
assert np.isclose(area_real, expected_area_real, rtol=1e-12, atol=1e-12)
assert np.isclose(area_recip, expected_area_recip, rtol=1e-12, atol=1e-12)

# Explicit duality check.
duality = np.array(
    [
        [np.dot(square_a1, b1), np.dot(square_a1, b2)],
        [np.dot(square_a2, b1), np.dot(square_a2, b2)],
    ]
)

assert np.allclose(
    duality,
    2.0 * np.pi * np.eye(2),
    rtol=1e-12,
    atol=1e-12,
)

print("[PASS] square reciprocal vectors and areas")


# ============================================================
# 3. TRIANGULAR RECIPROCAL LATTICE
# ============================================================

section("3. TRIANGULAR RECIPROCAL-LATTICE TEST")

tri_a1 = np.array([1.0, 0.0])
tri_a2 = np.array([0.5, np.sqrt(3.0) / 2.0])

tri_b1, tri_b2, tri_area_real, tri_area_recip = reciprocal_vectors_from_unitcell(
    tri_a1,
    tri_a2,
)

expected_tri_b1 = np.array(
    [
        2.0 * np.pi,
        -2.0 * np.pi / np.sqrt(3.0),
    ]
)

expected_tri_b2 = np.array(
    [
        0.0,
        4.0 * np.pi / np.sqrt(3.0),
    ]
)

expected_tri_area_real = np.sqrt(3.0) / 2.0
expected_tri_area_recip = 8.0 * np.pi**2 / np.sqrt(3.0)

print("triangular b1 =", tri_b1)
print("triangular b2 =", tri_b2)
print("triangular A_real =", tri_area_real)
print("triangular A_recip =", tri_area_recip)

assert np.allclose(
    tri_b1,
    expected_tri_b1,
    rtol=1e-12,
    atol=1e-12,
)

assert np.allclose(
    tri_b2,
    expected_tri_b2,
    rtol=1e-12,
    atol=1e-12,
)

assert np.isclose(
    tri_area_real,
    expected_tri_area_real,
    rtol=1e-12,
    atol=1e-12,
)

assert np.isclose(
    tri_area_recip,
    expected_tri_area_recip,
    rtol=1e-12,
    atol=1e-12,
)

print("[PASS] triangular reciprocal vectors and areas")


# ============================================================
# 4. GENERAL RECIPROCAL-CELL GRID
# ============================================================

section("4. GENERAL RECIPROCAL-CELL GRID TEST")

Nu_test = 5
Nv_test = 7
shift_test = (0.23, 0.17)

grid = reciprocal_unitcell_grid(
    a1=tri_a1,
    a2=tri_a2,
    Nu=Nu_test,
    Nv=Nv_test,
    shift=shift_test,
)

assert grid["qx"].shape == (Nv_test, Nu_test)
assert grid["qy"].shape == (Nv_test, Nu_test)
assert grid["U"].shape == (Nv_test, Nu_test)
assert grid["V"].shape == (Nv_test, Nu_test)

# q = U*b1 + V*b2 must hold point-by-point.
qx_reconstructed = (
    grid["U"] * grid["b1"][0]
    + grid["V"] * grid["b2"][0]
)

qy_reconstructed = (
    grid["U"] * grid["b1"][1]
    + grid["V"] * grid["b2"][1]
)

assert np.allclose(grid["qx"], qx_reconstructed, rtol=1e-13, atol=1e-13)
assert np.allclose(grid["qy"], qy_reconstructed, rtol=1e-13, atol=1e-13)

# Each point represents reciprocal area A_recip/(Nu*Nv).
expected_dA_q = tri_area_recip / (Nu_test * Nv_test)
expected_weight = 1.0 / (tri_area_real * Nu_test * Nv_test)

assert np.isclose(
    grid["dA_q"],
    expected_dA_q,
    rtol=1e-12,
    atol=1e-12,
)

assert np.isclose(
    grid["integration_weight"],
    expected_weight,
    rtol=1e-12,
    atol=1e-12,
)

# Fractional coordinates must live inside [-1/2, +1/2).
assert np.all(grid["u"] >= -0.5)
assert np.all(grid["u"] < 0.5)
assert np.all(grid["v"] >= -0.5)
assert np.all(grid["v"] < 0.5)

print("grid shape =", grid["qx"].shape)
print("integration weight =", grid["integration_weight"])
print("[PASS] arbitrary-unit-cell reciprocal grid")


# ============================================================
# 5. HOMOGENEOUS REFERENCE REGRESSION
# ============================================================

section("5. HOMOGENEOUS REFERENCE REGRESSION")

P_reference = homogeneous_reference_power_xy(
    wavelength=WAVELENGTH,
    refractive_index=N_REFERENCE,
    current=CURRENT,
)

print("P_reference =", P_reference)
print("expected    =", EXPECTED_P_REFERENCE)

assert np.isclose(
    P_reference,
    EXPECTED_P_REFERENCE,
    rtol=1e-13,
    atol=1e-13,
)

print("[PASS] homogeneous reference remains pi/4")


# ============================================================
# 6. OLD CARTESIAN MAP VS NEW UNIT-CELL MAP
# ============================================================

section("6. OLD CARTESIAN MAP VS NEW UNIT-CELL MAP")

cl = build_square_slab()

qmin = -np.pi / A
qmax = +np.pi / A
dq = (qmax - qmin) / NQ_REGRESSION

qx_old = qmin + (np.arange(NQ_REGRESSION) + 0.5) * dq
qy_old = qx_old.copy()

print("Calculating old Cartesian map ...")

old_map = dipole_power_map(
    crystal=cl,
    source_after="left_half",
    wavelength=WAVELENGTH,
    qx_values=qx_old,
    qy_values=qy_old,
    position=POSITION,
    current=CURRENT,
    backend="dense",
    verbose=False,
)

print("Calculating new general unit-cell map ...")

new_map = dipole_power_map_unitcell(
    crystal=cl,
    source_after="left_half",
    wavelength=WAVELENGTH,
    a1=(A, 0.0),
    a2=(0.0, A),
    Nu=NQ_REGRESSION,
    Nv=NQ_REGRESSION,
    position=POSITION,
    current=CURRENT,
    shift=(0.0, 0.0),
    backend="dense",
    verbose=False,
    return_point_details=False,
)

# The general square grid should exactly reproduce the old Cartesian grid.
assert np.allclose(
    new_map["qx"][0, :],
    qx_old,
    rtol=1e-13,
    atol=1e-13,
)

assert np.allclose(
    new_map["qy"][:, 0],
    qy_old,
    rtol=1e-13,
    atol=1e-13,
)

# The actual RCWA power maps should be the same.
for key in ("P_left", "P_right", "P_total"):
    difference = np.max(
        np.abs(
            old_map[key]
            - new_map[key]
        )
    )

    print(f"max |old-new| for {key} = {difference}")

    assert np.allclose(
        old_map[key],
        new_map[key],
        rtol=1e-11,
        atol=1e-12,
    )

print("[PASS] new unit-cell map reproduces old Cartesian map")


# ============================================================
# 7. OLD INTEGRAL VS NEW UNIT-CELL INTEGRAL
# ============================================================

section("7. OLD INTEGRAL VS NEW UNIT-CELL INTEGRAL")

P_old = integrate_dipole_power_map(
    power_map=old_map["P_total"],
    qx_values=qx_old,
    qy_values=qy_old,
)

P_new = integrate_dipole_power_unitcell(
    map_result=new_map,
    component="P_total",
)

print("P_old      =", P_old)
print("P_new      =", P_new)
print("P_expected =", EXPECTED_P_STRUCTURE)

assert np.isclose(
    P_old,
    P_new,
    rtol=1e-12,
    atol=1e-12,
)

assert np.isclose(
    P_new,
    EXPECTED_P_STRUCTURE,
    rtol=1e-11,
    atol=1e-12,
)

print("relative error vs validated slab value =", relative_error(P_new, EXPECTED_P_STRUCTURE))
print("[PASS] new reciprocal-cell integration reproduces validated slab result")


# ============================================================
# 8. OLD PURCELL WRAPPER VS NEW UNIT-CELL WRAPPER
# ============================================================

section("8. PURCELL WRAPPER REGRESSION")

old_pf = purcell_factor_from_map(
    power_map=old_map["P_total"],
    qx_values=qx_old,
    qy_values=qy_old,
    wavelength=WAVELENGTH,
    reference_index=N_REFERENCE,
    current=CURRENT,
)

new_pf = purcell_factor_from_unitcell_map(
    map_result=new_map,
    wavelength=WAVELENGTH,
    reference_index=N_REFERENCE,
    current=CURRENT,
)

print("old Fp      =", old_pf["F_purcell"])
print("new Fp      =", new_pf["F_purcell"])
print("expected Fp =", EXPECTED_FP)

assert np.isclose(
    old_pf["F_purcell"],
    new_pf["F_purcell"],
    rtol=1e-12,
    atol=1e-12,
)

assert np.isclose(
    new_pf["F_purcell"],
    EXPECTED_FP,
    rtol=1e-11,
    atol=1e-12,
)

assert np.isclose(
    new_pf["P_reference"],
    EXPECTED_P_REFERENCE,
    rtol=1e-13,
    atol=1e-13,
)

print("[PASS] new Purcell wrapper reproduces validated Cartesian result")


# ============================================================
# 9. SHIFT-AVERAGED GENERAL ROUTINE + BACKWARD WRAPPER
# ============================================================

section("9. SHIFT-AVERAGED GENERAL ROUTINE")

# Use a fresh slab object to make this test independent of prior source state.
cl_shift = build_square_slab()

print("Calculating one shifted general unit-cell result ...")

general_shift = purcell_factor_shift_averaged_unitcell(
    crystal=cl_shift,
    source_after="left_half",
    wavelength=WAVELENGTH,
    a1=(A, 0.0),
    a2=(0.0, A),
    Nu=NQ_SHIFT_TEST,
    Nv=NQ_SHIFT_TEST,
    position=POSITION,
    current=CURRENT,
    reference_index=N_REFERENCE,
    shifts=[SHIFT_TEST],
    backend="dense",
    verbose=False,
)

# One shift must have exactly zero sample standard deviation by construction.
assert general_shift["F_std"] == 0.0
assert general_shift["P_structure_std"] == 0.0
assert general_shift["F_spread"] == 0.0
assert general_shift["F_spread_percent"] == 0.0
assert np.isfinite(general_shift["F_purcell"])
assert general_shift["F_purcell"] > 0.0

print("general shifted Fp =", general_shift["F_purcell"])
print("general F_std =", general_shift["F_std"])

print("Calculating backward-compatible square wrapper ...")

cl_shift_wrapper = build_square_slab()

square_shift = purcell_factor_shift_averaged(
    crystal=cl_shift_wrapper,
    source_after="left_half",
    wavelength=WAVELENGTH,
    a=A,
    Nq=NQ_SHIFT_TEST,
    position=POSITION,
    current=CURRENT,
    reference_index=N_REFERENCE,
    shifts=[SHIFT_TEST],
    backend="dense",
    verbose=False,
)

print("square wrapper Fp =", square_shift["F_purcell"])

assert np.isclose(
    general_shift["P_structure"],
    square_shift["P_structure"],
    rtol=1e-12,
    atol=1e-12,
)

assert np.isclose(
    general_shift["F_purcell"],
    square_shift["F_purcell"],
    rtol=1e-12,
    atol=1e-12,
)

print("[PASS] general shift averaging and old square wrapper agree")


# ============================================================
# FINAL SUMMARY
# ============================================================

section("UPDATED DIPOLE TEST PASSED")

print("Validated:")
print("  reciprocal vectors for square lattice")
print("  reciprocal vectors for triangular lattice")
print("  arbitrary reciprocal primitive-cell grid")
print("  reciprocal-cell integration weight")
print("  homogeneous reference power")
print("  old Cartesian map vs new unit-cell map")
print("  old Cartesian integral vs new unit-cell integral")
print("  old Purcell wrapper vs new unit-cell Purcell wrapper")
print("  one-shift standard-deviation handling")
print("  backward-compatible square shift wrapper")

print("\nValidated square-slab values:")
print("P_structure =", P_new)
print("P_reference =", new_pf["P_reference"])
print("F_purcell   =", new_pf["F_purcell"])

print("\nThe updated general unit-cell implementation is ready for")
print("triangular-lattice / PRB photonic-crystal testing.")
