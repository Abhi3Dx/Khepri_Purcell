import numpy as np

from khepri.crystal import Crystal
from khepri.alternative import free_space_eigenmodes
from khepri.fields import layer_eigenbasis_matrix

from khepri.dipole import (
    dipole_fourier_current,
    dipole_source_jump_xy,
    solve_dipole_source_plane,
    dipole_source_plane_fields,
    dipole_exterior_amplitudes,
    dipole_outgoing_flux,
)

# ============================================================
# PRINTING OPTIONS
# ============================================================

np.set_printoptions(
    precision=6,
    suppress=True
)


# ============================================================
# PARAMETERS
# ============================================================

pw = (3, 3)

lattice_pitch = 1.0

wavelength = 2.0

# Bloch wavevector
kp = (0.0, 0.0)


# ============================================================
# MATERIAL PARAMETERS
# ============================================================
#
# Structure:
#
#              AIR
#               |
#        ----------------
#        epsilon = 2.25
#        thickness = 0.5
#        ----------------
#             DIPOLE
#        ----------------
#        epsilon = 2.25
#        thickness = 0.5
#        ----------------
#               |
#              AIR
#
# The two halves together form one dielectric slab
# of total thickness = 1.0.
#
# The dipole is located exactly at its center.
# ============================================================

epsilon_air = 1.0
epsilon_slab = 2.25

half_thickness = 0.5


# ============================================================
# 1. CREATE CRYSTAL
# ============================================================

cl = Crystal(
    pw=pw,
    lattice="square",
    lattice_pitch=lattice_pitch,
    epsi=epsilon_air,
    epse=epsilon_air,
)


# ============================================================
# 2. ADD LEFT HALF OF DIELECTRIC SLAB
# ============================================================

cl.add_layer_uniform(
    "left_half",
    epsilon=epsilon_slab,
    depth=half_thickness,
)


# ============================================================
# 3. ADD RIGHT HALF OF DIELECTRIC SLAB
# ============================================================

cl.add_layer_uniform(
    "right_half",
    epsilon=epsilon_slab,
    depth=half_thickness,
)


# ============================================================
# 4. DEFINE DEVICE STACK
# ============================================================
#
# fields_mask=True for left_half ensures that Khepri retains
# the partial scattering matrices at the plane immediately
# after left_half.
#
# That plane is exactly where we place the dipole.
# ============================================================

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
# 5. DEFINE WAVELENGTH AND BLOCH VECTOR
# ============================================================
#
# The TE/TM values here only satisfy the normal Crystal source
# interface. They are NOT the dipole source.
#
# The dipole will be inserted separately below.
# ============================================================

cl.set_source(
    wavelength=wavelength,
    te=1.0,
    tm=0.0,
    kp=kp,
)


# ============================================================
# 6. SOLVE THE ORDINARY RCWA STACK
# ============================================================

cl.solve(
    backend="dense"
)

print("\n============================================")
print("RCWA STACK")
print("============================================")

print("\nRCWA stack solved.")

print("\nGlobal stacking:")
print(cl.global_stacking)


# ============================================================
# 7. LOCATE THE SOURCE PLANE
# ============================================================
#
# Expected:
#
# global_stacking =
#
# [
#     'Sref',
#     'left_half',
#     'right_half',
#     'Strans'
# ]
#
# The dipole is placed AFTER left_half.
# ============================================================

source_index = cl.global_stacking.index(
    "left_half"
)

print("\nSource-plane index:")
print(source_index)


# ============================================================
# 8. EXTRACT PARTIAL SCATTERING MATRICES
# ============================================================
#
# Sl = complete stack to LEFT of dipole plane
#
#      air + left_half
#
#
# Sr = complete stack to RIGHT of dipole plane
#
#      right_half + air
#
# ============================================================

Sl = cl.stacking_matrices[
    source_index
]

Sr = cl.stacking_reverse_matrices[
    source_index
]


if Sl is None:
    raise RuntimeError(
        "Left partial scattering matrix was not retained."
    )

if Sr is None:
    raise RuntimeError(
        "Right partial scattering matrix was not retained."
    )


print("\nSl shape:")
print(Sl.shape)

print("\nSr shape:")
print(Sr.shape)


# ============================================================
# 9. CONSTRUCT COMMON REFERENCE BASIS R0
# ============================================================
#
# Khepri scattering matrices are expressed in the W0,V0
# reference eigenbasis.
#
# Therefore the physical source jump b must be converted
# through R0.
# ============================================================

Kx, Ky, _ = cl.expansion.k_vectors(
    kp,
    wavelength,
)

W0, V0 = free_space_eigenmodes(
    Kx,
    Ky,
)

R0 = layer_eigenbasis_matrix(
    W0,
    V0,
)


print("\nR0 shape:")
print(R0.shape)


# ============================================================
# 10. DEFINE DIPOLE
# ============================================================
#
# x-oriented electric-current dipole
#
# Position:
#
# x = 0
# y = 0
# z = 0.5
#
# The z coordinate indicates the physical source location.
# It does not enter the in-plane Fourier phase.
# ============================================================

position = (
    0.0,   # xd
    0.0,   # yd
    half_thickness,   # zd
)


current = (
    1.0,   # jx
    0.0,   # jy
    0.0,   # jz
)


# ============================================================
# 11. CONSTRUCT FOURIER DIPOLE CURRENT
# ============================================================

Jx, Jy, Jz = dipole_fourier_current(
    expansion=cl.expansion,
    kp=kp,
    position=position,
    current=current,
)


print("\n============================================")
print("DIPOLE CURRENT")
print("============================================")

print("\nJx:")
print(Jx)

print("\nJy:")
print(Jy)

print("\nJz:")
print(Jz)


# ============================================================
# 12. CONSTRUCT MAXWELL SOURCE JUMP
# ============================================================
#
# For an in-plane electric current:
#
# b =
#
# [ Delta Ex ]
# [ Delta Ey ]
# [ Delta Hx ]
# [ Delta Hy ]
#
# =
#
# [  0  ]
# [  0  ]
# [ Jy  ]
# [-Jx  ]
#
# ============================================================

b = dipole_source_jump_xy(
    Jx,
    Jy,
)


print("\nSource-vector shape:")
print(b.shape)


# ============================================================
# 13. SOLVE DIPOLE SOURCE-PLANE PROBLEM
# ============================================================
#
# c_plus:
#     emitted toward +z / right
#
# c_minus:
#     emitted toward -z / left
#
# Multiple reflections from both halves of the slab are
# automatically included.
# ============================================================

c_plus, c_minus = solve_dipole_source_plane(
    R0=R0,
    Sl=Sl,
    Sr=Sr,
    b=b,
)


print("\nc_plus shape:")
print(c_plus.shape)

print("\nc_minus shape:")
print(c_minus.shape)


# ============================================================
# 14. RECONSTRUCT FIELDS ON BOTH SIDES OF DIPOLE
# ============================================================

psi_left, psi_right = dipole_source_plane_fields(
    R0=R0,
    Sl=Sl,
    Sr=Sr,
    c_plus=c_plus,
    c_minus=c_minus,
)


print("\npsi_left shape:")
print(psi_left.shape)

print("\npsi_right shape:")
print(psi_right.shape)


# ============================================================
# 15. TEST MAXWELL JUMP CONDITION
# ============================================================
#
# Fundamental condition:
#
#      psi_right - psi_left = b
#
# ============================================================

jump_calculated = (
    psi_right
    -
    psi_left
)

jump_error = (
    jump_calculated
    -
    b
)


absolute_error = np.linalg.norm(
    jump_error
)


relative_error = (
    absolute_error
    /
    np.linalg.norm(b)
)


print("\n============================================")
print("MAXWELL JUMP TEST")
print("============================================")

print(
    "\nAbsolute jump error :",
    absolute_error,
)

print(
    "Relative jump error :",
    relative_error,
)


# ============================================================
# 16. EXTRACT REFLECTION MATRICES SEEN BY DIPOLE
# ============================================================

R_left = Sl[1, 1]

R_right = Sr[0, 0]


left_reflection_norm = np.linalg.norm(
    R_left
)

right_reflection_norm = np.linalg.norm(
    R_right
)


print("\n============================================")
print("REFLECTION TEST")
print("============================================")

print(
    "\nLeft reflection norm  :",
    left_reflection_norm,
)

print(
    "Right reflection norm :",
    right_reflection_norm,
)


# ============================================================
# 17. COMPARE LEFT AND RIGHT REFLECTION STRENGTH
# ============================================================
#
# Since the dipole is at the center of a symmetric slab,
# both sides should have the same reflection strength.
#
# We compare the norms rather than requiring every matrix
# element to be identical.
# ============================================================

reflection_norm_symmetry_error = abs(
    left_reflection_norm
    -
    right_reflection_norm
) / max(
    left_reflection_norm,
    right_reflection_norm,
    1e-30,
)


print(
    "\nReflection-norm symmetry error :",
    reflection_norm_symmetry_error,
)


# ============================================================
# 18. CHECK DIPOLE EMISSION SYMMETRY
# ============================================================
#
# Because the environment is mirror-symmetric around the
# dipole plane:
#
#      ||c_plus|| = ||c_minus||
#
# ============================================================

plus_norm = np.linalg.norm(
    c_plus
)

minus_norm = np.linalg.norm(
    c_minus
)


symmetry_error = abs(
    plus_norm
    -
    minus_norm
) / max(
    plus_norm,
    minus_norm,
    1e-30,
)


print("\n============================================")
print("EMISSION SYMMETRY TEST")
print("============================================")

print(
    "\n||c_plus||  =",
    plus_norm,
)

print(
    "||c_minus|| =",
    minus_norm,
)

print(
    "Symmetry error =",
    symmetry_error,
)


# ============================================================
# 19. AUTOMATIC TESTS
# ============================================================

print("\n============================================")
print("AUTOMATIC CHECKS")
print("============================================")


# ------------------------------------------------------------
# Jump condition must be extremely accurate
# ------------------------------------------------------------

assert relative_error < 1e-10, (
    "\nFAILED:\n"
    "Dipole Maxwell jump condition is not satisfied."
)


print(
    "\n[PASS] Maxwell jump condition"
)


# ------------------------------------------------------------
# Dielectric-air interfaces must cause reflection
# ------------------------------------------------------------

assert left_reflection_norm > 1e-12, (
    "\nFAILED:\n"
    "Left reflection is unexpectedly zero."
)


assert right_reflection_norm > 1e-12, (
    "\nFAILED:\n"
    "Right reflection is unexpectedly zero."
)


print(
    "[PASS] Non-zero slab reflections"
)


# ------------------------------------------------------------
# Symmetric slab → same reflection strength on both sides
# ------------------------------------------------------------

assert reflection_norm_symmetry_error < 1e-10, (
    "\nFAILED:\n"
    "Left and right reflection strengths are not symmetric."
)


print(
    "[PASS] Reflection symmetry"
)


# ------------------------------------------------------------
# Symmetric environment → symmetric dipole emission
# ------------------------------------------------------------

assert symmetry_error < 1e-10, (
    "\nFAILED:\n"
    "Dipole emission toward +z and -z is not symmetric."
)


print(
    "[PASS] Dipole emission symmetry"
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n============================================")
print("ALL DIELECTRIC DIPOLE TESTS PASSED")
print("============================================")

print(
    "\nThe dipole source is now interacting with "
    "non-zero reflections from the dielectric slab."
)

print(
    "The source jump, multiple reflections, "
    "and left/right symmetry are all consistent."
)

# ============================================================
# 20. PROPAGATE DIPOLE RADIATION TO EXTERIOR HALF-SPACES
# ============================================================

c_left_out, c_right_out = dipole_exterior_amplitudes(
    Sl=Sl,
    Sr=Sr,
    c_plus=c_plus,
    c_minus=c_minus,
)


print("\n============================================")
print("EXTERIOR DIPOLE AMPLITUDES")
print("============================================")

print(
    "\nc_left_out shape  =",
    c_left_out.shape
)

print(
    "c_right_out shape =",
    c_right_out.shape
)


# ============================================================
# 21. CONVERT MODAL AMPLITUDES TO ELECTRIC FIELDS
# ============================================================
#
# This is exactly analogous to Crystal.poynting_flux_end(),
# which multiplies the exterior modal coefficients by Wref
# or Wtrans before calling poynting_fluxes().
# ============================================================

Wref = cl.layers["Sref"].W

Wtrans = cl.layers["Strans"].W


E_left_out = (
    Wref @ c_left_out
)

E_right_out = (
    Wtrans @ c_right_out
)


print(
    "\nE_left_out shape  =",
    E_left_out.shape
)

print(
    "E_right_out shape =",
    E_right_out.shape
)


# ============================================================
# 22. CALCULATE OUTWARD DIPOLE POWER
# ============================================================

P_left, P_left_orders = dipole_outgoing_flux(
    expansion=cl.expansion,
    electric_field=E_left_out,
    kp=kp,
    wavelength=wavelength,
    epsilon=epsilon_air,
    return_orders=True,
)


P_right, P_right_orders = dipole_outgoing_flux(
    expansion=cl.expansion,
    electric_field=E_right_out,
    kp=kp,
    wavelength=wavelength,
    epsilon=epsilon_air,
    return_orders=True,
)


P_total = (
    P_left
    +
    P_right
)


print("\n============================================")
print("DIPOLE RADIATED POWER")
print("============================================")

print(
    "\nP_left  =",
    P_left
)

print(
    "P_right =",
    P_right
)

print(
    "P_total =",
    P_total
)


# ============================================================
# 23. LEFT / RIGHT POWER SYMMETRY
# ============================================================
#
# Dipole is at the center of a symmetric dielectric slab,
# therefore:
#
#       P_left = P_right
#
# ============================================================

power_symmetry_error = abs(
    P_left - P_right
) / max(
    abs(P_left),
    abs(P_right),
    1e-30,
)


print(
    "\nPower symmetry error =",
    power_symmetry_error
)


assert power_symmetry_error < 1e-10, (
    "\nFAILED:\n"
    "Left/right radiated dipole powers are not symmetric."
)


print(
    "\n[PASS] Dipole radiated-power symmetry"
)


# ============================================================
# 24. SHOW POWER IN EACH DIFFRACTION ORDER
# ============================================================

print("\n============================================")
print("POWER PER DIFFRACTION ORDER")
print("============================================")

print(
    "\nLeft:"
)

print(
    P_left_orders
)

print(
    "\nRight:"
)

print(
    P_right_orders
)


# ============================================================
# FINAL POWER TEST
# ============================================================

print("\n============================================")
print("DIPOLE POWER TEST PASSED")
print("============================================")