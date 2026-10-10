import numpy as np

from khepri.alternative import free_space_eigenmodes
from khepri.fields import layer_eigenbasis_matrix


# ============================================================
# NORMAL INCIDENCE, ONE FOURIER ORDER
# ============================================================

Kx = np.array([0.0])
Ky = np.array([0.0])

W, V = free_space_eigenmodes(
    Kx,
    Ky
)

R = layer_eigenbasis_matrix(
    W,
    V
)

print("W =")
print(W)

print("\nV =")
print(V)

print("\nR =")
print(R)


# ============================================================
# FORWARD x-POLARIZED MODE
# ============================================================
#
# There are 2 transverse modal coefficients:
#
# mode 0 -> Ex polarization
# mode 1 -> Ey polarization
#
# plus another 2 for backward waves.
# ============================================================

c = np.array([
    1.0,   # forward x-polarized mode
    0.0,
    0.0,
    0.0,
], dtype=complex)


psi = R @ c

Ex, Ey, Ux, Uy = psi


print("\n============================================")
print("FORWARD x-POLARIZED MODE")
print("============================================")

print("Ex =", Ex)
print("Ey =", Ey)
print("Ux =", Ux)
print("Uy =", Uy)


# ============================================================
# EXPECTED PHYSICAL MAXWELL FIELD
# ============================================================
#
# For propagation along +z:
#
# E = x-hat
#
# H = y-hat
#
# in normalized vacuum units.
#
# If Khepri used H directly, we would expect:
#
# Hx = 0
# Hy = +1
#
# If Khepri instead uses U = -i H:
#
# Ux = 0
# Uy = -i
# ============================================================

assert np.allclose(
    Ex,
    1.0
)

assert np.allclose(
    Ey,
    0.0
)

assert np.allclose(
    Ux,
    0.0
)

assert np.allclose(
    Uy,
    -1j
)


print("\n[PASS] Khepri uses U = -i H")


# ============================================================
# SECOND TEST: FORWARD y-POLARIZED MODE
# ============================================================
#
# For:
#
# E = y-hat
#
# propagation +z
#
# physical H = -x-hat
#
# Therefore if U = -iH:
#
# Ux = +i
# Uy = 0
# ============================================================

c = np.array([
    0.0,
    1.0,
    0.0,
    0.0,
], dtype=complex)


psi = R @ c

Ex, Ey, Ux, Uy = psi


print("\n============================================")
print("FORWARD y-POLARIZED MODE")
print("============================================")

print("Ex =", Ex)
print("Ey =", Ey)
print("Ux =", Ux)
print("Uy =", Uy)


assert np.allclose(
    Ex,
    0.0
)

assert np.allclose(
    Ey,
    1.0
)

assert np.allclose(
    Ux,
    1j
)

assert np.allclose(
    Uy,
    0.0
)


print("\n[PASS] y-polarized mode confirms U = -i H")


print("\n============================================")
print("KHEPRI FIELD CONVENTION VERIFIED")
print("============================================")