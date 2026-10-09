import numpy as np

from khepri.crystal import Crystal
from khepri.expansion import Expansion
from khepri.alternative import free_space_eigenmodes
from khepri.fields import layer_eigenbasis_matrix

from khepri.dipole import (
    dipole_fourier_current,
    dipole_source_jump_xy,
    solve_dipole_source_plane,
    dipole_source_plane_fields,
    dipole_exterior_amplitudes,
    dipole_outgoing_flux,
    dipole_total_nearfield_power,
)


# ============================================================
# SETTINGS
# ============================================================

np.set_printoptions(
    precision=10,
    suppress=True,
)

PW = (3, 3)

LATTICE_PITCH = 1.0

WAVELENGTH = 2.0

KP = (0.0, 0.0)

HALF_THICKNESS = 0.5

EPSILON_AIR = 1.0

EPSILON_SLAB = 2.25


# ============================================================
# TOLERANCES
# ============================================================

TOL = 1e-10

TOL_ZERO = 1e-12

TOL_POWER = 1e-10


# ============================================================
# TEST OUTPUT HELPERS
# ============================================================

def section(title):
    print("\n")
    print("=" * 60)
    print(title)
    print("=" * 60)


def passed(message):
    print(f"[PASS] {message}")


def relative_difference(a, b):
    """
    Relative difference between two scalar quantities.
    """

    return abs(a - b) / max(
        abs(a),
        abs(b),
        1e-30,
    )


# ============================================================
# COMPLETE DIPOLE SOLVER FOR ONE TEST CASE
# ============================================================

def solve_test_case(
    epsilon_slab,
    current=(1.0, 0.0, 0.0),
    position=(0.0, 0.0, HALF_THICKNESS),
):
    """
    Construct a symmetric slab:

               air
                |
          epsilon_slab
             d = 0.5
        ----------------
             DIPOLE
        ----------------
          epsilon_slab
             d = 0.5
                |
               air

    and solve the dipole radiation problem.

    Returns all quantities required for self-checking.
    """

    # --------------------------------------------------------
    # Crystal
    # --------------------------------------------------------

    cl = Crystal(
        pw=PW,
        lattice="square",
        lattice_pitch=LATTICE_PITCH,
        epsi=EPSILON_AIR,
        epse=EPSILON_AIR,
    )


    # --------------------------------------------------------
    # Two identical halves
    # --------------------------------------------------------

    cl.add_layer_uniform(
        "left_half",
        epsilon=epsilon_slab,
        depth=HALF_THICKNESS,
    )

    cl.add_layer_uniform(
        "right_half",
        epsilon=epsilon_slab,
        depth=HALF_THICKNESS,
    )


    # --------------------------------------------------------
    # Retain partial matrices at source plane
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Bloch vector / wavelength
    # --------------------------------------------------------

    cl.set_source(
        wavelength=WAVELENGTH,
        te=1.0,
        tm=0.0,
        kp=KP,
    )


    # --------------------------------------------------------
    # Ordinary RCWA solve
    # --------------------------------------------------------

    cl.solve(
        backend="dense"
    )


    # --------------------------------------------------------
    # Source plane
    # --------------------------------------------------------

    source_index = cl.global_stacking.index(
        "left_half"
    )

    Sl = cl.stacking_matrices[
        source_index
    ]

    Sr = cl.stacking_reverse_matrices[
        source_index
    ]


    if Sl is None:
        raise RuntimeError(
            "Sl was not retained."
        )

    if Sr is None:
        raise RuntimeError(
            "Sr was not retained."
        )


    # --------------------------------------------------------
    # Common Khepri reference basis
    # --------------------------------------------------------

    Kx, Ky, _ = cl.expansion.k_vectors(
        KP,
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


    # --------------------------------------------------------
    # Fourier dipole current
    # --------------------------------------------------------

    Jx, Jy, Jz = dipole_fourier_current(
        expansion=cl.expansion,
        kp=KP,
        position=position,
        current=current,
    )


    # --------------------------------------------------------
    # Khepri source jump
    # --------------------------------------------------------

    b = dipole_source_jump_xy(
        Jx,
        Jy,
    )


    # --------------------------------------------------------
    # Solve source-plane amplitudes
    # --------------------------------------------------------

    c_plus, c_minus = solve_dipole_source_plane(
        R0=R0,
        Sl=Sl,
        Sr=Sr,
        b=b,
    )


    # --------------------------------------------------------
    # Fields immediately around source
    # --------------------------------------------------------

    psi_left, psi_right = dipole_source_plane_fields(
        R0=R0,
        Sl=Sl,
        Sr=Sr,
        c_plus=c_plus,
        c_minus=c_minus,
    )


    # --------------------------------------------------------
    # Exterior amplitudes
    # --------------------------------------------------------

    c_left_out, c_right_out = dipole_exterior_amplitudes(
        Sl=Sl,
        Sr=Sr,
        c_plus=c_plus,
        c_minus=c_minus,
    )


    # --------------------------------------------------------
    # Convert exterior modal coefficients to transverse E
    # --------------------------------------------------------

    Wref = cl.layers["Sref"].W

    Wtrans = cl.layers["Strans"].W

    E_left_out = (
        Wref
        @
        c_left_out
    )

    E_right_out = (
        Wtrans
        @
        c_right_out
    )


    # --------------------------------------------------------
    # Exterior powers
    # --------------------------------------------------------

    P_ext_left, P_ext_left_orders = dipole_outgoing_flux(
        expansion=cl.expansion,
        electric_field=E_left_out,
        kp=KP,
        wavelength=WAVELENGTH,
        epsilon=EPSILON_AIR,
        return_orders=True,
    )

    P_ext_right, P_ext_right_orders = dipole_outgoing_flux(
        expansion=cl.expansion,
        electric_field=E_right_out,
        kp=KP,
        wavelength=WAVELENGTH,
        epsilon=EPSILON_AIR,
        return_orders=True,
    )

    P_ext_total = (
        P_ext_left
        +
        P_ext_right
    )


    # --------------------------------------------------------
    # Near-source powers
    # --------------------------------------------------------

    (
        P_near_left,
        P_near_right,
        P_near_total,
    ) = dipole_total_nearfield_power(
        psi_left,
        psi_right,
    )


    return {
        "crystal": cl,
        "Sl": Sl,
        "Sr": Sr,
        "R0": R0,

        "Jx": Jx,
        "Jy": Jy,
        "Jz": Jz,

        "b": b,

        "c_plus": c_plus,
        "c_minus": c_minus,

        "psi_left": psi_left,
        "psi_right": psi_right,

        "P_ext_left": P_ext_left,
        "P_ext_right": P_ext_right,
        "P_ext_total": P_ext_total,

        "P_ext_left_orders": P_ext_left_orders,
        "P_ext_right_orders": P_ext_right_orders,

        "P_near_left": P_near_left,
        "P_near_right": P_near_right,
        "P_near_total": P_near_total,
    }


# ============================================================
# TEST 1
#
# VERIFY KHEPRI FIELD CONVENTION
#
# U = -i H
# ============================================================

section(
    "TEST 1 — KHEPRI FIELD CONVENTION"
)


Kx_test = np.array([
    0.0
])

Ky_test = np.array([
    0.0
])


W, V = free_space_eigenmodes(
    Kx_test,
    Ky_test,
)


R = layer_eigenbasis_matrix(
    W,
    V,
)


# ------------------------------------------------------------
# Forward x-polarized mode
# ------------------------------------------------------------

c = np.array(
    [
        1.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=complex,
)


psi = R @ c


Ex, Ey, Ux, Uy = psi


assert np.allclose(
    Ex,
    1.0,
)

assert np.allclose(
    Ey,
    0.0,
)

assert np.allclose(
    Ux,
    0.0,
)

assert np.allclose(
    Uy,
    -1j,
)


passed(
    "x-polarized mode gives U = -i H"
)


# ------------------------------------------------------------
# Forward y-polarized mode
# ------------------------------------------------------------

c = np.array(
    [
        0.0,
        1.0,
        0.0,
        0.0,
    ],
    dtype=complex,
)


psi = R @ c


Ex, Ey, Ux, Uy = psi


assert np.allclose(
    Ex,
    0.0,
)

assert np.allclose(
    Ey,
    1.0,
)

assert np.allclose(
    Ux,
    1j,
)

assert np.allclose(
    Uy,
    0.0,
)


passed(
    "y-polarized mode independently confirms U = -i H"
)


# ============================================================
# TEST 2
#
# FOURIER DIPOLE CURRENT
# ============================================================

section(
    "TEST 2 — FOURIER DIPOLE CURRENT"
)


lattice = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
)


expansion = Expansion(
    PW,
    lattice,
)


N_G = np.prod(
    PW
)


# ------------------------------------------------------------
# Dipole at origin
# ------------------------------------------------------------

Jx, Jy, Jz = dipole_fourier_current(
    expansion=expansion,
    kp=(0.0, 0.0),
    position=(0.0, 0.0, 0.0),
    current=(1.0, 0.0, 0.0),
)


assert Jx.shape == (
    N_G,
)

assert np.allclose(
    Jx,
    1.0,
)

assert np.allclose(
    Jy,
    0.0,
)

assert np.allclose(
    Jz,
    0.0,
)


passed(
    "x-dipole at origin has expected Fourier current"
)


# ------------------------------------------------------------
# Displaced dipole
# ------------------------------------------------------------

Jx_displaced, _, _ = dipole_fourier_current(
    expansion=expansion,
    kp=(0.0, 0.0),
    position=(0.2, 0.3, 0.0),
    current=(1.0, 0.0, 0.0),
)


assert np.allclose(
    np.abs(
        Jx_displaced
    ),
    1.0,
)


assert not np.allclose(
    Jx_displaced,
    Jx,
)


passed(
    "dipole displacement produces phase without changing magnitude"
)


# ============================================================
# TEST 3
#
# SOURCE JUMP
# ============================================================

section(
    "TEST 3 — DIPOLE SOURCE JUMP"
)


b = dipole_source_jump_xy(
    Jx,
    Jy,
)


assert b.shape == (
    4 * N_G,
)


assert np.allclose(
    b[
        0:N_G
    ],
    0.0,
)


assert np.allclose(
    b[
        N_G:2 * N_G
    ],
    0.0,
)


assert np.allclose(
    b[
        2 * N_G:3 * N_G
    ],
    0.0,
)


assert np.allclose(
    b[
        3 * N_G:4 * N_G
    ],
    1j * Jx,
)


passed(
    "x-dipole source jump matches Khepri convention"
)


# ============================================================
# TEST 4
#
# VACUUM SOURCE PLANE
# ============================================================

section(
    "TEST 4 — HOMOGENEOUS VACUUM"
)


vacuum = solve_test_case(
    epsilon_slab=1.0,
)


# ------------------------------------------------------------
# Correct global stack
# ------------------------------------------------------------

assert vacuum[
    "crystal"
].global_stacking == [
    "Sref",
    "left_half",
    "right_half",
    "Strans",
]


passed(
    "vacuum stack constructed correctly"
)


# ------------------------------------------------------------
# Maxwell jump
# ------------------------------------------------------------

jump_error = np.linalg.norm(
    vacuum["psi_right"]
    -
    vacuum["psi_left"]
    -
    vacuum["b"]
) / np.linalg.norm(
    vacuum["b"]
)


print(
    "Jump relative error:",
    jump_error,
)


assert jump_error < TOL


passed(
    "vacuum Maxwell jump condition"
)


# ------------------------------------------------------------
# No reflections
# ------------------------------------------------------------

R_left_norm = np.linalg.norm(
    vacuum["Sl"][
        1,
        1
    ]
)

R_right_norm = np.linalg.norm(
    vacuum["Sr"][
        0,
        0
    ]
)


print(
    "Vacuum left reflection norm:",
    R_left_norm,
)

print(
    "Vacuum right reflection norm:",
    R_right_norm,
)


assert R_left_norm < TOL_ZERO

assert R_right_norm < TOL_ZERO


passed(
    "vacuum has zero reflections"
)


# ------------------------------------------------------------
# Modal symmetry
# ------------------------------------------------------------

modal_error = relative_difference(
    np.linalg.norm(
        vacuum["c_plus"]
    ),
    np.linalg.norm(
        vacuum["c_minus"]
    ),
)


assert modal_error < TOL


passed(
    "vacuum modal emission is symmetric"
)


# ------------------------------------------------------------
# Near-field power symmetry
# ------------------------------------------------------------

near_error = relative_difference(
    vacuum["P_near_left"],
    vacuum["P_near_right"],
)


assert near_error < TOL


passed(
    "vacuum near-field power is symmetric"
)


# ------------------------------------------------------------
# Exterior power symmetry
# ------------------------------------------------------------

ext_error = relative_difference(
    vacuum["P_ext_left"],
    vacuum["P_ext_right"],
)


assert ext_error < TOL


passed(
    "vacuum exterior power is symmetric"
)


# ------------------------------------------------------------
# Near power = exterior power
# ------------------------------------------------------------

energy_error = relative_difference(
    vacuum["P_near_total"],
    vacuum["P_ext_total"],
)


print(
    "Vacuum near/exterior power error:",
    energy_error,
)


assert energy_error < TOL_POWER


passed(
    "vacuum near-field and exterior powers agree"
)


# ============================================================
# TEST 5
#
# DIELECTRIC SLAB
# ============================================================

section(
    "TEST 5 — DIELECTRIC SLAB"
)


slab_x = solve_test_case(
    epsilon_slab=EPSILON_SLAB,
    current=(
        1.0,
        0.0,
        0.0,
    ),
)


# ------------------------------------------------------------
# Maxwell jump
# ------------------------------------------------------------

jump_error = np.linalg.norm(
    slab_x["psi_right"]
    -
    slab_x["psi_left"]
    -
    slab_x["b"]
) / np.linalg.norm(
    slab_x["b"]
)


print(
    "Jump relative error:",
    jump_error,
)


assert jump_error < TOL


passed(
    "dielectric Maxwell jump condition"
)


# ------------------------------------------------------------
# Nonzero reflections
# ------------------------------------------------------------

R_left_norm = np.linalg.norm(
    slab_x["Sl"][
        1,
        1
    ]
)

R_right_norm = np.linalg.norm(
    slab_x["Sr"][
        0,
        0
    ]
)


print(
    "Left reflection norm:",
    R_left_norm,
)

print(
    "Right reflection norm:",
    R_right_norm,
)


assert R_left_norm > TOL_ZERO

assert R_right_norm > TOL_ZERO


passed(
    "dielectric interfaces produce nonzero reflection"
)


# ------------------------------------------------------------
# Reflection symmetry
# ------------------------------------------------------------

reflection_error = relative_difference(
    R_left_norm,
    R_right_norm,
)


assert reflection_error < TOL


passed(
    "left/right reflection strengths are symmetric"
)


# ------------------------------------------------------------
# Modal symmetry
# ------------------------------------------------------------

modal_error = relative_difference(
    np.linalg.norm(
        slab_x["c_plus"]
    ),
    np.linalg.norm(
        slab_x["c_minus"]
    ),
)


assert modal_error < TOL


passed(
    "dielectric modal emission symmetry"
)


# ------------------------------------------------------------
# Near-field symmetry
# ------------------------------------------------------------

near_error = relative_difference(
    slab_x["P_near_left"],
    slab_x["P_near_right"],
)


assert near_error < TOL


passed(
    "dielectric near-field power symmetry"
)


# ------------------------------------------------------------
# Exterior symmetry
# ------------------------------------------------------------

ext_error = relative_difference(
    slab_x["P_ext_left"],
    slab_x["P_ext_right"],
)


assert ext_error < TOL


passed(
    "dielectric exterior power symmetry"
)


# ------------------------------------------------------------
# Energy conservation
# ------------------------------------------------------------

energy_error = relative_difference(
    slab_x["P_near_total"],
    slab_x["P_ext_total"],
)


print(
    "\nP_near_total =",
    slab_x["P_near_total"],
)

print(
    "P_ext_total  =",
    slab_x["P_ext_total"],
)

print(
    "Energy-conservation error =",
    energy_error,
)


assert energy_error < TOL_POWER


passed(
    "lossless slab conserves emitted power"
)


# ============================================================
# TEST 6
#
# DIFFRACTION ORDERS
# ============================================================

section(
    "TEST 6 — DIFFRACTION ORDERS"
)


left_orders = slab_x[
    "P_ext_left_orders"
]

right_orders = slab_x[
    "P_ext_right_orders"
]


print(
    "Left power orders:"
)

print(
    left_orders
)


print(
    "Right power orders:"
)

print(
    right_orders
)


propagating_left = np.where(
    left_orders
    >
    TOL_ZERO
)[0]


propagating_right = np.where(
    right_orders
    >
    TOL_ZERO
)[0]


central_order = (
    N_G - 1
) // 2


assert len(
    propagating_left
) == 1


assert len(
    propagating_right
) == 1


assert propagating_left[
    0
] == central_order


assert propagating_right[
    0
] == central_order


passed(
    "only the zero diffraction order propagates"
)


# ============================================================
# TEST 7
#
# x/y ROTATIONAL SYMMETRY
# ============================================================

section(
    "TEST 7 — x/y DIPOLE SYMMETRY"
)


slab_y = solve_test_case(
    epsilon_slab=EPSILON_SLAB,
    current=(
        0.0,
        1.0,
        0.0,
    ),
)


xy_near_error = relative_difference(
    slab_x["P_near_total"],
    slab_y["P_near_total"],
)


xy_ext_error = relative_difference(
    slab_x["P_ext_total"],
    slab_y["P_ext_total"],
)


print(
    "x/y near-field power error:",
    xy_near_error,
)

print(
    "x/y exterior power error:",
    xy_ext_error,
)


assert xy_near_error < TOL

assert xy_ext_error < TOL


passed(
    "uniform slab gives identical x/y dipole emission"
)


# ============================================================
# TEST 8
#
# IN-PLANE POSITION INVARIANCE
# ============================================================

section(
    "TEST 8 — IN-PLANE POSITION INVARIANCE"
)


slab_displaced = solve_test_case(
    epsilon_slab=EPSILON_SLAB,
    current=(
        1.0,
        0.0,
        0.0,
    ),
    position=(
        0.2,
        0.3,
        HALF_THICKNESS,
    ),
)


position_near_error = relative_difference(
    slab_x["P_near_total"],
    slab_displaced["P_near_total"],
)


position_ext_error = relative_difference(
    slab_x["P_ext_total"],
    slab_displaced["P_ext_total"],
)


print(
    "Position near-field error:",
    position_near_error,
)

print(
    "Position exterior error:",
    position_ext_error,
)


assert position_near_error < TOL

assert position_ext_error < TOL


passed(
    "homogeneous in-plane structure is position invariant"
)


# ============================================================
# TEST 9
#
# CURRENT AMPLITUDE SCALING
#
# Field ∝ J
# Power ∝ |J|^2
#
# doubling J must multiply power by 4.
# ============================================================

section(
    "TEST 9 — CURRENT AMPLITUDE SCALING"
)


slab_double = solve_test_case(
    epsilon_slab=EPSILON_SLAB,
    current=(
        2.0,
        0.0,
        0.0,
    ),
)


near_ratio = (
    slab_double["P_near_total"]
    /
    slab_x["P_near_total"]
)


ext_ratio = (
    slab_double["P_ext_total"]
    /
    slab_x["P_ext_total"]
)


print(
    "Near-field power ratio:",
    near_ratio,
)

print(
    "Exterior power ratio:",
    ext_ratio,
)


assert np.isclose(
    near_ratio,
    4.0,
    rtol=TOL,
    atol=TOL,
)


assert np.isclose(
    ext_ratio,
    4.0,
    rtol=TOL,
    atol=TOL,
)


passed(
    "power scales as |J|^2"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

section(
    "ALL DIPOLE TESTS PASSED"
)


print(
    "\nVerified:"
)

print(
    "  Khepri convention: U = -i H"
)

print(
    "  Fourier dipole phase"
)

print(
    "  x/y electric-current source jump"
)

print(
    "  source-plane modal solution"
)

print(
    "  Maxwell jump condition"
)

print(
    "  left/right multiple reflections"
)

print(
    "  vacuum reflection limit"
)

print(
    "  dielectric reflection symmetry"
)

print(
    "  exterior radiated flux"
)

print(
    "  near-source Poynting flux"
)

print(
    "  lossless power conservation"
)

print(
    "  diffraction-order selection"
)

print(
    "  x/y rotational symmetry"
)

print(
    "  in-plane translation invariance"
)

print(
    "  P proportional to |J|^2"
)


print(
    "\nReference dielectric-slab result:"
)

print(
    "P_near_total =",
    slab_x["P_near_total"],
)

print(
    "P_ext_total  =",
    slab_x["P_ext_total"],
)


print(
    "\nThe validated implementation currently supports "
    "in-plane x/y electric dipoles."
)

print(
    "A z-oriented dipole has NOT yet been implemented."
)