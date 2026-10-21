import numpy as np

from khepri.crystal import Crystal
from khepri.expansion import Expansion

from khepri.dipole import (
    dipole_power_map,
)


# ============================================================
# GLOBAL SETTINGS
# ============================================================

np.set_printoptions(
    precision=12,
    suppress=True,
)


# ============================================================
# PHYSICAL PARAMETERS
# ============================================================

a = 1.0

wavelength = 2.0

k0 = (
    2.0
    *
    np.pi
    /
    wavelength
)


epsilon_ref = 2.25

n_ref = np.sqrt(
    epsilon_ref
)


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
# SOLVER GRID
# ============================================================
#
# We deliberately use a modest grid for the RCWA validation.
#
# Nq = 7:
#
#   - contains Gamma exactly
#   - avoids an exact grazing/light-cone point
#   - requires only 49 solves per plane-wave truncation
#
# We will compare every RCWA point against an independent
# analytic homogeneous-medium result.
# ============================================================

Nq = 7


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


center = (
    Nq
    //
    2
)


# ============================================================
# TOLERANCES
# ============================================================

TOL_POINTWISE = 1e-8

TOL_PW = 1e-8

TOL_SYMMETRY = 1e-9

TOL_DISCRETE_INTEGRAL = 1e-8

TOL_COARSE_CONTINUUM = 5e-2

GRAZING_TOL = 1e-10


# ============================================================
# PRINTING HELPERS
# ============================================================

def section(title):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


def passed(message):

    print(
        f"[PASS] {message}"
    )


def global_relative_error(
    calculated,
    reference,
):
    """
    Stable global error.

    We do NOT calculate pointwise relative error because
    quantities close to zero would produce misleadingly
    enormous relative errors.
    """

    calculated = np.asarray(
        calculated
    )

    reference = np.asarray(
        reference
    )

    difference = np.abs(
        calculated
        -
        reference
    )

    scale = max(
        np.max(
            np.abs(
                reference
            )
        ),
        1e-30,
    )

    return (
        np.max(
            difference
        )
        /
        scale
    )


# ============================================================
# BUILD HOMOGENEOUS REFERENCE CRYSTAL
# ============================================================

def build_reference_crystal(
    pw,
):
    """
    Construct an effectively infinite homogeneous medium.

    All regions have exactly the same permittivity:

        epsilon_ref
             |
        epsilon_ref
             |
          DIPOLE
             |
        epsilon_ref
             |
        epsilon_ref

    The two finite half-layers merely give us an artificial
    source plane. There is no physical dielectric interface.
    """

    cl = Crystal(
        pw=pw,
        lattice="square",
        lattice_pitch=a,

        epsi=epsilon_ref,
        epse=epsilon_ref,
    )


    cl.add_layer_uniform(
        "left_half",
        epsilon=epsilon_ref,
        depth=half_thickness,
    )


    cl.add_layer_uniform(
        "right_half",
        epsilon=epsilon_ref,
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


    return cl


# ============================================================
# ANALYTIC HOMOGENEOUS q-RESOLVED POWER
# ============================================================

def analytic_homogeneous_power_q(
    expansion,
    kp,
    wavelength,
    epsilon,
    current,
):
    """
    Analytic emitted power for an in-plane electric-current
    sheet in an infinite homogeneous dielectric.

    This is written in the same normalized electromagnetic
    convention used by our Khepri dipole implementation.

    For each physical in-plane wavevector

        K_parallel = q + G,

    decompose the current into:

        J_parallel
        J_perpendicular

    relative to K_parallel.

    For a propagating harmonic:

        P_TE =
            |J_perpendicular|^2
            -------------------
                  2 kz

        P_TM =
            |J_parallel|^2 kz
            ------------------
                 2 epsilon

    where

        kz = sqrt(
            epsilon
            -
            |K_parallel / k0|^2
        )

    Evanescent harmonics carry zero net power in a lossless
    homogeneous medium.
    """

    qx_value, qy_value = kp

    jx, jy, jz = current


    if abs(
        jz
    ) > 1e-14:

        raise NotImplementedError(
            "This analytic test currently supports "
            "only in-plane dipoles."
        )


    k0_local = (
        2.0
        *
        np.pi
        /
        wavelength
    )


    g = np.asarray(
        expansion.g_vectors
    )


    Gx = g[0]

    Gy = g[1]


    # --------------------------------------------------------
    # Physical in-plane wavevectors
    # --------------------------------------------------------

    Kx = (
        qx_value
        +
        Gx
    )


    Ky = (
        qy_value
        +
        Gy
    )


    # --------------------------------------------------------
    # Normalize by k0
    # --------------------------------------------------------

    px = (
        Kx
        /
        k0_local
    )


    py = (
        Ky
        /
        k0_local
    )


    p2 = (
        px**2
        +
        py**2
    )


    # --------------------------------------------------------
    # Make sure the test grid does not accidentally evaluate
    # exactly on a light-cone / grazing condition.
    # --------------------------------------------------------

    grazing_distance = np.abs(
        epsilon
        -
        p2
    )


    if np.any(
        grazing_distance
        <
        GRAZING_TOL
    ):

        indices = np.where(
            grazing_distance
            <
            GRAZING_TOL
        )[0]

        raise RuntimeError(
            "Analytic validation grid encountered a grazing "
            "diffraction order at indices "
            f"{indices.tolist()}."
        )


    propagating = (
        p2
        <
        epsilon
    )


    power = np.zeros_like(
        p2,
        dtype=float,
    )


    # --------------------------------------------------------
    # Special case K_parallel = 0
    #
    # At normal incidence TE/TM are degenerate:
    #
    #     P_total =
    #     (|Jx|^2 + |Jy|^2)/(2 n)
    # --------------------------------------------------------

    normal = (
        propagating
        &
        (
            p2
            <
            1e-14
        )
    )


    n = np.sqrt(
        epsilon
    )


    power[
        normal
    ] = (
        (
            abs(jx)**2
            +
            abs(jy)**2
        )
        /
        (
            2.0
            *
            n
        )
    )


    # --------------------------------------------------------
    # General propagating harmonics
    # --------------------------------------------------------

    general = (
        propagating
        &
        (
            p2
            >=
            1e-14
        )
    )


    if np.any(
        general
    ):

        p = np.sqrt(
            p2[
                general
            ]
        )


        kz = np.sqrt(
            epsilon
            -
            p2[
                general
            ]
        )


        # ----------------------------------------------------
        # Current parallel to K_parallel
        # ----------------------------------------------------

        J_parallel = (
            jx
            *
            px[
                general
            ]
            +
            jy
            *
            py[
                general
            ]
        ) / p


        # ----------------------------------------------------
        # Current perpendicular to K_parallel
        #
        # This is the TE direction in the plane.
        # ----------------------------------------------------

        J_perpendicular = (
            -jx
            *
            py[
                general
            ]
            +
            jy
            *
            px[
                general
            ]
        ) / p


        P_TE = (
            np.abs(
                J_perpendicular
            )**2
            /
            (
                2.0
                *
                kz
            )
        )


        P_TM = (
            np.abs(
                J_parallel
            )**2
            *
            kz
            /
            (
                2.0
                *
                epsilon
            )
        )


        power[
            general
        ] = (
            P_TE
            +
            P_TM
        )


    # --------------------------------------------------------
    # Sum over reciprocal-lattice harmonics
    # --------------------------------------------------------

    return float(
        np.sum(
            power
        )
    )


# ============================================================
# ANALYTIC q MAP
# ============================================================

def analytic_power_map(
    expansion,
    qx_values,
    qy_values,
    wavelength,
    epsilon,
    current,
):

    P = np.zeros(
        (
            len(qy_values),
            len(qx_values),
        ),
        dtype=float,
    )


    for iy, qy_value in enumerate(
        qy_values
    ):

        for ix, qx_value in enumerate(
            qx_values
        ):

            P[
                iy,
                ix
            ] = analytic_homogeneous_power_q(
                expansion=expansion,

                kp=(
                    qx_value,
                    qy_value,
                ),

                wavelength=wavelength,

                epsilon=epsilon,

                current=current,
            )


    return P


# ============================================================
# EXACT CONTINUUM HOMOGENEOUS REFERENCE
# ============================================================

def analytic_reference_integral(
    wavelength,
    refractive_index,
    current,
):
    """
    Exact FBZ/full-momentum-space reference integral for an
    in-plane dipole in an infinite homogeneous dielectric.

    In the normalization used by our Khepri implementation:

        I_ref =
            k0^2 n
            --------
              6 pi

            * (|Jx|^2 + |Jy|^2)
    """

    jx, jy, jz = current


    if abs(
        jz
    ) > 1e-14:

        raise NotImplementedError(
            "Only in-plane dipoles are supported."
        )


    k0_local = (
        2.0
        *
        np.pi
        /
        wavelength
    )


    return (
        k0_local**2
        *
        refractive_index
        /
        (
            6.0
            *
            np.pi
        )
        *
        (
            abs(jx)**2
            +
            abs(jy)**2
        )
    )


# ============================================================
# TEST 1 — BASIC GRID
# ============================================================

section(
    "TEST 1 — REFERENCE q GRID"
)


print(
    "\nNq =",
    Nq,
)


print(
    "dq =",
    dq,
)


print(
    "qx ="
)

print(
    qx
)


assert np.isclose(
    qx[
        center
    ],
    0.0,
)


assert np.isclose(
    qy[
        center
    ],
    0.0,
)


passed(
    "Gamma is included exactly"
)


# ============================================================
# TEST 2 — EXACT HOMOGENEOUS GAMMA POWER
# ============================================================

section(
    "TEST 2 — HOMOGENEOUS GAMMA-POINT POWER"
)


expected_gamma = (
    (
        abs(
            current[0]
        )**2
        +
        abs(
            current[1]
        )**2
    )
    /
    (
        2.0
        *
        n_ref
    )
)


print(
    "\nAnalytic P(Gamma) =",
    expected_gamma,
)


# ============================================================
# TEST 3 — RCWA MAP WITH pw=(3,3)
# ============================================================

section(
    "TEST 3 — KHEPRI HOMOGENEOUS MAP, pw=(3,3)"
)


pw3 = (
    3,
    3,
)


cl3 = build_reference_crystal(
    pw3
)


result3 = dipole_power_map(
    crystal=cl3,

    source_after="left_half",

    wavelength=wavelength,

    qx_values=qx,

    qy_values=qy,

    position=position,

    current=current,

    backend="dense",

    verbose=False,
)


P3 = result3[
    "P_total"
]


P3_left = result3[
    "P_left"
]


P3_right = result3[
    "P_right"
]


print(
    "\nKhepri P(Gamma) =",
    P3[
        center,
        center
    ],
)


gamma_error = abs(
    P3[
        center,
        center
    ]
    -
    expected_gamma
) / expected_gamma


print(
    "Gamma relative error =",
    gamma_error,
)


assert gamma_error < TOL_POINTWISE


passed(
    "Khepri Gamma-point power matches analytic result"
)


# ============================================================
# TEST 4 — ANALYTIC FOLDED q MAP
# ============================================================

section(
    "TEST 4 — ANALYTIC q-RESOLVED HOMOGENEOUS POWER"
)


P3_analytic = analytic_power_map(
    expansion=cl3.expansion,

    qx_values=qx,

    qy_values=qy,

    wavelength=wavelength,

    epsilon=epsilon_ref,

    current=current,
)


map_error = global_relative_error(
    P3,
    P3_analytic,
)


difference = np.abs(
    P3
    -
    P3_analytic
)


worst = np.unravel_index(
    np.argmax(
        difference
    ),
    difference.shape,
)


print(
    "\nMaximum analytic power =",
    np.max(
        P3_analytic
    ),
)


print(
    "Maximum absolute RCWA/analytic difference =",
    np.max(
        difference
    ),
)


print(
    "Global normalized error =",
    map_error,
)


print(
    "\nWorst point:"
)


print(
    "qx =",
    qx[
        worst[1]
    ],
)


print(
    "qy =",
    qy[
        worst[0]
    ],
)


print(
    "Khepri =",
    P3[
        worst
    ],
)


print(
    "Analytic =",
    P3_analytic[
        worst
    ],
)


assert map_error < TOL_POINTWISE, (
    "\nFAILED:\n"
    "The Khepri homogeneous q-resolved dipole power does "
    "not agree with the independent analytic solution."
)


passed(
    "Khepri q-resolved power agrees with analytic Maxwell solution"
)


# ============================================================
# TEST 5 — LEFT / RIGHT SYMMETRY
# ============================================================

section(
    "TEST 5 — HOMOGENEOUS LEFT / RIGHT SYMMETRY"
)


lr_error = global_relative_error(
    P3_left,
    P3_right,
)


print(
    "\nGlobal left/right error =",
    lr_error,
)


assert lr_error < TOL_SYMMETRY


passed(
    "homogeneous dipole emits symmetrically toward +/- z"
)


# ============================================================
# TEST 6 — DISCRETE BZ INTEGRAL
# ============================================================

section(
    "TEST 6 — DISCRETE BZ INTEGRAL"
)


weight = (
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


I_rcwa = (
    np.sum(
        P3
    )
    *
    weight
)


I_analytic_discrete = (
    np.sum(
        P3_analytic
    )
    *
    weight
)


discrete_integral_error = abs(
    I_rcwa
    -
    I_analytic_discrete
) / max(
    abs(
        I_analytic_discrete
    ),
    1e-30,
)


print(
    "\nRCWA discrete integral =",
    I_rcwa,
)


print(
    "Analytic discrete integral =",
    I_analytic_discrete,
)


print(
    "Relative difference =",
    discrete_integral_error,
)


assert (
    discrete_integral_error
    <
    TOL_DISCRETE_INTEGRAL
)


passed(
    "RCWA and analytic midpoint integrals agree"
)


# ============================================================
# TEST 7 — EXACT CONTINUUM REFERENCE
# ============================================================

section(
    "TEST 7 — EXACT HOMOGENEOUS REFERENCE"
)


I_exact = analytic_reference_integral(
    wavelength=wavelength,

    refractive_index=n_ref,

    current=current,
)


expected_specific_value = (
    np.pi
    /
    4.0
)


print(
    "\nExact homogeneous reference =",
    I_exact,
)


print(
    "Expected for this test = pi/4 =",
    expected_specific_value,
)


assert np.isclose(
    I_exact,
    expected_specific_value,
    rtol=1e-14,
    atol=1e-14,
)


passed(
    "analytic reference gives pi/4 for n=1.5, lambda=2"
)


# ------------------------------------------------------------
# The coarse Nq=7 midpoint integral will not be exact because
# the homogeneous emissivity has an integrable light-cone
# singularity.
#
# We merely require this deliberately coarse quadrature to be
# reasonably close.
# ------------------------------------------------------------

coarse_quadrature_error = abs(
    I_analytic_discrete
    -
    I_exact
) / I_exact


print(
    "\nNq=7 midpoint integral =",
    I_analytic_discrete,
)


print(
    "Exact continuum integral =",
    I_exact,
)


print(
    "Coarse quadrature relative error =",
    coarse_quadrature_error,
)


assert (
    coarse_quadrature_error
    <
    TOL_COARSE_CONTINUUM
)


passed(
    "coarse midpoint integral is reasonably close to exact reference"
)


# ============================================================
# TEST 8 — PLANE-WAVE TRUNCATION pw=(5,5)
# ============================================================

section(
    "TEST 8 — PLANE-WAVE TRUNCATION"
)


pw5 = (
    5,
    5,
)


cl5 = build_reference_crystal(
    pw5
)


result5 = dipole_power_map(
    crystal=cl5,

    source_after="left_half",

    wavelength=wavelength,

    qx_values=qx,

    qy_values=qy,

    position=position,

    current=current,

    backend="dense",

    verbose=False,
)


P5 = result5[
    "P_total"
]


pw_error = global_relative_error(
    P5,
    P3,
)


print(
    "\nMaximum pw=(3,3) power =",
    np.max(
        P3
    ),
)


print(
    "Maximum pw=(5,5) power =",
    np.max(
        P5
    ),
)


print(
    "Global pw=(3,3) vs pw=(5,5) error =",
    pw_error,
)


assert pw_error < TOL_PW, (
    "\nFAILED:\n"
    "The homogeneous reference has not converged with "
    "respect to Fourier truncation."
)


passed(
    "pw=(3,3) and pw=(5,5) give the same emitted power"
)


# ============================================================
# TEST 9 — ANALYTIC HARMONIC COVERAGE
# ============================================================
#
# A 1x1 truncation contains only G=(0,0).
#
# For this particular case:
#
#     n k0 = 1.5*pi
#
# which extends outside the first square BZ.
#
# Therefore pw=(1,1) cannot reconstruct the entire physical
# homogeneous light cone.
#
# pw=(3,3) includes neighboring reciprocal cells and should
# contain all propagating harmonics.
# ============================================================

section(
    "TEST 9 — RECIPROCAL-LATTICE HARMONIC COVERAGE"
)


lattice = np.array(
    [
        [
            a,
            0.0,
        ],
        [
            0.0,
            a,
        ],
    ]
)


expansion1 = Expansion(
    (
        1,
        1,
    ),
    lattice,
)


P1_analytic = analytic_power_map(
    expansion=expansion1,

    qx_values=qx,

    qy_values=qy,

    wavelength=wavelength,

    epsilon=epsilon_ref,

    current=current,
)


I1 = (
    np.sum(
        P1_analytic
    )
    *
    weight
)


print(
    "\npw=(1,1) discrete reference =",
    I1,
)


print(
    "pw=(3,3) discrete reference =",
    I_analytic_discrete,
)


print(
    "Exact continuum reference =",
    I_exact,
)


assert (
    I1
    <
    I_analytic_discrete
)


passed(
    "pw=(1,1) misses folded propagating momentum states"
)


# ------------------------------------------------------------
# Analytic 3x3 and 5x5 maps should be identical because all
# additional harmonics in 5x5 lie outside the physical light
# cone for this example.
# ------------------------------------------------------------

P5_analytic = analytic_power_map(
    expansion=cl5.expansion,

    qx_values=qx,

    qy_values=qy,

    wavelength=wavelength,

    epsilon=epsilon_ref,

    current=current,
)


analytic_pw_error = global_relative_error(
    P5_analytic,
    P3_analytic,
)


print(
    "\nAnalytic 3x3 vs 5x5 error =",
    analytic_pw_error,
)


assert analytic_pw_error < 1e-14


passed(
    "3x3 contains all propagating harmonics for this reference"
)


# ============================================================
# TEST 10 — CURRENT AMPLITUDE NORMALIZATION
# ============================================================
#
# Exact homogeneous reference must scale as |J|^2.
# ============================================================

section(
    "TEST 10 — REFERENCE CURRENT SCALING"
)


I_double_current = analytic_reference_integral(
    wavelength=wavelength,

    refractive_index=n_ref,

    current=(
        2.0,
        0.0,
        0.0,
    ),
)


current_ratio = (
    I_double_current
    /
    I_exact
)


print(
    "\nI(J=2) / I(J=1) =",
    current_ratio,
)


assert np.isclose(
    current_ratio,
    4.0,
    rtol=1e-14,
    atol=1e-14,
)


passed(
    "homogeneous reference scales as |J|^2"
)


# ============================================================
# TEST 11 — x/y ISOTROPY
# ============================================================

section(
    "TEST 11 — HOMOGENEOUS x/y ISOTROPY"
)


I_y = analytic_reference_integral(
    wavelength=wavelength,

    refractive_index=n_ref,

    current=(
        0.0,
        1.0,
        0.0,
    ),
)


xy_error = abs(
    I_y
    -
    I_exact
) / I_exact


print(
    "\nIx =",
    I_exact,
)


print(
    "Iy =",
    I_y,
)


print(
    "x/y error =",
    xy_error,
)


assert xy_error < 1e-14


passed(
    "homogeneous reference is isotropic for x/y dipoles"
)


# ============================================================
# OPTIONAL — SHOW WHY NAIVE MIDPOINT 'CONVERGENCE'
# CAN OSCILLATE
# ============================================================
#
# This section uses ONLY the analytic formula, so it is cheap.
#
# Near the light cone the TE contribution behaves like 1/kz.
# It is integrable, but a midpoint grid can land unusually
# close to the singular boundary. Therefore successive Nq
# values need not approach the exact answer monotonically.
# ============================================================

section(
    "MIDPOINT QUADRATURE DIAGNOSTIC"
)


diagnostic_Nq = [
    7,
    9,
    11,
    13,
    17,
    21,
    31,
    41,
    61,
    81,
    101,
]


for N_test in diagnostic_Nq:

    dq_test = (
        qmax
        -
        qmin
    ) / N_test


    q_test = (
        qmin
        +
        (
            np.arange(
                N_test
            )
            +
            0.5
        )
        *
        dq_test
    )


    P_test = analytic_power_map(
        expansion=cl3.expansion,

        qx_values=q_test,

        qy_values=q_test,

        wavelength=wavelength,

        epsilon=epsilon_ref,

        current=current,
    )


    I_test = (
        np.sum(
            P_test
        )
        *
        dq_test
        *
        dq_test
        /
        (
            2.0
            *
            np.pi
        )**2
    )


    error_percent = (
        100.0
        *
        (
            I_test
            -
            I_exact
        )
        /
        I_exact
    )


    print(
        f"Nq={N_test:3d}   "
        f"I={I_test:.12f}   "
        f"error={error_percent:+.4f}%"
    )


print(
    "\nNOTE:"
)


print(
    "The oscillation above is expected for naive midpoint "
    "sampling near the homogeneous light-cone singularity."
)


print(
    "Therefore the exact analytic homogeneous reference is "
    "preferable to treating one arbitrary dense q grid as "
    "the normalization standard."
)


# ============================================================
# FINAL SUMMARY
# ============================================================

section(
    "ALL HOMOGENEOUS REFERENCE TESTS PASSED"
)


print(
    "\nValidated:"
)


print(
    "  Khepri homogeneous Gamma-point emission"
)


print(
    "  q-resolved RCWA power against analytic Maxwell solution"
)


print(
    "  left/right emission symmetry"
)


print(
    "  FBZ integration weight"
)


print(
    "  3x3 vs 5x5 plane-wave convergence"
)


print(
    "  reciprocal-lattice folding of homogeneous momentum space"
)


print(
    "  current-amplitude scaling"
)


print(
    "  x/y isotropy"
)


print(
    "\nExact homogeneous reference:"
)


print(
    "I_ref =",
    I_exact,
)


print(
    "\nFor this specific case:"
)


print(
    "I_ref = pi/4 =",
    np.pi
    /
    4.0,
)


print(
    "\nThis is the denominator that can be used for the "
    "next Purcell-factor validation for an in-plane unit "
    "current in n=1.5 at wavelength=2."
)