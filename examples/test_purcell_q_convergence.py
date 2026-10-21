import numpy as np
import matplotlib.pyplot as plt

from khepri.crystal import Crystal

from khepri.dipole import (
    dipole_power_map,
    homogeneous_reference_power_xy,
)


# ============================================================
# 1. PHYSICAL PARAMETERS
# ============================================================

a = 1.0

wavelength = 2.0

epsilon_air = 1.0

epsilon_slab = 2.25

n_reference = 1.5

half_thickness = 0.5


# ============================================================
# 2. DIPOLE PARAMETERS
# ============================================================

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
# 3. PLANE-WAVE TRUNCATIONS TO TEST
# ============================================================
#
# The purpose of this test is to change ONLY pw.
#
# Everything else stays fixed.
#
# ============================================================

PW_VALUES = [
    (3, 3),
    (5, 5),
]


# ============================================================
# 4. FIXED q-GRID
# ============================================================
#
# We already established that the shift-averaged q integration
# is well behaved.
#
# Nq = 31 is used here to keep the calculation reasonable.
#
# The same q grids are used for every pw value.
#
# ============================================================

Nq = 31


# ============================================================
# 5. SHIFTED q GRIDS
# ============================================================
#
# These are the same non-special shifts used in the previous
# convergence test.
#
# They avoid exact grazing / Rayleigh conditions.
#
# ============================================================

SHIFTS = [
    (0.00, 0.00),
    (0.23, 0.17),
    (0.41, 0.63),
    (0.77, 0.31),
]


# ============================================================
# 6. PLANE-WAVE CONVERGENCE CRITERION
# ============================================================
#
# Require the Purcell factors from consecutive pw values to
# agree within 0.5 percent.
#
# For a laterally uniform slab we actually expect much better
# agreement than this.
#
# ============================================================

PW_TOLERANCE_PERCENT = 0.5


# ============================================================
# 7. LIGHT-LINE TOLERANCE
# ============================================================

LIGHT_LINE_TOLERANCE = 1e-10


# ============================================================
# HELPER: SECTION HEADER
# ============================================================

def section(title):

    print("\n")
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# HELPER: BUILD SHIFTED q GRID
# ============================================================

def make_shifted_q_grid(
    Nq,
    shift_x,
    shift_y,
):
    """
    Construct a periodically shifted midpoint grid over

        [-pi/a, +pi/a)

    in qx and qy.

    shift_x and shift_y are fractions of one grid spacing.
    """

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


    # --------------------------------------------------------
    # Shifted midpoint coordinates
    # --------------------------------------------------------

    ix = (
        np.arange(Nq)
        +
        0.5
        +
        shift_x
    ) % Nq


    iy = (
        np.arange(Nq)
        +
        0.5
        +
        shift_y
    ) % Nq


    qx = (
        qmin
        +
        ix
        *
        dq
    )


    qy = (
        qmin
        +
        iy
        *
        dq
    )


    # --------------------------------------------------------
    # Sort after periodic wrapping
    # --------------------------------------------------------

    qx = np.sort(
        qx
    )

    qy = np.sort(
        qy
    )


    return (
        qx,
        qy,
        dq,
    )


# ============================================================
# HELPER: CHECK DISTANCE FROM EXTERIOR LIGHT LINE
# ============================================================

def diagnose_light_line(
    qx,
    qy,
):
    """
    Determine the minimum distance of the central diffraction
    order from

        |q| = k0.
    """

    k0 = (
        2.0
        *
        np.pi
        /
        wavelength
    )


    QX, QY = np.meshgrid(
        qx,
        qy,
    )


    q_radius = np.sqrt(
        QX**2
        +
        QY**2
    )


    distance = np.abs(
        q_radius
        -
        k0
    )


    index = np.unravel_index(
        np.argmin(
            distance
        ),
        distance.shape,
    )


    minimum_distance = float(
        distance[
            index
        ]
    )


    closest_qx = float(
        QX[
            index
        ]
    )


    closest_qy = float(
        QY[
            index
        ]
    )


    return (
        minimum_distance,
        closest_qx,
        closest_qy,
    )


# ============================================================
# HELPER: CHECK FOR NaN / INF
# ============================================================

def check_finite_power_map(
    P_map,
    qx,
    qy,
):
    """
    Stop with a useful diagnostic if NaN or inf occurs.
    """

    finite_mask = np.isfinite(
        P_map
    )


    if np.all(
        finite_mask
    ):

        return


    print("\n")
    print("!" * 78)
    print("NON-FINITE q-SPACE POWER DETECTED")
    print("!" * 78)


    bad_indices = np.argwhere(
        ~finite_mask
    )


    k0 = (
        2.0
        *
        np.pi
        /
        wavelength
    )


    print(
        "\nNumber of bad q points =",
        len(
            bad_indices
        ),
    )


    print(
        "k0 =",
        k0,
    )


    for bad_index in bad_indices[
        :20
    ]:

        iy = int(
            bad_index[0]
        )

        ix = int(
            bad_index[1]
        )


        qx_bad = float(
            qx[
                ix
            ]
        )


        qy_bad = float(
            qy[
                iy
            ]
        )


        q_abs = float(
            np.sqrt(
                qx_bad**2
                +
                qy_bad**2
            )
        )


        print(
            f"ix={ix:4d}, "
            f"iy={iy:4d}, "
            f"qx={qx_bad:+.12f}, "
            f"qy={qy_bad:+.12f}, "
            f"|q|={q_abs:.12f}, "
            f"|q|-k0={q_abs-k0:+.12e}"
        )


    raise RuntimeError(
        "Non-finite q-space power encountered."
    )


# ============================================================
# 8. HOMOGENEOUS REFERENCE
# ============================================================

section(
    "HOMOGENEOUS REFERENCE"
)


P_reference = homogeneous_reference_power_xy(
    wavelength=wavelength,
    refractive_index=n_reference,
    current=current,
)


expected_reference = (
    np.pi
    /
    4.0
)


print(
    "\nCalculated reference =",
    P_reference,
)


print(
    "Expected reference   =",
    expected_reference,
)


assert np.isclose(
    P_reference,
    expected_reference,
    rtol=1e-12,
    atol=1e-12,
)


print(
    "\n[PASS] homogeneous reference remains correct"
)


# ============================================================
# 9. PRECOMPUTE q GRIDS
# ============================================================
#
# We construct the grids only once.
#
# Therefore every plane-wave truncation sees EXACTLY the same
# q points.
#
# ============================================================

section(
    "BUILDING FIXED q GRIDS"
)


q_grids = []


for ishift, (
    shift_x,
    shift_y,
) in enumerate(
    SHIFTS,
    start=1,
):

    qx, qy, dq = make_shifted_q_grid(
        Nq=Nq,
        shift_x=shift_x,
        shift_y=shift_y,
    )


    # --------------------------------------------------------
    # Check spacing
    # --------------------------------------------------------

    assert np.allclose(
        np.diff(
            qx
        ),
        dq,
        rtol=1e-10,
        atol=1e-12,
    )


    assert np.allclose(
        np.diff(
            qy
        ),
        dq,
        rtol=1e-10,
        atol=1e-12,
    )


    # --------------------------------------------------------
    # Check light line
    # --------------------------------------------------------

    (
        distance,
        closest_qx,
        closest_qy,
    ) = diagnose_light_line(
        qx=qx,
        qy=qy,
    )


    print(
        f"\nShift {ishift}: "
        f"{SHIFTS[ishift-1]}"
    )


    print(
        "dq =",
        dq,
    )


    print(
        "Closest distance to air light line =",
        distance,
    )


    print(
        "Closest q =",
        (
            closest_qx,
            closest_qy,
        ),
    )


    if (
        distance
        <
        LIGHT_LINE_TOLERANCE
    ):

        raise RuntimeError(
            "A q grid lands essentially exactly on "
            "the exterior light line."
        )


    q_grids.append(
        {
            "qx": qx,
            "qy": qy,
            "dq": dq,
            "shift": (
                shift_x,
                shift_y,
            ),
        }
    )


print(
    "\n[PASS] all shifted q grids are valid"
)


# ============================================================
# 10. STORAGE
# ============================================================

results = {}


pw_labels = []

Fp_mean_values = []

Fp_std_values = []

P_mean_values = []


# ============================================================
# 11. LOOP OVER PLANE-WAVE TRUNCATION
# ============================================================

section(
    "PLANE-WAVE CONVERGENCE"
)


for pw in PW_VALUES:

    print("\n")
    print("#" * 78)

    print(
        "Testing pw =",
        pw,
    )


    print(
        "Nq =",
        Nq,
    )


    print(
        "Number of q points per shift =",
        Nq**2,
    )


    print(
        "Number of shifted grids =",
        len(
            SHIFTS
        ),
    )


    print(
        "Total q solves for this pw =",
        len(
            SHIFTS
        )
        *
        Nq**2,
    )


    print(
        "#" * 78
    )


    # ========================================================
    # BUILD CRYSTAL FOR THIS pw
    # ========================================================

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


    # ========================================================
    # STORAGE FOR SHIFTS
    # ========================================================

    individual_P = []

    individual_Fp = []

    individual_Pmax = []

    individual_qpeak = []


    # ========================================================
    # LOOP OVER THE SAME SHIFTED q GRIDS
    # ========================================================

    for ishift, grid in enumerate(
        q_grids,
        start=1,
    ):

        qx = grid[
            "qx"
        ]

        qy = grid[
            "qy"
        ]

        dq = grid[
            "dq"
        ]

        shift = grid[
            "shift"
        ]


        print("\n")
        print("-" * 78)


        print(
            f"pw = {pw}, "
            f"shift {ishift}/{len(q_grids)} = "
            f"{shift}"
        )


        # ----------------------------------------------------
        # q-resolved dipole power
        # ----------------------------------------------------

        map_result = dipole_power_map(
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


        P_map = map_result[
            "P_total"
        ]


        # ----------------------------------------------------
        # Shape
        # ----------------------------------------------------

        assert P_map.shape == (
            Nq,
            Nq,
        )


        # ----------------------------------------------------
        # NaN / inf
        # ----------------------------------------------------

        check_finite_power_map(
            P_map=P_map,
            qx=qx,
            qy=qy,
        )


        # ----------------------------------------------------
        # Min and max
        # ----------------------------------------------------

        P_min = float(
            np.min(
                P_map
            )
        )


        P_max = float(
            np.max(
                P_map
            )
        )


        # ----------------------------------------------------
        # Reject substantial negative power
        # ----------------------------------------------------

        negative_tolerance = (
            1e-9
            *
            max(
                abs(
                    P_max
                ),
                1.0,
            )
        )


        if (
            P_min
            <
            -negative_tolerance
        ):

            raise RuntimeError(
                f"Substantial negative power for pw={pw}.\n"
                f"P_min={P_min}\n"
                f"P_max={P_max}"
            )


        # ----------------------------------------------------
        # Peak location
        # ----------------------------------------------------

        peak_index = np.unravel_index(
            np.argmax(
                P_map
            ),
            P_map.shape,
        )


        peak_iy = int(
            peak_index[0]
        )


        peak_ix = int(
            peak_index[1]
        )


        peak_qx = float(
            qx[
                peak_ix
            ]
        )


        peak_qy = float(
            qy[
                peak_iy
            ]
        )


        # ----------------------------------------------------
        # Integrate over BZ
        # ----------------------------------------------------

        P_structure = (
            np.sum(
                P_map
            )
            *
            dq**2
            /
            (
                2.0
                *
                np.pi
            )**2
        )


        # ----------------------------------------------------
        # Purcell factor
        # ----------------------------------------------------

        Fp = (
            P_structure
            /
            P_reference
        )


        # ----------------------------------------------------
        # Sanity checks
        # ----------------------------------------------------

        if not np.isfinite(
            P_structure
        ):

            raise RuntimeError(
                "P_structure is not finite."
            )


        if not np.isfinite(
            Fp
        ):

            raise RuntimeError(
                "Fp is not finite."
            )


        if P_structure <= 0:

            raise RuntimeError(
                "P_structure is not positive."
            )


        if Fp <= 0:

            raise RuntimeError(
                "Fp is not positive."
            )


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        individual_P.append(
            float(
                P_structure
            )
        )


        individual_Fp.append(
            float(
                Fp
            )
        )


        individual_Pmax.append(
            float(
                P_max
            )
        )


        individual_qpeak.append(
            (
                peak_qx,
                peak_qy,
            )
        )


        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print(
            "P_min =",
            P_min,
        )


        print(
            "P_max =",
            P_max,
        )


        print(
            "Peak q =",
            (
                peak_qx,
                peak_qy,
            ),
        )


        print(
            "P_structure =",
            P_structure,
        )


        print(
            "F_purcell =",
            Fp,
        )


    # ========================================================
    # 12. SHIFT AVERAGE FOR THIS pw
    # ========================================================

    individual_P = np.asarray(
        individual_P,
        dtype=float,
    )


    individual_Fp = np.asarray(
        individual_Fp,
        dtype=float,
    )


    individual_Pmax = np.asarray(
        individual_Pmax,
        dtype=float,
    )


    P_mean = float(
        np.mean(
            individual_P
        )
    )


    P_std = float(
        np.std(
            individual_P,
            ddof=1,
        )
    )


    Fp_mean = float(
        np.mean(
            individual_Fp
        )
    )


    Fp_std = float(
        np.std(
            individual_Fp,
            ddof=1,
        )
    )


    Fp_min = float(
        np.min(
            individual_Fp
        )
    )


    Fp_max = float(
        np.max(
            individual_Fp
        )
    )


    Fp_spread_percent = (
        (
            Fp_max
            -
            Fp_min
        )
        /
        abs(
            Fp_mean
        )
        *
        100.0
    )


    Fp_std_percent = (
        Fp_std
        /
        abs(
            Fp_mean
        )
        *
        100.0
    )


    # ========================================================
    # STORE RESULT
    # ========================================================

    results[
        pw
    ] = {

        "P_mean":
            P_mean,

        "P_std":
            P_std,

        "Fp_mean":
            Fp_mean,

        "Fp_std":
            Fp_std,

        "Fp_std_percent":
            Fp_std_percent,

        "Fp_spread_percent":
            Fp_spread_percent,

        "individual_P":
            individual_P,

        "individual_Fp":
            individual_Fp,

        "individual_Pmax":
            individual_Pmax,

        "individual_qpeak":
            individual_qpeak,
    }


    pw_labels.append(
        f"{pw[0]}x{pw[1]}"
    )


    P_mean_values.append(
        P_mean
    )


    Fp_mean_values.append(
        Fp_mean
    )


    Fp_std_values.append(
        Fp_std
    )


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n")
    print("=" * 78)

    print(
        f"SUMMARY FOR pw = {pw}"
    )

    print("=" * 78)


    for index, (
        shift,
        Fp_value,
    ) in enumerate(
        zip(
            SHIFTS,
            individual_Fp,
        ),
        start=1,
    ):

        print(
            f"Shift {index}: "
            f"{shift}   "
            f"Fp = {Fp_value:.12f}"
        )


    print(
        "\nMean P_structure =",
        P_mean,
    )


    print(
        "Mean F_purcell =",
        Fp_mean,
    )


    print(
        "Relative shift std =",
        Fp_std_percent,
        "%",
    )


    print(
        "Full shift spread =",
        Fp_spread_percent,
        "%",
    )


# ============================================================
# 13. FINAL PLANE-WAVE CONVERGENCE TABLE
# ============================================================

section(
    "PLANE-WAVE CONVERGENCE SUMMARY"
)


print(
    "\n"
    f"{'pw':>12} "
    f"{'Mean P':>18} "
    f"{'Mean Fp':>18} "
    f"{'Shift std (%)':>16} "
    f"{'Shift spread (%)':>18}"
)


print(
    "-" * 86
)


for pw in PW_VALUES:

    result = results[
        pw
    ]


    print(
        f"{str(pw):>12} "
        f"{result['P_mean']:18.12f} "
        f"{result['Fp_mean']:18.12f} "
        f"{result['Fp_std_percent']:16.6f} "
        f"{result['Fp_spread_percent']:18.6f}"
    )


# ============================================================
# 14. COMPARE CONSECUTIVE pw VALUES
# ============================================================

section(
    "PLANE-WAVE DIFFERENCE"
)


pw_relative_changes = []


for i in range(
    1,
    len(
        PW_VALUES
    ),
):

    pw_previous = PW_VALUES[
        i - 1
    ]


    pw_current = PW_VALUES[
        i
    ]


    Fp_previous = results[
        pw_previous
    ][
        "Fp_mean"
    ]


    Fp_current = results[
        pw_current
    ][
        "Fp_mean"
    ]


    relative_change_percent = (
        abs(
            Fp_current
            -
            Fp_previous
        )
        /
        abs(
            Fp_current
        )
        *
        100.0
    )


    pw_relative_changes.append(
        relative_change_percent
    )


    print(
        f"\n{pw_previous} -> {pw_current}"
    )


    print(
        "Fp previous =",
        Fp_previous,
    )


    print(
        "Fp current  =",
        Fp_current,
    )


    print(
        "Absolute difference =",
        abs(
            Fp_current
            -
            Fp_previous
        ),
    )


    print(
        "Relative difference =",
        relative_change_percent,
        "%",
    )


# ============================================================
# 15. FINAL CONVERGENCE DECISION
# ============================================================

section(
    "FINAL PLANE-WAVE CONVERGENCE DIAGNOSTIC"
)


final_pw_change = (
    pw_relative_changes[
        -1
    ]
)


finest_pw = PW_VALUES[
    -1
]


finest_result = results[
    finest_pw
]


print(
    "\nFinest pw =",
    finest_pw,
)


print(
    "Finest shift-averaged P_structure =",
    finest_result[
        "P_mean"
    ],
)


print(
    "Finest shift-averaged F_purcell =",
    finest_result[
        "Fp_mean"
    ],
)


print(
    "Plane-wave relative difference =",
    final_pw_change,
    "%",
)


print(
    "Required tolerance =",
    PW_TOLERANCE_PERCENT,
    "%",
)


if (
    final_pw_change
    <
    PW_TOLERANCE_PERCENT
):

    print("\n")
    print("=" * 78)
    print("PLANE-WAVE TRUNCATION APPEARS CONVERGED")
    print("=" * 78)


    print(
        "\nThe change in Purcell factor between"
    )


    print(
        PW_VALUES[
            -2
        ],
        "and",
        PW_VALUES[
            -1
        ],
        "is below",
        PW_TOLERANCE_PERCENT,
        "%."
    )


    print(
        "\n[PASS] plane-wave convergence criterion satisfied"
    )


else:

    print("\n")
    print("=" * 78)
    print("PLANE-WAVE TRUNCATION NOT YET CONVERGED")
    print("=" * 78)


    print(
        "\nThe difference is larger than",
        PW_TOLERANCE_PERCENT,
        "%."
    )


    print(
        "Add pw=(7,7) before proceeding."
    )


# ============================================================
# 16. PLOT — PURCELL FACTOR VS PLANE-WAVE TRUNCATION
# ============================================================

x_positions = np.arange(
    len(
        PW_VALUES
    )
)


fig, ax = plt.subplots(
    figsize=(
        6,
        5,
    )
)


ax.errorbar(
    x_positions,
    Fp_mean_values,
    yerr=Fp_std_values,
    marker="o",
    capsize=4,
)


ax.set_xticks(
    x_positions
)


ax.set_xticklabels(
    pw_labels
)


ax.set_xlabel(
    "Plane-wave truncation"
)


ax.set_ylabel(
    r"$F_P$"
)


ax.set_title(
    "Purcell-factor plane-wave convergence"
)


ax.grid(
    True,
)


fig.tight_layout()


plt.show()


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

section(
    "TEST COMPLETED"
)


print(
    "\nReference power =",
    P_reference,
)


for pw in PW_VALUES:

    result = results[
        pw
    ]


    print(
        f"\npw = {pw}"
    )


    print(
        "P_structure =",
        result[
            "P_mean"
        ],
    )


    print(
        "F_purcell =",
        result[
            "Fp_mean"
        ],
    )


    print(
        "Shift std (%) =",
        result[
            "Fp_std_percent"
        ],
    )


print(
    "\nFinal plane-wave relative difference =",
    final_pw_change,
    "%",
)


if (
    final_pw_change
    <
    PW_TOLERANCE_PERCENT
):

    print(
        "\n[PASS] The slab Purcell calculation is converged "
        "with respect to plane-wave truncation."
    )


    print(
        "\nThe homogeneous/slab validation chain is now "
        "complete."
    )


else:

    print(
        "\n[WARNING] Plane-wave convergence has not yet "
        "been demonstrated."
    )


    print(
        "The next test should include pw=(7,7)."
    )