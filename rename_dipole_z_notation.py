"""
Rename left/right dipole notation to explicit -z/+z notation.

Run this from the repository root:

    python rename_dipole_z_notation.py

It edits:
    khepri/dipole.py

and creates:
    khepri/dipole.py.bak

Physics and equations are NOT changed.

The public result dictionaries will use:
    P_minus_z
    P_plus_z
    P_total

For compatibility, legacy aliases P_left/P_right are retained.
Likewise, return_details keeps both psi_minus_z/psi_plus_z and the
legacy psi_left/psi_right aliases.
"""

from pathlib import Path
import shutil
import py_compile

TARGET = Path("khepri/dipole.py")
BACKUP = Path("khepri/dipole.py.bak")

if not TARGET.exists():
    raise FileNotFoundError(
        "Could not find khepri/dipole.py. "
        "Run this script from the Khepri_Purcell repository root."
    )

text = TARGET.read_text(encoding="utf-8")
original = text

# ------------------------------------------------------------------
# 1. Internal reflection notation: left/right -> -z/+z
# ------------------------------------------------------------------
text = text.replace(
"""    # Wave travelling left sees this reflection operator.
    R_left = Sl[
        1,
        1,
    ]

    # Wave travelling right sees this reflection operator.
    R_right = Sr[
        0,
        0,
    ]""",
"""    # Wave travelling toward -z sees this reflection operator.
    R_minus_z = Sl[
        1,
        1,
    ]

    # Wave travelling toward +z sees this reflection operator.
    R_plus_z = Sr[
        0,
        0,
    ]"""
)

text = text.replace(
"""    # c_plus - R_left c_minus = s_plus
    #
    # R_right c_plus - c_minus = s_minus""",
"""    # c_plus - R_minus_z c_minus = s_plus
    #
    # R_plus_z c_plus - c_minus = s_minus"""
)

text = text.replace("-R_left,", "-R_minus_z,")
text = text.replace("R_right,", "R_plus_z,")

# ------------------------------------------------------------------
# 2. Source-plane fields: psi_left/right -> psi_minus_z/plus_z
# ------------------------------------------------------------------
text = text.replace(
"""    R_left = Sl[
        1,
        1,
    ]

    R_right = Sr[
        0,
        0,
    ]""",
"""    R_minus_z = Sl[
        1,
        1,
    ]

    R_plus_z = Sr[
        0,
        0,
    ]"""
)

text = text.replace(
"""    # LEFT side of source
    #
    # +z component:
    #     reflection of c_minus from left stack
    #
    # -z component:
    #     directly emitted c_minus""",
"""    # -z side of source: z = z0 - 0
    #
    # +z component:
    #     reflection of c_minus from the -z stack
    #
    # -z component:
    #     directly emitted c_minus"""
)

text = text.replace(
"""    left_modal = np.concatenate(
        [
            R_left @ c_minus,
            c_minus,
        ]
    )


    psi_left = (
        R0
        @
        left_modal
    )""",
"""    minus_z_modal = np.concatenate(
        [
            R_minus_z @ c_minus,
            c_minus,
        ]
    )


    psi_minus_z = (
        R0
        @
        minus_z_modal
    )"""
)

text = text.replace(
"""    # RIGHT side of source
    #
    # +z component:
    #     directly emitted c_plus
    #
    # -z component:
    #     reflection of c_plus from right stack""",
"""    # +z side of source: z = z0 + 0
    #
    # +z component:
    #     directly emitted c_plus
    #
    # -z component:
    #     reflection of c_plus from the +z stack"""
)

text = text.replace(
"""    right_modal = np.concatenate(
        [
            c_plus,
            R_right @ c_plus,
        ]
    )


    psi_right = (
        R0
        @
        right_modal
    )


    return psi_left, psi_right""",
"""    plus_z_modal = np.concatenate(
        [
            c_plus,
            R_plus_z @ c_plus,
        ]
    )


    psi_plus_z = (
        R0
        @
        plus_z_modal
    )


    return psi_minus_z, psi_plus_z"""
)

# Docstring wording in source-plane field helper.
text = text.replace(
"""    psi_left : ndarray
        Fourier field state immediately to the left of source:""",
"""    psi_minus_z : ndarray
        Fourier field state at z = z0 - 0:"""
)
text = text.replace(
"""    psi_right : ndarray
        Fourier field state immediately to the right of source:""",
"""    psi_plus_z : ndarray
        Fourier field state at z = z0 + 0:"""
)
text = text.replace(
"        psi_right - psi_left = b",
"        psi_plus_z - psi_minus_z = b"
)

# ------------------------------------------------------------------
# 3. Total near-field power helper
# ------------------------------------------------------------------
text = text.replace(
"""def dipole_total_nearfield_power(
    psi_left,
    psi_right,
):""",
"""def dipole_total_nearfield_power(
    psi_minus_z,
    psi_plus_z,
):"""
)

text = text.replace(
"""    Coordinate convention:

        +z -> right / emergence
        -z -> left  / incidence

    Therefore:

        P_left  = -Sz_left
        P_right = +Sz_right""",
"""    Coordinate convention:

        psi_minus_z is evaluated at z = z0 - 0
        psi_plus_z  is evaluated at z = z0 + 0

    Therefore the outward powers are:

        P_minus_z = -Sz(z0 - 0)
        P_plus_z  = +Sz(z0 + 0)"""
)

text = text.replace(
"""    psi_left : ndarray
        Field state immediately to the left of the source.

    psi_right : ndarray
        Field state immediately to the right of the source.""",
"""    psi_minus_z : ndarray
        Field state at z = z0 - 0.

    psi_plus_z : ndarray
        Field state at z = z0 + 0."""
)

text = text.replace(
"""    P_left : float
        Outward power emitted toward -z.

    P_right : float
        Outward power emitted toward +z.""",
"""    P_minus_z : float
        Outward power emitted toward -z.

    P_plus_z : float
        Outward power emitted toward +z."""
)

text = text.replace(
"""    Sz_left = dipole_nearfield_flux(
        psi_left
    )

    Sz_right = dipole_nearfield_flux(
        psi_right
    )


    P_left = float(
        -Sz_left
    )

    P_right = float(
        Sz_right
    )

    P_total = (
        P_left
        +
        P_right
    )


    return (
        P_left,
        P_right,
        P_total,
    )""",
"""    Sz_minus_z = dipole_nearfield_flux(
        psi_minus_z
    )

    Sz_plus_z = dipole_nearfield_flux(
        psi_plus_z
    )


    P_minus_z = float(
        -Sz_minus_z
    )

    P_plus_z = float(
        Sz_plus_z
    )

    P_total = (
        P_minus_z
        +
        P_plus_z
    )


    return (
        P_minus_z,
        P_plus_z,
        P_total,
    )"""
)

# ------------------------------------------------------------------
# 4. High-level dipole_power_q names
# ------------------------------------------------------------------
text = text.replace(
"""    psi_left, psi_right = dipole_source_plane_fields(""",
"""    psi_minus_z, psi_plus_z = dipole_source_plane_fields("""
)

text = text.replace(
"""        P_left,
        P_right,
        P_total,
    ) = dipole_total_nearfield_power(
        psi_left,
        psi_right,
    )""",
"""        P_minus_z,
        P_plus_z,
        P_total,
    ) = dipole_total_nearfield_power(
        psi_minus_z,
        psi_plus_z,
    )"""
)

text = text.replace(
"""    jump_difference = (
        psi_right
        -
        psi_left
        -
        b
    )""",
"""    jump_difference = (
        psi_plus_z
        -
        psi_minus_z
        -
        b
    )"""
)

# Main result: new canonical keys + compatibility aliases.
text = text.replace(
"""    result = {
        "P_left": float(P_left),
        "P_right": float(P_right),
        "P_total": float(P_total),
    }""",
"""    result = {
        # Canonical z-direction notation
        "P_minus_z": float(P_minus_z),
        "P_plus_z": float(P_plus_z),
        "P_total": float(P_total),

        # Backward-compatible aliases
        "P_left": float(P_minus_z),
        "P_right": float(P_plus_z),
    }"""
)

text = text.replace(
"""                "psi_left": psi_left,
                "psi_right": psi_right,""",
"""                # Canonical z-direction notation
                "psi_minus_z": psi_minus_z,
                "psi_plus_z": psi_plus_z,

                # Backward-compatible aliases
                "psi_left": psi_minus_z,
                "psi_right": psi_plus_z,"""
)

# Result docstring.
text = text.replace(
"""            result["P_left"]
            result["P_right"]
            result["P_total"]""",
"""            result["P_minus_z"]
            result["P_plus_z"]
            result["P_total"]

        Legacy aliases ``P_left`` and ``P_right`` are retained for
        backward compatibility."""
)

# ------------------------------------------------------------------
# 5. Cartesian power-map helper
# ------------------------------------------------------------------
text = text.replace(
"""    P_left = np.zeros(
        shape,
        dtype=float,
    )

    P_right = np.zeros(
        shape,
        dtype=float,
    )""",
"""    P_minus_z = np.zeros(
        shape,
        dtype=float,
    )

    P_plus_z = np.zeros(
        shape,
        dtype=float,
    )"""
)

text = text.replace(
"""            P_left[
                iy,
                ix
            ] = result_q[
                "P_left"
            ]

            P_right[
                iy,
                ix
            ] = result_q[
                "P_right"
            ]""",
"""            P_minus_z[
                iy,
                ix
            ] = result_q[
                "P_minus_z"
            ]

            P_plus_z[
                iy,
                ix
            ] = result_q[
                "P_plus_z"
            ]"""
)

text = text.replace(
"""        "P_left": P_left,
        "P_right": P_right,
        "P_total": P_total,""",
"""        "P_minus_z": P_minus_z,
        "P_plus_z": P_plus_z,
        "P_total": P_total,

        # Backward-compatible aliases
        "P_left": P_minus_z,
        "P_right": P_plus_z,"""
)

# ------------------------------------------------------------------
# 6. Reciprocal-unit-cell power map helper
# ------------------------------------------------------------------
text = text.replace(
"""    P_left = np.empty(
        (Nv_eff, Nu_eff),
        dtype=float,
    )

    P_right = np.empty(
        (Nv_eff, Nu_eff),
        dtype=float,
    )""",
"""    P_minus_z = np.empty(
        (Nv_eff, Nu_eff),
        dtype=float,
    )

    P_plus_z = np.empty(
        (Nv_eff, Nu_eff),
        dtype=float,
    )"""
)

text = text.replace(
"""            p_left = float(np.real(point_result["P_left"]))
            p_right = float(np.real(point_result["P_right"]))
            p_total = float(np.real(point_result["P_total"]))""",
"""            p_minus_z = float(np.real(point_result["P_minus_z"]))
            p_plus_z = float(np.real(point_result["P_plus_z"]))
            p_total = float(np.real(point_result["P_total"]))"""
)

text = text.replace(
"""                        p_left,
                        p_right,
                        p_total,""",
"""                        p_minus_z,
                        p_plus_z,
                        p_total,"""
)

text = text.replace(
"""            P_left[iv, iu] = p_left
            P_right[iv, iu] = p_right
            P_total[iv, iu] = p_total""",
"""            P_minus_z[iv, iu] = p_minus_z
            P_plus_z[iv, iu] = p_plus_z
            P_total[iv, iu] = p_total"""
)

text = text.replace(
"""            "P_left": P_left,
            "P_right": P_right,
            "P_total": P_total,""",
"""            "P_minus_z": P_minus_z,
            "P_plus_z": P_plus_z,
            "P_total": P_total,

            # Backward-compatible aliases
            "P_left": P_minus_z,
            "P_right": P_plus_z,"""
)

# Component selector: canonical new names + legacy old aliases.
text = text.replace(
"""    if component not in (
        "P_total",
        "P_left",
        "P_right",
    ):
        raise ValueError(
            "component must be 'P_total', 'P_left', or 'P_right'."
        )""",
"""    component_aliases = {
        "P_left": "P_minus_z",
        "P_right": "P_plus_z",
    }
    component = component_aliases.get(component, component)

    if component not in (
        "P_total",
        "P_minus_z",
        "P_plus_z",
    ):
        raise ValueError(
            "component must be 'P_total', 'P_minus_z', or 'P_plus_z'."
        )"""
)

# ------------------------------------------------------------------
# 7. Purely descriptive wording
# ------------------------------------------------------------------
text = text.replace("toward +z / right / emergence", "toward +z / emergence")
text = text.replace("toward -z / left / incidence", "toward -z / incidence")
text = text.replace("left and right parts of", "-z and +z parts of")
text = text.replace("Left and right partial scattering matrices.", "-z and +z partial scattering matrices.")
text = text.replace("left of the dipole source plane", "-z side of the dipole source plane")
text = text.replace("right of the dipole source plane", "+z side of the dipole source plane")
text = text.replace("Left partial scattering matrix.", "-z-side partial scattering matrix.")
text = text.replace("Right partial scattering matrix.", "+z-side partial scattering matrix.")

# ------------------------------------------------------------------
# Validate that meaningful changes occurred.
# ------------------------------------------------------------------
if text == original:
    raise RuntimeError(
        "No changes were made. The file may already use the new notation "
        "or may differ from the expected version."
    )

# Back up once.
if not BACKUP.exists():
    shutil.copy2(TARGET, BACKUP)

TARGET.write_text(text, encoding="utf-8")

# Syntax check the edited module.
py_compile.compile(str(TARGET), doraise=True)

print("Updated:", TARGET)
print("Backup :", BACKUP)
print()
print("Canonical names are now:")
print("  psi_minus_z, psi_plus_z")
print("  P_minus_z, P_plus_z, P_total")
print()
print("Legacy result aliases P_left/P_right and psi_left/psi_right")
print("are retained so existing scripts should keep working.")
print()
print("No electromagnetic equations or numerical operations were changed.")
