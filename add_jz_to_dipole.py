"""
Patch khepri/dipole.py to add z-oriented electric-dipole support
following Dyakov et al., PRB 108, 155416 (2023), Appendix B.

Run from the repository root:

    python add_jz_to_dipole.py

It edits:
    khepri/dipole.py

and creates:
    khepri/dipole.py.pre_jz.bak

The patch is compatible with both the original left/right naming and the
new -z/+z naming patch. It does NOT alter the existing validated Jx/Jy
source terms.
"""

from pathlib import Path
import re
import shutil
import py_compile

TARGET = Path("khepri/dipole.py")
BACKUP = Path("khepri/dipole.py.pre_jz.bak")

if not TARGET.exists():
    raise FileNotFoundError(
        "Could not find khepri/dipole.py. "
        "Run this script from the Khepri_Purcell repository root."
    )

text = TARGET.read_text(encoding="utf-8")
original = text

# ---------------------------------------------------------------------
# 1. Replace the transverse-only source builder with a general XYZ
#    builder, while retaining dipole_source_jump_xy as a compatibility
#    wrapper.
# ---------------------------------------------------------------------

source_section_pattern = re.compile(
    r"# ============================================================\n"
    r"# 2\. ELECTROMAGNETIC SOURCE JUMP\n"
    r"# ============================================================\n\n"
    r"def dipole_source_jump_xy\(.*?\n"
    r"(?=# ============================================================\n# 3\.)",
    re.S,
)

new_source_section = '''# ============================================================
# 2. ELECTROMAGNETIC SOURCE JUMP
# ============================================================

def _apply_inverse_epsilon_operator(
    inverse_epsilon_operator,
    Jz,
):
    """
    Apply the source-layer inverse-permittivity Fourier operator
    to the z-directed current hypervector.

    For a uniform isotropic layer,

        inverse_epsilon_operator = 1 / epsilon.

    For a patterned isotropic RCWA layer, Khepri stores

        IC = inv(C),

    where C is the Fourier convolution matrix of epsilon(x,y).
    """

    Jz = _as_complex_1d(Jz, "Jz")

    op = np.asarray(
        inverse_epsilon_operator,
        dtype=np.complex128,
    )

    if op.ndim == 0:
        return op.item() * Jz

    if op.ndim != 2:
        raise ValueError(
            "inverse_epsilon_operator must be a scalar or "
            "a two-dimensional square matrix."
        )

    expected = (len(Jz), len(Jz))

    if op.shape != expected:
        raise ValueError(
            "inverse_epsilon_operator has incompatible shape. "
            f"Expected {expected}, got {op.shape}."
        )

    return op @ Jz


def dipole_source_jump_xyz(
    Jx,
    Jy,
    Jz,
    Kx,
    Ky,
    inverse_epsilon_operator,
):
    """
    Construct the complete electric-dipole source jump in Khepri's
    transverse-field convention.

    This is the scalar-isotropic specialization of the source
    hypervector in Eqs. (B7)-(B8) of Dyakov et al.,
    Phys. Rev. B 108, 155416 (2023).

    Khepri's ``Expansion.k_vectors`` already returns Kx/k0 and Ky/k0.
    For the scalar-isotropic layers used here,

        eps_tilde_33 -> inverse_epsilon_operator.

    Khepri uses

        psi = [Ex, Ey, Ux, Uy],   U = -i H.

    Therefore

        Delta Ex = Kx * epsinv(Jz)
        Delta Ey = Ky * epsinv(Jz)
        Delta Ux = -i Jy
        Delta Uy = +i Jx

    Notes
    -----
    This implementation assumes the source layer is scalar and isotropic.
    It does not implement a fully anisotropic epsilon tensor with nonzero
    epsilon_13 or epsilon_23 blocks.
    """

    Jx = _as_complex_1d(Jx, "Jx")
    Jy = _as_complex_1d(Jy, "Jy")
    Jz = _as_complex_1d(Jz, "Jz")
    Kx = _as_complex_1d(Kx, "Kx")
    Ky = _as_complex_1d(Ky, "Ky")

    for array, name in (
        (Jy, "Jy"),
        (Jz, "Jz"),
        (Kx, "Kx"),
        (Ky, "Ky"),
    ):
        _check_same_shape(Jx, array, "Jx", name)

    epsinv_Jz = _apply_inverse_epsilon_operator(
        inverse_epsilon_operator,
        Jz,
    )

    delta_Ex = Kx * epsinv_Jz
    delta_Ey = Ky * epsinv_Jz
    delta_Ux = -1j * Jy
    delta_Uy = +1j * Jx

    return np.concatenate(
        [
            delta_Ex,
            delta_Ey,
            delta_Ux,
            delta_Uy,
        ]
    )


def dipole_source_jump_xy(
    Jx,
    Jy,
):
    """
    Backward-compatible wrapper for the validated in-plane source.

    This remains exactly

        b = [0, 0, -i Jy, +i Jx].
    """

    Jx = _as_complex_1d(Jx, "Jx")
    Jy = _as_complex_1d(Jy, "Jy")
    _check_same_shape(Jx, Jy, "Jx", "Jy")

    zeros = np.zeros_like(Jx, dtype=np.complex128)

    return dipole_source_jump_xyz(
        Jx=Jx,
        Jy=Jy,
        Jz=zeros,
        Kx=zeros,
        Ky=zeros,
        inverse_epsilon_operator=1.0,
    )


'''

text, n = source_section_pattern.subn(new_source_section, text, count=1)
if n != 1:
    raise RuntimeError(
        "Could not replace dipole_source_jump_xy section. "
        "Your dipole.py differs from the expected version."
    )

# ---------------------------------------------------------------------
# 2. Remove the old Jz NotImplemented guard.
# ---------------------------------------------------------------------

guard_pattern = re.compile(
    r'\n    if abs\(current\[2\]\) > 1e-14:\n'
    r'        raise NotImplementedError\(\n'
    r'            "z-oriented electric dipoles are not implemented "\n'
    r'            "yet\. Use current=\(jx, jy, 0\)\."\n'
    r'        \)\n',
    re.S,
)

text, n_guard = guard_pattern.subn("\n", text, count=1)
if n_guard != 1 and "z-oriented electric dipoles are not implemented" in text:
    raise RuntimeError(
        "Found a Jz NotImplemented guard, but it did not match the expected form."
    )

# ---------------------------------------------------------------------
# 3. Replace the source-builder call inside dipole_power_q.
# ---------------------------------------------------------------------

old_call_pattern = re.compile(
    r"    # --------------------------------------------------------\n"
    r"    # Build Maxwell source jump\n"
    r"    # --------------------------------------------------------\n\n"
    r"    b = dipole_source_jump_xy\(\n"
    r"        Jx,\n"
    r"        Jy,\n"
    r"    \)\n",
    re.S,
)

new_call = '''    # --------------------------------------------------------
    # Build Maxwell source jump
    #
    # For Jz != 0 we need the inverse-permittivity Fourier
    # operator of the physical layer containing the source.
    # ``source_after`` is the split layer immediately on the
    # z0-0 side of the dipole plane.
    # --------------------------------------------------------

    if abs(current[2]) > 1e-14:
        source_layer = crystal.layers[source_after]

        inverse_epsilon_operator = getattr(
            source_layer,
            "IC",
            None,
        )

        if inverse_epsilon_operator is None:
            raise RuntimeError(
                "A z-oriented dipole requires the inverse-permittivity "
                "operator of the source layer, but source_layer.IC is "
                "not available. Retain fields for the layer immediately "
                "before the dipole plane, e.g. fields_mask=[True, False] "
                "for a two-half split layer."
            )
    else:
        # For Jz=0 the operator is multiplied by zero and is irrelevant.
        inverse_epsilon_operator = 1.0

    b = dipole_source_jump_xyz(
        Jx=Jx,
        Jy=Jy,
        Jz=Jz,
        Kx=Kx,
        Ky=Ky,
        inverse_epsilon_operator=inverse_epsilon_operator,
    )
'''

text, n = old_call_pattern.subn(new_call, text, count=1)
if n != 1:
    raise RuntimeError(
        "Could not find the Maxwell source-jump call in dipole_power_q."
    )

# ---------------------------------------------------------------------
# 4. Replace the in-plane-only homogeneous reference by a general one.
# ---------------------------------------------------------------------

reference_pattern = re.compile(
    r"# ============================================================\n"
    r"# 11\. HOMOGENEOUS REFERENCE POWER FOR IN-PLANE DIPOLE\n"
    r"# ============================================================\n\n"
    r"def homogeneous_reference_power_xy\(.*?\n"
    r"(?=# ============================================================\n# 12\.)",
    re.S,
)

new_reference = '''# ============================================================
# 11. HOMOGENEOUS REFERENCE POWER
# ============================================================

def homogeneous_reference_power(
    wavelength,
    refractive_index,
    current,
):
    """
    Exact homogeneous-medium reference power in Khepri's dipole
    normalization.

    For an isotropic homogeneous medium,

        P_ref = k0^2 n / (6 pi)
                * (|Jx|^2 + |Jy|^2 + |Jz|^2).

    Equal-amplitude x, y, and z dipoles therefore have the same
    integrated homogeneous reference power.
    """

    if len(current) != 3:
        raise ValueError(
            "current must contain exactly (jx, jy, jz)."
        )

    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")

    if refractive_index <= 0:
        raise ValueError("refractive_index must be positive.")

    jx, jy, jz = current

    current_squared = (
        abs(jx)**2
        + abs(jy)**2
        + abs(jz)**2
    )

    if current_squared == 0:
        raise ValueError("The dipole current cannot be zero.")

    k0 = 2.0 * np.pi / wavelength

    P_ref = (
        k0**2
        * refractive_index
        / (6.0 * np.pi)
        * current_squared
    )

    return float(np.real(P_ref))


def homogeneous_reference_power_xy(
    wavelength,
    refractive_index,
    current,
):
    """
    Backward-compatible alias for ``homogeneous_reference_power``.

    The historical function name is retained so existing scripts keep
    working, but z-oriented dipoles are now supported too.
    """

    return homogeneous_reference_power(
        wavelength=wavelength,
        refractive_index=refractive_index,
        current=current,
    )


'''

text, n = reference_pattern.subn(new_reference, text, count=1)
if n != 1:
    raise RuntimeError(
        "Could not replace homogeneous_reference_power_xy section."
    )

# ---------------------------------------------------------------------
# 5. Update stale explanatory text.
# ---------------------------------------------------------------------

text = text.replace(
    "Only x/y-oriented dipoles are currently supported.",
    "x-, y-, and z-oriented electric dipoles are supported for scalar-isotropic source layers.",
)
text = text.replace(
    "z-oriented electric dipoles are not implemented yet.",
    "z-oriented electric dipoles use the source-layer inverse-permittivity operator.",
)
text = text.replace(
    "Currently only in-plane dipoles are supported.",
    "x-, y-, and z-oriented dipoles are supported for scalar-isotropic source layers.",
)

# ---------------------------------------------------------------------
# 6. Add the source operator to return_details when possible.
# ---------------------------------------------------------------------

details_anchor = '''                "Jz": Jz,

                "b": b,
'''
if details_anchor in text:
    text = text.replace(
        details_anchor,
        '''                "Jz": Jz,

                "inverse_epsilon_operator":
                    inverse_epsilon_operator,

                "b": b,
''',
        1,
    )

# ---------------------------------------------------------------------
# 7. Write and syntax-check.
# ---------------------------------------------------------------------

if text == original:
    raise RuntimeError("No changes were made.")

if not BACKUP.exists():
    shutil.copy2(TARGET, BACKUP)

TARGET.write_text(text, encoding="utf-8")
py_compile.compile(str(TARGET), doraise=True)

print("Jz support added successfully.")
print(f"Updated: {TARGET}")
print(f"Backup : {BACKUP}")
print()
print("Implemented source:")
print("  Delta Ex = Kx * eps^{-1}_Fourier @ Jz")
print("  Delta Ey = Ky * eps^{-1}_Fourier @ Jz")
print("  Delta Ux = -i Jy")
print("  Delta Uy = +i Jx")
print()
print("Existing Jx/Jy source terms are unchanged.")
print("Next run: python examples/test_dipole_jz.py")
