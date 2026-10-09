import numpy as np


# ============================================================
# INTERNAL VALIDATION HELPERS
# ============================================================

def _as_complex_1d(array, name):
    """
    Convert input to a one-dimensional complex NumPy array.
    """

    array = np.asarray(
        array,
        dtype=np.complex128,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional array."
        )

    return array


def _check_same_shape(a, b, name_a, name_b):
    """
    Ensure two arrays have identical shapes.
    """

    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must have the same shape. "
            f"Got {a.shape} and {b.shape}."
        )


# ============================================================
# 1. FOURIER REPRESENTATION OF THE DIPOLE CURRENT
# ============================================================

def dipole_fourier_current(
    expansion,
    kp,
    position,
    current,
):
    """
    Construct the Fourier components of a periodic array
    of electric dipoles.

    For each reciprocal-lattice harmonic G,

        J_alpha(G)
            = j_alpha
              exp[-i (q + G) . rho_d]

    where

        q     = (qx, qy)
        G     = (Gx, Gy)
        rho_d = (xd, yd)

    Parameters
    ----------
    expansion : khepri.expansion.Expansion
        Khepri reciprocal-space expansion.

    kp : tuple of float
        Bloch in-plane wavevector

            (qx, qy)

        in the same reciprocal-space units as
        ``expansion.g_vectors``.

    position : tuple of float
        Dipole position

            (xd, yd, zd)

        Only xd and yd enter the Fourier phase.
        zd determines the source plane in the layer stack.

    current : tuple of complex
        Electric-current amplitudes

            (jx, jy, jz)

    Returns
    -------
    Jx, Jy, Jz : ndarray
        Fourier current components for every retained
        reciprocal-lattice harmonic.

    Notes
    -----
    This function generates Jz as well, but the source-jump
    implementation below currently supports only x/y dipoles.
    """

    if len(kp) != 2:
        raise ValueError(
            "kp must contain exactly (qx, qy)."
        )

    if len(position) != 3:
        raise ValueError(
            "position must contain exactly (xd, yd, zd)."
        )

    if len(current) != 3:
        raise ValueError(
            "current must contain exactly (jx, jy, jz)."
        )

    # --------------------------------------------------------
    # Bloch vector
    # --------------------------------------------------------

    qx, qy = kp


    # --------------------------------------------------------
    # Dipole position
    # --------------------------------------------------------

    xd, yd, _zd = position


    # --------------------------------------------------------
    # Dipole current
    # --------------------------------------------------------

    jx, jy, jz = current


    # --------------------------------------------------------
    # Reciprocal-lattice vectors
    #
    # shape:
    #
    #     (2, N_G)
    #
    # row 0 -> Gx
    # row 1 -> Gy
    # --------------------------------------------------------

    g = np.asarray(
        expansion.g_vectors
    )

    if g.ndim != 2 or g.shape[0] != 2:
        raise ValueError(
            "expansion.g_vectors must have shape (2, N_G)."
        )

    Gx = g[0]
    Gy = g[1]


    # --------------------------------------------------------
    # Fourier phase
    # --------------------------------------------------------

    phase = np.exp(
        -1j
        * (
            (qx + Gx) * xd
            +
            (qy + Gy) * yd
        )
    )


    # --------------------------------------------------------
    # Fourier current
    # --------------------------------------------------------

    Jx = np.asarray(
        jx * phase,
        dtype=np.complex128,
    )

    Jy = np.asarray(
        jy * phase,
        dtype=np.complex128,
    )

    Jz = np.asarray(
        jz * phase,
        dtype=np.complex128,
    )


    return Jx, Jy, Jz


# ============================================================
# 2. ELECTROMAGNETIC SOURCE JUMP
# ============================================================

def dipole_source_jump_xy(
    Jx,
    Jy,
):
    """
    Construct the source jump for an in-plane electric dipole
    using Khepri's transverse-field convention.

    Khepri's transverse state is

        psi = [Ex, Ey, Ux, Uy]

    where numerical verification gives

        U = -i H.

    For an electric surface current,

        Delta Hx = +Jy
        Delta Hy = -Jx

    therefore

        Delta Ux = -i Jy
        Delta Uy = +i Jx.

    Hence

        b = [
            Delta Ex,
            Delta Ey,
            Delta Ux,
            Delta Uy
        ]

          = [
            0,
            0,
            -i Jy,
            +i Jx
        ].

    Parameters
    ----------
    Jx, Jy : ndarray
        Fourier components of the in-plane electric current.

    Returns
    -------
    b : ndarray
        Khepri source-jump vector with shape

            (4 * N_G,)

    Notes
    -----
    This function intentionally supports only x/y dipoles.

    A z-oriented dipole requires additional terms involving
    Kx, Ky and the appropriate Fourier permittivity operator.
    """

    Jx = _as_complex_1d(
        Jx,
        "Jx",
    )

    Jy = _as_complex_1d(
        Jy,
        "Jy",
    )

    _check_same_shape(
        Jx,
        Jy,
        "Jx",
        "Jy",
    )


    zeros = np.zeros_like(
        Jx,
        dtype=np.complex128,
    )


    b = np.concatenate(
        [
            zeros,        # Delta Ex
            zeros,        # Delta Ey
            -1j * Jy,     # Delta Ux
            +1j * Jx,     # Delta Uy
        ]
    )


    return b


# ============================================================
# 3. SOLVE THE DIPOLE SOURCE PLANE
# ============================================================

def solve_dipole_source_plane(
    R0,
    Sl,
    Sr,
    b,
):
    """
    Solve the self-consistent dipole radiation problem at the
    source plane.

    Parameters
    ----------
    R0 : ndarray
        Khepri reference eigenbasis matrix

            R0 = [[W0,  W0],
                  [-V0, V0]]

        mapping reference modal amplitudes to

            [Ex, Ey, Ux, Uy].

    Sl : ndarray
        Partial scattering matrix containing everything
        to the LEFT of the dipole source plane.

    Sr : ndarray
        Partial scattering matrix containing everything
        to the RIGHT of the dipole source plane.

    b : ndarray
        Electromagnetic source jump in Khepri's convention

            [Delta Ex,
             Delta Ey,
             Delta Ux,
             Delta Uy].

    Returns
    -------
    c_plus : ndarray
        Dipole-generated modal amplitudes travelling
        toward +z / right / emergence.

    c_minus : ndarray
        Dipole-generated modal amplitudes travelling
        toward -z / left / incidence.

    Notes
    -----
    The source-plane fields satisfy

        psi_right - psi_left = b.

    Multiple reflections between the left and right parts of
    the structure are included self-consistently.
    """

    R0 = np.asarray(
        R0,
        dtype=np.complex128,
    )

    Sl = np.asarray(
        Sl,
        dtype=np.complex128,
    )

    Sr = np.asarray(
        Sr,
        dtype=np.complex128,
    )

    b = _as_complex_1d(
        b,
        "b",
    )


    # --------------------------------------------------------
    # Basic dimensional checks
    # --------------------------------------------------------

    if R0.ndim != 2:
        raise ValueError(
            "R0 must be a two-dimensional matrix."
        )

    if R0.shape[0] != R0.shape[1]:
        raise ValueError(
            "R0 must be square."
        )

    if R0.shape[0] != len(b):
        raise ValueError(
            "R0 dimension must match the source vector b."
        )

    if Sl.ndim != 4 or Sl.shape[:2] != (2, 2):
        raise ValueError(
            "Sl must have shape (2, 2, N, N)."
        )

    if Sr.ndim != 4 or Sr.shape[:2] != (2, 2):
        raise ValueError(
            "Sr must have shape (2, 2, N, N)."
        )


    # --------------------------------------------------------
    # Convert physical source jump to Khepri's common
    # reference modal basis.
    # --------------------------------------------------------

    source_modal = np.linalg.solve(
        R0,
        b,
    )


    if len(source_modal) % 2 != 0:
        raise ValueError(
            "Reference modal source must contain an even "
            "number of components."
        )


    s_plus, s_minus = np.split(
        source_modal,
        2,
    )


    n = len(
        s_plus
    )


    if Sl.shape[2:] != (n, n):
        raise ValueError(
            "Sl modal dimension is inconsistent with R0."
        )

    if Sr.shape[2:] != (n, n):
        raise ValueError(
            "Sr modal dimension is inconsistent with R0."
        )


    # --------------------------------------------------------
    # Reflection operators seen from source plane
    # --------------------------------------------------------

    # Wave travelling left sees this reflection operator.
    R_left = Sl[
        1,
        1,
    ]

    # Wave travelling right sees this reflection operator.
    R_right = Sr[
        0,
        0,
    ]


    # --------------------------------------------------------
    # Source-plane equations
    #
    # psi_right - psi_left = b
    #
    # gives:
    #
    # c_plus - R_left c_minus = s_plus
    #
    # R_right c_plus - c_minus = s_minus
    # --------------------------------------------------------

    I = np.eye(
        n,
        dtype=np.complex128,
    )


    A = np.block(
        [
            [
                I,
                -R_left,
            ],
            [
                R_right,
                -I,
            ],
        ]
    )


    rhs = np.concatenate(
        [
            s_plus,
            s_minus,
        ]
    )


    solution = np.linalg.solve(
        A,
        rhs,
    )


    c_plus = solution[
        :n
    ]

    c_minus = solution[
        n:
    ]


    return c_plus, c_minus


# ============================================================
# 4. RECONSTRUCT FIELDS IMMEDIATELY AROUND THE DIPOLE
# ============================================================

def dipole_source_plane_fields(
    R0,
    Sl,
    Sr,
    c_plus,
    c_minus,
):
    """
    Reconstruct transverse Fourier fields immediately on the
    two sides of the dipole plane.

    Parameters
    ----------
    R0 : ndarray
        Khepri reference eigenbasis matrix.

    Sl, Sr : ndarray
        Left and right partial scattering matrices.

    c_plus : ndarray
        Dipole-generated wave travelling toward +z.

    c_minus : ndarray
        Dipole-generated wave travelling toward -z.

    Returns
    -------
    psi_left : ndarray
        Fourier field state immediately to the left of source:

            [Ex, Ey, Ux, Uy].

    psi_right : ndarray
        Fourier field state immediately to the right of source:

            [Ex, Ey, Ux, Uy].

    Notes
    -----
    These fields should satisfy

        psi_right - psi_left = b

    where b is the source jump used in
    ``solve_dipole_source_plane``.
    """

    R0 = np.asarray(
        R0,
        dtype=np.complex128,
    )

    Sl = np.asarray(
        Sl,
        dtype=np.complex128,
    )

    Sr = np.asarray(
        Sr,
        dtype=np.complex128,
    )

    c_plus = _as_complex_1d(
        c_plus,
        "c_plus",
    )

    c_minus = _as_complex_1d(
        c_minus,
        "c_minus",
    )


    _check_same_shape(
        c_plus,
        c_minus,
        "c_plus",
        "c_minus",
    )


    R_left = Sl[
        1,
        1,
    ]

    R_right = Sr[
        0,
        0,
    ]


    # --------------------------------------------------------
    # LEFT side of source
    #
    # +z component:
    #     reflection of c_minus from left stack
    #
    # -z component:
    #     directly emitted c_minus
    # --------------------------------------------------------

    left_modal = np.concatenate(
        [
            R_left @ c_minus,
            c_minus,
        ]
    )


    psi_left = (
        R0
        @
        left_modal
    )


    # --------------------------------------------------------
    # RIGHT side of source
    #
    # +z component:
    #     directly emitted c_plus
    #
    # -z component:
    #     reflection of c_plus from right stack
    # --------------------------------------------------------

    right_modal = np.concatenate(
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


    return psi_left, psi_right


# ============================================================
# 5. PROPAGATE THE DIPOLE WAVES TO THE EXTERIORS
# ============================================================

def dipole_exterior_amplitudes(
    Sl,
    Sr,
    c_plus,
    c_minus,
):
    """
    Calculate the dipole-generated modal amplitudes reaching
    the two exterior half-spaces.

    Parameters
    ----------
    Sl : ndarray
        Left partial scattering matrix.

    Sr : ndarray
        Right partial scattering matrix.

    c_plus : ndarray
        Source wave travelling toward +z.

    c_minus : ndarray
        Source wave travelling toward -z.

    Returns
    -------
    c_left : ndarray
        Outgoing amplitudes entering the incidence exterior.

    c_right : ndarray
        Outgoing amplitudes entering the transmission exterior.
    """

    c_plus = _as_complex_1d(
        c_plus,
        "c_plus",
    )

    c_minus = _as_complex_1d(
        c_minus,
        "c_minus",
    )


    _check_same_shape(
        c_plus,
        c_minus,
        "c_plus",
        "c_minus",
    )


    # --------------------------------------------------------
    # Standard scattering convention:
    #
    # [b_left ]   [S11 S12] [a_left ]
    # [b_right] = [S21 S22] [a_right]
    #
    # Source enters LEFT stack from its right port.
    # --------------------------------------------------------

    c_left = (
        Sl[0, 1]
        @
        c_minus
    )


    # Source enters RIGHT stack from its left port.

    c_right = (
        Sr[1, 0]
        @
        c_plus
    )


    return c_left, c_right


# ============================================================
# 6. EXTERIOR FAR-FIELD POWER FLUX
# ============================================================

def dipole_outgoing_flux(
    expansion,
    electric_field,
    kp,
    wavelength,
    epsilon=1.0,
    return_orders=False,
    grazing_tolerance=1e-12,
):
    """
    Calculate outward Khepri-normalized power flux carried by
    Fourier diffraction orders in a homogeneous, lossless
    exterior medium.

    This follows the same normalization convention as the
    numerator of Khepri's ``poynting_fluxes()`` but does NOT
    normalize by an incident plane-wave flux.

    Parameters
    ----------
    expansion : Expansion
        Khepri reciprocal-space expansion.

    electric_field : ndarray
        Transverse Fourier electric field

            [Ex_1 ... Ex_N,
             Ey_1 ... Ey_N]

    kp : tuple
        Bloch wavevector (qx, qy).

    wavelength : float
        Vacuum wavelength.

    epsilon : float
        Real relative permittivity of the homogeneous exterior.

    return_orders : bool
        If True, return the individual diffraction-order
        contributions as well.

    grazing_tolerance : float
        Threshold below which |kz| is treated as a grazing
        diffraction condition.

    Returns
    -------
    total_flux : float
        Total outward Khepri-normalized flux.

    order_flux : ndarray, optional
        Flux contribution from every Fourier order.

    Notes
    -----
    This is not an absolute power in watts.

    Use it consistently for comparisons or normalize against
    an identically implemented reference system.

    Complex/absorbing exterior media are intentionally not
    supported by this helper.
    """

    electric_field = _as_complex_1d(
        electric_field,
        "electric_field",
    )


    epsilon_complex = complex(
        epsilon
    )

    if abs(
        epsilon_complex.imag
    ) > 1e-14:
        raise NotImplementedError(
            "dipole_outgoing_flux currently supports only "
            "lossless exterior media with real epsilon."
        )


    if len(
        electric_field
    ) % 2 != 0:
        raise ValueError(
            "electric_field must contain equal Ex and Ey blocks."
        )


    Ex, Ey = np.split(
        electric_field,
        2,
    )


    # --------------------------------------------------------
    # Khepri returns normalized wavevectors:
    #
    #     kx/k0, ky/k0, kz/k0
    # --------------------------------------------------------

    kx, ky, kz = expansion.k_vectors(
        kp,
        wavelength,
        float(
            epsilon_complex.real
        ),
    )


    if Ex.shape != kx.shape:
        raise ValueError(
            "Electric-field harmonic count does not match "
            "the reciprocal-space expansion."
        )


    # --------------------------------------------------------
    # Detect exact/near grazing diffraction orders.
    #
    # Ez = -(kx Ex + ky Ey)/kz becomes singular exactly at
    # a Rayleigh threshold, so silently evaluating there is
    # dangerous.
    # --------------------------------------------------------

    grazing = (
        np.abs(kz)
        <
        grazing_tolerance
    )


    if np.any(
        grazing
    ):
        indices = np.where(
            grazing
        )[0]

        raise ValueError(
            "A grazing diffraction order was encountered at "
            f"indices {indices.tolist()}. "
            "Shift wavelength or kp slightly away from the "
            "Rayleigh threshold."
        )


    # --------------------------------------------------------
    # Longitudinal E component from k . E = 0
    # --------------------------------------------------------

    Ez = -(
        kx * Ex
        +
        ky * Ey
    ) / kz


    # --------------------------------------------------------
    # Khepri-normalized outward flux per order.
    #
    # Lossless evanescent orders have Re(kz)=0 and therefore
    # contribute zero exterior power.
    # --------------------------------------------------------

    order_flux = (
        np.real(kz)
        *
        (
            np.abs(Ex) ** 2
            +
            np.abs(Ey) ** 2
            +
            np.abs(Ez) ** 2
        )
    )


    # Remove tiny numerical negative values.

    order_flux = np.real_if_close(
        order_flux
    ).real


    tiny_negative = (
        (order_flux < 0)
        &
        (
            np.abs(order_flux)
            <
            1e-13
        )
    )

    order_flux[
        tiny_negative
    ] = 0.0


    total_flux = float(
        np.sum(
            order_flux
        ).real
    )


    if return_orders:
        return (
            total_flux,
            order_flux,
        )


    return total_flux


# ============================================================
# 7. NEAR-FIELD POYNTING FLUX AT THE SOURCE PLANE
# ============================================================

def dipole_nearfield_flux(
    psi,
):
    """
    Calculate the z-directed Khepri-normalized Poynting flux
    from the transverse Fourier field state.

    Khepri state ordering:

        psi = [Ex, Ey, Ux, Uy]

    with

        U = -i H.

    Since

        H = i U,

    the usual Poynting expression

        Re(Ex Hy* - Ey Hx*)

    becomes

        Im(Ex Uy* - Ey Ux*).

    Parameters
    ----------
    psi : ndarray
        Transverse Fourier field state

            [Ex, Ey, Ux, Uy].

    Returns
    -------
    Sz : float
        Net +z-directed Khepri-normalized flux summed over all
        retained Fourier harmonics.

    Notes
    -----
    Positive Sz means net power flow toward +z.

    Negative Sz means net power flow toward -z.

    This is a normalized solver quantity, not watts.
    """

    psi = _as_complex_1d(
        psi,
        "psi",
    )


    if len(
        psi
    ) % 4 != 0:
        raise ValueError(
            "psi length must be divisible by four."
        )


    N = (
        len(psi)
        //
        4
    )


    Ex = psi[
        0:N
    ]

    Ey = psi[
        N:2 * N
    ]

    Ux = psi[
        2 * N:3 * N
    ]

    Uy = psi[
        3 * N:4 * N
    ]


    Sz = np.imag(
        np.sum(
            Ex * np.conj(Uy)
            -
            Ey * np.conj(Ux)
        )
    )


    return float(
        np.real(
            Sz
        )
    )


# ============================================================
# 8. TOTAL POWER LEAVING THE DIPOLE PLANE
# ============================================================

def dipole_total_nearfield_power(
    psi_left,
    psi_right,
):
    """
    Calculate the total Khepri-normalized power leaving the
    dipole plane.

    Coordinate convention:

        +z -> right / emergence
        -z -> left  / incidence

    Therefore:

        P_left  = -Sz_left
        P_right = +Sz_right

    Parameters
    ----------
    psi_left : ndarray
        Field state immediately to the left of the source.

    psi_right : ndarray
        Field state immediately to the right of the source.

    Returns
    -------
    P_left : float
        Outward power emitted toward -z.

    P_right : float
        Outward power emitted toward +z.

    P_total : float
        Sum of the two outward fluxes.
    """

    Sz_left = dipole_nearfield_flux(
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
    )