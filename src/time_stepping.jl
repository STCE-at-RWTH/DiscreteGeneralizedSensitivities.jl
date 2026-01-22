# LAX-FRIEDRICHS + FWD EULER

function f_conserved_lax_friedrichs(U, j, Δt, Δx, cfg)
    f = nonlinear_f(cfg)
    return 0.5 * (f(U[j]) + f(U[j+1])) - 0.5 * (Δx / Δt) * (U[j+1] - U[j])
end

function step_lax_friedrichs!(U_next, U, Δt, xs, cfg)
    f = nonlinear_f(cfg)
    C = Δt / (2 * step(xs))
    @tullio U_next[i] = (U[i-1] + U[i+1]) / 2 + C * f(U[i-1]) - f(U[i+1])
    # apply extrapolation
    U_next[begin] = U_next[begin+1]
    U_next[end] = U_next[end-1]
    # in-place!
    return nothing
end

function dΞ_dt_appx(Ξ, U, xs, Cr, cfg)
    U_shock = piecewise_constant_interp(U, xs)(Ξ)
    return nonlinear_df(cfg)(U_shock)
end

function dΞ_dt_appx(Ξ_dual::Dual{T}, U_dual, xs, Cr, cfg) where {T}
    Ξ = value(Ξ_dual)
    Ξdot = partials(Ξ_dual)
    U_const = piecewise_constant_interp(U_dual, xs)
    dΞ = nonlinear_df(cfg)(value(U_const(Ξ)))

    dx = step(xs)
    ΞL = Ξ - Cr * dx^alpha(cfg)
    ΞR = Ξ + Cr * dx^alpha(cfg)
    @assert(ΞL < ΞR)

    UL = value(U_const(ΞL))
    UdotL = partials(U_const(ΞL))
    UL_far = value(U_const(ΞL - 2 * dx))
    dUdx_L = (UL - UL_far) / (2 * dx)

    UR = value(U_const(ΞR))
    UdotR = partials(U_const(ΞR))
    UR_far = value(U_const(ΞR + 2 * dx))
    dUdx_R = (UR_far - UR) / (2 * dx)

    ΔU_shock = UL - UR

    fL = nonlinear_f(cfg)(UL)
    fR = nonlinear_f(cfg)(UR)
    dfL = nonlinear_df(cfg)(UL)
    dfR = nonlinear_df(cfg)(UR)

    A_L = (dfL * (dUdx_L * Ξdot + UdotL)) / ΔU_shock
    A_R = (dfR * (dUdx_R * Ξdot + UdotR)) / ΔU_shock
    B = (fL - fR) / (ΔU_shock^2)
    C_L = (dUdx_L * Ξdot + UdotL)
    C_R = (dUdx_R * Ξdot + UdotR)
    dΞdot = (A_L - A_R - B * (C_L - C_R))
    return Dual{T}(dΞ, dΞdot)
end

function next_shock_location(Ξ, U, xs, Δt, Cr, cfg)
    return Ξ + Δt * dΞ_dt_appx(Ξ, U, xs, Cr, cfg)
end

function next_shock_sensitivity(Ξ, Ξdot, U, Udot, xs, Δt, Cr, cfg)
    dx = step(xs)
    ΞL = Ξ - Cr * dx^alpha(cfg)
    ΞR = Ξ + Cr * dx^alpha(cfg)
    @assert(ΞL < ΞR)
    @assert(Δt > 0)

    U_const = piecewise_constant_interp(U, xs)
    Udot_const = piecewise_constant_interp(Udot, xs)

    UL = U_const(ΞL)
    UL_far = U_const(ΞL - 2 * dx)
    UR = U_const(ΞR)
    UR_far = U_const(ΞR + 2 * dx)
    ΔU_shock = UL - UR

    dUdx_L = (UL - UL_far) / (2 * dx)
    dUdx_R = (UR_far - UR) / (2 * dx)

    UdotL = Udot_const(ΞL)
    UdotR = Udot_const(ΞR)

    fL = nonlinear_f(cfg)(UL)
    fR = nonlinear_f(cfg)(UR)
    dfL = nonlinear_df(cfg)(UL)
    dfR = nonlinear_df(cfg)(UR)

    A_L = (dfL * (dUdx_L * Ξdot + UdotL)) / ΔU_shock
    A_R = (dfR * (dUdx_R * Ξdot + UdotR)) / ΔU_shock
    B = (fL - fR) / (ΔU_shock^2)
    C_L = (dUdx_L * Ξdot + UdotL)
    C_R = (dUdx_R * Ξdot + UdotR)
    return Ξdot + Δt * (A_L - A_R - B * (C_L - C_R))
end

# WEIGHTED ESSENTIALLY NON OSCILLATORY SCHEMES

# Smoothness measurement according to Jiang, Shu [1996]
# which improves over the original ENO paper

const _IS_2_k_FACTORS = @SArray [-1, 1]
const _IS_3_k_FACTORS = @SArray [1 -2 1; 1 -4 3;;; 1 -2 1; 1 0 -1;;; 1 -2 1; 3 -4 1]

function _weno_smoothness_measure_impl(stencil_arg, k, ::Val{2})
    # k doesn't matter here but for argument uniformity...
    return (_IS_2_k_FACTORS' * stencil_arg)^2
end

function _weno_smoothness_measure_impl(stencil_arg, k, ::Val{3})
    prefactors = @SVector [13 / 12, 1 / 4]
    return prefactors' * ((_IS_3_k_FACTORS[:, :, begin+k] * stencil_arg) .^ 2)
end

function _weno_smoothness_measure(stencil_arg, k, order::Val{R}) where {R}
    @assert length(stencil_arg) == R "Stencil size must match order!"
    @assert 0 ≤ k < R "k is an offset index; range 0 to R-1."
    return _weno_smoothness_measure_impl(stencil_arg, k, order)
end

const _WENO_OPTIMAL_WEIGHTS = @SMatrix [0.0 0.0 0.0; 1/3 2/3 0.0; 1/10 6/10 3/10]

# Table 1 in Jiang, Shu [1996]
#
# These look "off" because the interpolation happens at
# a half-grid point and the values are available at the full grid point

# 2.7 and 2.8 in Jiang, Shu [1996]
const _ENO_2_WEIGHTS = @SMatrix [
    -0.5 1.5
    0.5 0.5
]

const _ENO_3_WEIGHTS = @SMatrix [
    1/3 -7/6 11/6
    -1/6 5/6 1/3
    1/3 5/6 -1/6
]

function _eno_weights(order)
    if order == 2
        return _ENO_2_WEIGHTS
    elseif order == 3
        return _ENO_3_WEIGHTS
    else
        throw(ArgumentError("Weights for this order not available."))
    end
end

# estimate f^+ using the WENO scheme of order R
# given by a Val{R}() parameter
# f_num is the conservative numerical flux and has the form f_num_jplusonehalf(U_view, j, Δt, Δx)
function _estimate_f_pos(F, j, order::Val{R}; ε = 1.0e-6, p = 2) where {R}
    # Computation of stencil weighting by the smoothness measure
    # the "optimal weights" are chosen when all stencils are Very Smooth
    ISk = sacollect(
        SVector{R},
        (_weno_smoothness_measure(@view(F[(j-R+k+1):(j+k)]), k, order) for k = 0:R-1),
    )
    alpha_k = map(_WENO_OPTIMAL_WEIGHTS[R, :], ISk) do Ck, Ik
        return Ck * inv((ε + Ik)^p)
    end
    w = alpha_k / sum(alpha_k)

    res = zero(eltype(F))
    for k = 0:R-1
        for ℓ = 1:R
            res += w[begin+k] * _eno_weights(R)[begin+k, ℓ] * F[j+k-R+ℓ]
        end
    end
    return res
end

function _estimate_f_neg(F, j, order::Val{R}; ε = 1.0e-6, p = 2) where {R} end

function _weno_estimate_L_f_pos!(L, F, U, Δt, Δx, order::Val{R}, cfg) where {R}
    f = nonlinear_f(cfg)
    tforeach(axes(F, 1)[begin:end-1]) do j
        F[j] = 0.5 * (f(U[j]) + f(U[j+1])) - 0.5 * (Δx / Δt) * (U[j+1] - U[j])
    end
    # L = -inv(Δx)(f_hat_{j+1/2} - f_hat_{j-1/2})

    tforeach(axes(L, 1)[begin+R:end-R]) do j
        L[j] = -inv(Δx) * (_estimate_f_pos(F, j, order) - _estimate_f_pos(F, j - 1, order))
    end

    return L
end

function _weno_estimate_L_f_split!(f_pos, f_neg, L, U, Δt, Δx, order::Val{R}) where {R}
    return L
end

function _apply_extrapolation_bcs!(U, ::Val{R}) where {R}
    @view(U[begin:begin+R-1]) .= U[begin+R]
    @view(U[end-R+1:end]) .= U[end-R]
end

# mutates U★
function _TVD_RK3_1!(U★, U, L, F_buf, Δt, xs, weno_order::Val{R}, cfg) where {R}
    Δx = step(xs)
    _weno_estimate_L_f_pos!(L, F_buf, U, Δt, Δx, weno_order, cfg)
    @tullio U★[i] = U[i] + Δt * L[i]
    _apply_extrapolation_bcs!(U★, weno_order)
    return nothing
end

# mutates U★
function _TVD_RK3_2!(U★, U, L, F_buf, Δt, xs, weno_order::Val{R}, cfg) where {R}
    Δx = step(xs)
    _weno_estimate_L_f_pos!(L, F_buf, U★, Δt, Δx, weno_order, cfg)
    @tullio U★[i] = 0.75 * U[i] + 0.25 * U★[i] + 0.25 * Δt * L[i]
    _apply_extrapolation_bcs!(U★, weno_order)
    return nothing
end

#mutates U★
function _TVD_RK3_3!(U★, U, L, F_buf, Δt, xs, weno_order::Val{R}, cfg) where {R}
    Δx = step(xs)
    _weno_estimate_L_f_pos!(L, F_buf, U★, Δt, Δx, weno_order, cfg)
    @tullio U★[i] = (1 / 3) * U[i] + (2 / 3) * U★[i] + (2 / 3) * Δt * L[i]
    _apply_extrapolation_bcs!(U★, weno_order)
    return nothing
end

# mutates U_next and U★
function _TVD_RK3!(U_next, U, U★, L, F_buf, Δt, xs, weno_order::Val{R}, cfg) where {R}
    _TVD_RK3_1!(U★, U, L, F_buf, Δt, xs, weno_order, cfg)
    _TVD_RK3_2!(U★, U, L, F_buf, Δt, xs, weno_order, cfg)
    _TVD_RK3_3!(U★, U, L, F_buf, Δt, xs, weno_order, cfg)
    @. U_next = U★
    return nothing
end

# mutates U★ and stores Xi and Xi★ at the end of the passed-in buffers
# size of F should be one less than the size of U
# asserted in calling function?
function _TVD_RK3_1_with_shock!(
    U★,
    U,
    L,
    F_buf,
    Δt,
    xs,
    weno_order::Val{R},
    Cr,
    cfg,
) where {R}
    Δx = step(xs)
    U_data = @view U[begin:end-1]
    U★_data = @view U★[begin:end-1]
    L_U = @view L[begin:end-1]
    _weno_estimate_L_f_pos!(L_U, F_buf, U_data, Δt, Δx, weno_order, cfg)
    L[end] = dΞ_dt_appx(U[end], U_data, xs, Cr, cfg)
    @tullio U★[i] = U[i] + Δt * L[i]
    _apply_extrapolation_bcs!(U★_data, weno_order)
    return nothing
end

function _TVD_RK3_2_with_shock!(
    U★,
    U,
    L,
    F_buf,
    Δt,
    xs,
    weno_order::Val{R},
    Cr,
    cfg,
) where {R}
    Δx = step(xs)
    U★_data = @view U★[begin:end-1]
    L_U = @view L[begin:end-1]
    _weno_estimate_L_f_pos!(L_U, F_buf, U★_data, Δt, Δx, weno_order, cfg)
    L[end] = dΞ_dt_appx(U★[end], U★_data, xs, Cr, cfg)
    @tullio U★[i] = 0.75 * U[i] + 0.25 * U★[i] + 0.25 * Δt * L[i]
    _apply_extrapolation_bcs!(U★_data, weno_order)
    return nothing
end

function _TVD_RK3_3_with_shock!(
    U★,
    U,
    L,
    F_buf,
    Δt,
    xs,
    weno_order::Val{R},
    Cr,
    cfg,
) where {R}
    Δx = step(xs)
    U★_data = @view U★[begin:end-1]
    L_U = @view L[begin:end-1]
    _weno_estimate_L_f_pos!(L_U, F_buf, U★_data, Δt, Δx, weno_order, cfg)
    L[end] = dΞ_dt_appx(U★[end], U★_data, xs, Cr, cfg)
    @tullio U★[i] = (1 / 3) * U[i] + (2 / 3) * U★[i] + (2 / 3) * Δt * L[i]
    _apply_extrapolation_bcs!(U★_data, weno_order)
    return nothing
end

function _TVD_RK3_with_shock!(
    U_aug_next,
    U_aug,
    U_aug★,
    L,
    F_buf,
    Δt,
    xs,
    weno_order::Val{R},
    Cr,
    cfg,
) where {R}
    @assert length(U_aug) == length(L) && length(F_buf) + 1 == length(L)
    _TVD_RK3_1_with_shock!(U_aug★, U_aug, L, F_buf, Δt, xs, weno_order, Cr, cfg)
    _TVD_RK3_2_with_shock!(U_aug★, U_aug, L, F_buf, Δt, xs, weno_order, Cr, cfg)
    _TVD_RK3_3_with_shock!(U_aug★, U_aug, L, F_buf, Δt, xs, weno_order, Cr, cfg)
    @. U_aug_next = U_aug★
    return nothing
end

function solve_pde_weno!(
    U,
    xs::AbstractRange,
    T_end,
    cfg;
    recompute_dt = false,
    weno_order = Val{3}(),
)
    # compute Δt
    Δx = step(xs)
    Δt = compute_Ct(U, cfg) * Δx
    t = zero(T_end)
    # Work buffers
    U_next = similar(U)
    U★ = zero(U)
    L = zero(U)
    F = zero(U)
    stepping = true
    while stepping
        if recompute_dt
            Δt = compute_Ct(U, cfg) * Δx
        end
        if t + Δt >= T_end
            Δt = T_end - t
            stepping = false
        end
        _TVD_RK3!(U_next, U, U★, L, F, Δt, xs, weno_order, cfg)
        t += Δt
        U .= U_next
    end
    return U
end

function solve_pde_weno!(
    U,
    Ξ,
    xs::AbstractRange,
    T_end,
    Cr_0,
    cfg;
    recompute_dt = false,
    weno_order = Val{3}(),
)
    U_aug = similar(U, size(U) .+ (1,))
    U_aug[begin:end-1] .= U
    U_aug[end] = Ξ

    U★_aug = zero(U_aug)
    U_aug_next = similar(U_aug)

    L = zero(U_aug)
    F = zero(U)
    Δx = step(xs)
    Δt = compute_Ct(U, cfg) * Δx
    Cr = Cr_0
    t = zero(T_end)
    stepping = true
    while stepping
        if recompute_dt
            Δt = compute_Ct(U, cfg) * Δx
        end
        if t + Δt >= T_end
            Δt = T_end - t
            stepping = false
        end
        _TVD_RK3_with_shock!(U_aug_next, U_aug, U★_aug, L, F, Δt, xs, weno_order, Cr, cfg)
        t += Δt
        U_aug .= U_aug_next
    end
    U .= @view U_aug[begin:end-1]
    Ξ = U_aug[end]
    return (U, Ξ)
end

# END WEIGHTED NON OSCILLATORY SCHEMES
function solve_pde_weno!(
    U,
    Udot,
    Ξ,
    Ξdot,
    xs::AbstractRange,
    T_end,
    Cr_0,
    cfg;
    recompute_dt = false,
    weno_order = Val{3}(),
)
    U_aug = similar(U, size(U) .+ (1,))
    U_aug[begin:end-1] .= U
    U_aug[end] = Ξ

    Udot_aug = similar(Udot, size(Udot) .+ (1,))
    Udot_aug[begin:end-1] .= Udot
    Udot_aug[end] = Ξdot

    U★_aug = zero(U_aug)
    U_aug_next = similar(U_aug)
    Udot_aug_next = similar(Udot_aug)

    L = zero(U_aug)
    F = zero(U)
    Δx = step(xs)
    Δt = compute_Ct(U, cfg) * Δx
    Cr = Cr_0
    t = zero(T_end)
    stepping = true
    _prep = prepare_pushforward(
        _TVD_RK3_with_shock!,
        U_aug_next,
        fdiff_backend,
        U_aug,
        (Udot_aug,),
        Cache(U★_aug),
        Cache(L),
        Cache(F),
        Constant(Δt),
        Constant(xs),
        Constant(weno_order),
        Constant(Cr),
        Constant(cfg),
    )
    while stepping
        if recompute_dt
            Δt = compute_Ct(U, cfg) * Δx
        end
        if t + Δt >= T_end
            Δt = T_end - t
            stepping = false
        end
        value_and_pushforward!(
            _TVD_RK3_with_shock!,
            U_aug_next,
            (Udot_aug_next,),
            _prep,
            fdiff_backend,
            U_aug,
            (Udot_aug,),
            Cache(U★_aug),
            Cache(L),
            Cache(F),
            Constant(Δt),
            Constant(xs),
            Constant(weno_order),
            Constant(Cr),
            Constant(cfg),
        )
        t += Δt
        U_aug .= U_aug_next
        Udot_aug .= Udot_aug_next
    end
    U .= @view U_aug[begin:end-1]
    Udot .= @view Udot_aug[begin:end-1]
    Ξ = U_aug[end]
    Ξdot = Udot_aug[end]
    return (U, Udot, Ξ, Ξdot)
end
