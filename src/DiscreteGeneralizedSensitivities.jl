module DiscreteGeneralizedSensitivities

@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end DiscreteGeneralizedSensitivities

using Base: Fix1, Fix2, Fix

using DifferentiationInterface
using ForwardDiff: ForwardDiff

export DiscreteSensitivityProblemCfg, DiscreteSensitivityBurgers
export get_initial_conditions
export cfl_safety_factor, cs_scaling_factor, cr_scaling_factor, alpha
export nonlinear_f, nonlinear_df, u_exact, u_broad, u_gen
export ξ, dξ_dp, jump_size

export GTVClosure
export solve_pde!, solve_pde_and_estimate_Cr!, compute_Cr, compute_Cs, compute_Ct

const fdiff_backend = AutoForwardDiff()

"""
    Fixed_t_x{F, T, X}

Fix some callable with signature `f(t, x, p)` at a specific `t` and `x`. 
"""
struct Fixed_t_x{F,T,X}
    fn::F
    t::T
    x::X
end

function (fix_tx::Fixed_t_x{F,T,X})(p) where {F,T,X}
    return fix_tx.fn(fix_tx.t, fix_tx.x, p)
end

"""
    Fixed_t_p{F, T, X}

Fix some callable with signature `f(t, x, p)` at a specific `t` and `p`. 
"""
struct Fixed_t_p{F,T,P}
    fn::F
    t::T
    p::P
end

function (fix_tp::Fixed_t_p{F,T,P})(x) where {F,T,P}
    return fix_tp.fn(fix_tp.t, x, fix_tp.p)
end

abstract type DiscreteSensitivityProblemCfg{T} end

"""
    cfl_safety_factor(::DiscreteSensitivityProblemCfg)

Get the additional CFL safety factor for this problem configuration. Setting this closer to zero increases numerical dissipation.
"""
cfl_safety_factor(::DiscreteSensitivityProblemCfg{T}) where {T} = one(T)

"""
    cs_scaling_factor(::DiscreteSensitivityProblemCfg)

Get the additional scaling factor for ``C_s``, chosen to satisfy Assumption 8 in [Hüser's dissertation](https://doi.org/10.18154/RWTH-2022-06229).
"""
cs_scaling_factor(::DiscreteSensitivityProblemCfg{T}) where {T} = convert(T, 2)
"""
    cr_scaling_factor(::DiscreteSensitivityProblemCfg)

Get the additional scaling factor on ``C_r``, chosen to satisfy Assumptions 8, 9, 10 in [Hüser's dissertation](https://doi.org/10.18154/RWTH-2022-06229).
"""
cr_scaling_factor(::DiscreteSensitivityProblemCfg{T}) where {T} = one(T)

@doc raw"""
    alpha(::DiscreteSensitivityProblemCfg)

Get the value of alpha chosen so that ``\Xi + \Delta x^\alpha`` escapes the shock region.
"""
alpha(::DiscreteSensitivityProblemCfg{T}) where {T} = convert(T, 0.5)

f_burg(u) = u^2 / 2
const df_burg_fwdprep = prepare_derivative(f_burg, fdiff_backend, 1.0)
df_burg(u) = derivative(f_burg, df_burg_fwdprep, fdiff_backend, u)

u0_ramp(x, p) = (1 + p) * x * X(x, (zero(x), one(x)))

ξ_ramp(t, p) = sqrt(1 + (1 + p) * t)
const dξ_ramp_fwdprep = prepare_derivative(Fix1(ξ_ramp, 0.0), fdiff_backend, 0.0)
dξ_ramp_dp(t, p) = derivative(Fix1(ξ_ramp, t), fdiff_backend, p)

u_ramp(t, x, p) = ((1 + p) * x) / (1 + (1 + p) * t) * X(x, zero(x), ξ_ramp(t, p))
Δu_ramp(t, p) = (1 + p) / ξ_ramp(t, p)

function du_ramp_dp(t, x, p)
    u_fixed = Fixed_t_x(u_ramp, t, x)
    return X(x, 0, ξ_ramp(t, p)) * derivative(u_fixed, fdiff_backend, p)
end

"""
    DiscreteSensitivityBurgers

Configuration for computing discrete sensitivities to solutions to Burgers equation.
Equipped with the ramp intial condition and corresponding exact solution.

Allows setting the CFL safety factor as well as scaling on ``C_s`` and ``C_r``.
"""
struct DiscreteSensitivityBurgers <: DiscreteSensitivityProblemCfg{Float64}
    CFL_SAFETY::Float64
    CS_FACTOR::Float64
    CR_FACTOR::Float64
    ALPHA::Float64
end

cfl_safety_factor(cfg::DiscreteSensitivityBurgers) = cfg.CFL_SAFETY
cs_scaling_factor(cfg::DiscreteSensitivityBurgers) = cfg.CS_FACTOR
cr_scaling_factor(cfg::DiscreteSensitivityBurgers) = cfg.CR_FACTOR
alpha(cfg::DiscreteSensitivityBurgers) = cfg.ALPHA

"""
    nonlinear_f(::DiscreteSensitivityProblemCfg)

Get the flux function ``f(u)`` for the PDE.
"""
nonlinear_f(::DiscreteSensitivityBurgers) = f_burg

"""
    nonlinear_df(::DiscreteSensitivityProblemCfg)

Get the derivative of the nonlinear flux function ``f'(u)`` for the PDE.
"""
nonlinear_df(::DiscreteSensitivityBurgers) = df_burg

"""
    u0(::DiscreteSensitivityProblemCfg)

Get the initial conditions ``u(0, x, p)``.
"""
u0(::DiscreteSensitivityBurgers) = u0_ramp

"""
    u_exact(::DiscreteSensitivityProblemCfg)

Get the exact solution ``u(t, x, p)``.
"""
u_exact(::DiscreteSensitivityBurgers) = u_ramp

"""
    u_broad(::DiscreteSensitivityProblemCfg)

Get the analytic broad tangent ``u^{(1)}(t, x, p)``.
"""
u_broad(::DiscreteSensitivityBurgers) = du_ramp_dp

@doc raw"""
    ξ(::DiscreteSensitivityProblemCfg)

Get the position(s) ``\xi(t, p)`` of any discontinuities in the exact solution.
"""
ξ(::DiscreteSensitivityBurgers) = ξ_ramp

"""
    dξ_dp(::DiscreteSensitivityProblemCfg)

Get the sensitivity of the shock positions to the parameters.
"""
dξ_dp(::DiscreteSensitivityBurgers) = dξ_ramp_dp

"""
    jump_size(::DiscreteSensitivityProblemCfg)

Get the size(s) of the discontinuities in the exact solution.
"""
jump_size(::DiscreteSensitivityBurgers) = Δu_ramp

"""
    u_gen(cfg::::DiscreteSensitivityProblemCfg)

The analytic generalized tangent of the exact solution. See Section 5 in [Hüser's dissertation](https://doi.org/10.18154/RWTH-2022-06229).
"""
function u_gen(cfg)
    return (t, x, p, pdot) -> let cfg = cfg
        res = u_broad(cfg)(t, x, p) * pdot
        Δ = jump_size(cfg)(t, p)
        ξk = ξ(cfg)(t, p)
        ξkdot = dξ_dp(cfg)(t, p) * pdot
        if ξkdot > 0
            res += X(x, ξk, ξk + ξkdot) * Δ
        elseif ξkdot < 0
            res -= X(x, ξk + ξkdot, ξk) * Δ
        end
        return res
    end
end

@doc raw"""
    get_initial_conditions(xs, p, pdot, cfg)

Get the initial conditions ``(U, U_^{(1)}{broad}, \Xi, \Xi^{(1)})`` given ``(x_j, p, p^{(1)})``.
"""
function get_initial_conditions(xs, p, pdot, cfg)
    U = map(Fix2(u0(cfg), p), xs)
    Udot = map(xs) do x
        first(pushforward(Fix1(u0(cfg), x), fdiff_backend, p, (pdot,)))
    end
    Ξ = ξ(cfg)(0.0, p)
    Ξdot = dξ_dp(cfg)(0.0, p) * pdot
    return (U, Udot, Ξ, Ξdot)
end

struct GTVClosure{T,XT}
    xs::XT
    cr_dx_alpha::T
    U::Vector{T}
    Udot::Vector{T}
    Ξ::T
    Ξdot::T
end

function (gtv::GTVClosure{T,XT})(x) where {T,XT}
    # edges of shock region
    ΞL = gtv.Ξ - gtv.cr_dx_alpha
    ΞR = gtv.Ξ + gtv.cr_dx_alpha
    # discrete jump height
    U_disc = piecewise_constant_interp(gtv.U, gtv.xs)
    ΔU = U_disc(ΞL) - U_disc(ΞR)
    # evaluate the broad tangent with the shock cut
    U_broad = piecewise_constant_interp(gtv.Udot, gtv.xs)(x) * !X(x, ΞL, ΞR)
    # evaluate the shock shift itself
    Ξ_shift = gtv.Ξdot
    shock_shift = if Ξ_shift > 0
        ΔU * X(x, gtv.Ξ, gtv.Ξ + Ξ_shift)
    else
        -ΔU * X(x, gtv.Ξ + Ξ_shift, gtv.Ξ)
    end
    # return the value of the GTV
    return U_broad + shock_shift
end

function compute_Ct(U, cfg)
    return cfl_safety_factor(cfg) / maximum(abs ∘ nonlinear_df(cfg), U)
end

function compute_Ct(u, xs, cfg)
    return cfl_safety_factor(cfg) / maximum(abs ∘ nonlinear_df(cfg) ∘ u, xs)
end

compute_Cs(U, cfg) = cs_scaling_factor(cfg) * compute_Ct(U, cfg)
compute_Cs(u, xs, cfg) = cs_scaling_factor(cfg) * compute_Ct(u, xs, cfg)

function compute_Cr(U, t, xs, p, cfg)
    u_ex_fix = Fixed_t_p(u_exact(cfg), t, p)
    A1 = maximum((abs ∘ nonlinear_df(cfg) ∘ u_ex_fix), xs)
    A2 = 2 + maximum((abs ∘ nonlinear_df(cfg)), U)
    return cr_scaling_factor(cfg) * (max(A1, A2) + 1) * compute_Cs(U, cfg)
end

function fdiff_eps(arg::T) where {T<:Real}
    cbrt_eps = cbrt(eps(T))
    h = 2^(round(log2((1 + abs(arg)) * cbrt_eps)))
    return h
end

function _step_lax_friedrichs_serial_impl!(U_next, U, Δt, xs::AbstractRange, cfg)
    f = nonlinear_f(cfg)
    @views map!(U_next[begin+1:end-1], U[begin:end-2], U[begin+2:end]) do U_L, U_R
        return (U_L + U_R) / 2 + Δt / (2 * step(xs)) * (f(U_L) - f(U_R))
    end
    # apply extrapolation
    U_next[begin] = U_next[begin+1]
    U_next[end] = U_next[end-1]
    # in-place!
    return nothing
end

function _step_lax_friedrichs_threaded_impl!(U_next, U, Δt, xs::AbstractRange, cfg)
    f = nonlinear_f(cfg)
    tmap!(
        (U_L, U_R) -> let f = f, Δt = Δt, xs = xs
            (U_L + U_R) / 2 + Δt / (2 * step(xs)) * (f(U_L) - f(U_R))
        end,
        @view(U_next[begin+1:end-1]),
        @view(U[begin:end-2]),
        @view(U[begin+2:end]),
    )
    # apply extrapolation
    U_next[begin] = U_next[begin+1]
    U_next[end] = U_next[end-1]
    # in-place!
    return nothing
end

function step_lax_friedrichs!(U_next, U, Δt, xs, cfg; threading_thold = 10_000)
    if length(xs) > threading_thold
        return _step_lax_friedrichs_serial_impl!(U_next, U, Δt, xs, cfg)
    else
        return _step_lax_friedrichs_threaded_impl!(U_next, U, Δt, xs, cfg)
    end
end

function next_shock_location(Ξ, U, xs, Δt, cfg)
    U_shock = piecewise_constant_interp(U, xs)(Ξ)
    return Ξ + Δt * nonlinear_df(cfg)(U_shock)
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

function solve_pde!(U, xs::AbstractRange, T_end, cfg; recompute_dt = false)
    dx = step(xs)
    dt = compute_Ct(U, cfg) * dx
    t = zero(T_end)
    U_temp = similar(U)
    stepping = true
    while stepping
        if recompute_dt
            dt = compute_Ct(U, cfg) * dx
        end
        if t + dt ≥ T_end
            dt = T_end - t
            stepping = false
        end
        step_lax_friedrichs!(U_temp, U, dt, xs, cfg)
        t += dt
        U .= U_temp
    end
    return U
end

function solve_pde_and_estimate_Cr!(
    U,
    xs::AbstractRange,
    T_end,
    p,
    Cr_0,
    cfg;
    recompute_dt = false,
)
    dx = step(xs)
    dt = compute_Ct(U, cfg) * dx
    Cr = Cr_0
    t = zero(T_end)
    U_temp = similar(U)
    stepping = true
    while stepping
        if recompute_dt
            dt = compute_Ct(U, cfg) * dx
        end
        if t + dt ≥ T_end
            dt = T_end - t
            stepping = false
        end
        step_lax_friedrichs!(U_temp, U, dt, xs, cfg)
        t += dt
        U .= U_temp
        Cr = max(Cr, compute_Cr(U, t, xs, p, cfg))
    end
    return (U, Cr)
end

function solve_pde!(U, Udot, xs::AbstractRange, T_end, cfg; recompute_dt = false)
    dx = step(xs)
    dt = compute_Ct(U, cfg) * dx
    t = zero(T_end)
    U_temp = similar(U)
    Udot_temp = similar(Udot)
    pushforward_prep = prepare_pushforward(
        step_lax_friedrichs!,
        U_temp,
        fdiff_backend,
        U,
        (Udot,),
        Constant(dt),
        Constant(xs),
        Constant(cfg),
    )
    stepping = true
    while stepping
        if recompute_dt
            dt = compute_Ct(U, cfg) * dx
        end
        if t + dt ≥ T_end
            dt = T_end - t
            stepping = false
        end
        value_and_pushforward!(
            step_lax_friedrichs!,
            U_temp,
            (Udot_temp,),
            pushforward_prep,
            fdiff_backend,
            U,
            (Udot,),
            Constant(dt),
            Constant(xs),
            Constant(cfg),
        )
        t += dt
        U .= U_temp
        Udot .= Udot_temp
    end
    return (U, Udot)
end

function solve_pde!(
    U,
    Udot,
    Ξ,
    Ξdot,
    xs::AbstractRange,
    T_end,
    Cr_0,
    cfg;
    recompute_dt = false,
)
    dx = step(xs)
    dt = compute_Ct(U, cfg) * dx
    Cr = Cr_0
    t = zero(T_end)
    U_temp = similar(U)
    Udot_temp = similar(Udot)
    pushforward_prep = prepare_pushforward(
        step_lax_friedrichs!,
        U_temp,
        fdiff_backend,
        U,
        (Udot,),
        Constant(dt),
        Constant(xs),
        Constant(cfg),
    )
    stepping = true
    while stepping
        if recompute_dt
            dt = compute_Ct(U, cfg) * dx
        end
        if t + dt ≥ T_end
            dt = T_end - t
            stepping = false
        end
        if dt < 0
            break
        end
        Ξdot = next_shock_sensitivity(Ξ, Ξdot, U, Udot, xs, dt, Cr, cfg)
        Ξ = next_shock_location(Ξ, U, xs, dt, cfg)
        value_and_pushforward!(
            step_lax_friedrichs!,
            U_temp,
            (Udot_temp,),
            pushforward_prep,
            fdiff_backend,
            U,
            (Udot,),
            Constant(dt),
            Constant(xs),
            Constant(cfg),
        )
        t += dt
        U .= U_temp
        Udot .= Udot_temp
    end
    return (U, Udot, Ξ, Ξdot)
end

# Characteristic function of an interval [a, b]
X(x, a, b) = a ≤ x ≤ b
X(x, ival) = X(x, ival...)

function piecewise_constant_interp(data, xs::AbstractRange)
    # I am too lazy to write a functor. closure time.
    return (x,) -> let data = data, xs = xs
        x <= first(xs) && return first(data)
        x >= last(xs) && return last(data)
        offset = round(Int, (x - first(xs)) / step(xs))
        return data[begin+offset]
    end
end

end
