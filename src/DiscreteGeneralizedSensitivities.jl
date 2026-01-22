module DiscreteGeneralizedSensitivities

@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end DiscreteGeneralizedSensitivities

using Base: Fix1, Fix2, Fix

using DifferentiationInterface
using ForwardDiff: ForwardDiff, Dual, value, partials
using LoopVectorization
using StaticArrays
using StaticArrays: sacollect
using Tullio

export DiscreteSensitivityProblemCfg
export get_initial_conditions
export cfl_safety_factor, cs_scaling_factor, cr_scaling_factor, alpha
export nonlinear_f, nonlinear_df, u_exact, u_broad, u_gen
export ξ, dξ_dp, jump_size

export GTVClosure
export solve_pde!, solve_pde_and_estimate_Cr!, compute_Cr, compute_Cs, compute_Ct
export solve_pde_weno!

const fdiff_backend = AutoForwardDiff()

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

"""
    fixed_t_x(f, t, x)

Fix some callable with signature `f(t, x, p)` at a specific `t` and `x`. 
"""
fixed_t_x(f, t, x) = Fix1(Fix1(f, t), x)

"""
    fixed_t_p(f, t, p)

Fix some callable with signature `f(t, x, p)` at a specific `t` and `p`. 
"""

fixed_t_p(f, t, p) = Fix2(Fix1(f, t), p)

include("time_stepping.jl")

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

"""
    alpha(::DiscreteSensitivityProblemCfg)

Get the value of alpha chosen so that ``Ξ + Δx^α`` escapes the shock region.
"""
alpha(::DiscreteSensitivityProblemCfg{T}) where {T} = convert(T, 0.5)

"""
    nonlinear_f(cfg::DiscreteSensitivityProblemCfg)

Get the flux function ``f(u)`` for the PDE problem set up by `cfg`.
"""
function nonlinear_f end

"""
    nonlinear_df(::DiscreteSensitivityProblemCfg)

Get the derivative of the nonlinear flux function ``f'(u)`` for the PDE.
"""
function nonlinear_df end

"""
    u0(::DiscreteSensitivityProblemCfg)

Get the initial conditions ``u(0, x, p)``.
"""
function u0 end

"""
    u_exact(::DiscreteSensitivityProblemCfg)

Get the exact solution ``u(t, x, p)``.
"""
function u_exact end

"""
    u_broad(::DiscreteSensitivityProblemCfg)

Get the analytic broad tangent ``u^{(1)}(t, x, p)``.
"""
function u_broad end

"""
    ξ(::DiscreteSensitivityProblemCfg)

Get the position(s) ``ξ(t, p)`` of any discontinuities in the exact solution.
"""
function ξ end

"""
    dξ_dp(::DiscreteSensitivityProblemCfg)

Get the sensitivity of the shock positions to the parameters.
"""
function dξ_dp end

"""
    jump_size(::DiscreteSensitivityProblemCfg)

Get the size(s) of the discontinuities in the exact solution.
"""
function jump_size end

include("problem_configurations/burgers.jl")

using .Burgers: DiscreteSensitivityBurgers

export DiscreteSensitivityBurgers

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

function (gtv::GTVClosure{T,XT})(x, λ_pdot) where {T,XT}
    # edges of shock region
    ΞL = gtv.Ξ - gtv.cr_dx_alpha
    ΞR = gtv.Ξ + gtv.cr_dx_alpha
    # discrete jump height
    U_disc = piecewise_constant_interp(gtv.U, gtv.xs)
    ΔU = U_disc(ΞL) - U_disc(ΞR)
    # evaluate the broad tangent with the shock cut
    U_broad = piecewise_constant_interp(gtv.Udot, gtv.xs)(x) * λ_pdot * !X(x, ΞL, ΞR)
    # evaluate the shock shift itself
    Ξ_shift = gtv.Ξdot * λ_pdot
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
    u_ex_fix = fixed_t_p(u_exact(cfg), t, p)
    A1 = maximum((abs ∘ nonlinear_df(cfg) ∘ u_ex_fix), xs)
    A2 = 2 + maximum((abs ∘ nonlinear_df(cfg)), U)
    return cr_scaling_factor(cfg) * (max(A1, A2) + 1) * compute_Cs(U, cfg)
end

function fdiff_eps(arg::T) where {T<:Real}
    cbrt_eps = cbrt(eps(T))
    h = 2^(round(log2((1 + abs(arg)) * cbrt_eps)))
    return h
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

end
