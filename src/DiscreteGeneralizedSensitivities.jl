module DiscreteGeneralizedSensitivities

using Base: Fix1, Fix2, Fix

using DifferentiationInterface
using ForwardDiff: ForwardDiff
using Interpolations: Interpolations, scale, interpolate, BSpline

const fdiff_backend = AutoForwardDiff()

"""

"""
struct Fixed_t_x{F,T}
    fn::F
    t::T
    x::T
end

function (fix_tx::Fixed_t_x{F,T})(p) where {F,T}
    return fix_tx.fn(fix_tx.t, fix_tx.x, p)
end

struct Fixed_t_p{F,T,P}
    fn::F
    t::T
    p::P
end

function (fix_tp::Fixed_t_p{F,T,P})(x) where {F,T,P}
    return fix_tp.fn(fix_tp.t, x, fix_tp.p)
end

abstract type DiscreteSensitivityProblemCfg{T} end

cfl_safety_factor(::DiscreteSensitivityProblemCfg{T}) where {T} = one(T)
cs_scaling_factor(::DiscreteSensitivityProblemCfg{T}) where {T} = convert(T, 2)
cr_scaling_factor(::DiscreteSensitivityProblemCfg{T}) where {T} = one(T)
alpha(::DiscreteSensitivityProblemCfg{T}) where {T} = convert(T, 0.5)

f_burg(u) = u^2 / 2
const df_burg_fwdprep = prepare_derivative(f_burg, fdiff_backend, 1.0)
df_burg(u) = derivative(f_burg, df_burg_fwdprep, fdiff_backend, u)

u0_ramp(x, p) = (1 + p) * x * X(x, (zero(x), one(x)))

ξ_ramp(t, p) = sqrt(1 + (1 + p) * t)
const dξ_ramp_fwdprep = prepare_derivative(Fix1(ξ_ramp, 0.0), fdiff_backend, 0.0)
dξ_ramp_dp(t, p) = derivative(Fix1(ξ_ramp, t), dξ_ramp_fwdprep, fdiff_backend, p)

u_ramp(t, x, p) = ((1 + p) * x) / (1 + (1 + p) * t) * X(x, zero(x), ξ_ramp(t, p))
Δu_ramp(t, p) = (1 + p) / ξ_ramp(t, p)

function du_ramp_dp(t, x, p)
    u_fixed = Fixed_t_x(u_ramp, t, x)
    return X(x, 0, ξ_ramp(t, p)) * derivative(u_fixed, fdiff_backend, p)
end

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

nonlinear_f(::DiscreteSensitivityBurgers) = f_burg
nonlinear_df(::DiscreteSensitivityBurgers) = df_burg

u0(::DiscreteSensitivityBurgers) = u0_ramp
u_exact(::DiscreteSensitivityBurgers) = u_ramp
u_broad(::DiscreteSensitivityBurgers) = du_ramp_dp

ξ(::DiscreteSensitivityBurgers) = ξ_ramp
dξ_dp(::DiscreteSensitivityBurgers) = dξ_ramp_dp

jump_size(::DiscreteSensitivityBurgers) = Δu_ramp

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

function step_lax_friedrichs!(U_next, U, Δt, xs::AbstractRange, cfg)
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

function step_shock_location(Ξ, U, xs, Δt, cfg)
    U_shock = piecewise_constant_interp(U, xs)(Ξ)
    return Ξ + Δt * nonlinear_df(cfg)(U_shock)
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

function solve_pde!(U, Udot, Ξ, xs::AbstractRange, T_end, cfg; recompute_dt = false)
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
        Ξ = step_shock_location(Ξ, U, xs, dt, cfg)
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
    return (U, Udot, Ξ)
end

# Characteristic function of an interval [a, b]
X(x, a, b) = a ≤ x ≤ b
X(x, ival) = X(x, ival...)

function piecewise_constant_interp(data, xs)
    return scale(interpolate(data, BSpline(Interpolations.Constant())), xs)
end

end
