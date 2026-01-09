
module Burgers

using DifferentiationInterface

using DiscreteGeneralizedSensitivities: fdiff_backend, Fixed_t_x, Fixed_t_p, X

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

nonlinear_f(::DiscreteSensitivityBurgers) = Burgers.f_burg
nonlinear_df(::DiscreteSensitivityBurgers) = Burgers.df_burg

u0(::DiscreteSensitivityBurgers) = Burgers.u0_ramp
u_exact(::DiscreteSensitivityBurgers) = Burgers.u_ramp
u_broad(::DiscreteSensitivityBurgers) = Burgers.du_ramp_dp

ξ(::DiscreteSensitivityBurgers) = Burgers.ξ_ramp
dξ_dp(::DiscreteSensitivityBurgers) = Burgers.dξ_ramp_dp
jump_size(::DiscreteSensitivityBurgers) = Burgers.Δu_ramp

module Euler

using DifferentiationInterface
using StaticArrays

using DiscreteGeneralizedSensitivities

_pressure(u, γ) = (γ - 1) * (u[3] - u[2]^2 / (2 * u[1]))
_sound_speed(u, γ) = sqrt(γ * (_pressure(u, γ) / u[1]))

function F(u, γ)
    v = u[2] / u[1]
    P = _pressure(u, γ)
    return SVector(u[2], u[2] * v + P, v * (u[3] + P))
end

end

struct DiscreteSensitivitySod <: DiscreteSensitivityProblemCfg{Float64}
    CFL_SAFETY::Float64
    CS_FACTOR::Float64
    CR_FACTOR::Float64
    ALPHA::Float64
    GAMMA::Float64
end

cfl_safety_factor(cfg::DiscreteSensitivitySod) = cfg.CFL_SAFETY
cs_scaling_factor(cfg::DiscreteSensitivitySod) = cfg.CS_FACTOR
cr_scaling_factor(cfg::DiscreteSensitivitySod) = cfg.CR_FACTOR
alpha(cfg::DiscreteSensitivitySod) = cfg.ALPHA

nonlinear_f(cfg::DiscreteSensitivitySod) = Fix2(Euler.F, cfg.GAMMA)

function nonlinear_df(cfg::DiscreteSensitivitySod)
    return (u,) -> let γ = cfg.GAMMA
        jacobian(Euler.F, fdiff_backend, u, Constant(γ))
    end
end
