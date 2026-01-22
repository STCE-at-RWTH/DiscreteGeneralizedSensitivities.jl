module Burgers

using DifferentiationInterface

using DiscreteGeneralizedSensitivities
using DiscreteGeneralizedSensitivities: fdiff_backend, fixed_t_x, fixed_t_p, X
using Base: Fix1, Fix2

const f_burg(u) = u^2 / 2
const df_burg_fwdprep = prepare_derivative(f_burg, fdiff_backend, 1.0)
df_burg(u) = derivative(f_burg, df_burg_fwdprep, fdiff_backend, u)

u0_ramp(x, p) = (1 + p) * x * X(x, (zero(x), one(x)))

ξ_ramp(t, p) = sqrt(1 + (1 + p) * t)
const dξ_ramp_fwdprep = prepare_derivative(Fix1(ξ_ramp, 0.0), fdiff_backend, 0.0)
dξ_ramp_dp(t, p) = derivative(Fix1(ξ_ramp, t), fdiff_backend, p)

u_ramp(t, x, p) = ((1 + p) * x) / (1 + (1 + p) * t) * X(x, zero(x), ξ_ramp(t, p))
Δu_ramp(t, p) = (1 + p) / ξ_ramp(t, p)

function du_ramp_dp(t, x, p)
    u_fixed = fixed_t_x(u_ramp, t, x)
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

function DiscreteGeneralizedSensitivities.cfl_safety_factor(cfg::DiscreteSensitivityBurgers)
    cfg.CFL_SAFETY
end

function DiscreteGeneralizedSensitivities.cs_scaling_factor(cfg::DiscreteSensitivityBurgers)
    cfg.CS_FACTOR
end

function DiscreteGeneralizedSensitivities.cr_scaling_factor(cfg::DiscreteSensitivityBurgers)
    cfg.CR_FACTOR
end

function DiscreteGeneralizedSensitivities.alpha(cfg::DiscreteSensitivityBurgers)
    cfg.ALPHA
end

function DiscreteGeneralizedSensitivities.nonlinear_f(::DiscreteSensitivityBurgers)
    return f_burg
end

function DiscreteGeneralizedSensitivities.nonlinear_df(::DiscreteSensitivityBurgers)
    return df_burg
end

function DiscreteGeneralizedSensitivities.u0(::DiscreteSensitivityBurgers)
    return u0_ramp
end

function DiscreteGeneralizedSensitivities.u_exact(::DiscreteSensitivityBurgers)
    return u_ramp
end

function DiscreteGeneralizedSensitivities.u_broad(::DiscreteSensitivityBurgers)
    return du_ramp_dp
end

function DiscreteGeneralizedSensitivities.ξ(::DiscreteSensitivityBurgers)
    return ξ_ramp
end

function DiscreteGeneralizedSensitivities.dξ_dp(::DiscreteSensitivityBurgers)
    return dξ_ramp_dp
end

function DiscreteGeneralizedSensitivities.jump_size(::DiscreteSensitivityBurgers)
    return Δu_ramp
end

end
