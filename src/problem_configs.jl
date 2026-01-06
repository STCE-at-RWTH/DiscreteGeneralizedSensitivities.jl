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

nonlinear_f(::DiscreteSensitivityBurgers) = f_burg
nonlinear_df(::DiscreteSensitivityBurgers) = df_burg

u0(::DiscreteSensitivityBurgers) = u0_ramp
u_exact(::DiscreteSensitivityBurgers) = u_ramp
u_broad(::DiscreteSensitivityBurgers) = du_ramp_dp

ξ(::DiscreteSensitivityBurgers) = ξ_ramp
dξ_dp(::DiscreteSensitivityBurgers) = dξ_ramp_dp
jump_size(::DiscreteSensitivityBurgers) = Δu_ramp

struct DiscreteSensitivitySod <: DiscreteSensitivityProblemCfg{Float64}
end
