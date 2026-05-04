#=
Fog & Air-Quality Impact-Based Forecasting Bayesian Network — RxInfer.jl

Mirrors bn-ibf/flood_ibf/flood_bn_ibf_v1.jl. Same engine, same CRMA cost-loss
trigger, same soft-evidence virtual-evidence channels. The flood pipeline's
five evidence parents (antecedent rainfall, exceedance, spatial coverage,
trend, tail) are replaced by fog/AQ-specific parents:

  aer  : aerosol load          (Low / Moderate / High)        — S5P AOD
  mois : column moisture       (Dry / Moderate / Moist)        — S5P TCWV (or IFS r_pl)
  fog  : ensemble fog-prob     (Very_Low..Very_High, 5 states) — IFS-derived
  stag : stagnation trend      (Improving / Stable / Stagnating) — IFS wind slope
  tail : extreme fog tail      (Nil / Low / Moderate / High)   — ens-max fog index

Hidden node:
  fog_aq_risk : Minimal / Low / Moderate / High / Extreme (5)

Decision (deterministic, cost-loss):
  crma_state : Monitor / Evaluate / Assess / Actionable_Risk

Usage:
    julia --project=. fog_bn_ibf_v1.jl \
        --input bn_inputs/fog_inputs_2025-01-01.csv \
        --output output/fog_bn_v1_2025-01-01.csv \
        --include-tail-risk --soft-evidence
=#

using LinearAlgebra
using Printf
using ArgParse
using CSV
using DataFrames
using RxInfer

# ============================================================================
# CONSTANTS — state labels
# ============================================================================

const AER_STATES      = ["Low", "Moderate", "High"]                              # 3
const MOIS_STATES     = ["Dry", "Moderate", "Moist"]                             # 3
const FOG_STATES      = ["Very_Low", "Low", "Medium", "High", "Very_High"]       # 5
const STAG_STATES     = ["Improving", "Stable", "Stagnating"]                    # 3
const TAIL_STATES     = ["Nil", "Low", "Moderate", "High"]                       # 4
const RISK_STATES     = ["Minimal", "Low", "Moderate", "High", "Extreme"]        # 5
const CRMA_STATES     = ["Monitor", "Evaluate", "Assess", "Actionable_Risk"]     # 4
const TRAFFIC_LIGHT   = Dict(
    "Monitor"         => "Green",
    "Evaluate"        => "Yellow",
    "Assess"          => "Orange",
    "Actionable_Risk" => "Red",
)

# Threshold cutoffs — must match fog_data_prep.py exactly so soft and hard
# evidence channels produce equivalent posteriors at one-hot limit.
const AER_THRESHOLDS  = (0.3, 0.6)                # AOD
const MOIS_THRESHOLDS = (15.0, 30.0)              # kg m⁻²
const FOG_THRESHOLDS  = (0.2, 0.4, 0.6, 0.8)      # P(fog conditions met)
const STAG_BAND_MS    = 0.5                       # m s⁻¹ slope band → "Stable"
const TAIL_THRESHOLDS = (0.5, 0.75, 0.9)          # ens-max F p95

# ============================================================================
# CATEGORIZE
# ============================================================================

function categorize_aer(aod::Float64)::Int
    isnan(aod) && return 2
    aod < AER_THRESHOLDS[1] && return 1
    aod < AER_THRESHOLDS[2] && return 2
    return 3
end

function categorize_mois(tcwv::Float64)::Int
    isnan(tcwv) && return 2
    tcwv < MOIS_THRESHOLDS[1] && return 1
    tcwv < MOIS_THRESHOLDS[2] && return 2
    return 3
end

function categorize_fog(p::Float64)::Int
    isnan(p) && return 1
    p < FOG_THRESHOLDS[1] && return 1
    p < FOG_THRESHOLDS[2] && return 2
    p < FOG_THRESHOLDS[3] && return 3
    p < FOG_THRESHOLDS[4] && return 4
    return 5
end

function categorize_stag(trend::Union{String,Float64})::Int
    if isa(trend, AbstractString)
        t = lowercase(trend)
        t == "improving"  && return 1
        t == "stagnating" && return 3
        return 2
    end
    isnan(trend) && return 2
    trend > STAG_BAND_MS && return 1     # Improving
    trend < -STAG_BAND_MS && return 3    # Stagnating
    return 2                             # Stable
end

function categorize_tail(p95::Float64)::Int
    isnan(p95) && return 1
    p95 < TAIL_THRESHOLDS[1] && return 1
    p95 < TAIL_THRESHOLDS[2] && return 2
    p95 < TAIL_THRESHOLDS[3] && return 3
    return 4
end

# ============================================================================
# CRMA — cost-loss trigger from posterior
# ============================================================================

"""
Compute CRMA risk-assessment state (1..4) from fog_aq_risk posterior.

Trigger rules (cost-loss based; default C/L = 0.2 → pre-positioned stockpile
or AQ advisory issuance):

  Actionable_Risk : P(High) + P(Extreme) ≥ C/L
  Assess          : P(Mod) + P(High) + P(Extreme) ≥ max(2·C/L, 0.4)
  Evaluate        : P(Low) + P(Mod) + P(High) + P(Extreme) ≥ max(3·C/L, 0.3)
  Monitor         : otherwise

Returns (state_idx, explanation_string).
"""
function compute_crma_state(risk_probs::Vector{Float64};
                            cost_loss_ratio::Float64=0.2)::Tuple{Int,String}
    p_he = risk_probs[4] + risk_probs[5]
    p_mhe = risk_probs[3] + risk_probs[4] + risk_probs[5]
    p_lmhe = risk_probs[2] + risk_probs[3] + risk_probs[4] + risk_probs[5]

    if p_he >= cost_loss_ratio
        return 4, @sprintf("P(High)+P(Extreme)=%.2f ≥ C/L=%.2f", p_he, cost_loss_ratio)
    elseif p_mhe >= max(2 * cost_loss_ratio, 0.4)
        return 3, @sprintf("P(Mod+High+Ext)=%.2f ≥ %.2f", p_mhe, max(2 * cost_loss_ratio, 0.4))
    elseif p_lmhe >= max(3 * cost_loss_ratio, 0.3)
        return 2, @sprintf("P(Low..Ext)=%.2f ≥ %.2f", p_lmhe, max(3 * cost_loss_ratio, 0.3))
    else
        return 1, @sprintf("P(High+Ext)=%.2f < C/L=%.2f", p_he, cost_loss_ratio)
    end
end

# ============================================================================
# EXPERT-RULE CPT for fog_aq_risk | aer, mois, fog, stag, tail
# ============================================================================

"""
Compute P(fog_aq_risk | aer, mois, fog, stag, tail) as a 5-vector.

Indices are 1-based, matching the state arrays above. The base score
combines four physically-grounded signals:

  • fog (forecast)        : strongest driver of fog risk            (weight ×1.0)
  • mois (observed)       : amplifies fog when moist; mutes when dry (weight ×0.6)
  • aer (observed)        : amplifies AQ side when high              (weight ×0.6)
  • stag (forecast)       : modifies severity via ventilation        (weight ±0.4)
  • tail (forecast)       : ensemble-max tail risk modifier          (weight ±0.6)

Then rule-based overrides for canonical scenarios:
  R1: Moist + High fog forecast + Stagnating → Extreme combined fog+AQ
  R2: High AOD + Stagnating + High fog forecast → Extreme (severe AQ event)
  R3: Dry + Low fog forecast + Improving → Minimal (clear conditions)
  R4: Low fog forecast but High tail risk + High AOD → Moderate (AQ sub-rule)

The base "score → distribution" mapping is shared with the flood model
to keep the calibration discipline consistent across hazards.
"""
function compute_risk_probs(aer::Int, mois::Int, fog::Int, stag::Int, tail::Int)::Vector{Float64}
    # Base score: shift by 1 so indices map to 0..N-1 contributions
    base = 0.0
    base += 0.9 * (fog - 1)       # 0..3.6
    base += 0.6 * (mois - 1)      # 0..1.2
    base += 0.6 * (aer - 1)       # 0..1.2

    # Stagnation modifier (1=Improving → -, 2=Stable → 0, 3=Stagnating → +)
    if stag == 3
        base += 0.5
    elseif stag == 1
        base -= 0.4
    end

    # Tail modifier (1=Nil → 0, 2=Low → +, 3=Moderate → ++, 4=High → +++)
    if tail == 4
        base += 0.7
    elseif tail == 3
        base += 0.4
    elseif tail == 2
        base += 0.15
    end

    # Expert rules — physically-motivated overrides
    probs = if mois >= 3 && fog >= 4 && stag == 3
        # R1: moist + high fog forecast + stagnating
        if aer >= 2
            [0.0, 0.0, 0.05, 0.20, 0.75]
        else
            [0.0, 0.0, 0.10, 0.40, 0.50]
        end
    elseif aer == 3 && stag == 3 && fog >= 3
        # R2: high AOD + stagnating + medium-or-better fog forecast → severe AQ episode
        [0.0, 0.0, 0.10, 0.45, 0.45]
    elseif fog >= 3 && tail >= 3
        # R3: medium fog forecast + high tail risk
        [0.0, 0.05, 0.30, 0.45, 0.20]
    elseif fog <= 2 && tail >= 3 && aer >= 3
        # R4: low forecast but tail risk + high AOD → AQ-driven moderate risk
        [0.05, 0.20, 0.45, 0.25, 0.05]
    elseif mois <= 1 && fog <= 2 && stag == 1
        # R5: dry + low forecast + improving → clear
        [0.65, 0.30, 0.05, 0.0, 0.0]
    elseif aer == 1 && fog <= 2 && stag != 3
        # R6: low aerosol + low forecast + not stagnating → minimal
        [0.55, 0.35, 0.10, 0.0, 0.0]
    elseif fog == 5 && mois <= 1
        # R7: high forecast but dry — possible cold-fog mismatch, reduce confidence
        [0.10, 0.30, 0.40, 0.15, 0.05]
    elseif base < 1.0
        [0.55, 0.35, 0.10, 0.0, 0.0]
    elseif base < 2.0
        [0.15, 0.40, 0.35, 0.10, 0.0]
    elseif base < 3.0
        [0.05, 0.20, 0.45, 0.25, 0.05]
    elseif base < 4.0
        [0.0, 0.05, 0.25, 0.50, 0.20]
    else
        [0.0, 0.0, 0.10, 0.40, 0.50]
    end

    return probs ./ sum(probs)
end

"""
Build the full risk CPT as a 6-D tensor for RxInfer's DiscreteTransition.
Axis order: (risk=5, aer=3, mois=3, fog=5, stag=3, tail=4) — total 5×3×3×5×3×4 = 2700 entries.
"""
function build_risk_cpt_tensor(; include_tail::Bool=true)::Array{Float64}
    if include_tail
        T = zeros(Float64, 5, 3, 3, 5, 3, 4)
        for tl in 1:4, st in 1:3, fg in 1:5, mo in 1:3, ae in 1:3
            T[:, ae, mo, fg, st, tl] = compute_risk_probs(ae, mo, fg, st, tl)
        end
        return T
    else
        T = zeros(Float64, 5, 3, 3, 5, 3)
        for st in 1:3, fg in 1:5, mo in 1:3, ae in 1:3
            T[:, ae, mo, fg, st] = compute_risk_probs(ae, mo, fg, st, 1)
        end
        return T
    end
end

# ============================================================================
# RxInfer MODEL — 5-parent BN with virtual-evidence channels (soft evidence)
# ============================================================================

@model function fog_bn_model_5parent(T, aer_data, mois_data, fog_data, stag_data, tail_data, risk_data)
    aer  ~ Categorical(fill(1/3, 3))
    mois ~ Categorical(fill(1/3, 3))
    fog  ~ Categorical(fill(1/5, 5))
    stag ~ Categorical(fill(1/3, 3))
    tail ~ Categorical(fill(1/4, 4))
    aer_data  ~ DiscreteTransition(aer,  diageye(3))
    mois_data ~ DiscreteTransition(mois, diageye(3))
    fog_data  ~ DiscreteTransition(fog,  diageye(5))
    stag_data ~ DiscreteTransition(stag, diageye(3))
    tail_data ~ DiscreteTransition(tail, diageye(4))
    risk ~ DiscreteTransition(aer, T, mois, fog, stag, tail)
    risk_data ~ DiscreteTransition(risk, diageye(5))
end

@model function fog_bn_model_4parent(T, aer_data, mois_data, fog_data, stag_data, risk_data)
    aer  ~ Categorical(fill(1/3, 3))
    mois ~ Categorical(fill(1/3, 3))
    fog  ~ Categorical(fill(1/5, 5))
    stag ~ Categorical(fill(1/3, 3))
    aer_data  ~ DiscreteTransition(aer,  diageye(3))
    mois_data ~ DiscreteTransition(mois, diageye(3))
    fog_data  ~ DiscreteTransition(fog,  diageye(5))
    stag_data ~ DiscreteTransition(stag, diageye(3))
    risk ~ DiscreteTransition(aer, T, mois, fog, stag)
    risk_data ~ DiscreteTransition(risk, diageye(5))
end

_init_5 = @initialization begin
    q(aer)  = Categorical(fill(1/3, 3))
    q(mois) = Categorical(fill(1/3, 3))
    q(fog)  = Categorical(fill(1/5, 5))
    q(stag) = Categorical(fill(1/3, 3))
    q(tail) = Categorical(fill(1/4, 4))
    q(risk) = Categorical(fill(1/5, 5))
end

_init_4 = @initialization begin
    q(aer)  = Categorical(fill(1/3, 3))
    q(mois) = Categorical(fill(1/3, 3))
    q(fog)  = Categorical(fill(1/5, 5))
    q(stag) = Categorical(fill(1/3, 3))
    q(risk) = Categorical(fill(1/5, 5))
end

"""
Soft-evidence inference via RxInfer. Each `*_ev` argument is a probability
vector over its node's states; pass a one-hot vector for hard evidence.
Returns the 5-vector posterior over fog_aq_risk.
"""
function infer_rxinfer_soft(
    aer_ev::Vector{Float64}, mois_ev::Vector{Float64},
    fog_ev::Vector{Float64}, stag_ev::Vector{Float64};
    tail_ev::Union{Nothing,Vector{Float64}}=nothing,
    risk_cpt_tensor::AbstractArray{Float64},
    iterations::Int=10,
)::Vector{Float64}
    if tail_ev === nothing
        r = infer(
            model = fog_bn_model_4parent(T = risk_cpt_tensor),
            data  = (aer_data = aer_ev, mois_data = mois_ev,
                     fog_data = fog_ev, stag_data = stag_ev,
                     risk_data = missing),
            iterations     = iterations,
            initialization = _init_4,
        )
    else
        r = infer(
            model = fog_bn_model_5parent(T = risk_cpt_tensor),
            data  = (aer_data = aer_ev, mois_data = mois_ev,
                     fog_data = fog_ev, stag_data = stag_ev,
                     tail_data = tail_ev, risk_data = missing),
            iterations     = iterations,
            initialization = _init_5,
        )
    end
    return Vector{Float64}(last(r.posteriors[:risk]).p)
end

"""
Fast direct tensor contraction — equivalent to RxInfer for fixed CPTs.
Used as the default path when the prep CSV carries no soft-evidence columns
because RxInfer per-call setup dominates for thousands of zones.
"""
function infer_matmul(
    aer_ev::Vector{Float64}, mois_ev::Vector{Float64},
    fog_ev::Vector{Float64}, stag_ev::Vector{Float64},
    tail_ev::Vector{Float64}, T::Array{Float64,6},
)::Vector{Float64}
    risk = zeros(Float64, 5)
    @inbounds for tl in 1:4, st in 1:3, fg in 1:5, mo in 1:3, ae in 1:3
        w = aer_ev[ae] * mois_ev[mo] * fog_ev[fg] * stag_ev[st] * tail_ev[tl]
        @inbounds for r in 1:5
            risk[r] += T[r, ae, mo, fg, st, tl] * w
        end
    end
    s = sum(risk)
    s > 0 && (risk ./= s)
    return risk
end

onehot(idx::Int, k::Int) = (v = zeros(Float64, k); v[idx] = 1.0; v)

# ============================================================================
# CSV-driven runner
# ============================================================================

function _maybe_soft(prefix::String, k::Int, row, colnames)
    needed = ["$(prefix)_p$i" for i in 1:k]
    all(c in colnames for c in needed) || return nothing
    return Float64[row[c] for c in needed]
end

function run_csv(input_path::String, output_path::String;
                 include_tail::Bool=true,
                 cost_loss_ratio::Float64=0.2,
                 use_rxinfer::Bool=false,
                 iterations::Int=10)
    df = CSV.read(input_path, DataFrame)
    cols = names(df)
    n = nrow(df)
    @info "[bn] read $n rows, $(length(cols)) cols  ($(input_path))"

    has_tail = include_tail &&
               ("extreme_fog_tail_p95" in cols || "tail_p1" in cols)
    if include_tail && !has_tail
        @info "[bn] no tail-risk column found — running 4-parent model"
    end

    T = build_risk_cpt_tensor(; include_tail=has_tail)

    out = Vector{NamedTuple}(undef, n)

    for (i, row) in enumerate(eachrow(df))
        # Hard categorisation indices (used as fallback for any missing soft column)
        ae_idx = categorize_aer(Float64(coalesce(get(row, :antecedent_aerosol_aod, NaN), NaN)))
        mo_idx = categorize_mois(Float64(coalesce(get(row, :antecedent_moisture_kgm2, NaN), NaN)))
        fg_idx = categorize_fog(Float64(coalesce(get(row, :ifs_fog_prob, NaN), NaN)))
        st_idx = "stagnation_trend" in cols ?
                 categorize_stag(String(row.stagnation_trend)) :
                 categorize_stag(Float64(coalesce(get(row, :stagnation_slope_ms, NaN), NaN)))
        tl_idx = has_tail ?
                 categorize_tail(Float64(coalesce(get(row, :extreme_fog_tail_p95, NaN), NaN))) : 1

        aer_ev  = something(_maybe_soft("aer",  3, row, cols), onehot(ae_idx, 3))
        mois_ev = something(_maybe_soft("mois", 3, row, cols), onehot(mo_idx, 3))
        fog_ev  = something(_maybe_soft("fog",  5, row, cols), onehot(fg_idx, 5))
        stag_ev = something(_maybe_soft("stag", 3, row, cols), onehot(st_idx, 3))
        tail_ev = has_tail ? something(_maybe_soft("tail", 4, row, cols),
                                       onehot(tl_idx, 4)) : nothing

        risk_probs = if use_rxinfer
            infer_rxinfer_soft(aer_ev, mois_ev, fog_ev, stag_ev;
                               tail_ev = tail_ev,
                               risk_cpt_tensor = T,
                               iterations = iterations)
        else
            infer_matmul(aer_ev, mois_ev, fog_ev, stag_ev,
                         has_tail ? tail_ev : onehot(1, 4), T)
        end

        crma_idx, crma_expl = compute_crma_state(risk_probs; cost_loss_ratio)
        crma_state = CRMA_STATES[crma_idx]
        traffic = TRAFFIC_LIGHT[crma_state]

        out[i] = (
            boundary_id      = String(row.id),
            boundary_name    = String(row.name),
            country          = String(coalesce(get(row, :country, "Unknown"), "Unknown")),
            target_date      = String(coalesce(get(row, :target_date, ""), "")),
            antecedent_aerosol_state  = AER_STATES[ae_idx],
            antecedent_moisture_state = MOIS_STATES[mo_idx],
            stagnation_trend          = STAG_STATES[st_idx],
            risk_level                = RISK_STATES[argmax(risk_probs)],
            risk_minimal              = round(risk_probs[1], digits=4),
            risk_low                  = round(risk_probs[2], digits=4),
            risk_moderate             = round(risk_probs[3], digits=4),
            risk_high                 = round(risk_probs[4], digits=4),
            risk_extreme              = round(risk_probs[5], digits=4),
            crma_state       = crma_state,
            traffic_light    = traffic,
            crma_explanation = crma_expl,
        )
        i % 50 == 0 && @info "[bn]   processed $i / $n boundaries"
    end

    out_df = DataFrame(out)
    out_path = output_path
    mkpath(dirname(out_path))
    CSV.write(out_path, out_df)
    @info "[bn] wrote $out_path  rows=$(nrow(out_df))  cols=$(ncol(out_df))"

    # Summary
    counts = combine(groupby(out_df, :crma_state), nrow => :n)
    @info "[bn] CRMA distribution:" counts
    return out_df
end

# ============================================================================
# Self-test
# ============================================================================

function self_test()
    @info "[test] Running self-test ..."
    T = build_risk_cpt_tensor(; include_tail=true)

    # Worst case: high AOD + moist + Very_High fog + stagnating + High tail
    rp1 = infer_matmul(onehot(3,3), onehot(3,3), onehot(5,5), onehot(3,3), onehot(4,4), T)
    @info "[test] worst case → " risk=RISK_STATES[argmax(rp1)] probs=round.(rp1, digits=3)
    @assert RISK_STATES[argmax(rp1)] == "Extreme"

    # Best case: low AOD + dry + Very_Low fog + improving + Nil tail
    rp2 = infer_matmul(onehot(1,3), onehot(1,3), onehot(1,5), onehot(1,3), onehot(1,4), T)
    @info "[test] best case → " risk=RISK_STATES[argmax(rp2)] probs=round.(rp2, digits=3)
    @assert RISK_STATES[argmax(rp2)] == "Minimal"

    # Stagnation amplifies: medium fog, stagnating, high AOD vs improving same fog
    rp_stag = infer_matmul(onehot(3,3), onehot(2,3), onehot(3,5), onehot(3,3), onehot(2,4), T)
    rp_imp  = infer_matmul(onehot(3,3), onehot(2,3), onehot(3,5), onehot(1,3), onehot(2,4), T)
    @info "[test] stagnation effect" stag=round.(rp_stag, digits=3) improving=round.(rp_imp, digits=3)

    # Soft evidence widens posterior
    rp_hard = infer_matmul(onehot(2,3), onehot(2,3), onehot(3,5), onehot(2,3), onehot(2,4), T)
    soft_aer = [0.3, 0.5, 0.2]
    rp_soft = infer_matmul(soft_aer, onehot(2,3), onehot(3,5), onehot(2,3), onehot(2,4), T)
    e_hard = -sum(p * log(max(p, 1e-12)) for p in rp_hard)
    e_soft = -sum(p * log(max(p, 1e-12)) for p in rp_soft)
    @info "[test] soft evidence increases entropy?" hard=round(e_hard, digits=3) soft=round(e_soft, digits=3)
    @assert e_soft >= e_hard - 1e-9

    @info "[test] all self-tests passed"
end

# ============================================================================
# CLI
# ============================================================================

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--input"
            help = "Path to fog_data_prep CSV (one row per adm-1)"
            arg_type = String
        "--output"
            help = "Path to BN output CSV"
            arg_type = String
        "--include-tail-risk"
            help = "Use the 5-parent CPT (default: true)"
            action = :store_true
            default = true
        "--no-tail-risk"
            help = "Drop the tail node and use the 4-parent CPT"
            action = :store_true
        "--cost-loss-ratio"
            help = "Decision threshold for CRMA Actionable_Risk"
            arg_type = Float64
            default = 0.2
        "--use-rxinfer"
            help = "Route through RxInfer message passing (slower; default uses tensor matmul)"
            action = :store_true
        "--iterations"
            help = "RxInfer iteration count when --use-rxinfer"
            arg_type = Int
            default = 10
        "--self-test"
            help = "Run self-test and exit"
            action = :store_true
    end
    return parse_args(s)
end

function main()
    args = parse_cli()
    if args["self-test"]
        self_test()
        return
    end
    if args["input"] === nothing || args["output"] === nothing
        @error "--input and --output are required (or use --self-test)"
        return 1
    end
    include_tail = args["include-tail-risk"] && !args["no-tail-risk"]
    run_csv(args["input"], args["output"];
            include_tail = include_tail,
            cost_loss_ratio = args["cost-loss-ratio"],
            use_rxinfer = args["use-rxinfer"],
            iterations = args["iterations"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
