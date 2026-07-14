"""
api.py : 轻量 Flask HTTP API
=============================
- POST /api/flavor        :  接受用户参量 -> 返回风味雷达 + 感官评分 + pH维度 + 综合
- POST /api/predict       :  接受当前 state + target_months -> 预测未来 state
- GET  /api/aging_curve   :  返回 0-120 月的时间序列(各参量演化)
- POST /api/compare       :  工艺/原料差异对比
- GET  /api/aaf/<int:day> :  AAF 阶段指定 day 的状态(1-18)
- GET  /api/aaf/<int:day>/advice?oxygen_upper=...&oxygen_lower=...&temperature=...
                            :  翻醅建议
- GET  /api/health        :  服务存活检查

输入载荷一律 JSON, 输出也是 JSON.
"""

import json
from dataclasses import asdict
from typing import Optional

from flask import Flask, jsonify, request

from vinegar_model.flavor_radar import (
    VinegarState,
    compute_flavor_profile,
    compute_sensory_score,
    compute_ph_profile,
    compute_overall_score,
)
from vinegar_model.aging_kinetics import (
    age_to_state,
    predict_at_age,
    predict_trajectory,
)
from vinegar_model.craft_effect import diff_craft
from vinegar_model.process_model import inspect_aaF_state, recommend_turning
from vinegar_model.aaf_kinetics import AAFKinetics
from vinegar_model.aging_mechanism import TMPReactionNetwork
from vinegar_model.hybrid_model import HybridVinegarModel


app = Flask(__name__)


def _err(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


def _state_from_request() -> tuple:
    """
    从请求中解析VinegarState参数，带参数验证
    返回: (VinegarState, None) 或 (None, error_dict)
    """
    if not request.is_json:
        return None, {"error": "请求必须是JSON格式", "code": "NOT_JSON"}

    payload = request.get_json(silent=True) or {}

    # 参数范围验证
    from vinegar_model.data_baseline import PARAMETER_RANGES
    validation_errors = []

    for param_name, (min_val, max_val) in PARAMETER_RANGES.items():
        if param_name in payload:
            val = payload[param_name]
            if not isinstance(val, (int, float)):
                validation_errors.append(f"{param_name}必须是数字, 当前值: {val}")
            elif val < min_val or val > max_val:
                validation_errors.append(f"{param_name}超出范围 [{min_val}, {max_val}], 当前值: {val}")

    if validation_errors:
        return None, {"error": "参数验证失败", "details": validation_errors, "code": "VALIDATION_ERROR"}

    try:
        state = VinegarState.from_dict(payload)
        return state, None
    except Exception as exc:
        return None, {"error": f"状态解析失败: {str(exc)}", "code": "PARSE_ERROR"}


# --------------------------------------------------------------------------- #
# 1. /api/flavor  实时风味映射
# --------------------------------------------------------------------------- #
@app.post("/api/flavor")
def api_flavor():
    s, err = _state_from_request()
    if err is not None:
        return jsonify(err), 400

    fp  = compute_flavor_profile(s)
    sc  = compute_sensory_score(s)
    php = compute_ph_profile(s)
    overall = compute_overall_score(s, fp, sc)

    return jsonify({
        "input": s.__dict__,
        "flavor_profile": fp.as_dict(),
        "sensory_score":  sc.as_dict(),
        "sensory_total":  round(sc.total(), 2),
        "sensory_total_pct": round(sc.total() / 350.0 * 100, 1),
        "ph_profile":     php.as_dict(),
        "overall_score":  overall,
    })


# --------------------------------------------------------------------------- #
# 2. /api/predict  给定当前 + 目标月份
# --------------------------------------------------------------------------- #
@app.post("/api/predict")
def api_predict():
    s, err = _state_from_request()
    if err is not None:
        return jsonify(err), 400
    payload = request.get_json(silent=True) or {}
    target_months = float(payload.get("target_months", s.vinegar_age_months))
    target_months = max(0.0, min(120.0, target_months))

    new_s = predict_at_age(s, target_months)

    fp_old = compute_flavor_profile(s)
    sc_old = compute_sensory_score(s)
    ov_old = compute_overall_score(s, fp_old, sc_old)

    fp_new = compute_flavor_profile(new_s)
    sc_new = compute_sensory_score(new_s)
    php_new = compute_ph_profile(new_s)
    ov_new = compute_overall_score(new_s, fp_new, sc_new)

    return jsonify({
        "current_state": {
            "input":   s.__dict__,
            "flavor_profile": fp_old.as_dict(),
            "sensory_score":  sc_old.as_dict(),
            "ph_profile":     compute_ph_profile(s).as_dict(),
            "overall_score":  ov_old,
        },
        "predicted_state": {
            "vinegar_age_months": new_s.vinegar_age_months,
            "ph":                new_s.ph,
            "total_acid":        new_s.total_acid,
            "non_volatile_acid": new_s.non_volatile_acid,
            "reducing_sugar":    new_s.reducing_sugar,
            "total_amino_acid":  new_s.total_amino_acid,
            "ethyl_acetate":     new_s.ethyl_acetate,
            "tmp":               new_s.tmp,
            "acetic_acid":       new_s.acetic_acid,
            "flavor_profile": fp_new.as_dict(),
            "sensory_score":  sc_new.as_dict(),
            "ph_profile":     php_new.as_dict(),
            "overall_score":  ov_new,
        },
        "delta": {
            "overall_score_change": round(ov_new - ov_old, 2),
            "ph_change":            round(new_s.ph - s.ph, 3),
            "total_acid_change":    round(new_s.total_acid - s.total_acid, 3),
            "tmp_change":           round(new_s.tmp - s.tmp, 2),
        },
    })


# --------------------------------------------------------------------------- #
# 3. /api/aging_curve  完整时间序列
# --------------------------------------------------------------------------- #
@app.get("/api/aging_curve")
def api_aging_curve():
    process       = request.args.get("process", "固态发酵")
    raw_material  = request.args.get("raw_material", "糯米")
    craft_style   = request.args.get("craft_style", "传统")
    step          = int(request.args.get("step", "6"))

    months = list(range(0, 121, max(1, step)))
    states = predict_trajectory(months, process=process,
                                raw_material=raw_material,
                                craft_style=craft_style)

    series = []
    for m, st in zip(months, states):
        fl = compute_flavor_profile(st)
        sc = compute_sensory_score(st)
        series.append({
            "months":           m,
            "ph":               round(st.ph, 3),
            "total_acid":       round(st.total_acid, 3),
            "non_volatile_acid":round(st.non_volatile_acid, 3),
            "reducing_sugar":   round(st.reducing_sugar, 3),
            "total_amino_acid": round(st.total_amino_acid, 3),
            "ethyl_acetate":    round(st.ethyl_acetate, 1),
            "tmp":              round(st.tmp, 2),
            "acetic_acid":      round(st.acetic_acid, 3),
            "flavor_avg":       round(sum(fl.as_array()) / 6.0, 2),
            "sensory_total":    round(sc.total(), 1),
        })
    return jsonify({
        "process":      process,
        "raw_material": raw_material,
        "craft_style":  craft_style,
        "series":       series,
    })


# --------------------------------------------------------------------------- #
# 4. /api/compare  工艺差异
# --------------------------------------------------------------------------- #
@app.post("/api/compare")
def api_compare():
    payload = request.get_json(silent=True) or {}
    months = float(payload.get("months", 60))
    base_proc  = payload.get("base_process", "固态发酵")
    targ_proc  = payload.get("target_process", "液态发酵")
    base_mat   = payload.get("base_material", "糯米")
    targ_mat   = payload.get("target_material", "糯米")
    base_style = payload.get("base_style", "传统")
    targ_style = payload.get("target_style", "传统")

    s_base = age_to_state(months, process=base_proc,
                           raw_material=base_mat, craft_style=base_style)
    s_targ = age_to_state(months, process=targ_proc,
                           raw_material=targ_mat, craft_style=targ_style)
    diff = diff_craft(s_base, s_targ)
    return jsonify({
        "months": months,
        "base":   {k: getattr(s_base, k) for k in
                    ["total_acid", "ph", "non_volatile_acid", "reducing_sugar",
                     "total_amino_acid", "ethyl_acetate", "tmp", "acetic_acid"]},
        "target": {k: getattr(s_targ, k) for k in
                    ["total_acid", "ph", "non_volatile_acid", "reducing_sugar",
                     "total_amino_acid", "ethyl_acetate", "tmp", "acetic_acid"]},
        "delta":  diff,
    })


# --------------------------------------------------------------------------- #
# 5. /api/aaf/<day>  AAF 状态
# --------------------------------------------------------------------------- #
@app.get("/api/aaf/<int:day>")
def api_aaf(day: int):
    state = inspect_aaF_state(day)
    return jsonify(state.as_dict())


@app.get("/api/aaf/<int:day>/advice")
def api_aaf_advice(day: int):
    o2u = float(request.args.get("oxygen_upper", 0)) or None
    o2l = float(request.args.get("oxygen_lower", 0)) or None
    t   = float(request.args.get("temperature", 0)) or None
    advice = recommend_turning(day,
                               current_oxygen_upper=o2u,
                               current_oxygen_lower=o2l,
                               current_temperature_upper=t)
    return jsonify(advice)


# --------------------------------------------------------------------------- #
# 6. /api/aaf_kinetics/<day>  AAF using Luedeking-Piret
# --------------------------------------------------------------------------- #
@app.get("/api/aaf_kinetics/<day>")
def api_aaf_kinetics(day):
    try:
        day = float(day)
    except (ValueError, TypeError):
        return jsonify({"error": "invalid day value"}), 400
    kinetics = AAFKinetics()
    state = kinetics.get_state_at(day)
    return jsonify(state)


@app.get("/api/tmp_mechanism")
def api_tmp_mechanism():
    t = float(request.args.get("t", 60.0))
    T = float(request.args.get("T", 25.0))
    mechanism = TMPReactionNetwork()
    t_arr = [t]
    tmp_pred = mechanism.predict_tmp(
        t_arr, initial_acetoin=80.0, initial_nh3=100.0, T=T, initial_precursor=150.0
    )
    t_long, acetoin_long, tmp_long = mechanism.simulate_ageing(t, T=T)
    return jsonify({
        "t_months": t,
        "T": T,
        "tmp": round(float(tmp_pred[0]), 2),
        "acetoin": round(float(acetoin_long[-1]), 2) if len(acetoin_long) > 0 else 80.0,
        "nh3": 100.0,
        "k1": mechanism.k1(T),
        "k2": mechanism.k2(T),
    })


@app.get("/api/hybrid/explain")
def api_hybrid_explain():
    t = float(request.args.get("t", 60.0))
    T = float(request.args.get("T", 25.0))
    model = HybridVinegarModel(T=T)
    mechanism_state = model.get_mechanism_prediction(t, T)
    explanation = model.explain_tmp_formation(t, T)
    aging_state = model.get_aging_state(t)
    return jsonify({
        "t_months": t,
        "T": T,
        "mechanism_prediction": mechanism_state,
        "logistic_prediction": {
            "tmp": aging_state.tmp,
            "ethyl_acetate": aging_state.ethyl_acetate,
        },
        "explanation": explanation,
    })


# --------------------------------------------------------------------------- #
# 7. /api/health
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def api_health():
    return jsonify({"status": "ok", "service": "vinegar_process_model"})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    print(f"醋发酵过程模型 API 已在 http://127.0.0.1:{port}/ 上启动")
    print("支持端点:")
    print("  POST /api/flavor")
    print("  POST /api/predict")
    print("  GET  /api/aging_curve")
    print("  POST /api/compare")
    print("  GET  /api/aaf/<day>")
    print("  GET  /api/aaf/<day>/advice")
    print("  GET  /api/aaf_kinetics/<day>")
    print("  GET  /api/tmp_mechanism?t=<months>&T=<temp>")
    print("  GET  /api/hybrid/explain?t=<months>&T=<temp>")
    print("  GET  /api/health")
    app.run(host="127.0.0.1", port=port, debug=False)
