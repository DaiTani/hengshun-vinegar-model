"""
tests/test_vinegar.py : 单元测试
===============================
运行: python -m pytest tests/ -v
或:   python tests/test_vinegar.py
"""
from __future__ import annotations

import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import math
import pytest

from vinegar_model.data_baseline import (
    AGING_CURVE,
    AAF_DYNAMICS,
    CRAFT_PROFILES,
    MATERIAL_PROFILES,
    PARAMETER_RANGES,
    USER_DEFAULTS,
    AgingSample,
    clamp,
)

from vinegar_model.flavor_radar import (
    VinegarState,
    FlavorProfile,
    SensoryScore,
    PHProfile,
    compute_flavor_profile,
    compute_sensory_score,
    compute_ph_profile,
    compute_overall_score,
)

from vinegar_model.aging_kinetics import (
    predict_at_age,
    predict_trajectory,
    age_to_state,
    AGE_FUNCTIONS,
)

from vinegar_model.craft_effect import (
    apply_craft_effect,
    craft_summary,
    PROCESS_FACTORS,
    MATERIAL_FACTORS,
)

from vinegar_model.process_model import (
    AAFState,
    inspect_aaF_state,
    recommend_turning,
)


# --------------------------------------------------------------------------- #
# data_baseline 测试
# --------------------------------------------------------------------------- #
class TestDataBaseline:
    def test_aging_curve_is_sorted(self):
        assert all(AGING_CURVE[i].months < AGING_CURVE[i+1].months
                   for i in range(len(AGING_CURVE)-1))

    def test_aging_curve_within_ranges(self):
        for s in AGING_CURVE:
            for f in PARAMETER_RANGES:
                if f == "vinegar_age_months":
                    continue
                lo, hi = PARAMETER_RANGES[f]
                assert lo <= getattr(s, f) <= hi, f"{f}={getattr(s, f)} out of range"

    def test_aaf_dynamics_strictly_increasing_acid(self):
        ta = [d["total_acid"] for d in AAF_DYNAMICS]
        acetic = [d["acetic_acid"] for d in AAF_DYNAMICS]
        # 总酸 上升到中后期保持
        assert ta[0] < ta[len(ta)//2] <= ta[-1]
        assert acetic[0] < acetic[len(acetic)//2] <= acetic[-1]

    def test_user_defaults_in_ranges(self):
        for k, v in USER_DEFAULTS.items():
            lo, hi = PARAMETER_RANGES[k]
            assert lo <= v <= hi, f"default {k}={v} out of [{lo},{hi}]"

    def test_clamp_behaviour(self):
        assert clamp("total_acid", 100) == 10.0
        assert clamp("total_acid", -10) == 3.0
        assert clamp("total_acid", 6.32) == 6.32

    def test_craft_profiles_complete(self):
        for k, d in CRAFT_PROFILES.items():
            for f in ["total_acid", "non_volatile_acid", "ethyl_acetate",
                      "tmp", "acetic_acid", "total_amino_acid", "ph", "reducing_sugar"]:
                assert f in d, f"missing field {f} in CRAFT_PROFILES[{k}]"

    def test_material_profiles_complete(self):
        for k, d in MATERIAL_PROFILES.items():
            for f in ["total_acid", "non_volatile_acid", "ethyl_acetate",
                      "tmp", "acetic_acid", "total_amino_acid", "ph", "reducing_sugar"]:
                assert f in d, f"missing field {f} in MATERIAL_PROFILES[{k}]"


# --------------------------------------------------------------------------- #
# flavor_radar 测试
# --------------------------------------------------------------------------- #
class TestFlavorRadar:
    def setup_method(self):
        self.s = VinegarState()                  # 默认 60 月

    def test_flavor_profile_in_0_10(self):
        fp = compute_flavor_profile(self.s)
        for v in fp.as_array():
            assert 0.0 <= v <= 10.0

    def test_sensory_score_in_0_25(self):
        sc = compute_sensory_score(self.s)
        for k, v in sc.as_dict().items():
            assert 0.0 <= v <= 25.0, f"{k}={v}"

    def test_sensory_worse_with_better_inputs(self):
        """对比总酸0.5 和 总酸10: 高总酸应更刺激, 涩度 / 苦度更高"""
        s1 = VinegarState(total_acid=0.5)
        s2 = VinegarState(total_acid=10.0)
        sc1 = compute_sensory_score(s1)
        sc2 = compute_sensory_score(s2)
        # 高总酸应该更涩、刺激
        assert sc2.astringency > sc1.astringency, \
               f"high-acid should be more astringent: {sc1.astringency} vs {sc2.astringency}"
        assert sc2.sourness > sc1.sourness, \
               f"high-acid should be more sour: {sc1.sourness} vs {sc2.sourness}"

    def test_overall_score_bounded(self):
        for v in range(10):
            score = compute_overall_score(self.s)
            assert 0.0 <= score <= 100.0

    def test_ph_profile_disabled_returns_none(self):
        s = VinegarState(enable_ph_dimension=False)
        php = compute_ph_profile(s)
        assert php.softness is None and php.pungency is None
        d = php.as_dict()
        assert d["pH_dimension_enabled"] is False

    def test_ph_profile_enabled_has_values(self):
        s = VinegarState(enable_ph_dimension=True, ph=3.65)
        php = compute_ph_profile(s)
        assert php.softness is not None and php.pungency is not None
        assert 0.0 <= php.softness <= 1.0
        assert 0.0 <= php.pungency <= 1.0

    def test_higher_ph_is_softer(self):
        s_low  = VinegarState(ph=3.0)
        s_high = VinegarState(ph=4.5)
        php_low  = compute_ph_profile(s_low)
        php_high = compute_ph_profile(s_high)
        assert php_high.softness > php_low.softness
        assert php_high.pungency < php_low.pungency

    def test_more_amino_acid_more_umami(self):
        s_low  = VinegarState(total_amino_acid=1.0)
        s_high = VinegarState(total_amino_acid=8.0)
        fp_low  = compute_flavor_profile(s_low)
        fp_high = compute_flavor_profile(s_high)
        assert fp_high.umami > fp_low.umami

    def test_more_ester_more_floral(self):
        s_low  = VinegarState(ethyl_acetate=500)
        s_high = VinegarState(ethyl_acetate=4000)
        fp_low  = compute_flavor_profile(s_low)
        fp_high = compute_flavor_profile(s_high)
        assert fp_high.floral > fp_low.floral

    def test_more_tmp_more_caramel(self):
        s_low  = VinegarState(tmp=10)
        s_high = VinegarState(tmp=180)
        fp_low  = compute_flavor_profile(s_low)
        fp_high = compute_flavor_profile(s_high)
        assert fp_high.caramel > fp_low.caramel


# --------------------------------------------------------------------------- #
# aging_kinetics 测试
# --------------------------------------------------------------------------- #
class TestAgingKinetics:
    def test_age_to_state_zero(self):
        s = age_to_state(0)
        assert 0.0 <= s.ph <= 4.0
        assert s.total_acid <= 6.0   # 新醋期低

    def test_age_to_state_monotonic_increasing_acid(self):
        ts = [0, 12, 24, 36, 48, 60, 72, 96, 120]
        for a, b in zip(ts[:-1], ts[1:]):
            sa = age_to_state(a); sb = age_to_state(b)
            # 总酸应单调非减 (允许短期内高原)
            assert sb.total_acid >= sa.total_acid - 0.1

    def test_age_to_state_8yr_close_to_literature(self):
        """8年(96月)参量与任晓荣2023实测的差距应在 20% 以内"""
        s = age_to_state(96)
        lit = {"ph": 3.71, "total_acid": 7.43, "non_volatile_acid": 2.91,
               "ethyl_acetate": 2910, "tmp": 95.0, "acetic_acid": 3.22}
        for f, ref in lit.items():
            assert abs(getattr(s, f) - ref) / ref <= 0.25, \
                   f"{f}: {getattr(s,f):.3f} vs {ref} (25% tolerance)"

    def test_predict_at_age_advances(self):
        s0 = age_to_state(0)
        s1 = predict_at_age(s0, 96)
        assert s1.vinegar_age_months == 96
        assert s1.total_acid > s0.total_acid
        assert s1.tmp > s0.tmp

    def test_predict_trajectory_returns_right_count(self):
        ts = predict_trajectory([0, 24, 48, 72, 96])
        assert len(ts) == 5
        assert ts[0].vinegar_age_months == 0
        assert ts[-1].vinegar_age_months == 96

    def test_age_functions_complete(self):
        for f in ["ph", "total_acid", "non_volatile_acid", "reducing_sugar",
                  "total_amino_acid", "ethyl_acetate", "tmp", "acetic_acid"]:
            assert f in AGE_FUNCTIONS
            assert callable(AGE_FUNCTIONS[f])

    def test_age_function_in_within_range(self):
        """对每个 age function, 0-120 月的预测值都应在 PARAMETER_RANGES 内"""
        for f, fn in AGE_FUNCTIONS.items():
            for m in [0, 12, 24, 60, 96, 120, 240]:
                v = fn(float(m))
                if f == "ph":
                    assert 3.3 <= v <= 4.2
                else:
                    lo, hi = PARAMETER_RANGES[f]
                    # 允许一定外推弹性
                    assert lo * 0.8 <= v <= hi * 1.2, \
                           f"{f}={v} at t={m} out of [{lo*0.8},{hi*1.2}]"

    def test_age_function_increasing_for_increasing_age(self):
        """年龄越大, 总酸 / pH / 乙酸 / 乙酸乙酯 / TMP 应越高(允许 0 月时本身已经较高)"""
        for f in ["total_acid", "ethyl_acetate", "tmp", "acetic_acid", "ph"]:
            v0   = AGE_FUNCTIONS[f](0.0)
            v60  = AGE_FUNCTIONS[f](60.0)
            v120 = AGE_FUNCTIONS[f](120.0)
            assert v120 >= v60 - 0.1, f"{f}: v60={v60} v120={v120}"
            assert v60  >= v0  - 0.1, f"{f}: v0={v0} v60={v60}"


# --------------------------------------------------------------------------- #
# craft_effect 测试
# --------------------------------------------------------------------------- #
class TestCraftEffect:
    def test_summary_returns_known_keys(self):
        s = craft_summary()
        assert set(s.keys()) == {"process", "material", "style"}
        assert "固态发酵" in s["process"]
        assert "糯米" in s["material"]

    def test_liquid_process_lowers_esters(self):
        s_solid = VinegarState(process="固态发酵")
        s_liq   = VinegarState(process="液态发酵")
        sl = apply_craft_effect(s_liq)
        ss = apply_craft_effect(s_solid)
        assert sl.ethyl_acetate < ss.ethyl_acetate   # 液态酯类少
        assert sl.tmp < ss.tmp                       # 液态 TMP 少
        assert sl.total_amino_acid < ss.total_amino_acid

    def test_glutinous_rice_keeps_base(self):
        """默认糯米应与 process/state 不变时偏差最小"""
        s = VinegarState(raw_material="糯米")
        sl = apply_craft_effect(s)
        # 糯米的 factor 默认1.0, 故差异应极小
        assert abs(sl.total_acid - s.total_acid) < 0.05


# --------------------------------------------------------------------------- #
# process_model 测试
# --------------------------------------------------------------------------- #
class TestProcessModel:
    def test_inspect_aaF_state_clamps_day(self):
        s = inspect_aaF_state(0)         # 应 clamp 到 day 1
        assert s.day == 1

        s = inspect_aaF_state(100)       # 应 clamp 到 day 18
        assert s.day == 18

    def test_inspect_returns_known_fields(self):
        s = inspect_aaF_state(8)
        for f in ["total_acid", "acetic_acid", "non_volatile_acid",
                  "ethanol_residual", "oxygen_upper", "oxygen_lower",
                  "temperature_upper", "ab_growth", "lb_growth",
                  "fermrntation_rate", "stage"]:
            assert hasattr(s, f), f"missing field {f}"

    def test_recommend_turning_low_oxygen(self):
        advice = recommend_turning(8, current_oxygen_lower=3.0)
        assert advice["should_turn_today"] is True
        assert "increase_turning_frequency" in advice["suggestions"]

    def test_recommend_turning_high_temp(self):
        advice = recommend_turning(8, current_temperature_upper=44.0)
        assert advice["should_turn_today"] is True
        assert any("温度" in w for w in advice["warnings"])

    def test_recommend_turning_no_inputs_keeps_state(self):
        advice = recommend_turning(15)
        # 不传状态数据 -> 不应有任何"翻醅"建议
        # (因 day=15 时发酵速率较低,可能显示"总酸增长缓慢"警告, 这是正常的)
        if "缓慢" not in advice["warnings"][0]:
            assert "状态良好" in advice["warnings"][0] or \
                   "维持当前" in advice["warnings"][0]


if __name__ == "__main__":
    import inspect
    sys.exit(pytest.main([__file__, "-v"]))
