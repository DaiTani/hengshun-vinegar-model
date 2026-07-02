"""
process_model : 醋酸发酵 (AAF) 阶段过程模型
========================================
本模块针对醋酸发酵的 18-20 天 (AAF stage)进行建模,

依据:
- 王超(2020): 0-21d 醋醅 水分/pH/总酸/还原糖/蛋白质/氨基酸/铵盐
- 刘卓非(2022): 上下层 O2 与温度 时间序列, WNN+ARIMA 拟合
- 李晓伟(2022): 转鼓式反应器, Logistic + Luedeking-Piret 动力学
- 樊苏皖(2021): NIR + PLSR 在线检测 (pH/总酸/不挥发酸)
- 简东振(2020): 煎醋 / 陈酿阶段的气味变化

模型能力
--------
1. inspect_aaF_state(day)  -> 当前 day 的发酵概况
2. recommend_turning(day, current_oxygen_upper, current_oxygen_lower)
   -> 给出翻醅时机建议
3. next_dynamics_step(day, delta_h)  -> 推进 delta_h 小时后的状态估计
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from .data_baseline import AAF_DYNAMICS
from .aaf_kinetics import AAFModel, AAFKinetics


# --------------------------------------------------------------------------- #
# 1. AAF 阶段类型
# --------------------------------------------------------------------------- #
@dataclass
class AAFState:
    """AAF 阶段 (day t) 的发酵状态快照"""
    day: int                              # 发酵天数(1-18)
    stage: str                            # "前期/中期/后期/结束"
    total_acid: float
    acetic_acid: float
    non_volatile_acid: float
    lactic_acid: float
    ethanol_residual: float
    oxygen_upper: float                  # 上层醋醅 O2 (%)
    oxygen_lower: float                  # 下层醋醅 O2 (%)
    temperature_upper: float
    ab_growth: float                     # 醋酸菌相对量
    lb_growth: float                     # 乳酸菌相对量
    fermrntation_rate: float             # 总酸增长速率(估算)

    def as_dict(self) -> dict:
        return {
            "day": self.day,
            "stage": self.stage,
            "total_acid_g100ml": round(self.total_acid, 3),
            "acetic_acid_g100ml": round(self.acetic_acid, 3),
            "non_volatile_acid_g100ml": round(self.non_volatile_acid, 3),
            "lactic_acid_g100ml": round(self.lactic_acid, 3),
            "ethanol_residual_pct": round(self.ethanol_residual, 3),
            "oxygen_upper_pct": round(self.oxygen_upper, 2),
            "oxygen_lower_pct": round(self.oxygen_lower, 2),
            "temperature_upper_c": round(self.temperature_upper, 1),
            "ab_growth": round(self.ab_growth, 3),
            "lb_growth": round(self.lb_growth, 3),
            "fermentation_rate_g100ml_per_day": round(self.fermrntation_rate, 3),
        }


# --------------------------------------------------------------------------- #
# 2. 阶段划分(按照文献的"过勺阶段"/"露底阶段"概念)
# --------------------------------------------------------------------------- #
def _stage_of(day: int) -> str:
    if day <= 3:
        return "启动期 (init)"
    if day <= 9:
        return "中期高活性 (high-activity)"
    if day <= 15:
        return "后期缓慢 (late)"
    return "末期平稳 (plateau)"


# --------------------------------------------------------------------------- #
# 3. 主对外接口
# --------------------------------------------------------------------------- #
def inspect_aaF_state(day: int) -> AAFState:
    """
    给出 day (1-18) 日的醋酸发酵状态.
    """
    day = max(1, min(18, int(day)))
    today = next(d for d in AAF_DYNAMICS if d["t_day"] == day)
    nxt = next((d for d in AAF_DYNAMICS if d["t_day"] == day + 1), today)
    rate = nxt["total_acid"] - today["total_acid"]
    return AAFState(
        day=day,
        stage=_stage_of(day),
        total_acid=today["total_acid"],
        acetic_acid=today["acetic_acid"],
        non_volatile_acid=today["non_volatile_acid"],
        lactic_acid=today["lactic_acid"],
        ethanol_residual=today["ethanol_residual"],
        oxygen_upper=today["oxygen_upper"],
        oxygen_lower=today["oxygen_lower"],
        temperature_upper=today["temperature_upper"],
        ab_growth=today["ab_growth"],
        lb_growth=today["lb_growth"],
        fermrntation_rate=rate,
    )


def recommend_turning(day: int,
                      current_oxygen_upper: Optional[float] = None,
                      current_oxygen_lower: Optional[float] = None,
                      current_temperature_upper: Optional[float] = None) -> dict:
    """
    根据当前醋醅状态给出翻醅建议.
    文献:
    - 传统工艺一天一翻 (刘卓非);
    - 基于预测模型可使工艺缩短 2.1 天;
    - 高温或低 O2 是"应翻醅" 的关键信号.
    """
    today = inspect_aaF_state(day)
    warnings = []
    suggestions = []
    if current_oxygen_lower is not None and current_oxygen_lower < 4.5:
        warnings.append(f"下层 O2 = {current_oxygen_lower:.1f}% 过低(<4.5%), 应立即翻醅")
        suggestions.append("increase_turning_frequency")
    if current_temperature_upper is not None and current_temperature_upper > 43.0:
        warnings.append(f"上层温度 {current_temperature_upper:.1f}°C 过高(>43°C), 应翻醅散热")
        suggestions.append("force_turning")
    if day <= 9 and current_oxygen_upper is not None and current_oxygen_upper < 12.0:
        warnings.append(f"上层 O2={current_oxygen_upper:.1f}% 偏低, 翻醅是必要的")
        suggestions.append("consider_turning_today")
    if today.fermrntation_rate < 0.05:
        warnings.append("总酸增长缓慢; 建议检查翻醅频次")

    if not warnings:
        warnings.append("状态良好, 维持当前翻醅节奏")

    if not warnings:
        warnings.append("状态良好, 维持当前翻醅节奏")

    return {
        "day": day,
        "should_turn_today": any(s in suggestions for s in
                                 ["increase_turning_frequency", "force_turning"]),
        "warnings": warnings,
        "suggestions": suggestions,
        "stage": today.stage,
        "traditional_cadence": "每 24 h 翻醅一次",
        "notes": "基于刘卓非(2022)的预测模型可优化翻醅频次约 2.1 天",
    }


def get_aaf_from_kinetics(day: float) -> AAFState:
    """
    Get AAF state using AAFModel (王超2020数据拟合).
    """
    model = AAFModel()
    state = model.get_state_at(day)
    return AAFState(
        day=state.day,
        stage=state.stage,
        total_acid=state.total_acid,
        acetic_acid=state.acetic_acid,
        non_volatile_acid=state.non_volatile_acid,
        lactic_acid=state.lactic_acid,
        ethanol_residual=state.ethanol_residual,
        oxygen_upper=state.oxygen_upper,
        oxygen_lower=state.oxygen_lower,
        temperature_upper=state.temperature_upper,
        ab_growth=state.ab_growth,
        lb_growth=state.lb_growth,
        fermrntation_rate=state.acid_rate,
    )


def next_dynamics_step(current_day: int,
                       current_hours_remaining: float,
                       target_hours: float) -> int:
    """
    给定 current_day 与"剩余发酵时长(小时)" -> 经过 target_hours 小时后的天数.
    例如 now=10d12h, target=10h -> 11d-2h, i.e. 仍为 day 11 内.
    """
    new_total_h = current_day * 24.0 - current_hours_remaining + target_hours
    new_day = max(1, min(18, int(new_total_h // 24)))
    return new_day
