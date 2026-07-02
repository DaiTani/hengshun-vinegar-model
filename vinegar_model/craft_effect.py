"""
craft_effect : 工艺修正
========================
基于沈广玥(2023)、孙宗保(2020)与余宁华等的"工艺判别"
研究建立的乘子模型.

不同工艺 / 原料 / 糖化方式 的影响通过乘子表达
(乘子在 data_baseline 中给出, 这里封装对外调用.)
"""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from .flavor_radar import VinegarState
from .data_baseline import (
    CRAFT_PROFILES,
    MATERIAL_PROFILES,
    CRAFT_MODERN_TRADITIONAL,
)
from .aging_kinetics import _apply_meta_factors


# 别名: 供外部模块使用
PROCESS_FACTORS = CRAFT_PROFILES
MATERIAL_FACTORS = MATERIAL_PROFILES


# --------------------------------------------------------------------------- #
# 主对外接口
# --------------------------------------------------------------------------- #
def apply_craft_effect(state: VinegarState) -> VinegarState:
    """复制 state 并应用工艺 / 原料 / 糖化 乘子."""
    return _apply_meta_factors(state)


# --------------------------------------------------------------------------- #
# 描述性输出
# --------------------------------------------------------------------------- #
def craft_summary() -> dict:
    """返回所有工艺profile的简介, 便于调试/前端显示"""
    return {
        "process":  list(CRAFT_PROFILES.keys()),
        "material": list(MATERIAL_PROFILES.keys()),
        "style":    list(CRAFT_MODERN_TRADITIONAL.keys()),
    }


# --------------------------------------------------------------------------- #
# 风味画像与工艺差异
# --------------------------------------------------------------------------- #
def diff_craft(baseline: VinegarState, target: VinegarState) -> dict:
    """计算 target 相对 baseline 的偏移"""
    out = {}
    fields = ["ph", "total_acid", "non_volatile_acid", "reducing_sugar",
              "total_amino_acid", "ethyl_acetate", "tmp", "acetic_acid"]
    for f in fields:
        a = getattr(baseline, f)
        b = getattr(target, f)
        out[f] = {"baseline": a, "target": b,
                  "delta": round(b - a, 3),
                  "ratio": round(b / a, 3) if a else None}
    return out
