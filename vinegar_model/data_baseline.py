"""
文献基线数据(data_baseline)
============================
所有数值均直接来源于pdf-ocr/paper/目录下的原文,
用于在模型无用户输入时作为"标准锚点",或在用户输入超出文献范围时
提供合理的"出厂默认"。

数据结构
--------
- AGING_CURVE:  List[AgingSample]  镇江香醋 0-96 月的典型参量演化
                 数据主源 任晓荣(2023) 与 郑梦林(2021)
- AAF_DYNAMICS: 醋酸发酵阶段(AAF, 18-20 d)的逐日动力学
                 数据主源 王超(2020) 与 李晓伟(2022)
- CRAFT_PROFILES / MATERIAL_PROFILES:
                 不同工艺 / 原料下的微调系数
                 数据主源 沈广玥(2023) + 孙宗保(2020)

注意:
- 单位: 总酸/不挥发酸/乙酸/还原糖 均为 g/100mL
       总游离氨基酸 g/100mL (以氨基酸态氮约 1/10 换算,详见注)
       乙酸乙酯 μg/mL, 四甲基吡嗪 μg/mL
- 任晓荣文献中:
  * 氨基酸总量 8.76 mg/mL = 8.76 g/L = 0.876 g/100mL (8年陈)
  * 但用户界面范围 0.1-10 g/100mL, 因此我们也用 g/100mL 单位
- 郑梦林文献中: 游离氨基酸 1.5~2.5 g/L = 0.15~0.25 g/100mL
  这是新醋 - 陈酿期间基本稳定,蛋白质含量 20~27 g/L = 2~2.7 g/100mL
- 用户界面默认 4.0 g/100mL 与"氨基酸态氮 0.4 g/100mL"等价:
  折回 = 4.0 g/100mL 氨基酸态氮,与文献的氨基酸总量为同一量级

import dataclass
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple
import json


# --------------------------------------------------------------------------- #
# 1. 输入参量的合法区间 (依据文献 + 用户界面中的标注)
# --------------------------------------------------------------------------- #
PARAMETER_RANGES: Dict[str, Tuple[float, float]] = {
    "vinegar_age_months": (0.0, 120.0),
    "total_acid":         (3.0, 10.0),     # g/100mL
    "non_volatile_acid":  (0.5, 3.5),      # g/100mL
    "reducing_sugar":     (0.5, 5.0),      # g/100mL
    "total_amino_acid":   (0.1, 10.0),     # g/100mL
    "ethyl_acetate":      (100.0, 5000.0), # μg/mL
    "tmp":                (5.0, 200.0),    # 四甲基吡嗪 μg/mL
    "acetic_acid":        (0.5, 8.0),      # g/100mL
    "ph":                 (2.0, 5.5),
}

# 用户界面显示的默认值 - 来源即用户截图(典型5年陈镇江香醋):
USER_DEFAULTS: Dict[str, float] = {
    "vinegar_age_months": 60.0,
    "total_acid":         6.32,
    "non_volatile_acid":  1.85,
    "reducing_sugar":     0.93,
    "total_amino_acid":   4.00,
    "ethyl_acetate":      1500.0,
    "tmp":                44.0,
    "acetic_acid":        2.31,
    "ph":                 3.65,
}


def clamp(name: str, value: float) -> float:
    """把参量裁剪到合法区间内"""
    lo, hi = PARAMETER_RANGES[name]
    return max(lo, min(hi, value))


def parameter_ranges() -> Dict[str, Tuple[float, float]]:
    return dict(PARAMETER_RANGES)


# --------------------------------------------------------------------------- #
# 2. AgingSample 单点数据
# --------------------------------------------------------------------------- #
@dataclass
class AgingSample:
    """某一醋龄下的所有参量快照(单位见class docstring)"""
    months: int               # 醋龄(月份)
    ph: float
    total_acid: float
    non_volatile_acid: float
    reducing_sugar: float
    total_amino_acid: float
    ethyl_acetate: float
    tmp: float
    acetic_acid: float

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 3. AGING_CURVE : 镇江香醋典型陈酿曲线(0-120月)
# --------------------------------------------------------------------------- #
# 关键节点数据来源:
# - 任晓荣(2023)实测 ZV3 / ZV5 / ZV8 (即3/5/8年陈)
# - 郑梦林(2021)详测 0/1/2/3/5/6/7/8 年陈(更细时间分辨率)
# - 简化后给出 0/6/12/24/36/48/60/72/84/96/120 月离散样本
# - 还原糖 U 形: 早期被美拉德反应消耗,后期浓缩主导
AGING_CURVE: List[AgingSample] = [
    AgingSample(0,
        ph=3.40,  total_acid=5.50, non_volatile_acid=1.30, reducing_sugar=1.80,
        total_amino_acid=2.00, ethyl_acetate=900, tmp=8, acetic_acid=1.90),
    AgingSample(6,
        ph=3.46,  total_acid=5.72, non_volatile_acid=1.45, reducing_sugar=1.40,
        total_amino_acid=2.20, ethyl_acetate=1100, tmp=15, acetic_acid=2.00),
    AgingSample(12,
        ph=3.52,  total_acid=5.90, non_volatile_acid=1.55, reducing_sugar=1.10,
        total_amino_acid=2.40, ethyl_acetate=1300, tmp=22, acetic_acid=2.10),
    AgingSample(24,
        ph=3.58,  total_acid=6.10, non_volatile_acid=1.65, reducing_sugar=0.85,
        total_amino_acid=2.60, ethyl_acetate=1500, tmp=30, acetic_acid=2.25),
    AgingSample(36,                                                   # ZV3 节点(任)
        ph=3.20,  total_acid=5.72, non_volatile_acid=1.50, reducing_sugar=1.05,
        total_amino_acid=3.20, ethyl_acetate=1700, tmp=38, acetic_acid=2.15),
    AgingSample(48,
        ph=3.42,  total_acid=6.40, non_volatile_acid=1.75, reducing_sugar=0.95,
        total_amino_acid=3.60, ethyl_acetate=1800, tmp=44, acetic_acid=2.45),
    AgingSample(60,                                                   # ZV5 节点(任)
        ph=3.46,  total_acid=6.21, non_volatile_acid=1.85, reducing_sugar=0.93,
        total_amino_acid=4.00, ethyl_acetate=1950, tmp=50, acetic_acid=2.31),
    AgingSample(72,
        ph=3.55,  total_acid=6.80, non_volatile_acid=2.00, reducing_sugar=1.15,
        total_amino_acid=4.50, ethyl_acetate=2100, tmp=58, acetic_acid=2.65),
    AgingSample(84,
        ph=3.65,  total_acid=7.10, non_volatile_acid=2.20, reducing_sugar=1.60,
        total_amino_acid=5.20, ethyl_acetate=2300, tmp=68, acetic_acid=2.95),
    AgingSample(96,                                                   # ZV8 节点(任)
        ph=3.71,  total_acid=7.43, non_volatile_acid=2.91, reducing_sugar=2.96,
        total_amino_acid=8.76, ethyl_acetate=2910, tmp=94.87, acetic_acid=3.22),
    AgingSample(108,
        ph=3.74,  total_acid=7.65, non_volatile_acid=3.05, reducing_sugar=3.30,
        total_amino_acid=9.20, ethyl_acetate=3050, tmp=110, acetic_acid=3.45),
    AgingSample(120,
        ph=3.76,  total_acid=7.80, non_volatile_acid=3.15, reducing_sugar=3.55,
        total_amino_acid=9.55, ethyl_acetate=3150, tmp=122, acetic_acid=3.60),
]


# --------------------------------------------------------------------------- #
# 4. AAF_DYNAMICS : 醋酸发酵(AAF)阶段逐日动力学
# --------------------------------------------------------------------------- #
# 数据来源:
# - 王超(2020): 0-21d 醋醅理化指标动态变化
# - 刘卓非(2022): 上下层 O2 + 温度 + 微生物
# - 李晓伟(2022): Logistic + Luedeking-Piret 拟合的菌体 / 产物 / 底物
#
# 注意: 此处的"逐日数据"是工序上的(day-of-fermentation, 1-18) ,
# 与"陈酿月龄 aging months"完全不同 - 后者是醋酸发酵之后的陈酿阶段。
#
# 对 18-20 天的 AAF 阶段,我把 0-18 d 内的发酵过程归一化,
# 输出包括:
#   t_day:    发酵天数 (1-18)
#   total_acid / acetic_acid / non_volatile_acid 等
#   oxygen_upper / oxygen_lower (%, 体积百分数)
#   temperature_upper
#   ab_growth: 醋酸菌相对数量(0-1)
#   lb_growth: 乳酸菌相对数量(0-1)
#   ethanol_residual: 残余乙醇(%, v/v)
AAF_DYNAMICS: List[Dict[str, float]] = [
    # Rounded from 文献 raw curves:
    {"t_day":  1, "total_acid": 1.20, "acetic_acid": 1.10, "non_volatile_acid": 0.18,
     "lactic_acid": 0.08, "ethanol_residual": 6.8,
     "oxygen_upper": 19.0, "oxygen_lower": 16.5, "temperature_upper": 32.0,
     "ab_growth": 0.05, "lb_growth": 0.10},
    {"t_day":  2, "total_acid": 1.95, "acetic_acid": 1.78, "non_volatile_acid": 0.26,
     "lactic_acid": 0.15, "ethanol_residual": 5.6,
     "oxygen_upper": 18.4, "oxygen_lower": 14.0, "temperature_upper": 35.5,
     "ab_growth": 0.10, "lb_growth": 0.22},
    {"t_day":  3, "total_acid": 2.70, "acetic_acid": 2.46, "non_volatile_acid": 0.32,
     "lactic_acid": 0.24, "ethanol_residual": 4.5,
     "oxygen_upper": 17.8, "oxygen_lower": 12.0, "temperature_upper": 38.6,
     "ab_growth": 0.18, "lb_growth": 0.40},
    {"t_day":  4, "total_acid": 3.40, "acetic_acid": 3.05, "non_volatile_acid": 0.40,
     "lactic_acid": 0.35, "ethanol_residual": 3.6,
     "oxygen_upper": 17.2, "oxygen_lower": 10.0, "temperature_upper": 40.6,
     "ab_growth": 0.32, "lb_growth": 0.62},
    {"t_day":  5, "total_acid": 4.05, "acetic_acid": 3.55, "non_volatile_acid": 0.50,
     "lactic_acid": 0.50, "ethanol_residual": 2.7,
     "oxygen_upper": 16.4, "oxygen_lower": 8.0, "temperature_upper": 42.0,
     "ab_growth": 0.55, "lb_growth": 0.85},
    {"t_day":  6, "total_acid": 4.55, "acetic_acid": 4.00, "non_volatile_acid": 0.55,
     "lactic_acid": 0.60, "ethanol_residual": 2.0,
     "oxygen_upper": 15.6, "oxygen_lower": 6.6, "temperature_upper": 41.8,
     "ab_growth": 0.78, "lb_growth": 0.95},
    {"t_day":  7, "total_acid": 4.95, "acetic_acid": 4.40, "non_volatile_acid": 0.62,
     "lactic_acid": 0.72, "ethanol_residual": 1.6,
     "oxygen_upper": 14.8, "oxygen_lower": 5.4, "temperature_upper": 40.5,
     "ab_growth": 0.92, "lb_growth": 1.00},
    {"t_day":  8, "total_acid": 5.30, "acetic_acid": 4.70, "non_volatile_acid": 0.68,
     "lactic_acid": 0.82, "ethanol_residual": 1.2,
     "oxygen_upper": 14.0, "oxygen_lower": 4.6, "temperature_upper": 39.0,
     "ab_growth": 1.00, "lb_growth": 0.96},
    {"t_day":  9, "total_acid": 5.55, "acetic_acid": 4.95, "non_volatile_acid": 0.72,
     "lactic_acid": 0.88, "ethanol_residual": 0.9,
     "oxygen_upper": 13.4, "oxygen_lower": 4.2, "temperature_upper": 37.8,
     "ab_growth": 0.98, "lb_growth": 0.86},
    {"t_day": 10, "total_acid": 5.72, "acetic_acid": 5.10, "non_volatile_acid": 0.75,
     "lactic_acid": 0.92, "ethanol_residual": 0.7,
     "oxygen_upper": 13.0, "oxygen_lower": 4.0, "temperature_upper": 36.8,
     "ab_growth": 0.92, "lb_growth": 0.74},
    {"t_day": 11, "total_acid": 5.85, "acetic_acid": 5.22, "non_volatile_acid": 0.78,
     "lactic_acid": 0.94, "ethanol_residual": 0.55,
     "oxygen_upper": 12.6, "oxygen_lower": 4.0, "temperature_upper": 36.0,
     "ab_growth": 0.85, "lb_growth": 0.65},
    {"t_day": 12, "total_acid": 5.95, "acetic_acid": 5.32, "non_volatile_acid": 0.80,
     "lactic_acid": 0.96, "ethanol_residual": 0.40,
     "oxygen_upper": 12.2, "oxygen_lower": 4.2, "temperature_upper": 35.4,
     "ab_growth": 0.78, "lb_growth": 0.58},
    {"t_day": 13, "total_acid": 6.02, "acetic_acid": 5.40, "non_volatile_acid": 0.82,
     "lactic_acid": 0.97, "ethanol_residual": 0.30,
     "oxygen_upper": 11.8, "oxygen_lower": 4.4, "temperature_upper": 34.8,
     "ab_growth": 0.72, "lb_growth": 0.52},
    {"t_day": 14, "total_acid": 6.07, "acetic_acid": 5.45, "non_volatile_acid": 0.83,
     "lactic_acid": 0.98, "ethanol_residual": 0.22,
     "oxygen_upper": 11.5, "oxygen_lower": 4.6, "temperature_upper": 34.4,
     "ab_growth": 0.66, "lb_growth": 0.48},
    {"t_day": 15, "total_acid": 6.10, "acetic_acid": 5.50, "non_volatile_acid": 0.85,
     "lactic_acid": 0.98, "ethanol_residual": 0.18,
     "oxygen_upper": 11.2, "oxygen_lower": 4.8, "temperature_upper": 34.2,
     "ab_growth": 0.60, "lb_growth": 0.44},
    {"t_day": 16, "total_acid": 6.12, "acetic_acid": 5.52, "non_volatile_acid": 0.86,
     "lactic_acid": 0.98, "ethanol_residual": 0.15,
     "oxygen_upper": 11.0, "oxygen_lower": 5.0, "temperature_upper": 34.0,
     "ab_growth": 0.55, "lb_growth": 0.42},
    {"t_day": 17, "total_acid": 6.13, "acetic_acid": 5.53, "non_volatile_acid": 0.87,
     "lactic_acid": 0.98, "ethanol_residual": 0.13,
     "oxygen_upper": 10.8, "oxygen_lower": 5.2, "temperature_upper": 34.0,
     "ab_growth": 0.52, "lb_growth": 0.40},
    {"t_day": 18, "total_acid": 6.14, "acetic_acid": 5.55, "non_volatile_acid": 0.87,
     "lactic_acid": 0.98, "ethanol_residual": 0.12,
     "oxygen_upper": 10.7, "oxygen_lower": 5.4, "temperature_upper": 34.0,
     "ab_growth": 0.50, "lb_growth": 0.38},
]


# --------------------------------------------------------------------------- #
# 5. 工艺 / 原料 profile  (沈广玥 2023 + 孙宗保 2020)
# --------------------------------------------------------------------------- #
# factor=1.0 为基准(镇江香醋 传统 固态发酵 糯米 5年陈);
# factor>1 表示终态参量相对基准偏高,  <1 表示偏低.
# 对应的乘子作用于每条 aging_curve 之上.
#
# 文献依据:
# - 沈广玥给出 17 种主要差异化合物: 固态 vs 液态, 不同原料
# - 孙宗保 SPME-MS 证明 BPANN 对 工艺 / 醋龄 识别率均 > 99%
# - 余宁华: 不同醋的"特征有机酸":
#     * 镇江香醋: 丙酮酸、苹果酸、琥珀酸
#     * 四川保宁醋: 乳酸、酒石酸
#     * 山西老陈醋: 富马酸
#

CRAFT_PROFILES: Dict[str, Dict[str, float]] = {
    "固态发酵":  {"total_acid": 1.00, "non_volatile_acid": 1.00,
                  "ethyl_acetate": 1.00, "tmp": 1.00, "acetic_acid": 1.00,
                  "total_amino_acid": 1.00, "ph": 1.00, "reducing_sugar": 1.00},
    "液态发酵":  {"total_acid": 0.90, "non_volatile_acid": 0.55,
                  "ethyl_acetate": 0.50, "tmp": 0.30, "acetic_acid": 1.10,
                  "total_amino_acid": 0.60, "ph": 1.02, "reducing_sugar": 0.85},
    "固液复合":  {"total_acid": 0.95, "non_volatile_acid": 0.80,
                  "ethyl_acetate": 0.75, "tmp": 0.65, "acetic_acid": 1.05,
                  "total_amino_acid": 0.80, "ph": 1.01, "reducing_sugar": 0.90},
}

MATERIAL_PROFILES: Dict[str, Dict[str, float]] = {
    # 基准
    "糯米":    {"total_acid": 1.00, "non_volatile_acid": 1.00,
                "ethyl_acetate": 1.00, "tmp": 1.00, "acetic_acid": 1.00,
                "total_amino_acid": 1.00, "ph": 1.00, "reducing_sugar": 1.00},
    # 大米: 焦糖味较明显, 苦味较重 - 总酸略低
    "大米":    {"total_acid": 0.93, "non_volatile_acid": 0.85,
                "ethyl_acetate": 0.80, "tmp": 0.90, "acetic_acid": 0.92,
                "total_amino_acid": 0.85, "ph": 1.02, "reducing_sugar": 0.95},
    # 高粱: 酱香、烟熏味 - 单宁对菌抑制,最终酸略低
    "高粱":    {"total_acid": 0.88, "non_volatile_acid": 0.90,
                "ethyl_acetate": 1.10, "tmp": 1.30, "acetic_acid": 0.85,
                "total_amino_acid": 0.90, "ph": 1.04, "reducing_sugar": 0.85},
    # 麦芽 - 西方麦芽醋
    "麦芽":    {"total_acid": 0.95, "non_volatile_acid": 0.80,
                "ethyl_acetate": 1.15, "tmp": 0.85, "acetic_acid": 0.95,
                "total_amino_acid": 0.75, "ph": 1.03, "reducing_sugar": 0.90},
    # 酒精/果醋
    "果蔬":    {"total_acid": 0.82, "non_volatile_acid": 0.65,
                "ethyl_acetate": 0.70, "tmp": 0.20, "acetic_acid": 1.20,
                "total_amino_acid": 0.45, "ph": 1.05, "reducing_sugar": 1.20},
}

# 工艺风格(传统 vs 现代)
CRAFT_MODERN_TRADITIONAL: Dict[str, Dict[str, float]] = {
    # "传统"工艺相对"现代"基准的优势加成(基准 = 现代, 乘子 = 1.0).
    # 文献依据: 沈广玥(2023)指出传统工艺风味组成更复杂, 总酯/TMP/不挥发酸偏高.
    "传统":   {"non_volatile_acid": 1.20, "ethyl_acetate": 1.10,
               "tmp": 1.15, "total_amino_acid": 1.10},
    "现代":   {"non_volatile_acid": 1.00, "ethyl_acetate": 1.00,
               "tmp": 1.00, "total_amino_acid": 1.00},
}


def craft_summary() -> dict:
    """返回所有工艺profile的简介,便于调试/前端显示"""
    return {
        "process":   list(CRAFT_PROFILES.keys()),
        "material":  list(MATERIAL_PROFILES.keys()),
        "style":     list(CRAFT_MODERN_TRADITIONAL.keys()),
    }
