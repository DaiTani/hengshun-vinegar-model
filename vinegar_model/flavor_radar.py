"""
flavor_radar : 实时风味映射
===========================
给定 {总酸, 不挥发酸, 还原糖, 总氨基酸, 乙酸乙酯, 四甲基吡嗪,
      乙酸, pH, 醋龄, 工艺/原料/糖化方式}

输出
----
1. FlavorProfile: 6 维风味雷达 0-10
   酸感 / 甜感 / 鲜感 / 醇厚 / 花果香 / 焦糖香

2. SensoryScore: 14 维感官评分(基于郑梦林/任晓荣/范梦蝶等文献的
   国标GB/T18623-2011+地理标志评价模型,各项满分25)
   色泽/体态/米醋香/果香/酯香/焦糖香/酸度/甜度/鲜度
   /柔和度/涩度/苦度/回味/协调性

3. PHProfile: pH 维度评分 0-1(开启 pH 维度评分时)
   柔和度 / 刺激感

4. OverallScore: 0-100 综合评分

设计思想
--------
每个指标通过 sigmoid 响应函数映射到 0-10,具有
- 理想中心 ideal
- 衰减速率 sharpness
- 上限/下限 floor / ceiling
- 理想区间宽度 tolerance

然后用加权求和合成维度分数。权重来自:
- 沈广玥(2023) XGBoost 在 工艺 / 原料 判别任务上的特征重要性
- 简东振(2020) GC-O 香气贡献度
- 余宁华 等  主成分分析 因变量载荷
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------- #
# 1. 工具函数: sigmoid 响应
# --------------------------------------------------------------------------- #
def _sigmoid(value: float, ideal: float, sharpness: float = 1.0,
             floor: float = 0.0, ceiling: float = 10.0) -> float:
    """把 value 映射到 [0, 10], 在 ideal 处峰值 10,衰减速率 sharpness"""
    return floor + (ceiling - floor) / (1.0 + math.exp(sharpness * (value - ideal)))


def _band(value: float, lo: float, hi: float, ideal_center: float,
          sharpness: float = 1.0, peak: float = 9.0) -> float:
    """
    区间响应: 在 [lo, hi] 区间内取理想, 区间外衰减.
    用于"既有上限又有下限"的指标(例如总酸、有机酸).

    简化: 在区间内 sigmoid(差)=mid, 区间外双侧衰减.
    默认返回 [0, peak] 以避免指标在中心处总是到 10.
    """
    if value < lo:
        delta = (lo - value) / max(lo, 1e-6)
    elif value > hi:
        delta = (value - hi) / max(hi, 1e-6)
    else:
        delta = abs(value - ideal_center) / max((hi - lo), 1e-6)
    score = peak * math.exp(-sharpness * delta * delta)
    return max(0.0, min(10.0, score))


def _weighted_mean(values: List[float], weights: List[float]) -> float:
    s = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / s if s else 0.0


# --------------------------------------------------------------------------- #
# 2. 输入 schema
# --------------------------------------------------------------------------- #
@dataclass
class VinegarState:
    """模型完整输入状态"""
    # 9 个核心参量 (与用户界面一致)
    vinegar_age_months: float = 60.0
    total_acid:         float = 6.32      # g/100mL
    non_volatile_acid:  float = 1.85      # g/100mL
    reducing_sugar:     float = 0.93      # g/100mL
    total_amino_acid:   float = 4.00      # g/100mL
    ethyl_acetate:      float = 1500.0    # μg/mL
    tmp:                float = 44.0      # 四甲基吡嗪 μg/mL
    acetic_acid:        float = 2.31      # g/100mL
    ph:                 float = 3.65
    # 工艺参量
    process:            str   = "固态发酵"   # 固态/液态/固液复合
    raw_material:       str   = "糯米"      # 糯米/大米/高粱/麦芽/果蔬
    craft_style:        str   = "传统"      # 传统/现代
    # 状态开关
    enable_ph_dimension: bool = True

    @staticmethod
    def from_dict(d: dict) -> "VinegarState":
        allowed = {f for f in VinegarState.__dataclass_fields__.keys()}
        return VinegarState(**{k: v for k, v in d.items() if k in allowed})


# --------------------------------------------------------------------------- #
# 3. FlavorProfile : 6维风味雷达
# --------------------------------------------------------------------------- #
@dataclass
class FlavorProfile:
    """6 维风味雷达,各项 0-10"""
    acidity:   float = 0.0    # 酸感(酸味的强度)
    sweetness: float = 0.0    # 甜感
    umami:     float = 0.0    # 鲜感
    body:      float = 0.0    # 醇厚
    floral:    float = 0.0    # 花果香
    caramel:   float = 0.0    # 焦糖香

    def as_dict(self) -> dict:
        return {
            "酸感":   round(self.acidity,   2),
            "甜感":   round(self.sweetness, 2),
            "鲜感":   round(self.umami,     2),
            "醇厚":   round(self.body,      2),
            "花果香": round(self.floral,    2),
            "焦糖香": round(self.caramel,   2),
        }

    def as_array(self) -> List[float]:
        return [self.acidity, self.sweetness, self.umami,
                self.body,   self.floral,    self.caramel]


# --------------------------------------------------------------------------- #
# 4. 维度内部对应(各指标对该维度的贡献)
# --------------------------------------------------------------------------- #

def _acid_dimension(s: VinegarState) -> float:
    """
    酸感 (acidity)
    ---------------
    主要由 总酸 + 乙酸 + pH 三者共同决定.
    文献:
    - 余宁华等: 乙酸 TAV(口感活性值) 在镇江香醋中高达 385~634, 是
      最主要的酸味来源;
    - 任晓荣(2023): 5年陈 pH 3.46 / TA 6.21 时, 口感较柔和;
    - 郑梦林: pH 与柠檬酸、苹果酸、琥珀酸共同贡献酸感.

    这里设计一个 理想区间 + pH调制:
    - 总酸在 [5.5, 7.0] 范围内 酸感最舒适, 6.5 接近满分10;
    - pH 越低 同样总酸下酸感越强 (TAV 提高).
    """
    ta_score = _band(s.total_acid, 3.0, 10.0, ideal_center=6.5, sharpness=0.85)
    acetic_score = _band(s.acetic_acid, 0.5, 8.0, ideal_center=3.5, sharpness=0.85)
    # pH 调制: 低 pH 让酸感更强烈
    ph_mod = (4.4 - s.ph) / (4.4 - 2.0)            # pH=2 → 1, pH=4.4 → 0
    ph_mod = max(0.0, min(1.0, ph_mod))
    base = _weighted_mean([ta_score, acetic_score], [0.55, 0.45])
    return max(0.0, min(10.0, base + ph_mod * 1.4))


def _sweet_dimension(s: VinegarState) -> float:
    """
    甜感 (sweetness)
    -----------------
    主要由 还原糖 决定; 同时丙氨酸等甜味氨基酸会增强甜感.
    文献:
    - 还原糖在陈酿中先减后增, 新醋时较多 (1.5~2.0), 中期最低 (0.6~0.9),
      长期因浓缩上升 (3年陈 ~1.05, 5年 ~0.93, 8年 2.96).
    - 王争争等: 甜味氨基酸(丙氨酸)在老陈醋陈酿期间显著上升.
    """
    sugar = _band(s.reducing_sugar, 0.5, 5.0, ideal_center=2.2, sharpness=0.6)
    # 氨基酸的部分会贡献甜味, 取 0~10 g/100mL 下的甜味氨基酸比例 (~25%)
    amino_sweet = 0.25 * s.total_amino_acid * 2.0   # peak at amino ≈ 4
    amino_sweet = max(0.0, min(10.0, amino_sweet))
    return _weighted_mean([sugar, amino_sweet], [0.7, 0.3])


def _umami_dimension(s: VinegarState) -> float:
    """
    鲜感 (umami)
    ------------
    来自 总氨基酸 + 琥珀酸(同时有酸味和鲜味的协同效应).
    文献:
    - 谷氨酸是食醋中TAV最高的鲜味氨基酸 (TAV~5.88, 郑梦林);
    - GMP TAV 1.0~2.1, 与氨基酸协同增鲜;
    - 琥珀酸 TAV 25~42, 同时具有酸味和鲜味.
    """
    amino = _band(s.total_amino_acid, 0.1, 10.0, ideal_center=4.5, sharpness=0.6)
    succinic_proxy = (s.non_volatile_acid - s.acetic_acid * 0.0) * 0.6 + s.total_amino_acid * 0.4
    succinic_proxy = max(0.0, min(10.0, succinic_proxy * 1.4))
    # 乙酸乙酯 / 苯乙醇系酯也贡献鲜甜协调感(沈广玥 17 种差异物质)
    ester = min(10.0, s.ethyl_acetate / 600.0)
    return _weighted_mean([amino, succinic_proxy, ester], [0.55, 0.25, 0.20])


def _body_dimension(s: VinegarState) -> float:
    """
    醇厚感 (body / mellowness)
    ---------------------------
    "酸而不涩、香而微甜" 的关键指标. 文献:
    - 8年陈 样品 ZV8: 有机酸总量 5.37 g/100mL 远高于新醋; 
      其中乳酸占 25.82%, 使口感绵柔醇厚;
    - 不挥发酸/总酸比 越高 越绵柔;
    - pH 越高 越柔和;
    - 氨基酸 / TMP 协同 也能提高醇厚感.
    """
    nv_ratio = s.non_volatile_acid / max(s.total_acid, 1e-6)
    nv_score = _band(nv_ratio, 0.05, 0.55, ideal_center=0.30, sharpness=4.0)
    # pH 影响(同样总酸下 pH 越高越柔和)
    ph_mod = (s.ph - 2.0) / (5.5 - 2.0)            # 0~1
    ph_score = 1.0 + ph_mod * 8.0                   # 0..9
    # 氨基酸 + TMP 增强 醇厚感
    extras = 0.35 * min(10.0, s.total_amino_acid * 1.0) \
           + 0.30 * _band(s.tmp, 5.0, 200.0, 80.0, 0.6) \
           + 0.20 * _band(s.reducing_sugar, 0.5, 5.0, 2.0, 0.5)
    extras = max(0.0, min(10.0, extras))
    return _weighted_mean([nv_score, ph_score, extras], [0.45, 0.30, 0.25])


def _floral_dimension(s: VinegarState) -> float:
    """
    花果香 (floral)
    ---------------
    文献:
    - 沈广玥: 苯乙醇与乙酸苯乙酯是糯米食醋的特征香气,
      其在糯米为原料醋中高于其他原料;
    - 孙宗保 SPME-MS: 不同生产工艺/醋龄可用 LDA/BPANN 区分,
      工艺主要影响酯类化合物;
    - 酯类物质含量随陈酿年份显著增加 (任晓荣: 8年陈 总酯 2.91);
    - 苯甲醛/苯乙醛 等花香化合物在中后期陈酿中逐渐生成.
    """
    # 用 sigmoid 风格做单调响应, 让分数区分度更好
    ea_log = math.log(s.ethyl_acetate + 1.0)
    ideal_log = math.log(1500.0)               # 在用户界面默认 1500 时取峰
    sharpness = 1.0
    ester = 9.0 / (1.0 + math.exp(-sharpness * (ea_log - ideal_log)))
    # 醋龄调制: 长期陈酿有助于花香类物质积累
    age_mod = 1.0 + 0.18 * (1.0 - math.exp(-s.vinegar_age_months / 36.0))
    return max(0.0, min(10.0, ester * age_mod))


def _caramel_dimension(s: VinegarState) -> float:
    """
    焦糖香 (caramel)
    ----------------
    由 四甲基吡嗪(TMP/川芎嗪)主导, 美拉德反应的标志产物.
    文献:
    - TMP 主要通过煎醋阶段美拉德反应大量生成,
      然后在陈酿过程中通过乙偶姻 + NH4+ 反应持续积累 (刘卓非);
    - 任晓荣(2023): TMP 含量随陈酿年份增加 (3年 < 5年 < 8年);
    - 乙偶姻含量随陈酿下降 (前体消耗), TMP上升 (产物积累);
    - 8年陈 样品 ZV8 的总酯 / TMP 显著优于新醋.
    """
    tmp_score = _band(s.tmp, 5.0, 200.0, ideal_center=110.0, sharpness=0.018)
    # 醋龄调制: 38-72 月之间 TMP 增长最快
    age_boost = 0.6 + 0.4 * min(1.0, s.vinegar_age_months / 96.0)
    # 还原糖的存在是 美拉德反应的前提
    sugar_factor = 0.6 + 0.4 * _band(s.reducing_sugar, 0.5, 5.0, 1.5, 1.5) / 10.0
    return max(0.0, min(10.0, tmp_score * age_boost * sugar_factor))


def compute_flavor_profile(s: VinegarState) -> FlavorProfile:
    """计算 6 维风味雷达"""
    return FlavorProfile(
        acidity=_acid_dimension(s),
        sweetness=_sweet_dimension(s),
        umami=_umami_dimension(s),
        body=_body_dimension(s),
        floral=_floral_dimension(s),
        caramel=_caramel_dimension(s),
    )


# --------------------------------------------------------------------------- #
# 5. SensoryScore : 14 维感官评分
# --------------------------------------------------------------------------- #
@dataclass
class SensoryScore:
    """14 维感官评分,每项满分 25"""
    color:         float = 0.0   # 色泽
    appearance:    float = 0.0   # 体态
    rice_aroma:    float = 0.0   # 米醋/米香
    fruit_aroma:   float = 0.0   # 果香
    ester_aroma:   float = 0.0   # 酯香
    caramel_aroma: float = 0.0   # 焦糖香
    sourness:      float = 0.0   # 酸味强度
    sweetness:     float = 0.0   # 甜味
    umami:         float = 0.0   # 鲜味
    softness:      float = 0.0   # 柔和度
    astringency:   float = 0.0   # 涩度(越低越好)
    bitterness:    float = 0.0   # 苦度(越低越好)
    aftertaste:    float = 0.0   # 回味
    harmony:       float = 0.0   # 协调性

    def as_dict(self) -> dict:
        return {
            "色泽":   round(self.color,         2),
            "体态":   round(self.appearance,    2),
            "米醋香": round(self.rice_aroma,    2),
            "果香":   round(self.fruit_aroma,   2),
            "酯香":   round(self.ester_aroma,   2),
            "焦糖香": round(self.caramel_aroma, 2),
            "酸味强度": round(self.sourness,     2),
            "甜味":   round(self.sweetness,     2),
            "鲜味":   round(self.umami,         2),
            "柔和度": round(self.softness,      2),
            "涩度(越低越好)": round(self.astringency, 2),
            "苦度(越低越好)": round(self.bitterness,  2),
            "回味":   round(self.aftertaste,    2),
            "协调性": round(self.harmony,       2),
        }

    def total(self) -> float:
        return (self.color + self.appearance + self.rice_aroma + self.fruit_aroma
                + self.ester_aroma + self.caramel_aroma
                + self.sourness + self.sweetness + self.umami
                + self.softness + (25 - self.astringency) + (25 - self.bitterness)
                + self.aftertaste + self.harmony)


def compute_sensory_score(s: VinegarState) -> SensoryScore:
    """14 维感官评分"""
    f = compute_flavor_profile(s)

    # 1. 色泽: 醋龄长 + 总酸高 -> 颜色深(深褐/红棕)
    color = 14.0 \
          + 0.30 * min(8.0, s.vinegar_age_months / 12.0) \
          + 1.5  * _band(s.total_acid, 3.0, 10.0, 6.0, 0.8) / 10.0 * 7.0
    color = max(0.0, min(25.0, color))

    # 2. 体态: 还原糖 适中 + 适当浓缩 -> 清亮
    appearance = 16.0 + 0.6 * _band(s.reducing_sugar, 0.5, 5.0, 1.5, 1.0) / 10.0 * 7.0
    if s.total_acid > 9.0:
        appearance -= (s.total_acid - 9.0) * 2.0
    appearance = max(0.0, min(25.0, appearance))

    # 3. 米醋香: 糯米 + 适中陈酿 -> 米香浓郁
    rice = 12.0
    if s.raw_material == "糯米":
        rice += 4.0
    rice += 0.25 * min(6.0, s.vinegar_age_months / 12.0)
    rice += 0.25 * s.ethyl_acetate / 200.0
    rice = max(0.0, min(25.0, rice))

    # 4. 果香: 乙酸乙酯 + 苯乙醇 -> 乙酸苯乙酯 主要贡献者
    fruit = 5.0 + 1.3 * s.ethyl_acetate / 500.0 + 1.0 * min(8.0, s.vinegar_age_months / 12.0)
    if s.raw_material in ("糯米", "果蔬"):
        fruit += 2.0
    fruit = max(0.0, min(25.0, fruit))

    # 5. 酯香: 总酯 主要由 乙酸乙酯 反映
    ester = 6.0 + 1.5 * s.ethyl_acetate / 400.0
    if s.process == "固态发酵":
        ester += 2.5
    if s.craft_style == "传统":
        ester += 1.5
    ester = max(0.0, min(25.0, ester))

    # 6. 焦糖香: TMP 主导
    caramel = 4.0 + 2.5 * s.tmp / 30.0
    if s.vinegar_age_months > 60:
        caramel += (min(96.0, s.vinegar_age_months) - 60) / 36.0 * 4.0
    caramel = max(0.0, min(25.0, caramel))

    # 7. 酸味: 高总酸 + 低 pH -> 强; 但过强会扣分
    sourness = 8.0 + 1.5 * s.total_acid
    sourness -= 0.6 * (s.total_acid - 6.0) * (s.total_acid > 6.0) * 2  # 高总酸刺激
    sourness += (5.5 - s.ph) * 2.0
    sourness = max(0.0, min(25.0, sourness))

    # 8. 甜味: 还原糖 + 部分氨基酸甜味
    sweetness = 4.0 + 1.5 * s.reducing_sugar + 0.4 * s.total_amino_acid
    sweetness = max(0.0, min(25.0, sweetness))

    # 9. 鲜味: 总氨基酸 + 部分乙酸乙酯(协同)
    umami = 5.0 + 1.4 * s.total_amino_acid + 0.4 * s.ethyl_acetate / 100.0
    umami = max(0.0, min(25.0, umami))

    # 10. 柔和度: 高 pH + 不挥发酸比例高 + 陈酿时间 -> 高分
    softness = 5.0 + 1.5 * (s.ph - 2.0) / 3.5 * 10.0  # pH 调制
    softness += 2.0 * (s.non_volatile_acid / max(s.total_acid, 1e-6)) * 10.0
    softness += 0.3 * min(8.0, s.vinegar_age_months / 12.0)
    softness = max(0.0, min(25.0, softness))

    # 11. 涩度(越低越好, 满分25代表无涩感)
    astringency = 11.0
    if s.total_acid > 8.0:
        astringency += (s.total_acid - 8.0) * 5.0
    if s.acetic_acid > 4.5:
        astringency += (s.acetic_acid - 4.5) * 3.5
    if s.tmp > 80:
        astringency -= 1.5
    astringency = max(0.0, min(25.0, astringency))

    # 12. 苦度(越低越好)
    bitterness = 9.0
    if s.total_amino_acid > 6.0:
        bitterness += (s.total_amino_acid - 6.0) * 1.4
    if s.ph < 3.0:
        bitterness += (3.0 - s.ph) * 4.0
    if s.raw_material == "高粱":
        bitterness += 1.5
    bitterness = max(0.0, min(25.0, bitterness))

    # 13. 回味: 酯香 + 不挥发酸 协同 留下多层次悠长余韵
    after = 5.0 + 0.6 * s.ethyl_acetate / 150.0 \
          + 1.5 * (s.non_volatile_acid / max(s.total_acid, 1e-6)) * 10.0
    if s.tmp > 50:
        after += 2.0
    after = max(0.0, min(25.0, after))

    # 14. 协调性: 各项均衡 = no dimension is too extreme
    # 用 f 的 6 维分数的方差的倒数
    fvals = f.as_array()
    fmean = sum(fvals) / 6.0
    fvar = sum((v - fmean) ** 2 for v in fvals) / 6.0
    balance = math.exp(-0.18 * fvar)
    harmony = 10.0 + 10.0 * balance + 0.20 * s.vinegar_age_months / 12.0
    harmony = max(0.0, min(25.0, harmony))

    sc = SensoryScore(
        color=color, appearance=appearance,
        rice_aroma=rice, fruit_aroma=fruit, ester_aroma=ester, caramel_aroma=caramel,
        sourness=sourness, sweetness=sweetness, umami=umami,
        softness=softness, astringency=astringency, bitterness=bitterness,
        aftertaste=after, harmony=harmony,
    )
    return sc


# --------------------------------------------------------------------------- #
# 6. PHProfile : pH 维度评分
# --------------------------------------------------------------------------- #
@dataclass
class PHProfile:
    """pH 维度评分 (0-1). 当 enable_ph_dimension = False 时, 两项都为 None."""
    softness: float = 0.0   # 柔和度 (高 pH 主导)
    pungency: float = 0.0   # 刺激感 (低 pH + 高乙酸 主导)

    def as_dict(self) -> dict:
        if self.softness is None and self.pungency is None:
            return {"柔和度": None, "刺激感": None,
                    "pH_dimension_enabled": False}
        return {
            "柔和度": round(self.softness, 3),
            "刺激感": round(self.pungency, 3),
            "pH_dimension_enabled": True,
        }


def compute_ph_profile(s: VinegarState) -> PHProfile:
    if not s.enable_ph_dimension:
        return PHProfile(softness=None, pungency=None)
    # 柔和度: 越高越好, 主导是 pH 高. pH=3.0 -> 0.20, pH=3.5 -> 0.50, pH=4.5 -> 0.93
    softness = 1.0 / (1.0 + math.exp(-1.6 * (s.ph - 3.5)))
    # 刺激感: pH 越低 乙酸越高 会刺激. pH=3.0 -> 0.81, pH=4.5 -> 0.05
    pung_ph = 1.0 / (1.0 + math.exp(2.2 * (s.ph - 3.20)))
    pung_ace = 1.0 / (1.0 + math.exp(-1.8 * (s.acetic_acid - 3.2)))
    pungency = 0.65 * pung_ph + 0.35 * pung_ace
    return PHProfile(softness=round(softness, 3),
                     pungency=round(max(0.0, min(1.0, pungency)), 3))


# --------------------------------------------------------------------------- #
# 7. OverallScore : 综合评分 (0-100)
# --------------------------------------------------------------------------- #
def compute_overall_score(s: VinegarState,
                          profile: FlavorProfile = None,
                          sensory: SensoryScore = None) -> float:
    """
    综合评分 0-100:
      40% 来自感官评分(转 0-100); 40% 来自风味雷达;
      10% 来自 pH 维度 (关闭时从感官评分里抽样取);
      10% 来自工艺 / 醋龄 / 原料 的元数据加成.
    """
    if profile is None:
        profile = compute_flavor_profile(s)
    if sensory is None:
        sensory = compute_sensory_score(s)

    sensory_total = sensory.total()            # 0~350
    sensory_100 = sensory_total / 350.0 * 100.0

    flavor_100 = sum(profile.as_array()) / 6.0 * 10.0  # 0~100

    # pH 维度评分贡献 (如果开启, 加 "柔和度 - 刺激感" 的差)
    if s.enable_ph_dimension:
        php = compute_ph_profile(s)
        ph_score = (php.softness - php.pungency + 1.0) / 2.0 * 100.0  # 0~100
    else:
        ph_score = (sensory_100 + flavor_100) / 2.0

    # 元数据奖励
    meta_bonus = 0.0
    if s.process == "固态发酵":
        meta_bonus += 3.0
    elif s.process == "固液复合":
        meta_bonus += 1.5
    if s.craft_style == "传统":
        meta_bonus += 1.0
    if 36 <= s.vinegar_age_months <= 96:
        meta_bonus += 2.0
    if s.raw_material == "糯米":
        meta_bonus += 1.0

    score = (sensory_100 * 0.40
           + flavor_100 * 0.40
           + ph_score    * 0.10
           + meta_bonus)
    return round(max(0.0, min(100.0, score)), 2)


# --------------------------------------------------------------------------- #
# 8. 雷达图绘制
# --------------------------------------------------------------------------- #
def radar_chart(profile: FlavorProfile, title: str = "风味雷达") -> "matplotlib.figure.Figure":
    """
    绘制 6 维风味雷达图. 返回 matplotlib Figure 对象.
    调用方: fig = radar_chart(profile); fig.savefig(...)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["酸感", "甜感", "鲜感", "醇厚", "花果香", "焦糖香"]
    values = profile.as_array()
    values = values + values[:1]                    # 闭合
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(7, 6.5), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color="#d6915a", alpha=0.45)
    ax.plot(angles, values, color="#a8623c", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], color="#666", fontsize=9)
    ax.set_title(title, fontsize=14, pad=18)
    ax.grid(linestyle="--", alpha=0.5)
    return fig
