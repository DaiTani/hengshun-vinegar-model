"""
process_model.py : 镇江香醋五工序生产过程模型
==============================================

基于原料量的化学计量生产模型

化学计量基础:
- 淀粉 → 葡萄糖: 1g淀粉 → 1.11g葡萄糖 (淀粉水解系数)
- 葡萄糖 → 乙醇: 1g葡萄糖 → 0.51g乙醇 (酵母发酵收率)
- 乙醇 → 乙酸: 1g乙醇 → 1.30g乙酸 (醋酸菌氧化)

文献依据:
- 薛茂云(2018): 糯米淀粉含量~70%, 糖化转化率88%
- 丁乾坤(2019): 乙醇收率0.42-0.48
- 王超(2020): AAF发酵R²=0.998
- 任晓荣(2023): 陈酿模型
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


# 原料成分数据 (淀粉含量, 文献值)
RAW_MATERIAL_COMPOSITION = {
    # 谷物类原料 (淀粉基)
    # 参数: 淀粉%, 蛋白质%, 水分%, 类型, 糖化转化率, 酵母发酵效率
    "糯米": {
        "starch_pct": 0.70, "protein_pct": 0.08, "moisture_pct": 0.14,
        "type": "starch", "saccharification_rate": 0.88, "fermentation_efficiency": 0.48,
        "leaching_eff": 0.78, "mash_density": 1.05,
        "ethyl_acetate_factor": 1.0, "tmp_factor": 1.0,
        "notes": "优质糯米, 糖化效率高"
    },
    "大米": {
        "starch_pct": 0.65, "protein_pct": 0.07, "moisture_pct": 0.14,
        "type": "starch", "saccharification_rate": 0.85, "fermentation_efficiency": 0.46,
        "leaching_eff": 0.76, "mash_density": 1.03,
        "ethyl_acetate_factor": 0.9, "tmp_factor": 0.9,
        "notes": "易于糖化, 发酵平稳"
    },
    "高粱": {
        "starch_pct": 0.60, "protein_pct": 0.10, "moisture_pct": 0.12,
        "type": "starch", "saccharification_rate": 0.75, "fermentation_efficiency": 0.42,
        "leaching_eff": 0.72, "mash_density": 1.08,
        "ethyl_acetate_factor": 1.1, "tmp_factor": 1.3,
        "notes": "单宁含量高, 发酵较慢但风味浓郁"
    },
    "小麦": {
        "starch_pct": 0.55, "protein_pct": 0.12, "moisture_pct": 0.13,
        "type": "starch", "saccharification_rate": 0.78, "fermentation_efficiency": 0.44,
        "leaching_eff": 0.74, "mash_density": 1.02,
        "ethyl_acetate_factor": 0.95, "tmp_factor": 0.95,
        "notes": "蛋白质高, 适合酿醋"
    },
    "玉米": {
        "starch_pct": 0.60, "protein_pct": 0.08, "moisture_pct": 0.14,
        "type": "starch", "saccharification_rate": 0.82, "fermentation_efficiency": 0.45,
        "leaching_eff": 0.75, "mash_density": 1.04,
        "ethyl_acetate_factor": 0.85, "tmp_factor": 0.80,
        "notes": "脂肪含量较高"
    },
    # 果蔬类原料 (糖基)
    # 非挥发性酸: 苹果酸(约6g/L在苹果汁中), 酒石酸(约4g/L在葡萄汁中)
    "苹果": {
        "sugar_pct": 0.12, "protein_pct": 0.003, "moisture_pct": 0.84,
        "type": "sugar", "fermentation_efficiency": 0.55,
        "leaching_eff": 0.85, "mash_density": 1.00,
        "acid_profile": "苹果酸为主", "aging_factor": 0.6,
        "ethyl_acetate_factor": 1.4, "tmp_factor": 0.2,
        "non_volatile_acid_initial": 6.0,  # 苹果酸 g/L (原料果汁中含量)
        "non_volatile_acid_type": "苹果酸",
        "notes": "液态发酵, 快速酿醋"
    },
    "葡萄": {
        "sugar_pct": 0.17, "protein_pct": 0.004, "moisture_pct": 0.80,
        "type": "sugar", "fermentation_efficiency": 0.58,
        "leaching_eff": 0.83, "mash_density": 1.00,
        "acid_profile": "酒石酸为主", "aging_factor": 0.5,
        "ethyl_acetate_factor": 1.3, "tmp_factor": 0.15,
        "non_volatile_acid_initial": 4.0,  # 酒石酸 g/L (原料果汁中含量)
        "non_volatile_acid_type": "酒石酸",
        "notes": "富含芳香物质"
    },
}

# 化学计量系数
STARCH_TO_GLUCOSE = 1.11    # 淀粉水解系数
GLUCOSE_TO_ETHANOL = 0.51  # 酵母发酵收率
ETHANOL_TO_ACETIC = 1.30   # 醋酸菌氧化系数

# 工艺参数
STARCH_CONVERSION_RATE = 0.88  # 糯米糖化转化率 (薛茂云, 2018)
YEAST_EFFICIENCY = 0.48        # 酵母乙醇转化效率 (丁乾坤, 2019)
AAF_ACETIC_RATIO = 0.90        # AAF产物中乙酸占比
LEACHING_EFFICIENCY = 0.75      # 淋醋提取效率

# 规模效应系数
# 实验室 (< 10kg 原料): 控制精确但损失大
# 中试 (10-500kg): 接近生产
# 生产 (> 500kg): 规模效益, 效率提升
SCALE_FACTORS = {
    "lab": {
        "saccharification": 0.82,  # 实验室糖化效率较低
        "fermentation": 0.85,      # 发酵控制差
        "leaching": 0.70,          # 提取损失大
        "description": "实验室规模 (<10kg原料)"
    },
    "pilot": {
        "saccharification": 0.88,
        "fermentation": 0.90,
        "leaching": 0.78,
        "description": "中试规模 (10-500kg原料)"
    },
    "production": {
        "saccharification": 0.92,  # 生产规模效率高
        "fermentation": 0.93,
        "leaching": 0.85,
        "description": "生产规模 (>500kg原料)"
    }
}


@dataclass
class ProductionInput:
    """生产输入参数"""
    raw_material_kg: float = 100.0       # 原料量 (kg)
    raw_material_type: str = "糯米"        # 原料类型
    scale_type: str = "pilot"            # 生产规模: lab, pilot, production
    water_ratio: float = 3.0              # 加水比 (原料:水)
    saccharification_hours: float = 60.0  # 糖化时间 (h)
    saccharification_temp: float = 60.0   # 糖化温度 (°C)
    alcohol_days: float = 6.0             # 酒精发酵天数
    alcohol_temp: float = 30.0            # 酒精发酵温度 (°C)
    aaf_days: float = 18.0                # AAF发酵天数
    aaf_turnover_times: int = 2           # AAF翻醅次数 (0-3)
    leaching_water_ratio: float = 1.5     # 淋醋加水比 (醅:水)
    leaching_hours: float = 16.0          # 淋醋时间 (h)
    aging_months: float = 60.0            # 陈酿月数
    aging_vessel: str = "陶缸"            # 陈酿容器: 陶缸, 不锈钢
    aging_temperature: str = "常温"       # 陈酿温度: 低温, 常温, 高温

# 陈酿容器效应系数
AGING_VESSEL_FACTORS = {
    "陶缸": {
        "ethyl_acetate_rate": 1.15,  # 透气性好, 酯化反应快
        "tmp_rate": 1.20,            # 氧化和美拉德反应促进TMP
        "acid_loss_rate": 0.97,       # 轻微挥发损失
        "description": "传统陶缸, 透气性好, 风味物质形成快"
    },
    "不锈钢": {
        "ethyl_acetate_rate": 0.90,  # 密闭, 酯化较慢
        "tmp_rate": 0.85,            # 氧化受限
        "acid_loss_rate": 1.00,       # 无挥发
        "description": "不锈钢罐, 密闭, 风味变化慢但稳定"
    }
}

# 陈酿温度效应系数
# 温度影响化学反应速率 (范特霍夫规则: 温度每升10°C, 速率增2-3倍)
AGING_TEMPERATURE_FACTORS = {
    "低温": {  # 10-15°C, 地窖陈酿
        "ethyl_acetate_rate": 0.70,
        "tmp_rate": 0.60,
        "description": "低温陈酿, 风味缓慢形成"
    },
    "常温": {  # 20-25°C, 室内陈酿
        "ethyl_acetate_rate": 1.00,
        "tmp_rate": 1.00,
        "description": "常温陈酿, 风味自然演化"
    },
    "高温": {  # 30-35°C, 加速陈酿
        "ethyl_acetate_rate": 1.40,
        "tmp_rate": 1.50,
        "description": "高温加速陈酿, 但风味可能不够细腻"
    }
}

# 翻醅次数对AAF的影响
# 翻醅可以促进氧气混合, 调节温度, 但过多翻醅会破坏菌丝体
TURNOVER_FACTORS = {
    0: {  # 不翻醅
        "efficiency_factor": 0.70,
        "description": "不翻醅, 发酵不均匀"
    },
    1: {  # 偶尔翻醅
        "efficiency_factor": 0.85,
        "description": "偶尔翻醅, 基本保证发酵"
    },
    2: {  # 正常翻醅 (每3-5天一次)
        "efficiency_factor": 1.00,
        "description": "正常翻醅, 发酵良好"
    },
    3: {  # 频繁翻醅
        "efficiency_factor": 0.92,
        "description": "频繁翻醅, 温度均匀但菌体扰动大"
    }
}


@dataclass
class StageOutput:
    """各工序产出"""
    stage_name: str = ""

    # 物料量 (kg 或 L)
    input_kg: float = 0.0
    output_kg: float = 0.0
    volume_L: float = 0.0

    # 关键成分
    starch_kg: float = 0.0
    glucose_kg: float = 0.0
    ethanol_kg: float = 0.0
    acetic_acid_kg: float = 0.0
    total_acid_kg: float = 0.0

    # 浓度 (g/L 或 %)
    glucose_conc_gL: float = 0.0
    ethanol_conc_pct: float = 0.0
    total_acid_conc_gL: float = 0.0

    # 效率/转化率
    conversion_rate: float = 0.0
    extraction_efficiency: float = 0.0

    def as_dict(self) -> Dict:
        return {
            "stage": self.stage_name,
            "input_kg": round(self.input_kg, 2),
            "output_kg": round(self.output_kg, 2),
            "volume_L": round(self.volume_L, 2),
            "starch_kg": round(self.starch_kg, 2),
            "glucose_kg": round(self.glucose_kg, 2),
            "ethanol_kg": round(self.ethanol_kg, 2),
            "acetic_acid_kg": round(self.acetic_acid_kg, 2),
            "total_acid_kg": round(self.total_acid_kg, 2),
            "glucose_conc_gL": round(self.glucose_conc_gL, 1),
            "ethanol_conc_pct": round(self.ethanol_conc_pct, 2),
            "total_acid_conc_gL": round(self.total_acid_conc_gL, 1),
            "conversion_rate": round(self.conversion_rate, 3),
        }


@dataclass
class FullProcessOutput:
    """完整生产流程产出"""
    input: ProductionInput

    saccharification: StageOutput = field(default_factory=StageOutput)
    alcohol: StageOutput = field(default_factory=StageOutput)
    aaf: StageOutput = field(default_factory=StageOutput)
    leaching: StageOutput = field(default_factory=StageOutput)
    aging_months: float = 0.0

    # 风味物质 (来自aging_kinetics, 单位μg/mL = mg/L)
    ethyl_acetate_ugmL: float = 0.0
    tmp_ugmL: float = 0.0
    overall_score: float = 0.0

    # 最终成品
    final_vinegar_L: float = 0.0
    final_total_acid_gL: float = 0.0
    final_ethyl_acetate_ugmL: float = 0.0
    final_tmp_ugmL: float = 0.0

    def as_dict(self) -> Dict:
        return {
            "input": {
                "raw_material_kg": self.input.raw_material_kg,
                "raw_material_type": self.input.raw_material_type,
                "water_ratio": self.input.water_ratio,
            },
            "saccharification": self.saccharification.as_dict(),
            "alcohol": self.alcohol.as_dict(),
            "aaf": self.aaf.as_dict(),
            "leaching": self.leaching.as_dict(),
            "aging_months": self.aging_months,
            "final": {
                "vinegar_L": round(self.final_vinegar_L, 1),
                "total_acid_gL": round(self.final_total_acid_gL, 1),
                "ethyl_acetate_ugmL": round(self.final_ethyl_acetate_ugmL, 1),
                "tmp_ugmL": round(self.final_tmp_ugmL, 1),
            }
        }


@dataclass
class SaccharificationState:
    """
    原料糖化阶段状态

    糖化阶段是镇江香醋五工序中的第一道工序，
    淀粉在α-淀粉酶和糖化酶作用下转化为葡萄糖。

    工艺参数 (薛茂云, 2018):
    - 温度: 60°C (最佳)
    - 时间: 24-48小时 (实际生产中可缩短至60分钟强化糖化)
    - pH值: 4.0-4.5 (最适合酶活性)
    - 糖化率: 75-88% (受原料种类、蒸煮程度、酶制剂添加量影响)
    """
    duration_hours: float        # 糖化时长(小时)
    temperature: float           # 糖化温度(°C)
    pH: float                    # pH值
    reducing_sugar: float        # 还原糖含量 (g/100mL)
    starch_conversion_rate: float  # 淀粉转化率 (0-1)
    raw_material: str            # 原料类型
    alpha_amylase_activity: float  # α-淀粉酶活性 (U/g)
    glucoamylase_activity: float  # 糖化酶活性 (U/g)
    starch_remaining: float      # 残余淀粉 (%)
    glucose_concentration: float  # 葡萄糖浓度 (g/L)

    def as_dict(self) -> Dict:
        return {
            "stage": "原料糖化",
            "duration_hours": round(self.duration_hours, 1),
            "temperature": round(self.temperature, 1),
            "pH": round(self.pH, 2),
            "reducing_sugar": round(self.reducing_sugar, 2),
            "starch_conversion_rate": round(self.starch_conversion_rate, 3),
            "raw_material": self.raw_material,
            "alpha_amylase_activity": round(self.alpha_amylase_activity, 1),
            "glucoamylase_activity": round(self.glucoamylase_activity, 1),
            "starch_remaining": round(self.starch_remaining, 1),
            "glucose_concentration": round(self.glucose_concentration, 1),
        }

    def get_stage_name(self) -> str:
        """根据转化率返回当前阶段名称"""
        conv = self.starch_conversion_rate * 100
        if conv >= 95:
            return "糖化完成"
        elif conv >= 72:
            return "还原糖释放阶段"
        elif conv >= 35:
            return "酶解糖化阶段"
        elif conv >= 10:
            return "降温拌曲阶段"
        else:
            return "蒸煮糊化阶段"


@dataclass
class AlcoholFermentationState:
    """酒精发酵阶段状态"""
    duration_days: float         # 发酵天数
    temperature: float           # 发酵温度(°C)
    ethanol: float               # 乙醇含量 (% v/v)
    reducing_sugar: float        # 残余还原糖 (g/100mL)
    yeast_viability: float       # 酵母活性 (0-1)
    CO2_production: float        # CO2产量 (估算, g/L)

    def as_dict(self) -> Dict:
        return {
            "stage": "酒精发酵",
            "duration_days": self.duration_days,
            "temperature": self.temperature,
            "ethanol": self.ethanol,
            "reducing_sugar": self.reducing_sugar,
            "yeast_viability": self.yeast_viability,
            "CO2_production": self.CO2_production,
        }


@dataclass
class AAFState:
    """醋酸发酵阶段状态 (AAFKinetics兼容)"""
    day: int
    stage: str
    total_acid: float            # g/100mL
    acetic_acid: float          # g/100mL
    non_volatile_acid: float    # g/100mL
    lactic_acid: float          # g/100mL
    ethanol_residual: float     # % v/v
    oxygen_upper: float         # % 上层溶氧
    oxygen_lower: float         # % 下层溶氧
    temperature_upper: float    # °C
    ab_growth: float            # 醋酸菌相对活性 (0-1)
    lb_growth: float            # 乳酸菌相对活性 (0-1)
    acid_rate: float            # g/100mL per day

    def as_dict(self) -> Dict:
        return {
            "stage": "醋酸发酵",
            "day": self.day,
            "stage_name": self.stage,
            "total_acid": self.total_acid,
            "acetic_acid": self.acetic_acid,
            "non_volatile_acid": self.non_volatile_acid,
            "lactic_acid": self.lactic_acid,
            "ethanol_residual": self.ethanol_residual,
            "oxygen_upper": self.oxygen_upper,
            "oxygen_lower": self.oxygen_lower,
            "temperature_upper": self.temperature_upper,
            "ab_growth": self.ab_growth,
            "lb_growth": self.lb_growth,
            "acid_rate": self.acid_rate,
        }


@dataclass
class LeachingState:
    """淋醋阶段状态"""
    water_ratio: float           # 加水比例 (醋醅:水)
    leaching_time: float         # 浸出时间 (小时)
    extraction_efficiency: float # 提取效率 (0-1)
    total_acid: float            # 淋出液总酸 (g/100mL)
    ethyl_acetate: float         # 淋出液乙酸乙酯 (μg/mL)
    reducing_sugar: float        # 淋出液还原糖 (g/100mL)

    def as_dict(self) -> Dict:
        return {
            "stage": "淋醋",
            "water_ratio": self.water_ratio,
            "leaching_time": self.leaching_time,
            "extraction_efficiency": self.extraction_efficiency,
            "total_acid": self.total_acid,
            "ethyl_acetate": self.ethyl_acetate,
            "reducing_sugar": self.reducing_sugar,
        }


@dataclass
class ProductionState:
    """完整生产流程状态"""
    saccharification: SaccharificationState
    alcohol_fermentation: AlcoholFermentationState
    aaf: AAFState
    leaching: LeachingState
    vinegar_age_months: float    # 陈酿月数

    def as_dict(self) -> Dict:
        return {
            "saccharification": self.saccharification.as_dict(),
            "alcohol_fermentation": self.alcohol_fermentation.as_dict(),
            "aaf": self.aaf.as_dict(),
            "leaching": self.leaching.as_dict(),
            "vinegar_age_months": self.vinegar_age_months,
        }


class SaccharificationModel:
    """
    原料糖化模型

    模拟淀粉在曲霉作用下糖化为还原糖的过程。
    采用一级反应动力学结合Arrhenius温度修正:

        d[糖]/dt = k * [淀粉]
        k = k₀ * exp(-Ea/R * (1/T - 1/T_ref))

    糖化阶段工艺参数 (薛茂云, 2018《镇江香醋糖化工艺的研究》):
    - 温度: 60°C (最佳, 58-62°C适宜区间)
    - 时间: 传统24-48小时; 强化工艺60分钟
    - pH值: 4.0-4.5 (最适合淀粉酶活性)
    - 酶制剂: 糖化酶100U/g, α-淀粉酶20U/g
    - 糖化率: 75-88% (受原料种类、蒸煮程度影响)

    原料差异:
    - 糯米: 淀粉含量75-80%, 蛋白质7-8%, 糖化效率最高(88%)
    - 大米: 淀粉含量65%, 易于糖化, 转化率82%
    - 高粱: 淀粉含量60%, 单宁含量高, 转化率较低(75%)
    - 小麦: 淀粉含量55%, 蛋白质较高, 转化率78%

    质量平衡:
    - 淀粉 → 葡萄糖: 1g淀粉 → 1.11g葡萄糖 (水解系数)
    - 葡萄糖 → 乙醇: 1g葡萄糖 → 0.51g乙醇 (酵母发酵收率)

    验证数据 (巩敏):
    - 糖化液葡萄糖: 36.39-58.02 g/L (≈3.6-5.8 g/100mL)
    - 与模型预测吻合良好

    创新点:
    - 引入pH对酶活性的影响修正
    - 考虑不同原料的淀粉结构差异
    - 提供完整的时间-温度-转化率轨迹
    """

    # 原料基准转化率 (薛茂云, 2018; 包启安)
    RAW_MATERIAL_CONVERSION = {
        "糯米": 0.88,   # 优质糯米，淀粉含量高
        "大米": 0.82,   # 易于糖化
        "高粱": 0.75,   # 单宁影响，转化率较低
        "小麦": 0.78,   # 蛋白质含量高
        "玉米": 0.80,   # 脂肪含量较高
    }

    # 酶活性的最佳pH范围
    OPTIMAL_PH_MIN = 4.0
    OPTIMAL_PH_MAX = 4.5
    OPTIMAL_PH = 4.25

    # 温度参数
    OPTIMAL_TEMP = 60.0   # °C (薛茂云, 2018)
    TEMP_RANGE_MIN = 45.0
    TEMP_RANGE_MAX = 70.0

    def __init__(self):
        self.base_k = 0.7    # 基准反应速率 (1/h), 60°C时约0.7 h⁻¹
        self.Ea = 42000      # 活化能 (J/mol), 淀粉糖化
        self.R = 8.314       # 气体常数 J/(mol·K)

    def _arrhenius_k(self, T: float) -> float:
        """
        计算给定温度下的反应速率常数
        使用Arrhenius方程进行温度修正
        """
        T_ref = 273.15 + 60.0  # 参考温度 60°C
        T_curr = T + 273.15
        return self.base_k * math.exp(self.Ea / self.R * (1 / T_ref - 1 / T_curr))

    def _ph_factor(self, pH: float) -> float:
        """
        计算pH对酶活性的影响因子
        淀粉酶在pH 4.0-4.5范围内活性最高
        偏离此范围活性下降
        """
        if self.OPTIMAL_PH_MIN <= pH <= self.OPTIMAL_PH_MAX:
            return 1.0
        dist_from_optimal = min(abs(pH - self.OPTIMAL_PH_MIN), abs(pH - self.OPTIMAL_PH_MAX))
        return max(0.5, 1.0 - 0.1 * dist_from_optimal)

    def _temperature_factor(self, T: float) -> float:
        """
        计算温度对酶活性的影响因子
        60°C为最佳温度
        """
        if self.TEMP_RANGE_MIN <= T <= self.TEMP_RANGE_MAX:
            optimal_range = 5.0  # ±5°C范围为最佳
            if abs(T - self.OPTIMAL_TEMP) <= optimal_range:
                return 1.0
            deviation = abs(T - self.OPTIMAL_TEMP) - optimal_range
            return max(0.6, 1.0 - 0.05 * deviation)
        return 0.7

    def get_state_at(self, hours: float, temperature: float = 60.0,
                     pH: float = 4.25, raw_material: str = "糯米",
                     alpha_amylase: float = 20.0,
                     glucoamylase: float = 100.0) -> SaccharificationState:
        """
        获取糖化指定时间后的状态

        参数:
            hours: 糖化时长(小时)
                - 传统工艺: 24-48小时
                - 强化工艺: 1-2小时 (高温短时)
            temperature: 糖化温度(°C), 默认60°C (薛茂云, 2018)
            pH: 糖化pH值, 默认4.25 (最佳范围4.0-4.5)
            raw_material: 原料类型 (糯米/大米/高粱/小麦/玉米)
            alpha_amylase: α-淀粉酶活性 (U/g), 默认20U/g
            glucoamylase: 糖化酶活性 (U/g), 默认100U/g

        返回:
            SaccharificationState: 包含还原糖、转化率等指标

        文献依据:
            薛茂云等(2018)《镇江香醋糖化工艺的研究》
            - 最佳条件: 60°C糖化60min, 糖化酶100U/g, α-淀粉酶20U/g
            - 蒸煮6秒后糖化60分钟
            - 最终酒精度可达12%

        模型验证:
            巩敏数据: 糖化液葡萄糖36.39-58.02 g/L
            本模型预测: 糯米在最佳条件下约51.4 g/L (5.14 g/100mL) ✓
        """
        # 获取原料基准转化率
        max_conversion = self.RAW_MATERIAL_CONVERSION.get(raw_material, 0.85)

        # 应用温度修正
        temp_factor = self._temperature_factor(temperature)
        # 应用pH修正
        ph_factor = self._ph_factor(pH)
        # 应用酶活性修正 (相对于基准20U/g α-淀粉酶, 100U/g糖化酶)
        enzyme_factor = min(1.0, (alpha_amylase / 20.0 + glucoamylase / 100.0) / 2)

        # 综合反应速率
        k_effective = self._arrhenius_k(temperature) * temp_factor * ph_factor * enzyme_factor

        # 一级反应动力学: conversion = max * (1 - exp(-k*t))
        conversion = max_conversion * (1 - math.exp(-k_effective * hours))

        # 还原糖计算 (基于巩敏数据校准)
        # 基准: 2.5 g/100mL (未转化时的背景值) + 3.0 * conversion
        reducing_sugar = 2.5 + 3.0 * conversion

        # 葡萄糖浓度 (g/L)
        glucose_concentration = reducing_sugar * 10  # 换算 g/100mL → g/L

        # 残余淀粉
        starch_remaining = (1 - conversion) * 100

        return SaccharificationState(
            duration_hours=hours,
            temperature=temperature,
            pH=pH,
            reducing_sugar=round(reducing_sugar, 2),
            starch_conversion_rate=round(conversion, 3),
            raw_material=raw_material,
            alpha_amylase_activity=alpha_amylase,
            glucoamylase_activity=glucoamylase,
            starch_remaining=round(starch_remaining, 1),
            glucose_concentration=round(glucose_concentration, 1),
        )

    def simulate_trajectory(self, max_hours: float = 6.0,
                           temperature: float = 60.0,
                           pH: float = 4.25,
                           raw_material: str = "糯米",
                           n_points: int = 100) -> Dict[str, List]:
        """
        模拟糖化过程的时间轨迹

        参数:
            max_hours: 最大糖化时间(小时)
            temperature: 糖化温度(°C)
            pH: 糖化pH值
            raw_material: 原料类型
            n_points: 轨迹点数量

        返回:
            Dict: 包含time, reducing_sugar, conversion_rate等轨迹数据
        """
        time_points = [i * max_hours / (n_points - 1) for i in range(n_points)]

        trajectory = {
            "time": [round(t, 2) for t in time_points],
            "reducing_sugar": [],
            "conversion_rate": [],
            "temperature": [],
        }

        for t in time_points:
            state = self.get_state_at(t, temperature, pH, raw_material)
            trajectory["reducing_sugar"].append(state.reducing_sugar)
            trajectory["conversion_rate"].append(state.starch_conversion_rate * 100)
            trajectory["temperature"].append(state.temperature)

        return trajectory

    def get_guidance(self, hours: float, temperature: float = 60.0,
                    pH: float = 4.25, raw_material: str = "糯米") -> Dict:
        """
        获取糖化工艺建议

        返回:
            Dict: 包含level和recommendations列表
        """
        state = self.get_state_at(hours, temperature, pH, raw_material)
        recommendations = []

        # 温度建议
        if abs(temperature - 60) > 5:
            recommendations.append(f"温度{temperature}°C偏离最佳值60°C，建议调整")
        else:
            recommendations.append(f"温度{temperature}°C在适宜范围内")

        # pH建议
        if pH < 4.0 or pH > 4.5:
            recommendations.append(f"pH {pH}偏离最佳范围4.0-4.5，建议调整")
        else:
            recommendations.append(f"pH {pH}在最佳范围内")

        # 时间建议
        if hours < 0.8:
            recommendations.append("糖化时间较短，建议延长至1小时以上")
        elif hours > 2.0:
            recommendations.append("糖化时间较长，可考虑优化工艺")

        # 转化率评估
        conv_pct = state.starch_conversion_rate * 100
        if conv_pct >= 85:
            level = "good"
            recommendations.append(f"转化率{conv_pct:.1f}%优秀")
        elif conv_pct >= 70:
            level = "medium"
            recommendations.append(f"转化率{conv_pct:.1f}%良好")
        else:
            level = "poor"
            recommendations.append(f"转化率{conv_pct:.1f}%偏低，建议优化参数")

        return {
            "level": level,
            "recommendations": recommendations,
        }


class AlcoholFermentationModel:
    """
    酒精发酵模型

    模拟酵母菌将还原糖转化为乙醇的过程。
    采用修正Logistic模型 (Gompertz方程) 拟合乙醇生成过程。

    文献参数 (丁乾坤, 2019; 刘海英, 2017):
    - Gompertz模型 R² > 0.98
    - Logistic修正模型 R² > 0.96
    - 乙醇收率: 0.42-0.48 (42%-48%)
    - 最大比生成速率 ν_max = 0.1896 h⁻¹ (丁乾坤数据)
    - 酵母生长: 0-6h适应期, 6-16h对数期, 16-22h稳定期

    文献依据:
        丁乾坤(2019)《酒精发酵产物动力学模型的研究》
            - 初糖150-250 g/L, 35°C发酵
            - Gompertz模型 R²: 0.981-0.994
            - Logistic修正模型: P_max=67.2 g/L, ν_max=0.1896 h⁻¹ (初糖150 g/L)
        刘海英(2017)《响应面法优化紫薯酒精发酵条件及动力学研究》
            - 最优条件: pH4.06, 29.74°C, 接种量4.58×10⁶ cfu/mL
            - 糖消耗 DoseResp模型 R² = 0.99866
    """

    def __init__(self):
        self.mu_max = 0.30    # 最大比增长速率 (1/day)
        self.Xm = 1.0        # 最大菌体浓度 (归一化)
        self.Yps = 0.67      # 产物得率系数 (校准: 匹配丁乾坤P_max=67g/L at 150g/L初糖)
        self.beta = 0.02     # 维持系数
        self.nu_max = 0.19  # 最大乙醇比生成速率 (h⁻¹), 丁乾坤数据 0.1896

    def get_state_at(self, days: float, initial_sugar: float = 12.0,
                     temperature: float = 30.0) -> AlcoholFermentationState:
        """
        获取酒精发酵days天后的状态

        参数:
            days: 发酵天数
            initial_sugar: 初始还原糖 (g/100mL), 典型12-15 g/100mL (镇江香醋)
            temperature: 发酵温度(°C), 典型28-32°C

        文献依据:
            丁乾坤(2019): 初糖150-250 g/L, 乙醇收率0.42-0.48
            丁乾坤数据: ν_max=0.1896 h⁻¹, P_max=67.2 g/L (初糖150 g/L)
            换算: 67.2 g/L ≈ 8.5% w/w ≈ 10.7% v/v

        校准说明:
            根据丁乾坤数据，初糖150 g/L时最终乙醇约67 g/L (约10.7% v/v)
            但镇江香醋实际乙醇8-12%，故保持默认参数
        """
        if temperature < 25 or temperature > 35:
            temp_factor = 0.85
        else:
            temp_factor = 1.0

        hours = days * 24.0
        X = self.Xm / (1 + (self.Xm / 0.05 - 1) * math.exp(-self.mu_max * temp_factor * hours))
        # 丁乾坤数据: 150g/L初糖在96h达到最大乙醇67g/L
        # 换算: ν_eff ≈ 0.038 h⁻¹ 使5-7天达到最大值

        # 边糖化边发酵: 糖化酶持续产生糖,酵母持续消耗糖
        # 有效糖消耗 = initial_sugar + continuous_sugar_supply
        # continuous_sugar_supply ≈ 1.5 * initial_sugar (经验值,模拟持续糖供应)
        continuous_factor = 1.5
        effective_sugar = initial_sugar * (1 + continuous_factor)

        nu_eff = 0.038
        ethanol = min(12.0, self.Yps * effective_sugar * (1 - math.exp(-nu_eff * hours)))
        residual_sugar = max(0.5, initial_sugar * math.exp(-0.5 * hours / 24.0))
        yeast_viab = max(0.3, 1.0 - 0.03 * days)
        CO2 = ethanol * 1.92

        return AlcoholFermentationState(
            duration_days=days,
            temperature=temperature,
            ethanol=round(ethanol, 2),
            reducing_sugar=round(residual_sugar, 2),
            yeast_viability=round(yeast_viab, 3),
            CO2_production=round(CO2, 1),
        )


class LeachingModel:
    """
    淋醋模型

    模拟加水浸出醋醅中风味物质的过程。
    基于质量平衡和提取动力学:

    C_final = C_initial * extraction_efficiency * dilution_factor

    文献参数:
    - 加水比: 1:1 到 1:2 (醋醅:水)
    - 时间: 12-24小时
    - 总酸提取率: 70-85%

    注: 淋醋后得到的醋液总酸约为AAF醋醅的60-70%,
    因为醋醅中含有约30-40%的水分, 加水后提取的是稀释液,
    后续通过勾调达到成品醋标准(5-6g/100mL总酸)。
    """

    def __init__(self):
        self.base_extraction = 0.80  # 基准提取率 (总酸提取约80%)
        self.water_ratio = 1.2       # 基准加水比

    def get_state_at(self, aaf_state: AAFState,
                     water_ratio: float = 1.2,
                     leaching_time: float = 16.0) -> LeachingState:
        """
        基于AAF发酵状态计算淋醋结果

        参数:
            aaf_state: 醋酸发酵结束时的状态
            water_ratio: 加水比例 (醋醅:水), 实际生产中约1:0.8-1:1.2
            leaching_time: 浸出时间(小时)
        """
        time_factor = 1.0 - math.exp(-leaching_time / 24.0)
        extraction = self.base_extraction * time_factor

        dilution = 1.0 / (1 + water_ratio)
        total_acid = aaf_state.total_acid * extraction * (1 + water_ratio) * 0.9
        ethyl_acetate = aaf_state.acetic_acid * extraction * 300 * (1 + water_ratio) * 0.5
        reducing_sugar = 0.8 * extraction

        return LeachingState(
            water_ratio=water_ratio,
            leaching_time=leaching_time,
            extraction_efficiency=round(extraction, 3),
            total_acid=round(total_acid, 2),
            ethyl_acetate=round(ethyl_acetate, 1),
            reducing_sugar=round(reducing_sugar, 2),
        )


class VinegarProductionModel:
    """
    镇江香醋完整生产流程模型

    整合五个工序:
    1. 原料糖化 (SaccharificationModel)
    2. 酒精发酵 (AlcoholFermentationModel)
    3. 醋酸发酵 (AAFModel)
    4. 淋醋 (LeachingModel)
    5. 陈酿 (aging_kinetics.age_to_state)

    使用方式:
    ```python
    from vinegar_model.process_model import VinegarProductionModel

    model = VinegarProductionModel()

    # 模拟完整生产流程
    state = model.simulate_full_process(
        raw_material="糯米",
        saccharification_hours=60,
        alcohol_fermentation_days=6,
        aaf_days=18,
        water_ratio=1.2,
        aging_months=60,
    )
    print(state.aaf.total_acid)  # 醋酸发酵总酸
    print(state.leaching.total_acid)  # 淋醋后总酸
    ```
    """

    def __init__(self):
        self.saccharification = SaccharificationModel()
        self.alcohol = AlcoholFermentationModel()
        self.leaching = LeachingModel()

    def simulate_full_process(
        self,
        raw_material: str = "糯米",
        saccharification_hours: float = 60.0,
        saccharification_temp: float = 62.0,
        alcohol_fermentation_days: float = 6.0,
        alcohol_temp: float = 30.0,
        aaf_days: float = 18.0,
        water_ratio: float = 1.2,
        leaching_time: float = 16.0,
        aging_months: float = 0.0,
    ) -> ProductionState:
        """
        模拟完整生产流程

        参数:
            raw_material: 原料类型
            saccharification_hours: 糖化时间(小时)
            saccharification_temp: 糖化温度(°C)
            alcohol_fermentation_days: 酒精发酵天数
            alcohol_temp: 酒精发酵温度(°C)
            aaf_days: 醋酸发酵天数
            water_ratio: 淋醋加水比
            leaching_time: 淋醋时间(小时)
            aging_months: 陈酿月数

        返回:
            ProductionState: 完整生产流程状态
        """
        sac_state = self.saccharification.get_state_at(
            hours=saccharification_hours,
            temperature=saccharification_temp,
            pH=4.25,
            raw_material=raw_material
        )

        alc_state = self.alcohol.get_state_at(
            alcohol_fermentation_days, sac_state.reducing_sugar, alcohol_temp
        )

        from .aaf_kinetics import AAFModel
        aaf_model = AAFModel()
        aaf_state = aaf_model.get_state_at(aaf_days)

        leached_state = self.leaching.get_state_at(aaf_state, water_ratio, leaching_time)

        return ProductionState(
            saccharification=sac_state,
            alcohol_fermentation=alc_state,
            aaf=aaf_state,
            leaching=leached_state,
            vinegar_age_months=aging_months,
        )

    def get_stage_summary(self, state: ProductionState) -> Dict:
        """
        获取各工序关键指标摘要
        """
        return {
            "原料糖化": {
                "时长": f"{state.saccharification.duration_hours:.0f}h",
                "温度": f"{state.saccharification.temperature:.0f}°C",
                "还原糖": f"{state.saccharification.reducing_sugar:.1f} g/100mL",
                "淀粉转化率": f"{state.saccharification.starch_conversion_rate*100:.0f}%",
            },
            "酒精发酵": {
                "时长": f"{state.alcohol_fermentation.duration_days:.0f}天",
                "温度": f"{state.alcohol_fermentation.temperature:.0f}°C",
                "乙醇": f"{state.alcohol_fermentation.ethanol:.1f}%",
                "酵母活性": f"{state.alcohol_fermentation.yeast_viability*100:.0f}%",
            },
            "醋酸发酵": {
                "时长": f"{state.aaf.day:.0f}天",
                "阶段": state.aaf.stage,
                "总酸": f"{state.aaf.total_acid:.2f} g/100mL",
                "乙酸": f"{state.aaf.acetic_acid:.2f} g/100mL",
            },
            "淋醋": {
                "加水比": f"1:{state.leaching.water_ratio:.1f}",
                "提取效率": f"{state.leaching.extraction_efficiency*100:.0f}%",
                "总酸": f"{state.leaching.total_acid:.2f} g/100mL",
            },
            "陈酿": {
                "时长": f"{state.vinegar_age_months:.0f}月",
            }
        }


def inspect_aaF_state(day: int) -> AAFState:
    """
    兼容接口: 查询AAF阶段状态
    使用AAFKinetics模型
    """
    from .aaf_kinetics import AAFModel
    model = AAFModel()
    state = model.get_state_at(float(day))
    return state


def recommend_turning(day: int,
                      current_oxygen_upper: float = None,
                      current_oxygen_lower: float = None,
                      current_temperature_upper: float = None) -> Dict:
    """
    兼容接口: 翻醅建议
    """
    from .aaf_kinetics import AAFModel
    model = AAFModel()
    return model.recommend_turning(
        float(day),
        oxygen_upper=current_oxygen_upper,
        oxygen_lower=current_oxygen_lower,
        temperature=current_temperature_upper,
    )


def next_dynamics_step(current_day: int,
                       current_hours_remaining: float,
                       target_hours: float) -> int:
    """
    兼容接口: 推进发酵时间
    """
    new_total_h = current_day * 24.0 - current_hours_remaining + target_hours
    new_day = max(1, min(18, int(new_total_h // 24)))
    return new_day


class MaterialBasedModel:
    """
    基于原料量的生产模型

    化学计量计算:
    - 淀粉 → 葡萄糖: 淀粉量 × 1.11
    - 葡萄糖 → 乙醇: 葡萄糖量 × 0.51 × 发酵效率
    - 乙醇 → 乙酸: 乙醇量 × 1.30 × 氧化效率
    """

    def calculate_full_process(self, input_params: ProductionInput) -> FullProcessOutput:
        """
        计算完整生产流程

        参数:
            input_params: 生产输入参数

        返回:
            FullProcessOutput: 完整生产流程产出
        """
        comp = RAW_MATERIAL_COMPOSITION.get(
            input_params.raw_material_type,
            RAW_MATERIAL_COMPOSITION["糯米"]
        )

        # 获取原料个性化参数
        raw_type = comp.get("type", "starch")
        sac_rate = comp.get("saccharification_rate", 0.85)  # 糖化转化率
        ferm_eff = comp.get("fermentation_efficiency", 0.48)  # 发酵效率

        # 获取规模效应系数
        scale_type = input_params.scale_type if hasattr(input_params, 'scale_type') else "pilot"
        scale = SCALE_FACTORS.get(scale_type, SCALE_FACTORS["pilot"])

        # ===== 1. 糖化阶段 =====
        if raw_type == "sugar":
            # 糖基原料 (水果等): 直接使用糖分
            sugar_kg = input_params.raw_material_kg * comp.get("sugar_pct", 0.15)
            glucose_kg = sugar_kg * 1.05 * scale["saccharification"]  # 应用规模效应
            starch_kg = 0
            conversion_rate = 1.0 * scale["saccharification"]
        else:
            # 淀粉基原料: 淀粉水解为葡萄糖
            starch_kg = input_params.raw_material_kg * comp["starch_pct"]
            glucose_kg = starch_kg * STARCH_TO_GLUCOSE * sac_rate * scale["saccharification"]
            conversion_rate = sac_rate * scale["saccharification"]

        # 醪液体积 = 原料量 + 加水量 (近似, 密度≈1)
        mash_volume_L = input_params.raw_material_kg + input_params.raw_material_kg * input_params.water_ratio

        # 葡萄糖浓度
        glucose_conc_gL = glucose_kg / mash_volume_L * 1000

        sac_output = StageOutput(
            stage_name="糖化",
            input_kg=input_params.raw_material_kg,
            output_kg=mash_volume_L,
            volume_L=mash_volume_L,
            starch_kg=starch_kg,
            glucose_kg=glucose_kg,
            glucose_conc_gL=glucose_conc_gL,
            conversion_rate=conversion_rate,
        )

        # ===== 2. 酒精发酵阶段 =====
        # 乙醇量 = 葡萄糖量 × 乙醇转化率 × 原料个性化发酵效率 × 规模效应
        # (考虑边糖化边发酵, 有效糖量约为直接糖化的1.5倍)
        effective_glucose_factor = 1.5
        ethanol_kg = glucose_kg * effective_glucose_factor * GLUCOSE_TO_ETHANOL * ferm_eff * scale["fermentation"]

        # 乙醇浓度 (体积分数, 近似)
        ethanol_conc_pct = (ethanol_kg / mash_volume_L) * 100 / 0.789

        # CO2产量
        co2_kg = glucose_kg * effective_glucose_factor * 0.48 * scale["fermentation"]

        alc_output = StageOutput(
            stage_name="酒精发酵",
            input_kg=mash_volume_L,
            output_kg=mash_volume_L - co2_kg * 0.3,  # 近似损失
            volume_L=mash_volume_L,
            glucose_kg=glucose_kg,
            ethanol_kg=ethanol_kg,
            ethanol_conc_pct=ethanol_conc_pct,
            conversion_rate=YEAST_EFFICIENCY * scale["fermentation"],
        )

        # ===== 3. 醋酸发酵 (AAF) =====
        # 获取翻醅次数效应
        turnover_times = input_params.aaf_turnover_times if hasattr(input_params, 'aaf_turnover_times') else 2
        turnover_factor = TURNOVER_FACTORS.get(turnover_times, TURNOVER_FACTORS[2])["efficiency_factor"]

        # 使用基于乙醇量的AAF计算
        from .aaf_kinetics import calculate_from_ethanol
        aaf_result = calculate_from_ethanol(
            ethanol_kg=ethanol_kg,
            mash_volume_L=mash_volume_L,
            days=input_params.aaf_days
        )

        # 应用翻醅次数效应
        acetic_acid_kg = aaf_result["acetic_acid_kg"] * turnover_factor
        total_acid_kg = aaf_result["total_acid_kg"] * turnover_factor
        vinegar_mash_kg = aaf_result["vinegar_mash_L"]
        total_acid_conc_gL = aaf_result["total_acid_gL"] * turnover_factor

        aaf_output = StageOutput(
            stage_name="醋酸发酵",
            input_kg=mash_volume_L,
            output_kg=vinegar_mash_kg,
            volume_L=vinegar_mash_kg,
            ethanol_kg=ethanol_kg,
            acetic_acid_kg=acetic_acid_kg,
            total_acid_kg=total_acid_kg,
            total_acid_conc_gL=total_acid_conc_gL,
            conversion_rate=aaf_result["efficiency"],
        )

        # ===== 4. 淋醋 =====
        # 获取原料的淋醋特性和醋醅密度
        mat_leaching_eff = comp.get("leaching_eff", 0.75)  # 原料特有淋醋效率
        mash_density = comp.get("mash_density", 1.05)  # 醋醅密度 (kg/L)

        # 醋醅实际体积 (L)
        vinegar_mash_L = vinegar_mash_kg / mash_density

        # 成品醋体积 = 醋醅体积 × 加水比 × 原料淋醋效率 × 规模效应
        effective_leaching_eff = mat_leaching_eff * scale["leaching"]
        vinegar_L = vinegar_mash_L * input_params.leaching_water_ratio * effective_leaching_eff

        # 淋醋后总酸浓度 (稀释后)
        final_acid_conc_gL = total_acid_kg / vinegar_L * 1000

        leach_output = StageOutput(
            stage_name="淋醋",
            input_kg=vinegar_mash_kg,
            output_kg=vinegar_L,
            volume_L=vinegar_L,
            acetic_acid_kg=acetic_acid_kg,
            total_acid_kg=total_acid_kg,
            total_acid_conc_gL=final_acid_conc_gL,
            extraction_efficiency=effective_leaching_eff,
        )

        # ===== 5. 陈酿 (风味物质计算) =====
        # 陈酿过程中:
        # - 总酸保持稳定 (不增加)
        # - 乙酸乙酯和TMP从陈酿基准值开始增加
        from .aging_kinetics import age_to_state

        # 获取原料的陈酿特性
        aging_factor = comp.get("aging_factor", 1.0)  # 果蔬类有独立的陈酿因子
        acid_profile = comp.get("acid_profile", "乙酸为主")  # 酸 profile

        # 总酸保持淋醋后的值 (g/L)
        final_acid_gL = final_acid_conc_gL

        # 陈酿计算
        if raw_type == "sugar":
            # 果蔬类醋的陈酿特点:
            # - 风味物质形成更快 (aging_factor < 1)
            # - 以果香酯类为主, 非四甲基吡嗪
            # - 酸度稳定, 不需要长期陈酿
            aging_months_adj = input_params.aging_months * aging_factor

            # 果蔬醋的风味物质演化
            # 陈酿温度效应
            temp_type = input_params.aging_temperature if hasattr(input_params, 'aging_temperature') else "常温"
            temp = AGING_TEMPERATURE_FACTORS.get(temp_type, AGING_TEMPERATURE_FACTORS["常温"])

            # 初始乙酸乙酯较高 (果香)
            initial_ethyl_baseline = 1500.0 * temp["ethyl_acetate_rate"]  # 温度影响
            ethyl_acetate_ugmL = initial_ethyl_baseline + aging_months_adj * 20 * temp["ethyl_acetate_rate"]
            ethyl_acetate_ugmL = min(ethyl_acetate_ugmL, 4000.0)

            # TMP在果蔬醋中不重要
            initial_tmp_baseline = 5.0 * temp["tmp_rate"]
            tmp_ugmL = initial_tmp_baseline + aging_months_adj * 0.5 * temp["tmp_rate"]
            tmp_ugmL = min(tmp_ugmL, 50.0)

            # 果蔬醋的process用于风味计算
            aging_process = "液态发酵" if "苹果" in input_params.raw_material_type or "葡萄" in input_params.raw_material_type else "固态发酵"
        else:
            # 谷物醋: 基准陈酿曲线 + 原料风味因子 + 容器效应 + 温度效应
            ethyl_factor = comp.get("ethyl_acetate_factor", 1.0)
            tmp_factor = comp.get("tmp_factor", 1.0)

            # 陈酿容器效应
            vessel_type = input_params.aging_vessel if hasattr(input_params, 'aging_vessel') else "陶缸"
            vessel = AGING_VESSEL_FACTORS.get(vessel_type, AGING_VESSEL_FACTORS["陶缸"])

            # 陈酿温度效应
            temp_type = input_params.aging_temperature if hasattr(input_params, 'aging_temperature') else "常温"
            temp = AGING_TEMPERATURE_FACTORS.get(temp_type, AGING_TEMPERATURE_FACTORS["常温"])

            aging_ref_0 = age_to_state(0, "固态发酵", input_params.raw_material_type, "传统")
            aging_ref_t = age_to_state(input_params.aging_months, "固态发酵", input_params.raw_material_type, "传统")

            # 乙酸乙酯增量 (受原料风味因子、容器效应和温度影响)
            total_ethyl_factor = ethyl_factor * vessel["ethyl_acetate_rate"] * temp["ethyl_acetate_rate"]
            ethyl_increase = (aging_ref_t.ethyl_acetate - aging_ref_0.ethyl_acetate) * total_ethyl_factor
            initial_ethyl_baseline = 800.0 * ethyl_factor
            ethyl_acetate_ugmL = max(initial_ethyl_baseline, min(initial_ethyl_baseline + ethyl_increase, 5000.0))

            # TMP增量计算 (受原料风味因子、容器效应和温度影响)
            total_tmp_factor = tmp_factor * vessel["tmp_rate"] * temp["tmp_rate"]
            tmp_increase = (aging_ref_t.tmp - aging_ref_0.tmp) * total_tmp_factor
            initial_tmp_baseline = 10.0 * tmp_factor
            tmp_ugmL = max(initial_tmp_baseline, min(initial_tmp_baseline + tmp_increase * 2, 200.0))

            aging_process = "固态发酵"

        # 综合评分 (基于实际陈酿参数)
        from .flavor_radar import VinegarState, compute_flavor_profile, compute_sensory_score, compute_overall_score
        from .aging_kinetics import age_to_state as get_aging_ref

        # 获取陈酿参考状态
        aging_ref = get_aging_ref(input_params.aging_months, aging_process, input_params.raw_material_type, "传统")

        # 非挥发性酸计算
        if raw_type == "sugar":
            # 果蔬原料: 使用原料中的初始有机酸含量
            non_volatile_initial = comp.get("non_volatile_acid_initial", 1.0)
            # 稀释效应后保留约70%
            non_volatile_acid = non_volatile_initial * 0.7 * (100 / vinegar_L)
        else:
            # 谷物原料: 使用陈酿参考值
            non_volatile_acid = aging_ref.non_volatile_acid

        # 创建实际陈酿状态
        aging_state = VinegarState(
            vinegar_age_months=input_params.aging_months,
            total_acid=final_acid_gL / 10.0,  # 转为 g/100mL
            non_volatile_acid=non_volatile_acid,
            reducing_sugar=aging_ref.reducing_sugar if raw_type == "starch" else 2.0,
            total_amino_acid=aging_ref.total_amino_acid if raw_type == "starch" else 0.1,
            ethyl_acetate=ethyl_acetate_ugmL,
            tmp=tmp_ugmL,
            acetic_acid=acetic_acid_kg / vinegar_L * 10,
            ph=aging_ref.ph if raw_type == "starch" else 3.5,
            process=aging_process,
            raw_material=input_params.raw_material_type,
            craft_style="传统"
        )

        profile = compute_flavor_profile(aging_state)
        sensory = compute_sensory_score(aging_state)
        overall = compute_overall_score(aging_state, profile, sensory)

        return FullProcessOutput(
            input=input_params,
            saccharification=sac_output,
            alcohol=alc_output,
            aaf=aaf_output,
            leaching=leach_output,
            aging_months=input_params.aging_months,
            ethyl_acetate_ugmL=ethyl_acetate_ugmL,
            tmp_ugmL=tmp_ugmL,
            overall_score=overall,
            final_vinegar_L=vinegar_L,
            final_total_acid_gL=final_acid_gL,  # 使用陈酿后的总酸
            final_ethyl_acetate_ugmL=ethyl_acetate_ugmL,
            final_tmp_ugmL=tmp_ugmL,
        )


def calculate_from_raw_material(
    raw_material_kg: float = 100.0,
    raw_material_type: str = "糯米",
    scale_type: str = "pilot",
    aging_vessel: str = "陶缸",
    aging_temperature: str = "常温",
    aaf_turnover_times: int = 2,
    water_ratio: float = 3.0,
    saccharification_hours: float = 60.0,
    alcohol_days: float = 6.0,
    aaf_days: float = 18.0,
    leaching_water_ratio: float = 1.5,
    aging_months: float = 60.0,
) -> FullProcessOutput:
    """
    基于原料量计算完整生产流程

    示例:
    ```
    result = calculate_from_raw_material(
        raw_material_kg=100,
        raw_material_type="糯米",
        scale_type="pilot",
        aging_vessel="陶缸",
        aging_temperature="常温",
        aaf_turnover_times=2,
        water_ratio=3.0,
        alcohol_days=6,
        aaf_days=18,
        aging_months=60
    )
    print(f"成品醋: {result.final_vinegar_L:.1f} L")
    print(f"总酸: {result.final_total_acid_gL:.1f} g/L")
    ```
    """
    model = MaterialBasedModel()
    input_params = ProductionInput(
        raw_material_kg=raw_material_kg,
        raw_material_type=raw_material_type,
        scale_type=scale_type,
        aging_vessel=aging_vessel,
        aging_temperature=aging_temperature,
        aaf_turnover_times=aaf_turnover_times,
        water_ratio=water_ratio,
        saccharification_hours=saccharification_hours,
        alcohol_days=alcohol_days,
        aaf_days=aaf_days,
        leaching_water_ratio=leaching_water_ratio,
        aging_months=aging_months,
    )
    return model.calculate_full_process(input_params)


if __name__ == "__main__":
    print("=" * 60)
    print("  镇江香醋五工序生产模型")
    print("=" * 60)

    model = VinegarProductionModel()

    state = model.simulate_full_process(
        raw_material="糯米",
        saccharification_hours=60,
        alcohol_fermentation_days=6,
        aaf_days=18,
        water_ratio=1.2,
        aging_months=60,
    )

    print("\n[各工序状态]")
    summary = model.get_stage_summary(state)
    for stage, metrics in summary.items():
        print(f"\n  {stage}:")
        for key, val in metrics.items():
            print(f"    {key}: {val}")

    print("\n[完整状态JSON]")
    import json
    print(json.dumps(state.as_dict(), ensure_ascii=False, indent=2))
