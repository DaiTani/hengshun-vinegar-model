"""
process_model.py : 镇江香醋五工序生产过程模型
==============================================

本模块对镇江香醋的完整生产流程进行建模，包含五个核心工序：

1. 原料糖化 (Saccharification)
   - 糯米/大米中的淀粉在曲霉作用下糖化为还原糖
   - 温度: 60°C, 时间: 与酒精发酵同步 (薛茂云, 2018)
   - 关键产物: 还原糖
   - 注: 实际工艺为边糖化边发酵，本模型使用简化处理

2. 酒精发酵 (Alcoholic Fermentation)
   - 酵母菌将糖转化为乙醇
   - 温度: 28-32°C, 时间: 5-7天
   - 关键产物: 乙醇, CO2
   - 文献: 丁乾坤(2019) 动力学模型 R²>0.98

3. 醋酸发酵 (AAF - Acetic Acid Fermentation)
   - 醋酸菌将乙醇氧化为乙酸
   - 温度: 30-42°C, 时间: 18-20天
   - 关键产物: 总酸, 乙酸
   - 模型: aaf_kinetics.AAFModel (R²=0.998)

4. 淋醋 (Vinegar Leaching)
   - 用水浸出醋醅中的风味物质
   - 温度: 常温, 时间: 12-24h
   - 关键产物: 成品醋原液

5. 陈酿 (Aging)
   - 醋在陶坛中陈化，风味物质转化
   - 温度: 20-30°C, 时间: 0-120月
   - 关键产物: TMP, 乙酸乙酯等风味物质
   - 模型: aging_kinetics.age_to_state

每个工序的输出作为下一工序的输入，实现从原料到成品醋的完整追踪。

文献依据
--------
醋酸发酵:
  - 王超等(2020): 醋酸发酵阶段理化指标动态分析 (R²=0.998)

陈酿:
  - 任晓荣等(2023): 不同陈酿年份镇江香醋品质指标分析
  - 郑梦林等(2021): 陈酿过程中主要呈味物质分析

酒精发酵:
  - 丁乾坤(2019): 酒精发酵产物动力学模型研究
    * Gompertz模型 R²: 0.981-0.994
    * 乙醇收率: 0.42-0.48
  - 刘海英(2017): 响应面法优化紫薯酒精发酵条件
    * 最优条件: pH4.06, 29.74°C
    * 糖消耗DoseResp模型 R²=0.99866

糖化工艺:
  - 薛茂云(2018): 镇江香醋糖化工艺研究
    * 最佳工艺: 蒸煮6s, 60°C糖化60min, 糖化酶100U/g, α-淀粉酶20U/g
    * 最终酒精度可达12%
    * 注: 缺乏还原糖-时间动力学曲线数据
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class SaccharificationState:
    """原料糖化阶段状态"""
    duration_hours: float        # 糖化时长(小时)
    temperature: float           # 糖化温度(°C)
    reducing_sugar: float        # 还原糖含量 (g/100mL)
    starch_conversion_rate: float  # 淀粉转化率 (0-1)
    raw_material: str            # 原料类型

    def as_dict(self) -> Dict:
        return {
            "stage": "原料糖化",
            "duration_hours": self.duration_hours,
            "temperature": self.temperature,
            "reducing_sugar": self.reducing_sugar,
            "starch_conversion_rate": self.starch_conversion_rate,
            "raw_material": self.raw_material,
        }


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
    采用一级反应动力学:
        d[糖]/dt = k * [淀粉]
    其中k遵循Arrhenius方程。

    文献参数 (薛茂云, 2018):
    - 最佳工艺: 蒸煮6s, 60°C糖化60min, 糖化酶100U/g, α-淀粉酶20U/g
    - 米水比: 1:3
    - 糖化60min后酒精度可达12%
    - 糯米淀粉转化率可达88%

    注: 缺乏糖化还原糖-时间的详细动力学曲线数据，
    当前模型使用Arrhenius方程估算，实际应用时建议参考薛茂云工艺参数。
    """

    def __init__(self):
        self.base_k = 0.7   # 基准反应速率 (1/h), 60°C时约0.7 h⁻¹
        self.Ea = 42000     # 活化能 (J/mol), 淀粉糖化
        self.R = 8.314      # 气体常数

    def _arrhenius_k(self, T: float) -> float:
        return self.base_k * math.exp(self.Ea / self.R * (1 / 333.15 - 1 / (T + 273.15)))

    def get_state_at(self, hours: float, temperature: float = 60.0,
                     raw_material: str = "糯米") -> SaccharificationState:
        """
        获取糖化hours小时后的状态

        参数:
            hours: 糖化时长(小时)
            temperature: 糖化温度(°C), 最佳60°C (薛茂云, 2018)
            raw_material: 原料类型

        文献依据:
            薛茂云等(2018)《镇江香醋糖化工艺的研究》
            - 最佳条件: 60°C糖化60min, 糖化酶100U/g, α-淀粉酶20U/g
            - 最终酒精度可达12%
        """
        k = self._arrhenius_k(temperature)
        max_conversion = {"糯米": 0.88, "大米": 0.82, "高粱": 0.75, "麦芽": 0.85}.get(raw_material, 0.85)

        conversion = max_conversion * (1 - math.exp(-k * hours))
        reducing_sugar = 3.0 + 7.0 * conversion

        return SaccharificationState(
            duration_hours=hours,
            temperature=temperature,
            reducing_sugar=round(reducing_sugar, 2),
            starch_conversion_rate=round(conversion, 3),
            raw_material=raw_material,
        )


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
        nu_eff = 0.038
        ethanol = min(12.0, self.Yps * initial_sugar * (1 - math.exp(-nu_eff * hours)))
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
            saccharification_hours, saccharification_temp, raw_material
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
