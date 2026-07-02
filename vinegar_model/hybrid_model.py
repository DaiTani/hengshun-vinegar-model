"""
hybrid_model.py : 双层醋风味模型
================================
结合经验模型与发酵动力学，提供完整的醋酸发酵+陈酿预测能力。

层1 - 经验模型 (陈酿):
    基于任晓荣(2023)等实测数据拟合的Logistic曲线
    适用: 快速预测任意陈酿时间的风味组成

层2 - 发酵动力学 (AAF):
    基于王超(2020)实测数据的Logistic拟合
    适用: 发酵过程监测与翻醅建议

注意: 本模块不包含假的化学动力学机制。
TMP等陈酿产物的S形演化曲线是经验拟合结果，
其背后的化学机理（Maillard反应、Strecker降解等）
需要更深入的实验研究才能建立定量模型。
"""

from __future__ import annotations

from typing import Dict

from .aaf_kinetics import AAFModel
from .aging_kinetics import age_to_state, predict_trajectory
from .flavor_radar import VinegarState


class HybridVinegarModel:
    """
    双层醋风味模型

    提供:
    - AAF发酵状态 (基于王超2020数据)
    - 陈酿状态 (基于任晓荣2023数据)
    - 风味预测与评分
    """

    def __init__(self):
        self.aaf = AAFModel()

    def get_aaf_state(self, day: float) -> Dict:
        """
        获取第day天的发酵状态

        Returns:
            dict with keys: day, stage, total_acid, acetic_acid,
            non_volatile_acid, lactic_acid, ethanol_residual,
            oxygen_upper, oxygen_lower, temperature_upper, ab_growth, lb_growth
        """
        state = self.aaf.get_state_at(day)
        return state.as_dict()

    def get_aging_state(self, months: float, process: str = "固态发酵",
                        raw_material: str = "糯米",
                        craft_style: str = "现代") -> VinegarState:
        """
        获取陈酿months月后的风味状态
        """
        return age_to_state(months, process, raw_material, craft_style)

    def get_combined_state(self, day: float, months: float = 0.0,
                          process: str = "固态发酵",
                          raw_material: str = "糯米",
                          craft_style: str = "现代") -> Dict:
        """
        获取发酵+陈酿的完整状态

        实际生产中，day是发酵天数，months是陈酿月数。
        新醋(day=18发酵结束, months=0)经过陈酿后变成成品醋。
        """
        aaf_state = self.get_aaf_state(day)
        aging_state = self.get_aging_state(months, process, raw_material, craft_style)

        return {
            "fermentation": aaf_state,
            "aging": {
                "vinegar_age_months": aging_state.vinegar_age_months,
                "tmp": aging_state.tmp,
                "ethyl_acetate": aging_state.ethyl_acetate,
                "total_amino_acid": aging_state.total_amino_acid,
                "total_acid": aging_state.total_acid,
                "ph": aging_state.ph,
            }
        }

    def get_kinetic_parameters(self) -> Dict:
        """
        返回模型参数摘要

        AAF: 基于王超(2020)的Logistic拟合参数
        陈酿: 基于任晓荣(2023)的Logistic拟合参数
        """
        aaf_val = self.aaf.validate()

        return {
            "aaf_fermentation": {
                "model_type": "分段Logistic (王超2020)",
                "literature_source": "王超(2020) - 镇江香醋醋酸发酵过程中理化指标的动态分析研究",
                "validation_r2_total_acid": aaf_val["r2_total_acid"],
                "validation_r2_acetic_acid": aaf_val["r2_acetic_acid"],
                "total_acid_params": {
                    "K": self.aaf.ta_K,
                    "k": self.aaf.ta_k,
                    "t0": self.aaf.ta_t0,
                },
                "acetic_acid_params": {
                    "K": self.aaf.ac_K,
                    "k": self.aaf.ac_k,
                    "t0": self.aaf.ac_t0,
                },
            },
            "aging": {
                "model_type": "Logistic曲线 (任晓荣2023)",
                "note": "陈酿参数曲线见aging_kinetics.py中的AGE_FUNCTIONS",
            }
        }


if __name__ == "__main__":
    model = HybridVinegarModel()

    print("=" * 60)
    print("双层醋风味模型")
    print("=" * 60)

    print("\n[发酵状态 - AAF (王超2020)]")
    for day in [0, 3, 8, 11, 13, 17, 21]:
        state = model.get_aaf_state(day)
        print(f"  Day {day:2d}: 总酸={state['total_acid']:.2f}, "
              f"乙酸={state['acetic_acid']:.2f}, 阶段={state['stage']}")

    print("\n[陈酿状态 - Logistic (任晓荣2023)]")
    for months in [0, 36, 60, 96, 120]:
        state = model.get_aging_state(months)
        print(f"  {months:3d}月: TMP={state.tmp:.1f}, "
              f"乙酸乙酯={state.ethyl_acetate:.0f}, 总酸={state.total_acid:.2f}")

    print("\n[模型验证]")
    params = model.get_kinetic_parameters()
    aaf = params["aaf_fermentation"]
    print(f"  AAF R²(总酸): {aaf['validation_r2_total_acid']:.4f}")
    print(f"  AAF R²(乙酸): {aaf['validation_r2_acetic_acid']:.4f}")
