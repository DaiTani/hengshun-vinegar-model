"""
vinegar_model
=============
基于约80篇镇江香醋/山西老陈醋/食醋发酵与风味分析文献构建的食醋风味模型。

核心能力
--------
1. AAF发酵动力学 (aaf_kinetics):
   基于王超(2020)数据的Logistic拟合，R²=0.998
   用于发酵过程监测与翻醅建议

2. 陈酿动力学 (aging_kinetics):
   基于任晓荣(2023)等实测数据的Logistic曲线拟合
   用于预测任意陈酿时间的风味组成

3. 风味映射 (flavor_radar):
   8维理化指标 -> 6维风味雷达 + 14维感官评分 + 综合评分

4. REST API:
   基于Flask的轻量接口,JSON格式输入/输出

文献依据
--------
- 王超等(2020): 镇江香醋醋酸发酵过程中理化指标的动态分析研究
- 任晓荣等(2023): 不同陈酿年份镇江香醋品质指标和功能成分的比较
- 郑梦林等(2021): 镇江香醋陈酿过程中主要呈味物质的分析
- 刘卓非(2022): 食醋固态酿造过程氧含量监测及时序预测
- 李晓伟等(2022): 食醋固态发酵罐条件优化及发酵动力学分析
- GB/T 18623-2011: 地理标志产品镇江香醋
"""

from .aaf_kinetics import AAFModel, AAFKinetics
from .aging_kinetics import (
    age_to_state, predict_trajectory, predict_at_age, AGE_FUNCTIONS,
)
from .flavor_radar import (
    VinegarState, FlavorProfile, SensoryScore, PHProfile,
    compute_flavor_profile, compute_sensory_score,
    compute_ph_profile, compute_overall_score, radar_chart,
)
from .craft_effect import (
    apply_craft_effect, craft_summary,
    PROCESS_FACTORS, MATERIAL_FACTORS,
)
from .process_model import (
    AAFState, inspect_aaF_state, recommend_turning, next_dynamics_step,
)
from .hybrid_model import HybridVinegarModel

__version__ = "1.0.0"

__all__ = [
    "AAFModel",
    "AAFKinetics",
    "AAFState",
    "HybridVinegarModel",
    "age_to_state",
    "predict_trajectory",
    "predict_at_age",
    "AGE_FUNCTIONS",
    "inspect_aaF_state",
    "recommend_turning",
    "next_dynamics_step",
    "VinegarState",
    "FlavorProfile",
    "SensoryScore",
    "PHProfile",
    "compute_flavor_profile",
    "compute_sensory_score",
    "compute_ph_profile",
    "compute_overall_score",
    "radar_chart",
    "apply_craft_effect",
    "craft_summary",
    "PROCESS_FACTORS",
    "MATERIAL_FACTORS",
]
