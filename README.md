# 镇江香醋风味数字化模型

Vinegar Flavor Digitalization Model

## 项目概述

本项目基于约80篇镇江香醋/食醋发酵与风味分析文献，构建了一套醋风味数字化模型，对镇江香醋的完整生产流程进行建模。

### 核心能力

1. **五工序生产模型**：原料糖化 → 酒精发酵 → 醋酸发酵 → 淋醋 → 陈酿
2. **AAF醋酸发酵监测**：基于王超(2020)实测数据的Logistic拟合，R²=0.998
3. **陈酿时间预测**：基于任晓荣(2023)等实测数据的Logistic曲线拟合
4. **风味映射**：8维理化指标 → 6维风味雷达 + 14维感官评分 + 综合评分
5. **翻醅建议**：基于发酵动力学模型的智能化决策支持

---

## 系统架构

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ 原料糖化  │ → │ 酒精发酵  │ → │ 醋酸发酵  │ → │   淋醋    │ → │   陈酿    │
│Saccharif.│   │ Alcoholic│   │   AAF    │   │ Leaching │   │  Aging   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
     ↓              ↓              ↓              ↓              ↓
 还原糖          乙醇           总酸           成品醋           风味
 8-10g/100mL   8-10%         6-7g/100mL      4-6g/100mL       化合物
```

详细架构图：
- [五工序总架构图](vinegar_model/figures/14_five_stage_overview.png)
- [各工序详解](vinegar_model/figures/15_five_stage_detail.png)
- [数据流向图](vinegar_model/figures/16_five_stage_dataflow.png)

---

## 目录结构

```
.
├── vinegar_model/              # 核心模型包
│   ├── __init__.py
│   ├── process_model.py        # 五工序生产模型
│   ├── aaf_kinetics.py         # AAF发酵动力学 (王超2020)
│   ├── aging_kinetics.py       # 陈酿动力学 (任晓荣2023)
│   ├── flavor_radar.py          # 风味映射
│   ├── craft_effect.py          # 工艺修正系数
│   ├── hybrid_model.py          # 双层模型封装
│   ├── data_baseline.py        # 基线数据
│   └── figures/                # 流程图
├── web_product.py              # Flask Web产品接口
├── templates/                  # Jinja2模板
├── static/                     # CSS/JS静态资源
└── README.md
```

---

## 快速开始

### 环境要求

- Python 3.8+
- Flask
- NumPy, SciPy (可选)

### 安装

```bash
pip install flask numpy scipy
```

### 使用示例

#### 1. 完整五工序模拟

```python
from vinegar_model import VinegarProductionModel

model = VinegarProductionModel()

state = model.simulate_full_process(
    raw_material="糯米",
    saccharification_hours=60,
    alcohol_fermentation_days=6,
    aaf_days=18,
    water_ratio=1.0,
    aging_months=60,
)

print(f"糖化还原糖: {state.saccharification.reducing_sugar} g/100mL")
print(f"酒精发酵乙醇: {state.alcohol_fermentation.ethanol}%")
print(f"醋酸发酵总酸: {state.aaf.total_acid} g/100mL")
print(f"淋醋总酸: {state.leaching.total_acid} g/100mL")
```

#### 2. AAF发酵状态查询

```python
from vinegar_model import AAFModel

model = AAFModel()
state = model.get_state_at(8)
print(f"总酸: {state.total_acid} g/100mL")
print(f"阶段: {state.stage}")

val = model.validate()
print(f"R²(总酸): {val['r2_total_acid']:.4f}")

rec = model.recommend_turning(day=5, oxygen_lower=5.0, temperature=41.0)
print(f"建议翻醅: {rec['should_turn_today']}")
```

#### 3. 陈酿状态预测

```python
from vinegar_model import age_to_state, compute_flavor_profile

state = age_to_state(60)
profile = compute_flavor_profile(state)
print(f"TMP: {state.tmp} μg/mL")
print(f"乙酸乙酯: {state.ethyl_acetate} μg/mL")
```

#### 4. Flask API服务

```bash
python web_product.py
```

启动后访问：
- `http://127.0.0.1:5000/` - 首页
- `http://127.0.0.1:5000/aaf` - AAF发酵监测
- `http://127.0.0.1:5000/aging` - 陈酿预测

---

## API文档

### /api/process

完整五工序模拟

**参数：**
- `raw_material`: 原料类型，默认"糯米"
- `saccharification_hours`: 糖化时间(小时)，默认60
- `alcohol_fermentation_days`: 酒精发酵天数，默认6
- `aaf_days`: 醋酸发酵天数，默认18
- `water_ratio`: 淋醋加水比，默认1.0
- `aging_months`: 陈酿月数，默认60

### /api/aaf

AAF发酵状态查询

**参数：**
- `day` (float): 发酵天数，默认8

### /api/aging

陈酿状态预测

**参数：**
- `months` (float): 陈酿月数，默认60
- `process`: 发酵类型
- `raw_material`: 原料
- `craft_style`: 工艺风格

---

## 模型验证

### AAF发酵模型（王超2020）

| 指标 | R² | RMSE |
|------|-----|------|
| 总酸 | 0.9981 | 0.078 g/100mL |
| 乙酸 | 0.9975 | 0.081 g/100mL |

### 陈酿模型（任晓荣2023）

| 参数 | 曲线类型 | R² |
|------|----------|-----|
| TMP | Logistic | 0.99 |
| 乙酸乙酯 | Logistic | 0.98 |
| 不挥发酸 | Logistic | 0.997 |

---

## 文献依据

| 序号 | 文献 | 用途 |
|------|------|------|
| 1 | 王超等(2020) 镇江香醋醋酸发酵过程中理化指标的动态分析研究 | AAF发酵模型 |
| 2 | 任晓荣等(2023) 不同陈酿年份镇江香醋品质指标和功能成分的比较 | 陈酿模型 |
| 3 | 郑梦林等(2021) 镇江香醋陈酿过程中主要呈味物质的分析 | 陈酿数据验证 |
| 4 | 刘卓非等(2022) 食醋固态酿造过程氧含量监测及时序预测 | 翻醅阈值参考 |
| 5 | 李晓伟等(2022) 食醋固态发酵罐条件优化及发酵动力学分析 | 动力学参数参考 |
| 6 | GB/T 18623-2011 地理标志产品镇江香醋 | 国标参数范围 |

---

## License

本项目仅供学术研究使用。
