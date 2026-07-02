# 镇江香醋风味数字化模型

Vinegar Flavor Digitalization Model

## 项目概述

本项目基于约80篇镇江香醋/食醋发酵与风味分析文献，构建了一套醋风味数字化模型，用于：

- **AAF醋酸发酵过程监测**：基于王超(2020)实测数据的Logistic拟合模型，R²=0.998
- **陈酿时间预测**：基于任晓荣(2023)等实测数据的Logistic曲线拟合
- **风味映射**：8维理化指标 → 6维风味雷达 + 14维感官评分 + 综合评分
- **翻醅建议**：基于发酵动力学模型的智能化决策支持

---

## 系统架构

```
用户输入 → 数据封装 → 双层混合模型 → 输出
                              ↓
         ┌────────────────────┼────────────────────┐
         ↓                                         ↓
   经验模型层                                  机理模型层
   (陈酿预测)                                  (发酵监测)
         ↓                                         ↓
   Logistic曲线                               Logistic拟合
   + 工艺乘子修正                            + 翻醅建议
```

详细架构图见：
- [系统总架构图](vinegar_model/figures/11_overview.png)
- [经验模型详细](vinegar_model/figures/12_empirical_model.png)
- [发酵模型详细](vinegar_model/figures/13_mechanistic_model.png)

---

## 目录结构

```
.
├── vinegar_model/              # 核心模型包
│   ├── __init__.py
│   ├── aaf_kinetics.py         # AAF发酵动力学 (王超2020)
│   ├── aging_kinetics.py       # 陈酿动力学 (任晓荣2023)
│   ├── flavor_radar.py          # 风味映射 (雷达图/感官评分)
│   ├── craft_effect.py          # 工艺修正系数
│   ├── process_model.py         # AAF状态查询/翻醅建议
│   ├── hybrid_model.py          # 双层模型封装
│   ├── data_baseline.py        # 基线数据
│   ├── figures/                # 流程图与验证图
│   └── tests/                  # 单元测试
├── web_product.py              # Flask Web产品接口
├── templates/                  # Jinja2模板
├── static/                     # CSS/JS静态资源
├── pdf-ocr/                   # 文献PDF与OCR结果
└── README.md
```

---

## 快速开始

### 环境要求

- Python 3.8+
- Flask
- NumPy, SciPy, Matplotlib (可选)

### 安装

```bash
pip install flask numpy scipy matplotlib
```

### 使用示例

#### 1. AAF发酵状态查询

```python
from vinegar_model import AAFModel

model = AAFModel()

# 查询第8天发酵状态
state = model.get_state_at(8)
print(f"总酸: {state.total_acid} g/100mL")
print(f"乙酸: {state.acetic_acid} g/100mL")
print(f"阶段: {state.stage}")

# 模型验证
val = model.validate()
print(f"R²(总酸): {val['r2_total_acid']:.4f}")
print(f"R²(乙酸): {val['r2_acetic_acid']:.4f}")

# 翻醅建议
rec = model.recommend_turning(
    day=5,
    oxygen_lower=5.0,
    oxygen_upper=14.0,
    temperature=41.0
)
print(f"建议翻醅: {rec['should_turn_today']}")
print(f"原因: {rec['reasons']}")
```

输出：
```
总酸: 4.965 g/100mL
乙酸: 4.413 g/100mL
阶段: 高活性期
R²(总酸): 0.9981
R²(乙酸): 0.9975
建议翻醅: False
原因: ['状态良好，维持当前翻醅节奏']
```

#### 2. 陈酿状态预测

```python
from vinegar_model import age_to_state, compute_flavor_profile

# 预测60月陈酿状态
state = age_to_state(60)
print(f"TMP: {state.tmp} μg/mL")
print(f"乙酸乙酯: {state.ethyl_acetate} μg/mL")

# 风味雷达
profile = compute_flavor_profile(state)
print(f"酸味: {profile.sourness:.2f}")
print(f"鲜味: {profile.umami:.2f}")
```

#### 3. Flask API服务

```bash
python web_product.py
```

启动后访问：
- `http://127.0.0.1:5000/` - 首页
- `http://127.0.0.1:5000/aaf` - AAF发酵监测
- `http://127.0.0.1:5000/aging` - 陈酿预测

---

## API文档

### /api/aaf

AAF发酵状态查询

**参数：**
- `day` (float): 发酵天数，默认8

**返回：**
```json
{
  "day": 8,
  "state": {
    "total_acid_g100ml": 4.965,
    "acetic_acid_g100ml": 4.413,
    "oxygen_upper_pct": 15.4,
    "oxygen_lower_pct": 10.0,
    "stage": "高活性期"
  },
  "turning": {
    "should_turn_today": false,
    "reasons": ["状态良好，维持当前翻醅节奏"]
  },
  "simulation": { ... }
}
```

### /api/aging

陈酿状态预测

**参数：**
- `months` (float): 陈酿月数，默认60
- `process` (string): 发酵类型，默认"固态发酵"
- `raw_material` (string): 原料，默认"糯米"
- `craft_style` (string): 工艺风格，"传统"或"现代"，默认"传统"

### /api/trajectory

陈酿轨迹预测

返回0-120月的完整演化轨迹

### /api/flavor

风味评分计算

输入8维理化指标，返回6维风味雷达和14维感官评分

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
| 2 | 任晓荣等(2023) 不同陈酿年份镇江香醋品质指标和功能成分的比较 | 陈酿Logistic拟合 |
| 3 | 郑梦林等(2021) 镇江香醋陈酿过程中主要呈味物质的分析 | 陈酿数据验证 |
| 4 | 刘卓非(2022) 食醋固态酿造过程氧含量监测及时序预测 | 翻醅阈值参考 |
| 5 | 李晓伟等(2022) 食醋固态发酵罐条件优化及发酵动力学分析 | 动力学参数参考 |
| 6 | GB/T 18623-2011 地理标志产品镇江香醋 | 国标参数范围 |

---

## License

本项目仅供学术研究使用。
