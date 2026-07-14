"""
web_app.py - 恒顺香醋风味数字化监测系统
=======================================
Streamlit web application for vinegar flavor model demonstration.
Competition branding: 江苏恒顺醋业 × 挑战杯揭榜挂帅
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vinegar_model.flavor_radar import (
    VinegarState,
    compute_flavor_profile,
    compute_sensory_score,
    compute_ph_profile,
    compute_overall_score,
    radar_chart,
)
from vinegar_model.aging_kinetics import (
    predict_trajectory,
    predict_at_age,
    age_to_state,
)
from vinegar_model.aaf_kinetics import AAFKinetics
from vinegar_model.aging_mechanism import TMPReactionNetwork
from vinegar_model.aging_kinetics import MechanismAgingModel
from vinegar_model.hybrid_model import HybridVinegarModel

st.set_page_config(
    page_title="恒顺香醋风味数字化监测系统",
    page_icon="🍶",
    layout="wide",
    initial_sidebar_state="expanded"
)

COMPETITION_BRANDING = "江苏恒顺醋业 × 挑战杯揭榜挂帅"

PARAMETER_CONFIG = {
    "vinegar_age_months": {"label": "醋龄 (月)", "min": 0, "max": 120, "default": 60, "step": 1},
    "total_acid": {"label": "总酸 (g/100mL)", "min": 3.0, "max": 10.0, "default": 6.32, "step": 0.01},
    "non_volatile_acid": {"label": "不挥发酸 (g/100mL)", "min": 0.5, "max": 3.5, "default": 1.85, "step": 0.01},
    "reducing_sugar": {"label": "还原糖 (g/100mL)", "min": 0.5, "max": 5.0, "default": 0.93, "step": 0.01},
    "total_amino_acid": {"label": "总氨基酸 (g/100mL)", "min": 0.1, "max": 10.0, "default": 4.0, "step": 0.1},
    "ethyl_acetate": {"label": "乙酸乙酯 (μg/mL)", "min": 100, "max": 5000, "default": 1500, "step": 10},
    "tmp": {"label": "四甲基吡嗪 TMP (μg/mL)", "min": 5, "max": 200, "default": 44, "step": 1},
    "acetic_acid": {"label": "乙酸 (g/100mL)", "min": 0.5, "max": 8.0, "default": 2.31, "step": 0.01},
    "ph": {"label": "pH值", "min": 2.0, "max": 5.5, "default": 3.65, "step": 0.01},
}

PROCESS_OPTIONS = ["固态发酵", "液态发酵", "固液复合"]
MATERIAL_OPTIONS = ["糯米", "大米", "高粱", "麦芽", "果蔬"]
CRAFT_OPTIONS = ["传统", "现代"]

ANCHOR_MONTHS = [36, 60, 96]


def init_session_state():
    if "vinegar_state" not in st.session_state:
        st.session_state.vinegar_state = VinegarState()
    if "aaf_day" not in st.session_state:
        st.session_state.aaf_day = 9


def create_vinegar_state_from_inputs() -> VinegarState:
    return VinegarState(
        vinegar_age_months=st.session_state.get("vinegar_age_months", 60),
        total_acid=st.session_state.get("total_acid", 6.32),
        non_volatile_acid=st.session_state.get("non_volatile_acid", 1.85),
        reducing_sugar=st.session_state.get("reducing_sugar", 0.93),
        total_amino_acid=st.session_state.get("total_amino_acid", 4.0),
        ethyl_acetate=st.session_state.get("ethyl_acetate", 1500),
        tmp=st.session_state.get("tmp", 44),
        acetic_acid=st.session_state.get("acetic_acid", 2.31),
        ph=st.session_state.get("ph", 3.65),
        process=st.session_state.get("process", "固态发酵"),
        raw_material=st.session_state.get("raw_material", "糯米"),
        craft_style=st.session_state.get("craft_style", "传统"),
    )


@st.cache_data
def compute_all_metrics(state: VinegarState):
    profile = compute_flavor_profile(state)
    sensory = compute_sensory_score(state)
    ph_profile = compute_ph_profile(state)
    overall = compute_overall_score(state, profile, sensory)
    return profile, sensory, ph_profile, overall


def plot_radar_chart(profile):
    try:
        import plotly.graph_objects as go
        labels = ["酸感", "甜感", "鲜感", "醇厚", "花果香", "焦糖香"]
        values = profile.as_array()
        values_scaled = [v / 10.0 * 100 for v in values]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values_scaled + values_scaled[:1],
            theta=labels + [labels[0]],
            fill='toself',
            fillcolor='rgba(214,145,90,0.4)',
            line=dict(color='#a8623c', width=2),
            name='风味得分'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[20, 40, 60, 80, 100],
                    ticktext=['2', '4', '6', '8', '10']
                )
            ),
            showlegend=False,
            title=dict(text="6维风味雷达图", x=0.5, font_size=16),
            height=450
        )
        return fig
    except ImportError:
        import matplotlib.pyplot as plt
        fig = radar_chart(profile, "6维风味雷达图")
        return fig


def plot_aging_trajectories(process, raw_material, craft_style):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

    months = np.linspace(0, 120, 200).tolist()
    trajectory = predict_trajectory(months, process, raw_material, craft_style)

    params = ["total_acid", "ethyl_acetate", "tmp", "total_amino_acid",
              "non_volatile_acid", "reducing_sugar", "acetic_acid", "ph"]
    param_labels = ["总酸", "乙酸乙酯", "TMP", "氨基酸", "不挥发酸", "还原糖", "乙酸", "pH"]
    colors = ['#c0623f', '#7b9b3e', '#4886c1', '#9b6b9e', '#e8907a', '#d4a843', '#5c8a8a', '#c77eb8']

    try:
        fig = make_subplots(rows=4, cols=2, subplot_titles=param_labels)

        for i, (param, label) in enumerate(zip(params, param_labels)):
            row = i // 2 + 1
            col = i % 2 + 1
            values = [getattr(s, param) for s in trajectory]

            fig.add_trace(
                go.Scatter(x=months, y=values, mode='lines', name=label,
                          line=dict(color=colors[i], width=2)),
                row=row, col=col
            )

            for anchor in ANCHOR_MONTHS:
                state_at_anchor = age_to_state(anchor, process, raw_material, craft_style)
                anchor_val = getattr(state_at_anchor, param)
                fig.add_trace(
                    go.Scatter(x=[anchor], y=[anchor_val], mode='markers',
                              marker=dict(color=colors[i], size=12, symbol='star'),
                              showlegend=False),
                    row=row, col=col
                )

        fig.update_layout(height=800, showlegend=False, title_text="陈酿参数演化曲线 (★ 文献锚点)")
        return fig
    except ImportError:
        fig, axes = plt.subplots(4, 2, figsize=(14, 12))
        axes = axes.flatten()

        for i, (param, label) in enumerate(zip(params, param_labels)):
            values = [getattr(s, param) for s in trajectory]
            axes[i].plot(months, values, color=colors[i], linewidth=2)
            for anchor in ANCHOR_MONTHS:
                state_at_anchor = age_to_state(anchor, process, raw_material, craft_style)
                axes[i].scatter([anchor], [getattr(state_at_anchor, param)],
                               color=colors[i], s=100, marker='*', zorder=5)
            axes[i].set_title(label)
            axes[i].set_xlabel("醋龄 (月)")
            axes[i].grid(True, alpha=0.3)

        fig.suptitle("陈酿参数演化曲线 (★ 文献锚点)", fontsize=14)
        fig.tight_layout()
        return fig


def plot_aaf_kinetics():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

    kinetics = AAFKinetics()
    sim = kinetics.simulate(day_0_to_18=18, n_points=200)
    time = sim["time"]

    try:
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=["总酸/乙酸演化", "菌体生长曲线", "溶氧变化", "温度曲线"])

        fig.add_trace(go.Scatter(x=time, y=sim["total_acid"], name="总酸", line=dict(color='#c0623f')),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=sim["acetic_acid"], name="乙酸", line=dict(color='#7b9b3e')),
                     row=1, col=1)

        fig.add_trace(go.Scatter(x=time, y=sim["ab_growth"], name="醋酸菌", line=dict(color='#c0623f')),
                     row=1, col=2)
        fig.add_trace(go.Scatter(x=time, y=sim["lb_growth"], name="乳酸菌", line=dict(color='#7b9b3e')),
                     row=1, col=2)

        fig.add_trace(go.Scatter(x=time, y=sim["oxygen_upper"], name="上层O₂", line=dict(color='#4886c1')),
                     row=2, col=1)
        fig.add_trace(go.Scatter(x=time, y=sim["oxygen_lower"], name="下层O₂", line=dict(color='#e8907a')),
                     row=2, col=1)

        fig.add_trace(go.Scatter(x=time, y=sim["temperature_upper"], name="温度", line=dict(color='#d4a843')),
                     row=2, col=2)

        fig.update_layout(height=500, showlegend=True, title_text="Luedeking-Piret 动力学模拟")
        return fig
    except ImportError:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(time, sim["total_acid"], '#c0623f', label="总酸")
        axes[0, 0].plot(time, sim["acetic_acid"], '#7b9b3e', label="乙酸")
        axes[0, 0].set_title("总酸/乙酸演化")
        axes[0, 0].legend()

        axes[0, 1].plot(time, sim["ab_growth"], '#c0623f', label="醋酸菌")
        axes[0, 1].plot(time, sim["lb_growth"], '#7b9b3e', label="乳酸菌")
        axes[0, 1].set_title("菌体生长曲线")
        axes[0, 1].legend()

        axes[1, 0].plot(time, sim["oxygen_upper"], '#4886c1', label="上层O₂")
        axes[1, 0].plot(time, sim["oxygen_lower"], '#e8907a', label="下层O₂")
        axes[1, 0].set_title("溶氧变化")
        axes[1, 0].legend()

        axes[1, 1].plot(time, sim["temperature_upper"], '#d4a843', label="温度")
        axes[1, 1].set_title("温度曲线")
        axes[1, 1].legend()

        fig.suptitle("Luedeking-Piret 动力学模拟")
        fig.tight_layout()
        return fig


def plot_mechanism_comparison():
    try:
        import plotly.graph_objects as go
    except ImportError:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

    t_data = np.array([36.0, 60.0, 96.0])
    tmp_data = np.array([38.0, 50.0, 95.0])

    t_smooth = np.linspace(0, 120, 300)
    logistic_pred = 8.0 + (256.0 - 8.0) / (1.0 + np.exp(-0.0245 * (t_smooth - 121.78)))

    model = TMPReactionNetwork()
    mechanism_pred = model.predict_tmp(t_smooth, 80.0, 100.0, T=25, initial_precursor=150.0)

    try:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=t_smooth, y=logistic_pred, mode='lines',
                                 name='Logistic模型', line=dict(color='blue', dash='dash', width=2)))

        fig.add_trace(go.Scatter(x=t_smooth, y=mechanism_pred, mode='lines',
                                 name='2步反应机制模型', line=dict(color='green', width=2)))

        fig.add_trace(go.Scatter(x=t_data, y=tmp_data, mode='markers',
                                 name='文献实测 (任晓荣2023)',
                                 marker=dict(color='red', size=14, symbol='star')))

        fig.update_layout(
            title="TMP形成: Logistic vs 2步反应机制",
            xaxis_title="醋龄 (月)",
            yaxis_title="TMP (μg/mL)",
            height=400,
            legend=dict(x=0.6, y=0.95)
        )
        return fig
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_smooth, logistic_pred, 'b--', linewidth=2, label='Logistic模型')
        ax.plot(t_smooth, mechanism_pred, 'g-', linewidth=2, label='2步反应机制模型')
        ax.scatter(t_data, tmp_data, color='red', s=150, marker='*', zorder=5,
                  label='文献实测 (任晓荣2023)')
        ax.set_xlabel("醋龄 (月)")
        ax.set_ylabel("TMP (μg/mL)")
        ax.set_title("TMP形成: Logistic vs 2步反应机制")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig


def page_home():
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #8B0000, #CD5C5C); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; text-align: center; margin: 0;">恒顺香醋风味数字化监测系统</h1>
        <p style="color: #FFE4E1; text-align: center; margin: 10px 0 0 0; font-size: 18px;">🍶 {COMPETITION_BRANDING}</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 输入参量", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**核心参量**")
            for key in ["vinegar_age_months", "total_acid", "non_volatile_acid", "reducing_sugar"]:
                cfg = PARAMETER_CONFIG[key]
                st.session_state[key] = st.slider(
                    cfg["label"], cfg["min"], cfg["max"], cfg["default"], cfg["step"],
                    key=f"home_{key}"
                )

        with col2:
            st.markdown("**风味相关**")
            for key in ["total_amino_acid", "ethyl_acetate", "tmp", "acetic_acid"]:
                cfg = PARAMETER_CONFIG[key]
                st.session_state[key] = st.slider(
                    cfg["label"], cfg["min"], cfg["max"], cfg["default"], cfg["step"],
                    key=f"home_{key}"
                )

        with col3:
            st.markdown("**酸度指标**")
            st.session_state["ph"] = st.slider(
                PARAMETER_CONFIG["ph"]["label"],
                PARAMETER_CONFIG["ph"]["min"], PARAMETER_CONFIG["ph"]["max"],
                PARAMETER_CONFIG["ph"]["default"], PARAMETER_CONFIG["ph"]["step"],
                key="home_ph"
            )

            st.markdown("**工艺选择**")
            st.session_state["process"] = st.selectbox(
                "发酵工艺", PROCESS_OPTIONS, index=PROCESS_OPTIONS.index(st.session_state.get("process", "固态发酵")),
                key="home_process"
            )
            st.session_state["raw_material"] = st.selectbox(
                "原料", MATERIAL_OPTIONS, index=MATERIAL_OPTIONS.index(st.session_state.get("raw_material", "糯米")),
                key="home_material"
            )
            st.session_state["craft_style"] = st.selectbox(
                "酿造风格", CRAFT_OPTIONS, index=CRAFT_OPTIONS.index(st.session_state.get("craft_style", "传统")),
                key="home_craft"
            )

    state = create_vinegar_state_from_inputs()
    profile, sensory, ph_profile, overall = compute_all_metrics(state)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 6维风味雷达")
        try:
            fig = plot_radar_chart(profile)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"图表渲染失败: {e}")

    with col2:
        st.markdown("### 综合评分")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 40px; border-radius: 20px; text-align: center; margin: 10px 0;">
            <h2 style="color: white; font-size: 72px; margin: 0;">{overall:.1f}</h2>
            <p style="color: #E0E0E0; font-size: 18px;">综合评分 (0-100)</p>
        </div>
        """, unsafe_allow_html=True)

        if ph_profile.softness is not None:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 15px;">
                <h4>pH维度评分</h4>
                <p>柔和度: {ph_profile.softness:.3f} | 刺激感: {ph_profile.pungency:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 14维感官评分")
    sensory_dict = sensory.as_dict()

    try:
        import plotly.express as px
        df = pd.DataFrame({
            "指标": list(sensory_dict.keys()),
            "得分": list(sensory_dict.values())
        })
        df["满分"] = 25.0
        df["得分占比"] = df["得分"] / 25.0 * 100

        fig = px.bar(df, x="指标", y="得分", color="得分占比",
                    color_continuous_scale="RdYlGn", range_color=[0, 100],
                    title="14维感官评分 (满分25分)")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        names = list(sensory_dict.keys())
        values = list(sensory_dict.values())
        colors = plt.cm.RdYlGn([v/25 for v in values])
        bars = ax.bar(names, values, color=colors)
        ax.set_ylim(0, 25)
        ax.set_title("14维感官评分 (满分25分)")
        plt.xticks(rotation=45, ha='right')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        st.pyplot(fig)

    st.markdown("### 当前参量状态")
    state_dict = {
        "醋龄": f"{state.vinegar_age_months:.0f}月",
        "总酸": f"{state.total_acid:.2f} g/100mL",
        "不挥发酸": f"{state.non_volatile_acid:.2f} g/100mL",
        "还原糖": f"{state.reducing_sugar:.2f} g/100mL",
        "氨基酸": f"{state.total_amino_acid:.2f} g/100mL",
        "乙酸乙酯": f"{state.ethyl_acetate:.0f} μg/mL",
        "TMP": f"{state.tmp:.1f} μg/mL",
        "乙酸": f"{state.acetic_acid:.2f} g/100mL",
        "pH": f"{state.ph:.2f}",
        "工艺": state.process,
        "原料": state.raw_material,
        "风格": state.craft_style,
    }
    st.dataframe(pd.DataFrame(state_dict.items(), columns=["参数", "数值"]), hide_index=True)


def page_aging_prediction():
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #2E7D32, #4CAF50); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; text-align: center; margin: 0;">陈酿时间序列预测</h1>
        <p style="color: #E8F5E9; text-align: center; margin: 10px 0 0 0; font-size: 18px;">🍶 {COMPETITION_BRANDING}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 当前状态")
        state = create_vinegar_state_from_inputs()
        current_dict = {
            "醋龄": f"{state.vinegar_age_months:.0f}月",
            "总酸": f"{state.total_acid:.2f}",
            "乙酸乙酯": f"{state.ethyl_acetate:.0f}",
            "TMP": f"{state.tmp:.1f}",
            "氨基酸": f"{state.total_amino_acid:.2f}",
            "pH": f"{state.ph:.2f}",
        }
        st.dataframe(pd.DataFrame(current_dict.items(), columns=["参数", "当前值"]), hide_index=True)

        st.markdown("### 预测时间点")
        target_months = st.slider("选择预测醋龄", 0, 120, 60, 5)

        predicted_state = predict_at_age(state, target_months)
        st.markdown(f"**{target_months}月后预测状态**")
        pred_dict = {
            "醋龄": f"{predicted_state.vinegar_age_months:.0f}月",
            "总酸": f"{predicted_state.total_acid:.2f}",
            "乙酸乙酯": f"{predicted_state.ethyl_acetate:.0f}",
            "TMP": f"{predicted_state.tmp:.1f}",
            "氨基酸": f"{predicted_state.total_amino_acid:.2f}",
            "pH": f"{predicted_state.ph:.2f}",
        }
        st.dataframe(pd.DataFrame(pred_dict.items(), columns=["参数", "预测值"]), hide_index=True)

        profile_pred = compute_flavor_profile(predicted_state)
        overall_pred = compute_overall_score(predicted_state, profile_pred)
        st.markdown(f"**预测综合评分: {overall_pred:.1f}**")

    with col2:
        st.markdown("### 8参数演化曲线")
        fig = plot_aging_trajectories(state.process, state.raw_material, state.craft_style)
        try:
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.pyplot(fig)

    st.markdown("### 模型对比: Logistic vs 机制模型")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Logistic经验模型")
        st.markdown("""
        - 基于任晓荣(2023)实测数据拟合
        - R²=0.9904 (TMP)
        - 适用: 快速预测、实时监测
        """)
        st.markdown("""
        ```
        TMP = 8 + 248/(1 + exp(-0.0245*(t-121.78)))
        ```
        """)

    with col2:
        st.markdown("#### 2步反应机制模型")
        st.markdown("""
        - 前体 → 乙偶姻 → TMP
        - Step 1: Maillard反应 (Ea≈75 kJ/mol)
        - Step 2: Strecker降解 (Ea≈65 kJ/mol)
        - 适用: 机理解释、参数优化
        """)
        st.markdown("""
        ```
        d[TMP]/dt = k₂ × [Acetoin] × [NH₃]
        ```
        """)

    fig_mech = plot_mechanism_comparison()
    try:
        st.plotly_chart(fig_mech, use_container_width=True)
    except Exception:
        st.pyplot(fig_mech)


def page_aaf_kinetics():
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #FF8F00, #FFB300); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; text-align: center; margin: 0;">醋酸发酵阶段 (AAF) 动力学</h1>
        <p style="color: #FFF8E1; text-align: center; margin: 10px 0 0 0; font-size: 18px;">🍶 {COMPETITION_BRANDING}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 发酵进度")
        day = st.slider("选择发酵天数", 0, 18, st.session_state.aaf_day, 1)
        st.session_state.aaf_day = day

        kinetics = AAFKinetics()
        state_at_day = kinetics.get_state_at(day)

        st.markdown(f"**Day {day} - {state_at_day['stage']}**")
        state_display = {
            "总酸 (g/100mL)": f"{state_at_day['total_acid']:.3f}",
            "乙酸 (g/100mL)": f"{state_at_day['acetic_acid']:.3f}",
            "不挥发酸 (g/100mL)": f"{state_at_day['non_volatile_acid']:.3f}",
            "乳酸 (g/100mL)": f"{state_at_day['lactic_acid']:.3f}",
            "残余乙醇 (%)": f"{state_at_day['ethanol_residual']:.2f}",
            "上层O₂ (%)": f"{state_at_day['oxygen_upper']:.1f}",
            "下层O₂ (%)": f"{state_at_day['oxygen_lower']:.1f}",
            "上层温度 (°C)": f"{state_at_day['temperature_upper']:.1f}",
            "醋酸菌活性": f"{state_at_day['ab_growth']:.3f}",
            "乳酸菌活性": f"{state_at_day['lb_growth']:.3f}",
        }
        st.dataframe(pd.DataFrame(state_display.items(), columns=["参数", "值"]), hide_index=True)

        st.markdown("### 翻醅建议")
        oxygen_upper = st.number_input("上层O₂ (%)", 0.0, 20.0, 14.0, 0.5, key="aaf_o2_upper")
        oxygen_lower = st.number_input("下层O₂ (%)", 0.0, 20.0, 5.0, 0.5, key="aaf_o2_lower")
        temperature = st.number_input("上层温度 (°C)", 30.0, 50.0, 38.0, 0.5, key="aaf_temp")

        recommendation = kinetics.get_turning_recommendation(day, oxygen_lower, oxygen_upper, temperature)

        if recommendation["should_turn_today"]:
            st.error("⚠️ 建议立即翻醅")
        else:
            st.success("✓ 状态良好，维持当前节奏")

        for reason in recommendation["reasons"]:
            st.info(reason)

    with col2:
        st.markdown("### Luedeking-Piret 动力学模拟")
        fig = plot_aaf_kinetics()
        try:
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.pyplot(fig)

    st.markdown("### 模型对比")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 新动力学模型 (Luedeking-Piret)")
        st.markdown("""
        - 基于李晓伟(2022)转鼓式发酵罐数据
        - 两相模型: 生长期 + 衰亡期
        - α, β参数从CFU数据回归
        """)
        validation = kinetics.validate_against_wangchao2020()
        st.markdown(f"""
        | 指标 | R² | RMSE |
        |------|-----|-------|
        | 总酸 | {validation['r2_total_acid']:.4f} | {validation['rmse_total_acid']:.4f} |
        | 乙酸 | {validation['r2_acetic_acid']:.4f} | {validation['rmse_acetic_acid']:.4f} |
        """)

    with col2:
        st.markdown("#### 传统经验方法")
        st.markdown("""
        - 固定周期翻醅 (24h/次)
        - 经验判断翻醅时机
        - 依赖工人经验，无法优化
        """)
        st.markdown("""
        **问题**:
        - 发酵周期波动大
        - 能源浪费
        - 品质一致性差
        """)


def page_model_info():
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #1565C0, #42A5F5); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; text-align: center; margin: 0;">双层混合模型架构</h1>
        <p style="color: #E3F2FD; text-align: center; margin: 10px 0 0 0; font-size: 18px;">🍶 {COMPETITION_BRANDING}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 模型架构

    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                    双层混合模型 (Hybrid Model)               │
    ├─────────────────────────────────────────────────────────────┤
    │  Layer 1: 经验模型层 (Empirical - Logistic)                 │
    │  ├── 陈酿动力学: 8参数时间演化曲线                           │
    │  ├── 风味映射: 9输入 → 6维雷达 + 14维感官评分               │
    │  └── 适用场景: 实时监测、快速预测                            │
    ├─────────────────────────────────────────────────────────────┤
    │  Layer 2: 机制模型层 (Mechanistic - First Principles)       │
    │  ├── AAF发酵: Luedeking-Piret动力学                         │
    │  │   └── 醋酸菌/乳酸菌生长 + 产物生成                        │
    │  └── TMP陈酿: 2步反应网络 (前体→乙偶姻→TMP)                 │
    │      └── Maillard + Strecker降解                            │
    └─────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("### 动力学参数提取")

    hybrid_model = HybridVinegarModel()
    params = hybrid_model.get_kinetic_parameters()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### AAF发酵参数 (李晓伟2022)")
        aaf = params["aaf_fermentation"]
        st.markdown(f"""
        - **模型类型**: {aaf['model_type']}
        - **文献来源**: {aaf['literature_source']}
        - **醋酸菌μ_max**: {aaf['acetic_acid_bacteria']['mu_max_d_minus_1']} d⁻¹
        - **乳酸菌μ_max**: {aaf['lactic_acid_bacteria']['mu_max_growth_d_minus_1']} d⁻¹
        - **发酵周期**: {aaf['fermentation_duration_days']}天
        """)

    with col2:
        st.markdown("#### TMP陈酿参数 (任晓荣2023)")
        tmp = params["tmp_aging"]
        st.markdown(f"""
        - **模型类型**: {tmp['model_type']}
        - **文献来源**: {tmp['literature_source']}
        - **Step 1**: {tmp['step1']['pathway']} (Ea={tmp['step1']['Ea_kJ_mol']} kJ/mol)
        - **Step 2**: {tmp['step2']['pathway']} (Ea={tmp['step2']['Ea_kJ_mol']} kJ/mol)
        """)

    st.markdown("### R²验证结果")

    kinetics = AAFKinetics()
    val = kinetics.validate_against_wangchao2020()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总酸 R²", f"{val['r2_total_acid']:.4f}")
    with col2:
        st.metric("乙酸 R²", f"{val['r2_acetic_acid']:.4f}")
    with col3:
        st.metric("文献来源", "王超 (2020)")

    st.markdown("""
    ### 参考文献

    | 序号 | 文献 | 年份 | 贡献 |
    |------|------|------|------|
    | 1 | 任晓荣 等 | 2023 | 不同陈酿年份镇江香醋品质指标和功能成分的比较 |
    | 2 | 郑梦林 等 | 2021 | 镇江香醋陈酿过程中主要呈味物质的分析 |
    | 3 | 王超 等 | 2020 | 镇江香醋醋酸发酵过程中理化指标的动态分析研究 |
    | 4 | 李晓伟 | 2022 | 食醋固态发酵罐条件优化及发酵动力学分析 |
    | 5 | 沈广玥 | 2023 | 食醋风味及其与发酵工艺的相关性分析 |
    | 6 | 孙宗保 等 | 2020 | 基于SPME-MS技术识别不同生产工艺和醋龄的镇江香醋 |
    | 7 | 刘卓非 | 2022 | 食醋固态酿造过程氧含量监测及时序预测 |
    | 8 | 简东振 等 | 2020 | 镇江香醋陈酿香气变化及其影响因素研究 |
    | 9 | GB/T 18623-2011 | - | 地理标志产品镇江香醋 (国家标准) |

    ### 技术特点

    1. **双层架构**: 结合经验模型的实用性 + 机制模型的可解释性
    2. **多尺度建模**: 从发酵天数(0-18天)到陈酿月份(0-120月)
    3. **文献驱动**: 基于约80篇镇江香醋相关文献构建
    4. **实时预测**: 9个输入参数 → 6维风味 + 14维感官 + 综合评分
    """)


def main():
    init_session_state()

    pages = {
        "风味实时监测 (首页)": page_home,
        "陈酿预测": page_aging_prediction,
        "AAF发酵阶段": page_aaf_kinetics,
        "模型技术说明": page_model_info,
    }

    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(180deg, #8B0000, #CD5C5C); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
        <h3 style="color: white; text-align: center; margin: 0;">🍶 恒顺醋业</h3>
        <p style="color: #FFE4E1; text-align: center; font-size: 12px;">挑战杯揭榜挂帅</p>
    </div>
    """, unsafe_allow_html=True)

    selection = st.sidebar.radio("导航", list(pages.keys()))

    pages[selection]()


if __name__ == "__main__":
    main()