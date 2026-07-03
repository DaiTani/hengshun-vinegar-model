from flask import Flask, render_template, jsonify, request
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vinegar_model.flavor_radar import (
    VinegarState, compute_flavor_profile,
    compute_sensory_score, compute_overall_score
)
from vinegar_model.aging_kinetics import age_to_state, predict_trajectory
from vinegar_model.process_model import (
    recommend_turning, VinegarProductionModel,
    MaterialBasedModel, ProductionInput, calculate_from_raw_material,
    AGING_VESSEL_FACTORS, AGING_TEMPERATURE_FACTORS
)
from vinegar_model.aaf_kinetics import AAFModel

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process')
def process_view():
    return render_template('process.html')


@app.route('/aging')
def aging():
    return render_template('aging.html')


@app.route('/aaf')
def aaf():
    return render_template('aaf.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/api/flavor', methods=['GET'])
def api_flavor():
    params = {
        'vinegar_age_months': float(request.args.get('vinegar_age_months', 60)),
        'total_acid': float(request.args.get('total_acid', 6.32)),
        'non_volatile_acid': float(request.args.get('non_volatile_acid', 1.85)),
        'reducing_sugar': float(request.args.get('reducing_sugar', 0.93)),
        'total_amino_acid': float(request.args.get('total_amino_acid', 4.0)),
        'ethyl_acetate': float(request.args.get('ethyl_acetate', 1500)),
        'tmp': float(request.args.get('tmp', 44)),
        'acetic_acid': float(request.args.get('acetic_acid', 2.31)),
        'ph': float(request.args.get('ph', 3.65)),
        'process': request.args.get('process', '固态发酵'),
        'raw_material': request.args.get('raw_material', '糯米'),
        'craft_style': request.args.get('craft_style', '传统'),
    }
    
    state = VinegarState(**params)
    profile = compute_flavor_profile(state)
    sensory = compute_sensory_score(state)
    overall = compute_overall_score(state, profile, sensory)
    
    return jsonify({
        'flavor': profile.as_dict(),
        'sensory': sensory.as_dict(),
        'overall': overall,
        'state': {
            'vinegar_age_months': state.vinegar_age_months,
            'total_acid': state.total_acid,
            'non_volatile_acid': state.non_volatile_acid,
            'reducing_sugar': state.reducing_sugar,
            'total_amino_acid': state.total_amino_acid,
            'ethyl_acetate': state.ethyl_acetate,
            'tmp': state.tmp,
            'acetic_acid': state.acetic_acid,
            'ph': state.ph,
        }
    })


@app.route('/api/aging', methods=['GET'])
def api_aging():
    months = float(request.args.get('months', 60))
    process = request.args.get('process', '固态发酵')
    raw_material = request.args.get('raw_material', '糯米')
    craft_style = request.args.get('craft_style', '传统')
    aging_vessel = request.args.get('aging_vessel', '陶缸')
    aging_temperature = request.args.get('aging_temperature', '常温')

    # 陈酿容器和温度效应系数
    vessel_factors = AGING_VESSEL_FACTORS.get(aging_vessel, AGING_VESSEL_FACTORS['陶缸'])
    temp_factors = AGING_TEMPERATURE_FACTORS.get(aging_temperature, AGING_TEMPERATURE_FACTORS['常温'])
    ethyl_factor = vessel_factors["ethyl_acetate_rate"] * temp_factors["ethyl_acetate_rate"]
    tmp_factor = vessel_factors["tmp_rate"] * temp_factors["tmp_rate"]

    state = age_to_state(months, process, raw_material, craft_style)

    # 应用容器和温度效应到风味物质
    state.ethyl_acetate *= ethyl_factor
    state.tmp *= tmp_factor

    profile = compute_flavor_profile(state)
    sensory = compute_sensory_score(state)
    overall = compute_overall_score(state, profile, sensory)

    return jsonify({
        'months': months,
        'state': {
            'vinegar_age_months': state.vinegar_age_months,
            'total_acid': round(state.total_acid, 3),
            'non_volatile_acid': round(state.non_volatile_acid, 3),
            'reducing_sugar': round(state.reducing_sugar, 3),
            'total_amino_acid': round(state.total_amino_acid, 3),
            'ethyl_acetate': round(state.ethyl_acetate, 1),
            'tmp': round(state.tmp, 2),
            'acetic_acid': round(state.acetic_acid, 3),
            'ph': round(state.ph, 3),
        },
        'flavor': profile.as_dict(),
        'overall': overall
    })


@app.route('/api/aaf', methods=['GET'])
def api_aaf():
    day = float(request.args.get('day', 8))

    model = AAFModel()
    aaf_state = model.get_state_at(day)
    turning = model.recommend_turning(
        day,
        oxygen_upper=aaf_state.oxygen_upper,
        oxygen_lower=aaf_state.oxygen_lower,
        temperature=aaf_state.temperature_upper
    )

    sim_data = model.simulate(n_points=200)

    return jsonify({
        'day': day,
        'state': {
            'day': aaf_state.day,
            'stage': aaf_state.stage,
            'total_acid_g100ml': aaf_state.total_acid,
            'acetic_acid_g100ml': aaf_state.acetic_acid,
            'non_volatile_acid_g100ml': aaf_state.non_volatile_acid,
            'lactic_acid_g100ml': aaf_state.lactic_acid,
            'ethanol_residual_pct': aaf_state.ethanol_residual,
            'oxygen_upper_pct': aaf_state.oxygen_upper,
            'oxygen_lower_pct': aaf_state.oxygen_lower,
            'temperature_upper_c': aaf_state.temperature_upper,
            'ab_growth': aaf_state.ab_growth,
            'lb_growth': aaf_state.lb_growth,
        },
        'turning': turning,
        'simulation': {
            'time': sim_data['time'],
            'total_acid': sim_data['total_acid'],
            'acetic_acid': sim_data['acetic_acid'],
            'oxygen_upper': sim_data['oxygen_upper'],
            'oxygen_lower': sim_data['oxygen_lower'],
            'ab_growth': sim_data['ab_growth'],
        }
    })


@app.route('/api/trajectory', methods=['GET'])
def api_trajectory():
    process = request.args.get('process', '固态发酵')
    raw_material = request.args.get('raw_material', '糯米')
    craft_style = request.args.get('craft_style', '传统')
    aging_vessel = request.args.get('aging_vessel', '陶缸')
    aging_temperature = request.args.get('aging_temperature', '常温')

    # 陈酿容器和温度效应系数
    vessel_factors = AGING_VESSEL_FACTORS.get(aging_vessel, AGING_VESSEL_FACTORS['陶缸'])
    temp_factors = AGING_TEMPERATURE_FACTORS.get(aging_temperature, AGING_TEMPERATURE_FACTORS['常温'])

    months_list = [0, 6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
    trajectory = predict_trajectory(months_list, process, raw_material, craft_style)

    # 应用容器和温度效应
    ethyl_factor = vessel_factors["ethyl_acetate_rate"] * temp_factors["ethyl_acetate_rate"]
    tmp_factor = vessel_factors["tmp_rate"] * temp_factors["tmp_rate"]
    
    result = {
        'months': months_list,
        'parameters': {
            'total_acid': [],
            'non_volatile_acid': [],
            'reducing_sugar': [],
            'total_amino_acid': [],
            'ethyl_acetate': [],
            'tmp': [],
            'acetic_acid': [],
            'ph': [],
        },
        'overall': []
    }
    
    for state in trajectory:
        result['parameters']['total_acid'].append(round(state.total_acid, 3))
        result['parameters']['non_volatile_acid'].append(round(state.non_volatile_acid, 3))
        result['parameters']['reducing_sugar'].append(round(state.reducing_sugar, 3))
        result['parameters']['total_amino_acid'].append(round(state.total_amino_acid, 3))
        result['parameters']['ethyl_acetate'].append(round(state.ethyl_acetate * ethyl_factor, 1))
        result['parameters']['tmp'].append(round(state.tmp * tmp_factor, 2))
        result['parameters']['acetic_acid'].append(round(state.acetic_acid, 3))
        result['parameters']['ph'].append(round(state.ph, 3))
        profile = compute_flavor_profile(state)
        result['overall'].append(round(compute_overall_score(state, profile), 2))

    return jsonify(result)


@app.route('/api/process', methods=['GET'])
def api_process():
    model = VinegarProductionModel()

    state = model.simulate_full_process(
        raw_material=request.args.get('raw_material', '糯米'),
        saccharification_hours=float(request.args.get('saccharification_hours', 60)),
        saccharification_temp=float(request.args.get('saccharification_temp', 62)),
        alcohol_fermentation_days=float(request.args.get('alcohol_days', 6)),
        alcohol_temp=float(request.args.get('alcohol_temp', 30)),
        aaf_days=float(request.args.get('aaf_days', 18)),
        water_ratio=float(request.args.get('water_ratio', 1.0)),
        leaching_time=float(request.args.get('leaching_time', 16)),
        aging_months=float(request.args.get('aging_months', 60)),
    )

    summary = model.get_stage_summary(state)

    return jsonify({
        'summary': summary,
        'detail': {
            'saccharification': state.saccharification.as_dict(),
            'alcohol_fermentation': state.alcohol_fermentation.as_dict(),
            'aaf': state.aaf.as_dict(),
            'leaching': state.leaching.as_dict(),
            'vinegar_age_months': state.vinegar_age_months,
        }
    })


@app.route('/api/material', methods=['GET'])
def api_material():
    """
    基于原料量的镇江香醋生产计算

    支持原料类型:
    - 淀粉基: 糯米(70%), 大米(65%), 高粱(60%), 小麦(55%), 玉米(60%)
    - 糖基: 苹果(12%), 葡萄(17%)

    支持生产规模:
    - lab: 实验室规模 (<10kg原料)
    - pilot: 中试规模 (10-500kg)
    - production: 生产规模 (>500kg)

    支持陈酿容器:
    - 陶缸: 传统透气容器, 风味物质形成快
    - 不锈钢: 密闭容器, 风味变化慢但稳定

    支持陈酿温度:
    - 低温: 10-15°C, 地窖陈酿, 风味缓慢形成
    - 常温: 20-25°C, 室内陈酿, 风味自然演化
    - 高温: 30-35°C, 加速陈酿, 风味形成快但可能不够细腻

    支持AAF翻醅次数: 0-3次

    参数:
        raw_material_kg: 原料量 (kg), 默认100
        raw_material_type: 原料类型, 默认"糯米"
        scale_type: 生产规模, 默认"pilot"
        aging_vessel: 陈酿容器, 默认"陶缸"
        aging_temperature: 陈酿温度, 默认"常温"
        aaf_turnover_times: AAF翻醅次数, 默认2
        water_ratio: 加水比 (米:水), 默认3.0
        alcohol_days: 酒精发酵天数, 默认6
        aaf_days: AAF发酵天数, 默认18
        leaching_water_ratio: 淋醋加水比, 默认1.5
        aging_months: 陈酿月数, 默认60

    返回:
        各工序产出和最终成品指标
    """
    raw_material_kg = float(request.args.get('raw_material_kg', 100))
    raw_material_type = request.args.get('raw_material_type', '糯米')
    scale_type = request.args.get('scale_type', 'pilot')
    aging_vessel = request.args.get('aging_vessel', '陶缸')
    aging_temperature = request.args.get('aging_temperature', '常温')
    aaf_turnover_times = int(request.args.get('aaf_turnover_times', 2))
    water_ratio = float(request.args.get('water_ratio', 3.0))
    alcohol_days = float(request.args.get('alcohol_days', 6))
    aaf_days = float(request.args.get('aaf_days', 18))
    leaching_water_ratio = float(request.args.get('leaching_water_ratio', 1.5))
    aging_months = float(request.args.get('aging_months', 60))

    result = calculate_from_raw_material(
        raw_material_kg=raw_material_kg,
        raw_material_type=raw_material_type,
        scale_type=scale_type,
        aging_vessel=aging_vessel,
        aging_temperature=aging_temperature,
        aaf_turnover_times=aaf_turnover_times,
        water_ratio=water_ratio,
        alcohol_days=alcohol_days,
        aaf_days=aaf_days,
        leaching_water_ratio=leaching_water_ratio,
        aging_months=aging_months,
    )

    return jsonify({
        'input': {
            'raw_material_kg': result.input.raw_material_kg,
            'raw_material_type': result.input.raw_material_type,
            'water_ratio': result.input.water_ratio,
            'alcohol_days': result.input.alcohol_days,
            'aaf_days': result.input.aaf_days,
            'leaching_water_ratio': result.input.leaching_water_ratio,
            'aging_months': result.input.aging_months,
        },
        'stages': {
            'saccharification': {
                'stage': result.saccharification.stage_name,
                'input_kg': result.saccharification.input_kg,
                'volume_L': result.saccharification.volume_L,
                'starch_kg': result.saccharification.starch_kg,
                'glucose_kg': result.saccharification.glucose_kg,
                'glucose_conc_gL': result.saccharification.glucose_conc_gL,
                'conversion_rate': result.saccharification.conversion_rate,
            },
            'alcohol': {
                'stage': result.alcohol.stage_name,
                'ethanol_kg': result.alcohol.ethanol_kg,
                'ethanol_conc_pct': result.alcohol.ethanol_conc_pct,
            },
            'aaf': {
                'stage': result.aaf.stage_name,
                'acetic_acid_kg': result.aaf.acetic_acid_kg,
                'total_acid_kg': result.aaf.total_acid_kg,
                'total_acid_conc_gL': result.aaf.total_acid_conc_gL,
            },
            'leaching': {
                'stage': result.leaching.stage_name,
                'final_vinegar_L': result.final_vinegar_L,
                'total_acid_conc_gL': result.leaching.total_acid_conc_gL,
                'extraction_efficiency': result.leaching.extraction_efficiency,
            },
            'aging': {
                'months': result.aging_months,
                'ethyl_acetate_ugmL': result.ethyl_acetate_ugmL,
                'tmp_ugmL': result.tmp_ugmL,
                'overall_score': result.overall_score,
            }
        },
        'final': {
            'vinegar_L': round(result.final_vinegar_L, 1),
            'total_acid_gL': round(result.final_total_acid_gL, 1),
            'ethyl_acetate_ugmL': round(result.final_ethyl_acetate_ugmL, 1),
            'tmp_ugmL': round(result.final_tmp_ugmL, 1),
            'overall_score': round(result.overall_score, 1),
        }
    })


@app.route('/api/saccharification', methods=['GET'])
def api_saccharification():
    """原料糖化工序模拟"""
    model = VinegarProductionModel()
    hours = float(request.args.get('hours', 1.0))
    temp = float(request.args.get('temperature', 60.0))
    raw_material = request.args.get('raw_material', '糯米')

    state = model.saccharification.get_state_at(hours, temp, raw_material)

    return jsonify({
        'input': {
            'hours': hours,
            'temperature': temp,
            'raw_material': raw_material,
        },
        'output': {
            'reducing_sugar': state.reducing_sugar,
            'starch_conversion_rate': state.starch_conversion_rate,
            'duration_hours': state.duration_hours,
            'temperature': state.temperature,
        }
    })


@app.route('/api/alcohol', methods=['GET'])
def api_alcohol():
    """酒精发酵工序模拟"""
    model = VinegarProductionModel()
    days = float(request.args.get('days', 5.0))
    initial_sugar = float(request.args.get('initial_sugar', 10.0))
    temp = float(request.args.get('temperature', 30.0))

    state = model.alcohol.get_state_at(days, initial_sugar, temp)

    return jsonify({
        'input': {
            'days': days,
            'initial_sugar': initial_sugar,
            'temperature': temp,
        },
        'output': {
            'ethanol': state.ethanol,
            'reducing_sugar': state.reducing_sugar,
            'yeast_viability': state.yeast_viability,
            'CO2_production': state.CO2_production,
            'duration_days': state.duration_days,
            'temperature': state.temperature,
        }
    })


@app.route('/api/acid', methods=['GET'])
def api_acid():
    """醋酸发酵工序模拟"""
    model = AAFModel()
    days = float(request.args.get('days', 18))

    aaf_state = model.get_state_at(days)
    turning = model.recommend_turning(
        days,
        oxygen_upper=aaf_state.oxygen_upper,
        oxygen_lower=aaf_state.oxygen_lower,
        temperature=aaf_state.temperature_upper
    )

    sim_data = model.simulate(n_points=200)

    return jsonify({
        'input': {'days': days},
        'output': {
            'day': aaf_state.day,
            'stage': aaf_state.stage,
            'total_acid': aaf_state.total_acid,
            'acetic_acid': aaf_state.acetic_acid,
            'non_volatile_acid': aaf_state.non_volatile_acid,
            'lactic_acid': aaf_state.lactic_acid,
            'ethanol_residual': aaf_state.ethanol_residual,
            'oxygen_upper': aaf_state.oxygen_upper,
            'oxygen_lower': aaf_state.oxygen_lower,
            'temperature_upper': aaf_state.temperature_upper,
        },
        'turning': turning,
        'simulation': {
            'time': sim_data['time'],
            'total_acid': sim_data['total_acid'],
            'acetic_acid': sim_data['acetic_acid'],
        }
    })


@app.route('/api/leaching', methods=['GET'])
def api_leaching():
    """淋醋工序模拟"""
    model = VinegarProductionModel()
    aaf_days = float(request.args.get('aaf_days', 18))
    water_ratio = float(request.args.get('water_ratio', 1.2))
    leaching_time = float(request.args.get('leaching_time', 16.0))

    aaf_model = AAFModel()
    aaf_state = aaf_model.get_state_at(aaf_days)
    leached_state = model.leaching.get_state_at(aaf_state, water_ratio, leaching_time)

    return jsonify({
        'input': {
            'aaf_days': aaf_days,
            'water_ratio': water_ratio,
            'leaching_time': leaching_time,
        },
        'output': {
            'total_acid': leached_state.total_acid,
            'ethyl_acetate': leached_state.ethyl_acetate,
            'reducing_sugar': leached_state.reducing_sugar,
            'extraction_efficiency': leached_state.extraction_efficiency,
            'water_ratio': leached_state.water_ratio,
            'leaching_time': leached_state.leaching_time,
        }
    })


@app.route('/api/aging_stage', methods=['GET'])
def api_aging_stage():
    """陈酿工序模拟"""
    months = float(request.args.get('months', 12))
    process = request.args.get('process', '固态发酵')
    raw_material = request.args.get('raw_material', '糯米')
    craft_style = request.args.get('craft_style', '传统')

    state = age_to_state(months, process, raw_material, craft_style)
    profile = compute_flavor_profile(state)
    sensory = compute_sensory_score(state)
    overall = compute_overall_score(state, profile, sensory)

    return jsonify({
        'input': {
            'months': months,
            'process': process,
            'raw_material': raw_material,
            'craft_style': craft_style,
        },
        'output': {
            'vinegar_age_months': state.vinegar_age_months,
            'total_acid': state.total_acid,
            'ethyl_acetate': state.ethyl_acetate,
            'tmp': state.tmp,
            'ph': state.ph,
        },
        'flavor': profile.as_dict(),
        'overall': overall
    })


if __name__ == '__main__':
    print("=" * 60)
    print("  醋风味监测系统 - Vinegar Flavor Monitoring System")
    print("  五工序生产模型 v2.1.0")
    print("=" * 60)
    print("  Running at: http://127.0.0.1:2026")
    print("  Pages:")
    print("    /           - 首页 (风味监测)")
    print("    /aging      - 陈酿预测")
    print("    /aaf        - AAF发酵")
    print("    /about      - 关于")
    print("  API Endpoints:")
    print("    /api/saccharification - 原料糖化")
    print("    /api/alcohol          - 酒精发酵")
    print("    /api/acid             - 醋酸发酵")
    print("    /api/leaching         - 淋醋")
    print("    /api/aging_stage     - 陈酿")
    print("    /api/process          - 完整流程")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=2026)
