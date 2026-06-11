"""
혈액 부족 예측 및 헌혈 운영 최적화 — Streamlit 대시보드
UNIST Industrial Operations Management Term Project
Team 2: 노우찬 · 손준영 · 민예지
"""

import os, json, warnings, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from scraper import scrape_blood_inventory, KRC_RISK_LEVELS
from optimizer import optimize_response, estimate_daily_need, CAMPAIGN_TIERS

warnings.filterwarnings('ignore')
load_dotenv()

# ── 경로 설정 ─────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(BASE, 'processed')

# ══════════════════════════════════════════════════════════════════
# 다국어 텍스트 사전 (한국어 / English)
# ══════════════════════════════════════════════════════════════════
TEXT = {
    '한국어': {
        'page_title':        '혈액 공급 운영 대시보드',
        'subtitle':          '대한적십자사 혈액정보 실시간 연동 · AI 멀티에이전트 예측 시스템',
        'run_btn':           '▶ 분석 실행',
        'refresh_btn':       '🔄 다시 실행',
        'lang_label':        '🌐 언어',
        # KPI
        'kpi_stock':         '🩸 농축적혈구 보유량',
        'kpi_risk':          '위험 등급',
        'kpi_min':           '14일 예측 최저',
        'kpi_action':        '📢 권고 대응',
        # 탭
        'tab_plt_fc':        '📅 14일 예측 (혈소판)',
        'tab_total_fc':      '🩸 14일 예측 (적혈구)',
        'tab_comp_fc':       '🔬 제제별 예측',
        'tab_btype_fc':      '🅾️ 혈액형별 예측',
        'tab_blood_type':    '🩸 혈액형별 보유일수',
        'tab_comp_stock':    '📊 제제별 보유량',
        # 심층 분석
        'analysis_title':    '🔬 심층 분석',
        'tab_campaign':      '💉 캠페인 시뮬레이션',
        'tab_trend':         '📉 헌혈률 장기 추세',
        'tab_gap':           '⚖️ 공급-수요 갭',
        'tab_whatif':        '🎯 What-If 최적화',
        'tab_waste':         '🗑️ 폐기 분석',
        # 보고서/로그
        'report_title':      '📄 운영 보고서',
        'report_tab_current':'🎯 현재 유력 상황 보고서',
        'report_tab_scenario':'🔀 시나리오별 대응 보고서',
        'scenario_caption':  '각 위험 등급이 발생할 경우의 예상 상황과 대응 방안입니다. 현재 등급은 강조 표시됩니다.',
        'scenario_current':  '현재',
        'scenario_trigger':  '발생 조건',
        'scenario_est':      '예상 보유량',
        'scenario_action':   '대응 방안',
        'scenario_boost':    '목표 증가율',
        'risk_reasoning_title': '🧠 위험 평가 근거 (AI 추론)',
        'risk_key_factor':   '핵심 주의 요소',
        'risk_by_llm':       'gpt-4o-mini 추론',
        'risk_by_formula':   '규칙 기반 공식',
        'log_title':         '🤖 에이전트 실행 로그',
        # 차트 제목
        'chart_total_fc':    '농축적혈구(RBC) 14일 예측 — 오늘({date}) 실측 기준',
        'chart_comp_fc':     '제제별 보유량 14일 예측 (일별)',
        'chart_btype_fc':    '혈액형별 보유량 14일 예측 (전체 제제 합산)',
        'chart_blood_days':  '혈액형별 RBC 보유일수',
        'chart_comp_stock':  '제제별 보유량 (역사 평균 대비 %)',
        # 제제명
        'comp': {'RBC':'농축적혈구','PLT':'농축혈소판','FFP':'신선동결혈장','SDP':'성분채혈혈소판'},
        # 사이드바
        'sidebar_title':     '🩸 혈액 공급 운영 대시보드',
        'sidebar_caption':   'UNIST OM Team 2',
        'sidebar_krc':       'KRC 위험 단계 기준',
        'sidebar_pipeline':  '에이전트 파이프라인',
        'krc_table':         '| 단계 | 기준 |\n|---|---|\n| 🟢 정상 | 5일 이상 |\n| 🔵 관심 | 3~5일 |\n| 🟡 주의 | 2~3일 |\n| 🟠 경계 | 1~2일 |\n| 🔴 심각 | 1일 미만 |',
        # 스크래핑 출처
        'src_live':          '📡 실시간',
        'src_csv':           '📂 CSV',
        # 기타
        'today_actual':      '오늘 실측',
        'forecast_label':    '예측',
        'no_result':         '← 왼쪽 **분석 실행** 버튼을 눌러 에이전트를 시작하세요.',
    },
    'English': {
        'page_title':        'Blood Supply Operations Dashboard',
        'subtitle':          'Real-time KRC Data · AI Multi-Agent Forecast System',
        'run_btn':           '▶ Run Analysis',
        'refresh_btn':       '🔄 Rerun',
        'lang_label':        '🌐 Language',
        # KPI
        'kpi_stock':         '🩸 RBC Stock',
        'kpi_risk':          'Risk Level',
        'kpi_min':           '14-Day Min Forecast',
        'kpi_action':        '📢 Recommended Action',
        # Tabs
        'tab_plt_fc':        '📅 14-Day Forecast (Platelet)',
        'tab_total_fc':      '🩸 14-Day Forecast (RBC)',
        'tab_comp_fc':       '🔬 By Component',
        'tab_btype_fc':      '🅾️ By Blood Type',
        'tab_blood_type':    '🩸 Days by Blood Type',
        'tab_comp_stock':    '📊 Component Stock',
        # Deep analysis
        'analysis_title':    '🔬 Deep Analysis',
        'tab_campaign':      '💉 Campaign Simulation',
        'tab_trend':         '📉 Donation Rate Trend',
        'tab_gap':           '⚖️ Supply-Demand Gap',
        'tab_whatif':        '🎯 What-If Optimizer',
        'tab_waste':         '🗑️ Waste Analysis',
        # Report/Log
        'report_title':      '📄 Operations Report',
        'report_tab_current':'🎯 Current Situation Report',
        'report_tab_scenario':'🔀 Scenario Response Reports',
        'scenario_caption':  'Expected situations and responses if each risk level occurs. Current level is highlighted.',
        'scenario_current':  'CURRENT',
        'scenario_trigger':  'Trigger',
        'scenario_est':      'Est. Stock',
        'scenario_action':   'Response',
        'scenario_boost':    'Target Boost',
        'risk_reasoning_title': '🧠 Risk Assessment Rationale (AI Reasoning)',
        'risk_key_factor':   'Key Factor',
        'risk_by_llm':       'gpt-4o-mini reasoning',
        'risk_by_formula':   'Rule-based formula',
        'log_title':         '🤖 Agent Execution Log',
        # Chart titles
        'chart_total_fc':    'Red Blood Cells (RBC) · 14-Day Forecast (anchored {date})',
        'chart_comp_fc':     'Component-wise 14-Day Forecast (Daily)',
        'chart_btype_fc':    'Blood-Type 14-Day Forecast (All Components)',
        'chart_blood_days':  'RBC Days by Blood Type',
        'chart_comp_stock':  'Component Stock (% of Historical Avg)',
        # Component names
        'comp': {'RBC':'Red Blood Cells','PLT':'Platelets','FFP':'Fresh Frozen Plasma','SDP':'Apheresis Platelets'},
        # Sidebar
        'sidebar_title':     '🩸 Blood Supply Dashboard',
        'sidebar_caption':   'UNIST OM Team 2',
        'sidebar_krc':       'KRC Risk Levels',
        'sidebar_pipeline':  'Agent Pipeline',
        'krc_table':         '| Level | Threshold |\n|---|---|\n| 🟢 Normal | ≥ 5 days |\n| 🔵 Watch | 3–5 days |\n| 🟡 Caution | 2–3 days |\n| 🟠 Warning | 1–2 days |\n| 🔴 Critical | < 1 day |',
        # Scraping source
        'src_live':          '📡 Live',
        'src_csv':           '📂 CSV',
        # Others
        'today_actual':      'Today (Live)',
        'forecast_label':    'Forecast',
        'no_result':         '← Click **Run Analysis** in the sidebar to start.',
    },
}

# ── 페이지 설정 ───────────────────────────────────────────────────
st.set_page_config(
    page_title="혈액 공급 운영 대시보드",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 스타일 ────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.report-box {
    background: #f8f9fa;
    color: #1a1a1a !important;
    border-left: 4px solid #c62828;
    border-radius: 6px;
    padding: 16px 20px;
    font-size: 0.95rem;
    line-height: 1.7;
    white-space: pre-wrap;
}
.report-box * { color: #1a1a1a !important; }
.scenario-card {
    background: #ffffff;
    color: #1a1a1a !important;
    border-radius: 8px;
    padding: 14px 18px;
    line-height: 1.6;
    font-size: 0.9rem;
    border: 1px solid #e0e0e0;
    height: 100%;
}
.scenario-card * { color: #1a1a1a !important; }
.scenario-card h4 { margin: 0 0 8px 0; font-size: 1rem; }
.agent-log {
    font-family: monospace;
    font-size: 0.85rem;
    background: #1e1e1e;
    color: #d4d4d4;
    padding: 12px 16px;
    border-radius: 8px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 데이터 로드 (캐싱)
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def load_thresholds():
    with open(f'{PROCESSED}/thresholds.json', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_daily_inventory():
    df = pd.read_csv(f'{PROCESSED}/daily_inventory.csv', parse_dates=['date'])
    return df.sort_values('date').reset_index(drop=True)

@st.cache_data
def load_forecast():
    with open(f'{PROCESSED}/latest_forecast.json', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_monthly_by_type():
    return pd.read_csv(f'{PROCESSED}/monthly_inventory_by_type.csv', parse_dates=['date'])

# ── 정보공개 청구 데이터 (2021–2025 일별, 적십자사 공식) ──────────
@st.cache_data
def load_disclosure_inv_by_type():
    """혈액형별 일별 보유량 (O/A/B/AB, 전체 제제 합산)"""
    df = pd.read_csv(f'{PROCESSED}/inventory_by_type.csv', parse_dates=['date'])
    return df.set_index('date').asfreq('D').interpolate()

@st.cache_data
def load_disclosure_inv_by_component():
    """제제별 일별 보유량 (RBC/혈소판/혈장/백혈구여과제거혈소판)"""
    df = pd.read_csv(f'{PROCESSED}/inventory_by_component.csv', parse_dates=['date'])
    return df.set_index('date').asfreq('D').interpolate()

@st.cache_resource
def _train_hw_series(col: str, source: str):
    """정보공개 일별 시계열 1개에 대해 damped Holt-Winters 학습 (캐싱).
    source: 'type' | 'component'"""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    df = load_disclosure_inv_by_type() if source == 'type' else load_disclosure_inv_by_component()
    ts = df[col]
    model = ExponentialSmoothing(
        ts, trend='add', damped_trend=True, seasonal='add',
        seasonal_periods=7, initialization_method='estimated'
    ).fit(optimized=True)
    return model, ts, float(model.resid.std())

THRESHOLDS  = load_thresholds()

# KRC 공식 위험 단계 색상/이모지 (일수 기준)
RISK_COLOR  = {'NORMAL':'#4caf50','WATCH':'#1565c0','CAUTION':'#ffb300','WARNING':'#ff6d00','CRITICAL':'#d50000'}
RISK_EMOJI  = {'NORMAL':'🟢','WATCH':'🔵','CAUTION':'🟡','WARNING':'🟠','CRITICAL':'🔴'}
RISK_BG     = {'NORMAL':'#e8f5e9','WATCH':'#e3f2fd','CAUTION':'#fff8e1','WARNING':'#fff3e0','CRITICAL':'#ffebee'}

def classify_krc_risk(days: float) -> str:
    """KRC 공식 기준: 적혈구 보유일수 → 위험 등급"""
    if days < 1: return 'CRITICAL'
    if days < 2: return 'WARNING'
    if days < 3: return 'CAUTION'
    if days < 5: return 'WATCH'
    return 'NORMAL'

# ══════════════════════════════════════════════════════════════════
# LLM 설정
# ══════════════════════════════════════════════════════════════════
USE_LLM = bool(os.environ.get('OPENAI_API_KEY', ''))
if USE_LLM:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3, max_tokens=1024)
else:
    llm = None

# ══════════════════════════════════════════════════════════════════
# 에이전트 공통 유틸리티
# ══════════════════════════════════════════════════════════════════
def classify_risk(v):
    if v < THRESHOLDS['CRITICAL']: return 'CRITICAL'
    if v < THRESHOLDS['WARNING']:  return 'WARNING'
    if v < THRESHOLDS['CAUTION']:  return 'CAUTION'
    return 'NORMAL'

_ACTION_RULES = {
    'CRITICAL':('EMERGENCY','전국 긴급 전 채널 캠페인 즉시 실행 — SNS·TV·라디오 동시 송출 및 기업 단체헌혈 긴급 요청','전국','+15%'),
    'WARNING': ('HIGH',     'SNS 캠페인 집중 집행 및 지역 혈액원별 단체헌혈 협약기관 긴급 연락',                   '전국','+10%'),
    'CAUTION': ('MEDIUM',   '정기헌혈자 대상 SMS 발송 및 헌혈의집 방문 유도 이벤트 운영',                          '수도권·광역시','+5%'),
    'NORMAL':  ('NONE',     '현황 모니터링 유지 — 정기 헌혈 안내 채널 정상 운영',                                  '해당 없음','0%'),
}

# ══════════════════════════════════════════════════════════════════
# 에이전트 함수 정의
# ══════════════════════════════════════════════════════════════════
class BloodAgentState(TypedDict):
    run_date: str; current_inventory: int; last_data_date: str
    trend_7d_direction: str; trend_7d_rate: float
    current_season: str; is_risk_season: bool
    rbc_days: float          # 적혈구 보유일수 (KRC 기준)
    rbc_by_type: dict        # 혈액형별 RBC 보유량
    plt_rate: float          # 혈소판 보유율 (%)
    plt_total_units: float   # 혈소판 현재 보유 수 (개)
    scrape_success: bool     # 스크래핑 성공 여부
    forecast_14d: list; forecast_min_value: int; forecast_min_date: str
    shortage_probability: float; days_until_warning: int
    risk_level: str; risk_score: int; component_risks: dict; historical_context: str
    risk_reasoning: list; risk_key_factor: str; risk_method: str
    intervention_level: str; recommended_action: str; action_reasoning: str
    final_report: str; agent_logs: list

def sensing_agent(state):
    logs = state.get('agent_logs', [])
    logs.append("🔵 **Sensing Agent** — bloodinfo.net 실시간 스크래핑")

    # ── bloodinfo.net 실시간 스크래핑 ────────────────────────────
    scrape_ok = False
    rbc_days = 0.0; rbc_by_type = {}; plt_rate = 0.0
    try:
        live = scrape_blood_inventory()
        cur       = int(live['rbc']['total_units'])
        rbc_days  = live['rbc']['total_days']
        rbc_by_type = live['rbc']['units_by_type']
        plt_rate        = live['plt']['total_rate']
        plt_total_units = live['plt'].get('total_units', 0) or 0
        last_date = live['date']
        scrape_ok = True
        logs.append(f"  → 스크래핑 성공: **{cur:,} unit** ({last_date})")
        logs.append(f"  → RBC 보유일수: **{rbc_days}일**  |  PLT 보유율: {plt_rate}%")
    except Exception as e:
        # 스크래핑 실패 시 CSV 마지막 행으로 폴백
        logs.append(f"  → ⚠️ 스크래핑 실패 ({e}) — CSV 데이터로 대체")
        df = load_daily_inventory()
        ts = df.set_index('date')['inventory']
        cur = int(ts.iloc[-1])
        last_date = ts.index.max().strftime('%Y-%m-%d')
        # 보유일수 추정: 역사적 1일 소요량 평균으로 계산
        daily_need_avg  = 5052  # 역사적 평균 1일 소요량
        rbc_days        = round(cur / daily_need_avg, 1)
        plt_total_units = 0.0

    # ── 7일 추세 (역사적 CSV 기반) ────────────────────────────────
    df = load_daily_inventory()
    ts = df.set_index('date')['inventory']
    slope = float(np.polyfit(range(7), ts.iloc[-7:].values, 1)[0])
    direction = '증가' if slope > 200 else ('감소' if slope < -200 else '보합')

    # ── 계절 정보 ─────────────────────────────────────────────────
    month = pd.Timestamp.today().month
    season = {1:'겨울',2:'겨울',3:'봄',4:'봄',5:'봄',
              6:'여름',7:'여름',8:'여름',
              9:'가을',10:'가을',11:'가을',12:'겨울'}[month]
    is_risk = month in [3, 4, 10, 11, 12]

    logs.append(f"  → 7일 추세: {direction} ({slope:+.0f} unit/일)  |  시즌: {season}")

    return {
        'current_inventory':  cur,
        'last_data_date':     last_date,
        'trend_7d_direction': direction,
        'trend_7d_rate':      round(slope, 1),
        'current_season':     season,
        'is_risk_season':     is_risk,
        'rbc_days':           rbc_days,
        'rbc_by_type':        rbc_by_type,
        'plt_rate':           plt_rate,
        'plt_total_units':    plt_total_units,
        'scrape_success':     scrape_ok,
        'agent_logs':         logs,
    }

@st.cache_resource
def _train_hw_daily():
    """일별 RBC 보유량으로 Holt-Winters 학습 (주간 계절성).
    무거우므로 캐싱하여 재사용."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    df = load_daily_inventory()
    ts = df.set_index('date')['inventory'].sort_index().asfreq('D')
    ts = ts.interpolate()   # 결측 보간
    model = ExponentialSmoothing(
        ts, trend='add', damped_trend=True, seasonal='add',
        seasonal_periods=7, initialization_method='estimated'
    ).fit(optimized=True)
    resid_std = float(model.resid.std())
    return model, ts, resid_std


def forecasting_agent(state):
    logs = state.get('agent_logs', [])
    logs.append("🟡 **Forecasting Agent** — Holt-Winters 14일 예측 (실시간 앵커링)")

    today_actual = state['current_inventory']
    rbc_days     = state['rbc_days']
    daily_need   = (today_actual / rbc_days) if rbc_days > 0 else 5052

    today = pd.Timestamp.today().normalize()

    # ── Holt-Winters 학습 (일별, 캐싱) ───────────────────────────
    model, ts_hist, resid_std = _train_hw_daily()

    # 학습 데이터 끝 → 오늘까지의 일수만큼 예측한 뒤, 오늘 이후 14일 추출
    last_train_date = ts_hist.index.max()
    steps_to_today  = max((today - last_train_date).days, 0)
    total_steps     = steps_to_today + 14
    hw_fc_full      = model.forecast(total_steps)   # 미래 전체 예측

    # ── 실시간 앵커링: 오늘 시점 모델 예측값과 실측값의 차이로 전체 보정 ──
    if steps_to_today > 0 and steps_to_today <= len(hw_fc_full):
        model_today = hw_fc_full.iloc[steps_to_today - 1]
    else:
        model_today = float(ts_hist.iloc[-1])
    anchor_adj = today_actual - model_today          # 보정량

    # 오늘 이후 14일 구간만 추출 + 앵커 보정 적용
    hw_future = hw_fc_full.iloc[steps_to_today:steps_to_today + 14]

    fc_list = []
    for k, model_val in enumerate(hw_future.values, start=1):
        future_date = today + pd.Timedelta(days=k)
        fc_val      = max(0, round(float(model_val) + anchor_adj))   # ★ 앵커링
        fc_days     = round(fc_val / daily_need, 1) if daily_need > 0 else 0
        fc_list.append({
            'date':     future_date.strftime('%Y-%m-%d'),
            'forecast': fc_val,
            'lower_95': max(0, round(fc_val - 1.96 * resid_std)),
            'upper_95': round(fc_val + 1.96 * resid_std),
            'days':     fc_days,
            'risk':     classify_krc_risk(fc_days),
        })

    fc_vals   = [r['forecast'] for r in fc_list]
    fc_days_l = [r['days']     for r in fc_list]
    min_idx   = int(np.argmin(fc_days_l))
    d_warn    = next((i for i, d in enumerate(fc_days_l) if d < 2), len(fc_list))
    shortage_prob = sum(1 for d in fc_days_l if d < 3) / len(fc_list)

    logs.append(f"  → 예측 최저: **{fc_list[min_idx]['days']}일분** ({fc_list[min_idx]['date']})")
    logs.append(f"  → 주의 이하(3일분): {sum(1 for d in fc_days_l if d < 3)}일/14일")
    logs.append(f"  → 기준: {today.strftime('%Y-%m-%d')} 실제 **{today_actual:,} unit** 앵커")

    return {
        'forecast_14d':         fc_list,
        'forecast_min_value':   int(fc_vals[min_idx]),
        'forecast_min_date':    fc_list[min_idx]['date'],
        'shortage_probability': round(shortage_prob, 3),
        'days_until_warning':   d_warn,
        'agent_logs':           logs,
    }

def _compute_score_formula(rbc_days, fc_min_days, d_warn, sp, is_risk):
    """규칙 기반 위험 점수 (LLM 판단의 참고용 baseline)"""
    score = 0
    if   rbc_days < 1: score += 40
    elif rbc_days < 2: score += 30
    elif rbc_days < 3: score += 20
    elif rbc_days < 5: score += 10
    if   fc_min_days < 1: score += 30
    elif fc_min_days < 2: score += 22
    elif fc_min_days < 3: score += 14
    elif fc_min_days < 5: score += 6
    if   d_warn <= 2:  score += 20
    elif d_warn <= 5:  score += 15
    elif d_warn <= 10: score += 8
    score += int(sp * 5)
    if is_risk: score += 5
    return min(100, score)


def risk_agent(state):
    logs = state.get('agent_logs', [])
    logs.append("🟠 **Risk Agent** — LLM 추론 기반 위험 평가 (공식 점수 참고)")

    cur          = state['current_inventory']
    rbc_days     = state['rbc_days']
    fc_days_list = [r['days'] for r in state['forecast_14d']]
    fc_min_days  = min(fc_days_list) if fc_days_list else rbc_days
    d_warn       = state['days_until_warning']
    is_risk      = state['is_risk_season']
    sp           = state['shortage_probability']

    # ── 1) 규칙 기반 점수 계산 (참고용 baseline) ─────────────────
    formula_score = _compute_score_formula(rbc_days, fc_min_days, d_warn, sp, is_risk)
    formula_level = classify_krc_risk(rbc_days)

    # ── 2) 제제별 위험도 ─────────────────────────────────────────
    mbt = load_monthly_by_type()
    lm  = mbt['date'].max(); ld = mbt[mbt['date'] == lm]
    ha  = mbt[mbt['date'].dt.month == lm.month].groupby('component_code')['inventory'].mean()
    comp = {}
    for _, row in ld.iterrows():
        code = row['component_code']; val = row['inventory']
        avg  = ha.get(code, val); ratio = val / avg if avg > 0 else 1.0
        thr  = 0.85 if code in ('PLT', 'SDP') else 0.75
        level = 'WARNING' if ratio < thr - 0.10 else ('CAUTION' if ratio < thr else 'NORMAL')
        comp[code] = {'level': level, 'ratio': round(ratio, 3), 'value': int(val)}

    # ── 3) 전년 동기 맥락 ────────────────────────────────────────
    df = load_daily_inventory(); ts = df.set_index('date')['inventory']
    try:
        prev = ts[ts.index.year == (ts.index.max().year - 1)].iloc[-1]
        yd   = cur - int(prev)
        ctx  = f'전년 동기 대비 {yd:+,} unit ({yd/prev*100:+.1f}%)'
    except Exception:
        ctx = '전년 동기 데이터 없음'

    comp_str = ' / '.join(f"{k}:{v['level']}({v['ratio']*100:.0f}%)" for k, v in comp.items())

    # ── 4) LLM 추론 (공식 점수를 참고자료로 제공) ────────────────
    lvl, score, reasoning, key_factor, method = formula_level, formula_score, [], '', 'formula'

    if USE_LLM:
        prompt = f"""당신은 대한적십자사 혈액수급 위기 분석 전문가입니다.
아래 데이터를 종합적으로 해석하여 위험 등급을 판단하세요.

[현황 데이터]
- 농축적혈구(RBC) 현재 보유: {cur:,} unit ({rbc_days}일분)
- 최근 7일 추세: {state['trend_7d_direction']} ({state['trend_7d_rate']:+.0f} unit/일)
- 14일 예측 최저 보유일수: {fc_min_days}일분
- 경고(2일분) 도달까지: {d_warn}일
- 주의 이하 부족확률: {sp*100:.0f}%
- 계절: {state['current_season']} (위험시즌: {'예' if is_risk else '아니오'})
- 전년 동기: {ctx}
- 제제별 위험도: {comp_str}

[참고: 규칙 기반 공식 점수]
- 공식 산출 점수: {formula_score}/100
- 공식 산출 등급: {formula_level}
※ 이 공식 점수는 참고용입니다. 추세·계절·전년대비 등 맥락을 고려해
   필요하면 공식과 다르게 판단해도 됩니다. 단, 크게 벗어나면 근거를 명확히 하세요.

[KRC 공식 기준] 정상≥5일 / 관심3~5일 / 주의2~3일 / 경계1~2일 / 심각<1일

다음 JSON 형식으로만 응답하세요:
{{
  "risk_level": "NORMAL|WATCH|CAUTION|WARNING|CRITICAL 중 하나",
  "risk_score": 0~100 정수,
  "reasoning": ["판단 근거 1", "판단 근거 2", "판단 근거 3"],
  "key_factor": "가장 주의해야 할 핵심 요소 (1문장)"
}}"""
        try:
            raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
            if '```' in raw:
                raw = raw.split('```')[1]
                if raw.startswith('json'): raw = raw[4:]
            parsed = json.loads(raw.strip())
            lvl        = parsed.get('risk_level', formula_level)
            score      = int(parsed.get('risk_score', formula_score))
            reasoning  = parsed.get('reasoning', [])
            key_factor = parsed.get('key_factor', '')
            method     = 'LLM'
            logs.append(f"  → LLM 판단: {RISK_EMOJI.get(lvl,'❓')} **{lvl}** ({score}/100)  [공식: {formula_level} {formula_score}점]")
            if key_factor:
                logs.append(f"  → 핵심 요소: {key_factor}")
        except Exception as e:
            logs.append(f"  → ⚠️ LLM 파싱 실패 ({e}) → 공식 점수 사용")
            reasoning = [f'공식 점수 {formula_score}점 기준 {formula_level} 등급 판정']
    else:
        reasoning = [f'규칙 기반 공식 점수 {formula_score}점 → {formula_level} 등급']
        logs.append(f"  → 위험 점수: **{score}/100**  |  등급: {RISK_EMOJI.get(lvl,'❓')} **{lvl}** (공식)")

    logs.append(f"  → 전년 동기: {ctx}")

    return {
        'risk_level':         lvl,
        'risk_score':         score,
        'component_risks':    comp,
        'historical_context': ctx,
        'risk_reasoning':     reasoning,
        'risk_key_factor':    key_factor,
        'risk_method':        method,
        'agent_logs':         logs,
    }

def action_agent(state):
    logs = state.get('agent_logs', [])
    logs.append("🔴 **Action Agent** — 대응 방안 결정")
    risk_level = state['risk_level']
    if USE_LLM:
        comp_str='\n'.join(f"  - {k}: {v['level']} (평균 대비 {v['ratio']*100:.0f}%)" for k,v in state['component_risks'].items())
        prompt=f"""당신은 대한적십자사의 혈액 공급 관리 전문가입니다.
현황: 보유량 {state['current_inventory']:,} unit, 위험등급 {risk_level}({state['risk_score']}/100), 추세 {state['trend_7d_direction']}
예측: 14일 내 최저 {state['forecast_min_value']:,} unit({state['forecast_min_date']}), 부족확률 {state['shortage_probability']*100:.0f}%
제제별:\n{comp_str}

JSON으로만 응답:
{{"intervention_level":"NONE|LOW|MEDIUM|HIGH|EMERGENCY","recommended_action":"구체적 방안(2문장)","target_regions":"대상지역","expected_boost_pct":"+N%","reasoning":"근거(1문장)"}}"""
        try:
            resp=llm.invoke([HumanMessage(content=prompt)])
            raw=resp.content.strip()
            if '```' in raw: raw=raw.split('```')[1]; raw=raw[4:] if raw.startswith('json') else raw
            p=json.loads(raw.strip())
            lvl=p.get('intervention_level','NONE'); action=p.get('recommended_action',''); reasoning=p.get('reasoning','')
        except:
            lvl,action,_,_=_ACTION_RULES.get(risk_level,_ACTION_RULES['NORMAL']); reasoning='LLM 폴백'
    else:
        lvl,action,_,_=_ACTION_RULES.get(risk_level,_ACTION_RULES['NORMAL']); reasoning='Rule-based'
    logs.append(f"  → 대응 수준: **{lvl}**")
    logs.append(f"  → 실행 방안: {action[:60]}{'...' if len(action)>60 else ''}")
    return {'intervention_level':lvl,'recommended_action':action,'action_reasoning':reasoning,'agent_logs':logs}

def report_agent(state):
    logs = state.get('agent_logs', [])
    logs.append("📄 **Report Agent** — 운영 보고서 생성")
    if USE_LLM:
        comp_str=' / '.join(f"{k}:{v['level']}" for k,v in state.get('component_risks',{}).items())
        prompt=f"""당신은 대한적십자사 혈액수급 운영 책임자입니다. 아래 데이터로 의사결정용 운영 보고서를 작성하세요.

[현황] {state['last_data_date']} 농축적혈구(RBC) 보유량 {state['current_inventory']:,} unit ({state['rbc_days']}일분)
       최근 7일 추세 {state['trend_7d_direction']} ({state['trend_7d_rate']:+.0f} unit/일), {state['historical_context']}
[위험] 등급 {state['risk_level']} (위험점수 {state['risk_score']}/100), 제제별 {comp_str}
[예측] 향후 14일 최저 {state['forecast_min_value']:,} unit ({state['forecast_min_date']}), 부족확률 {state['shortage_probability']*100:.0f}%
[권고] {state.get('intervention_level','')} — {state.get('recommended_action','')}

다음 4개 소제목을 그대로 사용하고 각 항목을 1~2문장으로 작성하세요. 수치 근거를 반드시 포함하세요.
■ 현황 진단
■ 위험 평가
■ 14일 전망
■ 권고 조치
한국어로, 마크다운 강조(**)는 쓰지 마세요."""
        try:
            report=llm.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception:
            report=_rule_report(state)
    else:
        report=_rule_report(state)
    logs.append("  → 보고서 생성 완료 ✅")
    return {'final_report':report,'agent_logs':logs}


# ──────────────────────────────────────────────────────────────────
#  시나리오 보고서: 4개 위험 등급별 "만약 ~라면" 대응 시나리오
# ──────────────────────────────────────────────────────────────────
SCENARIO_DEFS = [
    {'level':'NORMAL',   'emoji':'🟢', 'days':'5일 이상', 'rep_days':6,
     'trigger':'RBC 보유량이 5일분 이상으로 안정적인 경우',
     'channel':'정기 채널 (헌혈앱 푸시, 홈페이지 배너)',
     'target':'전체 등록 헌혈자',
     'timing':'주간 정기 안내',
     'measures':[
        '레드커넥트 앱 정기 푸시 1회/주 발송',
        '헌혈의집 예약 정상 운영, 별도 인센티브 없음',
        '제제별 보유 추이 주 2회 모니터링',
        '단체헌혈 협약기관 정기 일정 유지',
     ],
     'boost':'0%'},
    {'level':'WATCH',    'emoji':'🔵', 'days':'3~5일', 'rep_days':4,
     'trigger':'RBC 보유량이 3~5일분으로 감소 징후가 보이는 경우',
     'channel':'SNS (인스타·카카오 채널) + 앱 푸시',
     'target':'최근 6개월 내 헌혈 이력자',
     'timing':'주 2~3회',
     'measures':[
        'SNS 헌혈 독려 카드뉴스 주 2회 게시',
        '직전 6개월 헌혈자 대상 앱 푸시 + 카카오 알림톡',
        'O형·A형 등 감소 폭 큰 혈액형 우선 호소',
        '일별 보유량 추이 모니터링으로 전환 (주간→일간)',
     ],
     'boost':'+3%'},
    {'level':'CAUTION',  'emoji':'🟡', 'days':'2~3일', 'rep_days':2.5,
     'trigger':'RBC 보유량이 2~3일분으로 부분적 부족이 발생한 경우',
     'channel':'SMS 일괄 발송 + 헌혈의집 현장 이벤트',
     'target':'전체 등록 헌혈자 + 휴면 헌혈자',
     'timing':'즉시, 이후 격일 반복',
     'measures':[
        '등록 헌혈자 전체 대상 SMS 일괄 발송 (예약 링크 포함)',
        '헌혈의집 방문 시 기념품·문화상품권 등 인센티브 제공',
        '대학·기업 단체헌혈 일정 1주 내 조기 편성 요청',
        '부족 혈액형 지정 호소 메시지 차별 발송',
        '인근 혈액원 재고 현황 공유 및 이송 가능성 사전 점검',
     ],
     'boost':'+5%'},
    {'level':'WARNING',  'emoji':'🟠', 'days':'1~2일', 'rep_days':1.5,
     'trigger':'RBC 보유량이 1~2일분으로 부족이 지속되는 경우',
     'channel':'전 디지털 채널 집중 + 협약기관 직접 연락',
     'target':'전체 헌혈자 + 단체헌혈 협약기관 + 지역 커뮤니티',
     'timing':'즉시, 매일 반복 집행',
     'measures':[
        'SNS·앱·SMS 동시 집중 캠페인 (1일 1회 이상)',
        '단체헌혈 협약 대학·기업·관공서 담당자 직접 전화 요청',
        '인근 권역 혈액원 간 재고 긴급 이송 실행',
        '주말·야간 헌혈의집 운영시간 한시 연장',
        '지역 언론(지역방송·신문)에 헌혈 협조 보도자료 배포',
        '부족 혈액형(특히 O형) 집중 호소 및 지정헌혈 유도',
     ],
     'boost':'+10%'},
    {'level':'CRITICAL', 'emoji':'🔴', 'days':'1일 미만', 'rep_days':0.5,
     'trigger':'RBC 보유량이 1일분 미만으로 수급 위기가 확대된 경우',
     'channel':'전국 전채널 (TV·라디오 포함) 비상 동원',
     'target':'전 국민 + 정부·지자체·군부대',
     'timing':'즉시, 위기 해소 시까지 상시',
     'measures':[
        '전국 긴급 전채널 캠페인 즉시 발령 (SNS·TV·라디오 동시 송출)',
        '보건복지부·지자체 협조 요청 및 공공기관 단체헌혈 긴급 동원',
        '군부대·경찰 등 대규모 단체헌혈 긴급 협조 요청',
        '전국 혈액원 간 재고 재배분 및 응급 수혈 우선순위 조정',
        '의료기관에 비응급 수술 일정 조정 협조 요청',
        '헌혈의집 전 지점 운영시간 최대 연장 및 임시 헌혈 차량 추가 배치',
     ],
     'boost':'+15%'},
]

def build_scenario_reports(result):
    """현재 상태 기준 + 5개 가상 시나리오 보고서 카드 데이터 반환"""
    rbc_days   = result.get('rbc_days', 0)
    daily_need = result['current_inventory'] / rbc_days if rbc_days > 0 else 5052
    cur_level  = result['risk_level']

    cards = []
    for sc in SCENARIO_DEFS:
        est_unit  = int(sc['rep_days'] * daily_need)
        # 목표 증가율 적용 시 일일 추가 확보량
        boost_pct = int(sc['boost'].replace('%', '').replace('+', '')) / 100
        extra_per_day = int(daily_need * boost_pct)
        cards.append({
            **sc,
            'est_unit':      est_unit,
            'extra_per_day': extra_per_day,
            'is_current':    (sc['level'] == cur_level),
        })
    return cards

def _rule_report(s):
    rl={'NORMAL':'정상','WATCH':'관심','CAUTION':'주의','WARNING':'경고','CRITICAL':'위기'}
    am=_ACTION_RULES.get(s.get('risk_level','NORMAL'),_ACTION_RULES['NORMAL'])
    return (
        f"■ 현황 진단\n"
        f"{s['last_data_date']} 기준 농축적혈구(RBC) 보유량은 {s['current_inventory']:,} unit"
        f"({s.get('rbc_days','-')}일분)이며, 최근 7일 추세는 {s['trend_7d_direction']}"
        f"({s['trend_7d_rate']:+.0f} unit/일)입니다. {s['historical_context']}.\n\n"
        f"■ 위험 평가\n"
        f"종합 위험점수는 {s['risk_score']}/100점으로 '{rl.get(s['risk_level'],s['risk_level'])}' 등급에 해당합니다.\n\n"
        f"■ 14일 전망\n"
        f"향후 14일 내 최저 보유량은 {s['forecast_min_value']:,} unit으로 예상되며"
        f"({s['forecast_min_date']}), 주의 수준 이하로 떨어질 확률은 {s['shortage_probability']*100:.0f}%입니다.\n\n"
        f"■ 권고 조치\n"
        f"{am[1]} (목표 증가율 {am[3]})."
    )

def risk_router(state):
    return 'action' if state['risk_level'] in ('CAUTION','WARNING','CRITICAL') else 'report'

# ══════════════════════════════════════════════════════════════════
# LangGraph 파이프라인
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def build_pipeline():
    wf = StateGraph(BloodAgentState)
    wf.add_node('sensing',     sensing_agent)
    wf.add_node('forecasting', forecasting_agent)
    wf.add_node('risk',        risk_agent)
    wf.add_node('action',      action_agent)
    wf.add_node('report',      report_agent)
    wf.set_entry_point('sensing')
    wf.add_edge('sensing',     'forecasting')
    wf.add_edge('forecasting', 'risk')
    wf.add_conditional_edges('risk', risk_router, {'action':'action','report':'report'})
    wf.add_edge('action',  'report')
    wf.add_edge('report',  END)
    return wf.compile()

def run_pipeline():
    app = build_pipeline()
    init = {
        'run_date':            pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'current_inventory':   0,
        'last_data_date':      '',
        'trend_7d_direction':  '',
        'trend_7d_rate':       0.0,
        'current_season':      '',
        'is_risk_season':      False,
        'rbc_days':            0.0,
        'rbc_by_type':         {},
        'plt_rate':            0.0,
        'plt_total_units':     0.0,
        'scrape_success':      False,
        'forecast_14d':        [],
        'forecast_min_value':  0,
        'forecast_min_date':   '',
        'shortage_probability':0.0,
        'days_until_warning':  14,
        'risk_level':          'NORMAL',
        'risk_score':          0,
        'component_risks':     {},
        'historical_context':  '',
        'risk_reasoning':      [],
        'risk_key_factor':     '',
        'risk_method':         '',
        'intervention_level':  'NONE',
        'recommended_action':  '',
        'action_reasoning':    '',
        'final_report':'','agent_logs':[],
    }
    return app.invoke(init)

# ══════════════════════════════════════════════════════════════════
# 차트 함수
# ══════════════════════════════════════════════════════════════════
def chart_forecast(result, lang='한국어'):
    """오늘 실측값 → 14일 예측 차트 (농축적혈구 RBC, 오늘부터만 표시)"""
    T          = TEXT[lang]
    fc_df      = pd.DataFrame(result['forecast_14d'])
    fc_dates   = pd.to_datetime(fc_df['date'])
    fc_vals    = fc_df['forecast'].values
    today      = pd.Timestamp.today().normalize()
    today_val  = result['current_inventory']
    rbc_days   = result.get('rbc_days', 0)
    daily_need = today_val / rbc_days if rbc_days > 0 else 5052

    all_x = [today] + list(fc_dates)
    all_y = [today_val] + list(fc_vals)

    fig = go.Figure()

    # 신뢰구간
    if 'lower_95' in fc_df.columns:
        fig.add_trace(go.Scatter(
            x=list(fc_dates) + list(fc_dates[::-1]),
            y=list(fc_df['upper_95']) + list(fc_df['lower_95'])[::-1],
            fill='toself', fillcolor='rgba(198,40,40,0.12)',
            line=dict(color='rgba(0,0,0,0)'), name='95% 신뢰구간',
            hoverinfo='skip', showlegend=True,
        ))

    # 예측 라인
    fig.add_trace(go.Scatter(
        x=all_x, y=all_y,
        name='14일 예측',
        line=dict(color='#c62828', width=2.5),
        mode='lines+markers',
        marker=dict(size=6, color='#c62828',
                    line=dict(color='white', width=1.5)),
        hovertemplate='%{x|%m/%d}<br><b>%{y:,} unit</b><extra></extra>'
    ))

    # 오늘 강조 마커
    fig.add_trace(go.Scatter(
        x=[today], y=[today_val],
        name=f'오늘 실측',
        mode='markers',
        marker=dict(color='#1565c0', size=13, symbol='circle',
                    line=dict(color='white', width=2.5)),
        hovertemplate=f'오늘({today.strftime("%m/%d")})<br><b>{today_val:,} unit ({rbc_days}일분)</b><extra></extra>'
    ))

    # KRC 위험 임계선 (4단계)
    x0, x1 = today - pd.Timedelta(hours=6), fc_dates.iloc[-1] + pd.Timedelta(hours=6)
    for label, days, color, dash in [
        ('관심 5일', 5, '#1565c0', 'dot'),
        ('주의 3일', 3, '#ffb300', 'dot'),
        ('경계 2일', 2, '#ff6d00', 'dash'),
        ('심각 1일', 1, '#d50000', 'longdash'),
    ]:
        v = days * daily_need
        fig.add_shape(type='line', x0=x0, x1=x1, y0=v, y1=v,
                      line=dict(color=color, width=1.3, dash=dash))
        fig.add_annotation(x=x1, y=v, text=label, showarrow=False,
                           xanchor='left', font=dict(size=9, color=color), xshift=4)

    fig.update_layout(
        title=dict(text='📅 ' + T['chart_total_fc'].format(date=today.strftime("%Y-%m-%d")),
                   font=dict(size=13, color='#333')),
        xaxis=dict(showgrid=False, tickformat='%m/%d',
                   range=[today - pd.Timedelta(hours=12),
                          fc_dates.iloc[-1] + pd.Timedelta(days=1)]),
        yaxis=dict(tickformat=',d', title='보유량 (unit)', showgrid=True,
                   gridcolor='#f0f0f0'),
        legend=dict(orientation='h', y=-0.18, x=0),
        height=340, margin=dict(l=0, r=80, t=45, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
        hovermode='x unified',
    )
    return fig

def chart_components(result):
    """제제별 보유량 — 역사 평균 대비 % (CSV 기반)"""
    comp   = result.get('component_risks', {})
    if not comp:
        return go.Figure()
    names  = list(comp.keys())
    ratios = [v['ratio'] * 100 for v in comp.values()]
    colors = [RISK_COLOR.get(v['level'], '#4caf50') for v in comp.values()]
    labels = {'RBC':'농축적혈구','PLT':'농축혈소판','FFP':'신선동결혈장','SDP':'성분채혈혈소판'}

    fig = go.Figure(go.Bar(
        x=[labels.get(n, n) for n in names], y=ratios,
        marker_color=colors, marker_line_color='white', marker_line_width=2,
        text=[f'{v:.0f}%' for v in ratios], textposition='outside',
        hovertemplate='%{x}<br><b>%{y:.1f}%</b><extra></extra>'
    ))
    for val, color, label in [
        (100, '#9e9e9e', '역사 평균'),
        (85,  '#ffb300', '혈소판 기준'),
        (75,  '#ff6d00', '일반 기준'),
    ]:
        fig.add_hline(y=val, line_dash='dot', line_color=color, line_width=1.3,
                      annotation_text=label, annotation_position='top right',
                      annotation_font_size=9, annotation_font_color=color)

    fig.update_layout(
        title=dict(text='제제별 보유량 (역사 평균 대비 %)', font=dict(size=13, color='#333')),
        yaxis=dict(range=[0, max(ratios) * 1.25 if ratios else 140], title='%',
                   showgrid=True, gridcolor='#f0f0f0'),
        height=340, margin=dict(l=0, r=60, t=45, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
        showlegend=False,
    )
    return fig


def chart_blood_types(result):
    """혈액형별 RBC 보유일수 차트 (스크래핑 실시간 데이터)"""
    rbc_by_type  = result.get('rbc_by_type', {})
    rbc_days_all = result.get('rbc_days', 0)
    daily_need   = result['current_inventory'] / rbc_days_all if rbc_days_all > 0 else 5052

    blood_types = ['A', 'B', 'O', 'AB']
    # 혈액형별 1일 소요량 비율 (대략 A:B:O:AB = 34:26:28:12)
    need_ratio  = {'A': 0.34, 'B': 0.26, 'O': 0.28, 'AB': 0.12}

    days_vals = []
    for bt in blood_types:
        units     = rbc_by_type.get(bt, 0) or 0
        bt_need   = daily_need * need_ratio.get(bt, 0.25)
        days_val  = round(units / bt_need, 1) if bt_need > 0 else 0
        days_vals.append(days_val)

    bar_colors = [
        '#d50000' if d < 1 else
        '#ff6d00' if d < 2 else
        '#ffb300' if d < 3 else
        '#1565c0' if d < 5 else
        '#43a047'
        for d in days_vals
    ]

    fig = go.Figure(go.Bar(
        x=blood_types, y=days_vals,
        marker_color=bar_colors, marker_line_color='white', marker_line_width=2,
        text=[f'{d}일' for d in days_vals], textposition='outside',
        hovertemplate='%{x}형<br><b>%{y}일분</b><extra></extra>'
    ))

    for days, color, label in [(5,'#1565c0','관심'), (3,'#ffb300','주의'), (2,'#ff6d00','경계')]:
        fig.add_hline(y=days, line_dash='dot', line_color=color, line_width=1.3,
                      annotation_text=label, annotation_position='top right',
                      annotation_font_size=9, annotation_font_color=color)

    fig.update_layout(
        title=dict(text='혈액형별 RBC 보유일수', font=dict(size=13, color='#333')),
        yaxis=dict(range=[0, max(days_vals) * 1.3 if days_vals else 10],
                   title='보유일수', showgrid=True, gridcolor='#f0f0f0'),
        xaxis=dict(title='혈액형'),
        height=340, margin=dict(l=0, r=60, t=45, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
        showlegend=False,
    )
    return fig

def chart_historical(n_years=3):
    daily = load_daily_inventory()
    ts    = daily.set_index('date')['inventory']
    fig   = go.Figure()
    years = sorted(ts.index.year.unique())[-n_years:]
    palette = ['#90caf9','#42a5f5','#1565c0']
    for yr, color in zip(years, palette):
        sub = ts[ts.index.year == yr]
        fig.add_trace(go.Scatter(
            x=sub.index.dayofyear, y=sub.values,
            name=str(yr), line=dict(color=color, width=1.4),
            hovertemplate=f'{yr} Day %{{x}}: %{{y:,}} unit<extra></extra>'
        ))
    for val, color, label in [
        (THRESHOLDS['CAUTION'],'#ffb300','주의선'),
        (THRESHOLDS['WARNING'],'#ff6d00','경고선'),
    ]:
        fig.add_hline(y=val, line_dash='dot', line_color=color, line_width=1.2,
                      annotation_text=label, annotation_font_size=9)
    fig.update_layout(
        title=dict(text=f'최근 {n_years}년 일별 보유량 비교', font=dict(size=13)),
        xaxis=dict(title='연중 일수 (Day of Year)', showgrid=False),
        yaxis=dict(tickformat=',d', title='보유량 (unit)'),
        height=300,
        legend=dict(orientation='h', y=-0.22),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='white', plot_bgcolor='#fafafa',
        hovermode='x unified',
    )
    return fig


# ──────────────────────────────────────────────────────────────────
#  일별 14일 예측 — 2x2 서브플롯 공통 빌더 (정보공개 일별 데이터 기반)
# ──────────────────────────────────────────────────────────────────
def _build_daily_forecast_grid(series_defs, title, lang='한국어'):
    """
    series_defs: [{'col','source','label','color','anchor'(optional)}] × 4
    각 시계열에 damped HW 학습 → 오늘 이후 14일 예측 (anchor 있으면 보정)
    """
    from plotly.subplots import make_subplots
    T     = TEXT[lang]
    today = pd.Timestamp.today().normalize()

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[s['label'] for s in series_defs],
                        vertical_spacing=0.18, horizontal_spacing=0.10)

    for i, sd in enumerate(series_defs):
        row, col = divmod(i, 2)
        row, col = row + 1, col + 1

        model, ts, _ = _train_hw_series(sd['col'], sd['source'])
        steps = max((today - ts.index.max()).days, 0)
        fc_full = model.forecast(steps + 14)
        fut = fc_full.iloc[steps:steps + 14]
        fut.index = pd.date_range(today + pd.Timedelta(days=1), periods=14)

        # 앵커 보정 (오늘 실측값이 있는 시계열만)
        anchor = sd.get('anchor')
        if anchor:
            model_today = fc_full.iloc[steps - 1] if steps > 0 else float(ts.iloc[-1])
            fut = fut + (anchor - model_today)

        # 예측 라인
        fig.add_trace(go.Scatter(
            x=fut.index, y=fut.values,
            name=sd['label'],
            line=dict(color=sd['color'], width=2, dash='dash'),
            mode='lines+markers', marker=dict(size=4), showlegend=False,
            hovertemplate=T['forecast_label'] + ' %{x|%m/%d}: <b>%{y:,.0f}</b><extra></extra>',
        ), row=row, col=col)

        # 오늘 실측 마커 (앵커가 있을 때)
        if anchor:
            fig.add_trace(go.Scatter(
                x=[today], y=[anchor],
                mode='markers',
                marker=dict(color=sd['color'], size=11, symbol='star',
                            line=dict(color='white', width=1.5)),
                name=T['today_actual'], showlegend=(i == 0),
                hovertemplate=T['today_actual'] + ': <b>%{y:,.0f}</b><extra></extra>',
            ), row=row, col=col)

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color='#333')),
        height=460, margin=dict(l=0, r=10, t=65, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
        legend=dict(orientation='h', y=-0.05),
        hovermode='x unified',
    )
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=11, color='#333')
    return fig


def chart_platelet_forecast(result, lang='한국어'):
    """혈소판(농축) 14일 예측 — 유통기한 5일, 가장 시급한 제제 (메인 차트)"""
    is_ko = (lang == '한국어')
    today = pd.Timestamp.today().normalize()

    plt_units = result.get('plt_total_units') or 0
    plt_rate  = result.get('plt_rate') or 0
    # 1일 소요량: 보유율(%) = 보유량/1일소요 → 역산, 실패 시 공시값 4,572
    daily_need = (plt_units / (plt_rate / 100)) if plt_units and plt_rate else 4572.0

    # damped HW 학습 (정보공개 일별 농축혈소판 계열) + 오늘 실측 앵커
    model, ts, resid_std = _train_hw_series('platelet', 'component')
    steps   = max((today - ts.index.max()).days, 0)
    fc_full = model.forecast(steps + 14)
    fut     = fc_full.iloc[steps:steps + 14]
    fut.index = pd.date_range(today + pd.Timedelta(days=1), periods=14)
    anchor_ok = bool(plt_units) and result.get('scrape_success')
    if anchor_ok:
        model_today = fc_full.iloc[steps - 1] if steps > 0 else float(ts.iloc[-1])
        fut = fut + (plt_units - model_today)

    all_x = ([today] + list(fut.index)) if anchor_ok else list(fut.index)
    all_y = ([plt_units] + list(fut.values)) if anchor_ok else list(fut.values)

    fig = go.Figure()
    # 신뢰구간
    fig.add_trace(go.Scatter(
        x=list(fut.index) + list(fut.index[::-1]),
        y=list(fut.values + 1.96 * resid_std) + list((fut.values - 1.96 * resid_std))[::-1],
        fill='toself', fillcolor='rgba(21,101,192,0.10)',
        line=dict(color='rgba(0,0,0,0)'),
        name=('95% 신뢰구간' if is_ko else '95% CI'), hoverinfo='skip',
    ))
    # 예측 라인
    fig.add_trace(go.Scatter(
        x=all_x, y=all_y,
        name=('14일 예측' if is_ko else '14-day forecast'),
        line=dict(color='#1565c0', width=2.5),
        mode='lines+markers',
        marker=dict(size=6, color='#1565c0', line=dict(color='white', width=1.5)),
        hovertemplate='%{x|%m/%d}<br><b>%{y:,.0f} unit</b><extra></extra>',
    ))
    # 오늘 실측
    if anchor_ok:
        fig.add_trace(go.Scatter(
            x=[today], y=[plt_units],
            name=('오늘 실측' if is_ko else 'Today (live)'),
            mode='markers',
            marker=dict(color='#c62828', size=13, symbol='circle',
                        line=dict(color='white', width=2.5)),
            hovertemplate=(f'오늘 {plt_units:,.0f} unit (보유율 {plt_rate:.0f}%)<extra></extra>'
                           if is_ko else f'Today {plt_units:,.0f}<extra></extra>'),
        ))

    # 보유율 임계선 (100% = 1일분)
    x0 = today - pd.Timedelta(hours=12)
    x1 = fut.index[-1] + pd.Timedelta(hours=12)
    for pct, color, dash in [(200, '#1565c0', 'dot'), (150, '#ffb300', 'dot'), (100, '#d50000', 'longdash')]:
        v = daily_need * pct / 100
        label = (f'보유율 {pct}% ({pct/100:.0f}일분)' if is_ko else f'{pct}% ({pct/100:.0f}d)')
        fig.add_shape(type='line', x0=x0, x1=x1, y0=v, y1=v,
                      line=dict(color=color, width=1.3, dash=dash))
        fig.add_annotation(x=x1, y=v, text=label, showarrow=False,
                           xanchor='left', font=dict(size=9, color=color), xshift=4)

    title = (f'농축혈소판(PLT) 14일 예측 — 유통기한 5일 · 오늘({today.strftime("%Y-%m-%d")}) 실측 기준'
             if is_ko else
             f'Platelet 14-Day Forecast — 5-day shelf life · anchored {today.strftime("%Y-%m-%d")}')
    fig.update_layout(
        title=dict(text='🩹 ' + title, font=dict(size=13, color='#333')),
        xaxis=dict(showgrid=False, tickformat='%m/%d',
                   range=[today - pd.Timedelta(days=1), fut.index[-1] + pd.Timedelta(days=2)]),
        yaxis=dict(tickformat=',d',
                   title=('보유량 (unit)' if is_ko else 'Stock (unit)'),
                   rangemode='tozero', showgrid=True, gridcolor='#f0f0f0'),
        legend=dict(orientation='h', y=-0.18, x=0),
        height=340, margin=dict(l=0, r=110, t=45, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
        hovermode='x unified',
    )
    return fig


def chart_component_forecast(result, lang='한국어'):
    """제제별 일별 14일 예측 (정보공개 데이터, RBC는 실시간 앵커)"""
    T = TEXT[lang]
    series_defs = [
        {'col': 'RBC',        'source': 'component', 'label': T['comp']['RBC'],
         'color': '#c62828',  'anchor': result.get('current_inventory') or None},
        {'col': 'platelet',   'source': 'component', 'label': T['comp']['PLT'],
         'color': '#1565c0',  'anchor': result.get('plt_total_units') or None},
        {'col': 'plasma',     'source': 'component', 'label': T['comp']['FFP'],
         'color': '#2e7d32'},
        {'col': 'F_platelet', 'source': 'component', 'label': T['comp']['SDP'],
         'color': '#6a1b9a'},
    ]
    return _build_daily_forecast_grid(series_defs, T['chart_comp_fc'], lang)


def chart_blood_type_forecast(result, lang='한국어'):
    """혈액형별(O/A/B/AB) 일별 14일 예측 (정보공개 데이터, 전체 제제 합산)"""
    T = TEXT[lang]
    colors = {'O': '#c62828', 'A': '#1565c0', 'B': '#2e7d32', 'AB': '#6a1b9a'}
    suffix = '형' if lang == '한국어' else ''
    series_defs = [
        {'col': bt, 'source': 'type', 'label': f'{bt}{suffix}', 'color': colors[bt]}
        for bt in ['O', 'A', 'B', 'AB']
    ]
    return _build_daily_forecast_grid(series_defs, T['chart_btype_fc'], lang)


# ──────────────────────────────────────────────────────────────────
#  분석 1: 캠페인 시뮬레이션
# ──────────────────────────────────────────────────────────────────
def chart_campaign_sim(result):
    """헌혈 캠페인 강도별 14일 보유량 시뮬레이션"""
    fc_df      = pd.DataFrame(result['forecast_14d'])
    today      = pd.Timestamp.today().normalize()
    today_val  = result['current_inventory']
    rbc_days   = result.get('rbc_days', 0)
    daily_need = today_val / rbc_days if rbc_days > 0 else 5052

    fc_dates = pd.to_datetime(fc_df['date'])
    all_x    = [today] + list(fc_dates)
    base_y   = [today_val] + list(fc_df['forecast'].values)

    fig = go.Figure()

    # 신뢰구간 (기준선)
    if 'lower_95' in fc_df.columns:
        fig.add_trace(go.Scatter(
            x=list(fc_dates) + list(fc_dates[::-1]),
            y=list(fc_df['upper_95']) + list(fc_df['lower_95'])[::-1],
            fill='toself', fillcolor='rgba(198,40,40,0.08)',
            line=dict(color='rgba(0,0,0,0)'), name='기준 95% CI',
            hoverinfo='skip', showlegend=False,
        ))

    # 기준선 (캠페인 없음)
    fig.add_trace(go.Scatter(
        x=all_x, y=base_y,
        name='현재 추세 (캠페인 없음)',
        line=dict(color='#c62828', width=2.5, dash='dash'),
        mode='lines+markers',
        marker=dict(size=5),
        hovertemplate='캠페인 없음<br>%{x|%m/%d}: <b>%{y:,} unit</b><extra></extra>'
    ))

    # 캠페인 시나리오 3개
    scenarios = [
        ('+5%  — SMS 정기헌혈자',   0.05, '#43a047', 'dot'),
        ('+10% — SNS 집중 캠페인', 0.10, '#1976d2', 'dash'),
        ('+15% — 긴급 전채널 캠페인', 0.15, '#7b1fa2', 'solid'),
    ]
    for name, rate, color, dash in scenarios:
        extra_per_day = daily_need * rate
        # k번째 날부터 누적으로 보유량 증가
        scenario_y = [today_val] + [
            fc_df.iloc[k]['forecast'] + extra_per_day * (k + 1)
            for k in range(len(fc_df))
        ]
        fig.add_trace(go.Scatter(
            x=all_x, y=scenario_y,
            name=name,
            line=dict(color=color, width=2, dash=dash),
            hovertemplate=f'{name}<br>%{{x|%m/%d}}: <b>%{{y:,}} unit</b><extra></extra>'
        ))

    # KRC 임계선
    for days, color, label in [
        (5, '#1565c0', '관심(5일)'),
        (3, '#ffb300', '주의(3일)'),
        (2, '#ff6d00', '경계(2일)'),
    ]:
        v = days * daily_need
        fig.add_hline(y=v, line_dash='dot', line_color=color, line_width=1.2,
                      annotation_text=label, annotation_position='bottom right',
                      annotation_font_size=9, annotation_font_color=color)

    # D+14 차이 비교 annotation
    d14_base = base_y[-1]
    for name, rate, color, _ in scenarios:
        d14_val = base_y[-1] + daily_need * rate * 14
        diff = d14_val - d14_base
        fig.add_annotation(
            x=all_x[-1], y=d14_val,
            text=f'+{diff:,.0f}',
            showarrow=False, xanchor='left', xshift=5,
            font=dict(size=9, color=color)
        )

    fig.update_layout(
        title=dict(text='캠페인 강도별 14일 혈액 보유량 시뮬레이션', font=dict(size=13, color='#333')),
        xaxis=dict(showgrid=False, tickformat='%m/%d',
                   range=[today - pd.Timedelta(hours=6),
                          fc_dates.iloc[-1] + pd.Timedelta(days=1)]),
        yaxis=dict(tickformat=',d', title='보유량 (unit)',
                   showgrid=True, gridcolor='#f0f0f0'),
        legend=dict(orientation='h', y=-0.22, x=0),
        height=380, margin=dict(l=0, r=80, t=45, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
        hovermode='x unified',
    )
    return fig


# ──────────────────────────────────────────────────────────────────
#  분석 2: 헌혈률 장기 추세 + 위기 시점 예측
# ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_yearly_donors():
    return pd.read_csv(f'{PROCESSED}/yearly_donors.csv')

def chart_donation_trend():
    """헌혈률 10년 추세 및 선형 회귀 기반 미래 예측"""
    import numpy as np
    from scipy.stats import linregress

    df   = load_yearly_donors()
    yrs  = df['year'].values
    rate = df['실제 국민 헌혈률 (%)'].values
    tots = df['총 헌혈실적 (건)'].values

    # 선형 회귀
    slope, intercept, r_val, _, _ = linregress(yrs, rate)
    proj_years = list(range(2015, 2041))
    proj_rate  = [slope * y + intercept for y in proj_years]

    # 위기 시점 (3.0% / 2.5% 도달)
    crisis_pts = {}
    for threshold in [3.0, 2.5, 2.0]:
        yr = next((y for y, r in zip(proj_years, proj_rate) if r < threshold), None)
        if yr:
            crisis_pts[threshold] = yr

    fig = go.Figure()

    # 실제 헌혈 건수 (보조 y축, 막대)
    fig.add_trace(go.Bar(
        x=yrs, y=tots,
        name='총 헌혈실적 (건)',
        marker_color='rgba(100,181,246,0.5)',
        yaxis='y2',
        hovertemplate='%{x}년<br>헌혈실적: <b>%{y:,}건</b><extra></extra>'
    ))

    # 실제 헌혈률 (선)
    fig.add_trace(go.Scatter(
        x=yrs, y=rate,
        name='실제 국민 헌혈률 (%)',
        line=dict(color='#c62828', width=2.5),
        mode='lines+markers',
        marker=dict(size=7, color='#c62828', line=dict(color='white', width=2)),
        hovertemplate='%{x}년<br>헌혈률: <b>%{y:.2f}%</b><extra></extra>'
    ))

    # 회귀선 + 예측 구간
    fig.add_trace(go.Scatter(
        x=proj_years, y=proj_rate,
        name=f'추세선 (R²={r_val**2:.2f})',
        line=dict(color='#ff6d00', width=1.8, dash='dash'),
        hovertemplate='%{x}년 예측: <b>%{y:.2f}%</b><extra></extra>'
    ))

    # 현재 / 미래 구분선
    fig.add_vline(x=2025.5, line_dash='dot', line_color='gray', line_width=1.2)
    fig.add_annotation(x=2026, y=max(rate) * 0.97,
                       text='← 실제  |  예측 →',
                       showarrow=False, font=dict(size=9, color='gray'))

    # 위기 임계선 + annotation
    for threshold, color, label in [
        (3.0, '#ff6d00', '주의선 3.0%'),
        (2.5, '#d50000', '위기선 2.5%'),
    ]:
        fig.add_hline(y=threshold, line_dash='longdash', line_color=color, line_width=1.3,
                      annotation_text=label, annotation_position='bottom right',
                      annotation_font_size=9, annotation_font_color=color)
        if threshold in crisis_pts:
            fig.add_annotation(
                x=crisis_pts[threshold], y=threshold,
                text=f'⚠️ {crisis_pts[threshold]}년',
                showarrow=True, arrowhead=2, arrowcolor=color,
                font=dict(size=10, color=color), ay=-30
            )

    fig.update_layout(
        title=dict(text='국민 헌혈률 장기 추세 및 위기 시점 예측 (2015–2040)',
                   font=dict(size=13, color='#333')),
        xaxis=dict(showgrid=False, dtick=2),
        yaxis=dict(title='헌혈률 (%)', range=[2.0, 5.0],
                   showgrid=True, gridcolor='#f0f0f0'),
        yaxis2=dict(title='총 헌혈실적 (건)', overlaying='y', side='right',
                    showgrid=False, tickformat=',d'),
        legend=dict(orientation='h', y=-0.2),
        height=400, margin=dict(l=0, r=60, t=45, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
        hovermode='x unified',
    )
    return fig, crisis_pts


# ──────────────────────────────────────────────────────────────────
#  분석 3: 공급-수요 갭 분석
# ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_monthly_donation_cached():
    return pd.read_csv(f'{PROCESSED}/monthly_donation.csv', parse_dates=['date'])

def chart_supply_demand():
    """월별 헌혈량(공급)과 RBC 보유량 변화(수요 초과 여부) 분석"""
    import numpy as np

    # RBC 월별 보유량
    mbt = pd.read_csv(f'{PROCESSED}/monthly_inventory_by_type.csv', parse_dates=['date'])
    rbc = mbt[mbt['component_code'] == 'RBC'][['date','inventory']].sort_values('date').reset_index(drop=True)
    rbc['delta'] = rbc['inventory'].diff()   # 전월 대비 증감

    # 월별 헌혈 (공급)
    don = load_monthly_donation_cached().sort_values('date').reset_index(drop=True)

    # 병합
    merged = pd.merge(
        rbc[['date','inventory','delta']],
        don[['date','donation']],
        on='date', how='inner'
    ).dropna()

    # 정규화 (헌혈 건수 → 0–100 스케일)
    don_norm = (merged['donation'] - merged['donation'].min()) / \
               (merged['donation'].max() - merged['donation'].min()) * 100
    inv_norm = (merged['inventory'] - merged['inventory'].min()) / \
               (merged['inventory'].max() - merged['inventory'].min()) * 100

    fig = go.Figure()

    # RBC 보유량 영역
    fig.add_trace(go.Scatter(
        x=merged['date'], y=merged['inventory'],
        name='RBC 보유량 (unit)',
        fill='tozeroy', fillcolor='rgba(198,40,40,0.1)',
        line=dict(color='#c62828', width=1.5),
        yaxis='y',
        hovertemplate='%{x|%Y-%m}<br>보유량: <b>%{y:,} unit</b><extra></extra>'
    ))

    # 월별 헌혈 건수 (막대, 보조 y축)
    colors_bar = ['#c62828' if d < 0 else '#43a047' for d in merged['delta']]
    fig.add_trace(go.Bar(
        x=merged['date'], y=merged['donation'],
        name='월별 헌혈 건수',
        marker_color='rgba(100,181,246,0.6)',
        yaxis='y2',
        hovertemplate='%{x|%Y-%m}<br>헌혈: <b>%{y:,}건</b><extra></extra>'
    ))

    # 보유량 감소 구간 (수요 > 공급) 하이라이트
    crisis_months = merged[merged['delta'] < -3000]  # 한 달에 3,000 이상 감소
    for _, row in crisis_months.iterrows():
        fig.add_vrect(
            x0=row['date'] - pd.Timedelta(days=15),
            x1=row['date'] + pd.Timedelta(days=15),
            fillcolor='rgba(198,40,40,0.07)',
            layer='below', line_width=0,
        )

    # 월별 순증감 꺾은선 (delta)
    fig.add_trace(go.Scatter(
        x=merged['date'], y=merged['delta'],
        name='월간 보유량 증감',
        line=dict(color='#7b1fa2', width=1.5, dash='dot'),
        yaxis='y',
        hovertemplate='%{x|%Y-%m}<br>증감: <b>%{y:+,} unit</b><extra></extra>'
    ))
    fig.add_hline(y=0, line_dash='solid', line_color='#7b1fa2',
                  line_width=0.8, yref='y')

    fig.update_layout(
        title=dict(text='월별 헌혈량(공급) vs RBC 보유량 변화 — 공급·수요 갭 분석',
                   font=dict(size=13, color='#333')),
        xaxis=dict(showgrid=False),
        yaxis=dict(title='RBC 보유량 / 증감 (unit)',
                   showgrid=True, gridcolor='#f0f0f0'),
        yaxis2=dict(title='월별 헌혈 건수', overlaying='y', side='right',
                    showgrid=False, tickformat=',d'),
        legend=dict(orientation='h', y=-0.2),
        height=400, margin=dict(l=0, r=60, t=45, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
        hovermode='x unified',
        barmode='overlay',
    )

    # 요약 통계
    surplus_months  = int((merged['delta'] > 0).sum())
    deficit_months  = int((merged['delta'] < 0).sum())
    worst_month_idx = merged['delta'].idxmin()
    worst           = merged.loc[worst_month_idx]

    return fig, surplus_months, deficit_months, worst


# ──────────────────────────────────────────────────────────────────
#  폐기량 분석 (정보공개 청구 데이터, 2021-2025 일별)
# ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_waste_by_component():
    return pd.read_csv(f'{PROCESSED}/waste_by_component.csv', parse_dates=['date']).set_index('date')

@st.cache_data
def load_donation_by_component_daily():
    return pd.read_csv(f'{PROCESSED}/donation_by_component.csv', parse_dates=['date']).set_index('date')


@st.cache_data
def get_waste_slopes():
    """제제별 [주간 평균 재고 → 주간 폐기량] 회귀 기울기 (실측 보정 계수).

    혈소판 0.039 / RBC 0.003 — 유통기한이 짧을수록 재고가 폐기로
    전환되는 비율이 높음. What-If 최적화와 캠페인 시뮬레이션에서
    '폐기 전환율' 보정에 사용."""
    inv_w = load_disclosure_inv_by_component().resample('W').mean()
    wst_w = load_waste_by_component().resample('W').sum()
    slopes = {}
    for comp in ['platelet', 'RBC', 'plasma']:
        df = pd.concat([inv_w[comp], wst_w[comp]], axis=1, keys=['inv', 'waste']).dropna()
        slope, _ = np.polyfit(df['inv'], df['waste'], 1)
        slopes[comp] = max(float(slope), 0.0)   # 음수(혈장)는 0으로
    return slopes


def chart_waste_trend(lang='한국어'):
    """분석 A: 연간 폐기량(제제별 스택) + 폐기율 추이"""
    is_ko = (lang == '한국어')
    wst = load_waste_by_component()
    don = load_donation_by_component_daily()

    wy = wst.resample('YE').sum()
    dy = don.resample('YE').sum()
    years = [d.year for d in wy.index]

    don_total  = dy[['whole_blood', 'apheresis_platelet', 'platelet_plasma', 'plasma']].sum(axis=1)
    waste_total= wy[['RBC', 'platelet', 'plasma']].sum(axis=1)
    rate = (waste_total.values / don_total.values) * 100

    fig = go.Figure()
    comp_labels = {'RBC': ('적혈구' if is_ko else 'RBC'),
                   'platelet': ('혈소판' if is_ko else 'Platelet'),
                   'plasma': ('혈장' if is_ko else 'Plasma')}
    colors = {'RBC': '#c62828', 'platelet': '#1565c0', 'plasma': '#2e7d32'}
    for comp in ['RBC', 'platelet', 'plasma']:
        fig.add_trace(go.Bar(
            x=years, y=wy[comp], name=comp_labels[comp],
            marker_color=colors[comp],
            hovertemplate='%{x}<br>' + comp_labels[comp] + ': <b>%{y:,.0f}</b><extra></extra>'
        ))
    fig.add_trace(go.Scatter(
        x=years, y=rate, name=('폐기율(폐기/헌혈)' if is_ko else 'Waste Rate'),
        line=dict(color='#ff6d00', width=2.5), mode='lines+markers',
        marker=dict(size=8), yaxis='y2',
        hovertemplate='%{x}<br>' + ('폐기율' if is_ko else 'Rate') + ': <b>%{y:.2f}%</b><extra></extra>'
    ))
    # 2022 피크 강조
    fig.add_annotation(x=2022, y=rate[1], yref='y2',
                       text=('⚠️ 2022 급등' if is_ko else '⚠️ 2022 spike'),
                       showarrow=True, arrowhead=2, ay=-35,
                       font=dict(size=10, color='#ff6d00'))
    fig.update_layout(
        barmode='stack',
        title=dict(text=('연간 부적격 폐기량 및 폐기율 (2021–2025)'
                         if is_ko else 'Annual Blood Disposal & Waste Rate (2021–2025)'),
                   font=dict(size=13, color='#333')),
        yaxis=dict(title=('폐기량 (unit)' if is_ko else 'Waste (unit)'),
                   tickformat=',d', showgrid=True, gridcolor='#f0f0f0'),
        yaxis2=dict(title=('폐기율 (%)' if is_ko else 'Rate (%)'),
                    overlaying='y', side='right', showgrid=False, range=[0, 6]),
        legend=dict(orientation='h', y=-0.18),
        height=380, margin=dict(l=0, r=50, t=45, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
        hovermode='x unified',
    )
    return fig


def chart_waste_scatter(lang='한국어'):
    """분석 B (핵심): 제제별 주간 재고 vs 폐기 산점도 — 유통기한 그라데이션"""
    from plotly.subplots import make_subplots
    from scipy.stats import pearsonr
    is_ko = (lang == '한국어')

    inv = load_disclosure_inv_by_component()
    wst = load_waste_by_component()
    inv_w = inv.resample('W').mean()
    wst_w = wst.resample('W').sum()

    comps = [
        ('platelet', ('혈소판 · 유통기한 5일' if is_ko else 'Platelet · 5d'),  '#c62828'),
        ('RBC',      ('적혈구 · 유통기한 35일' if is_ko else 'RBC · 35d'),     '#1565c0'),
        ('plasma',   ('혈장 · 유통기한 1년'   if is_ko else 'Plasma · 1y'),    '#9e9e9e'),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[c[1] for c in comps],
                        horizontal_spacing=0.08)

    stats = {}
    for i, (comp, label, color) in enumerate(comps, start=1):
        df = pd.concat([inv_w[comp], wst_w[comp]], axis=1, keys=['inv', 'waste']).dropna()
        r, p = pearsonr(df['inv'], df['waste'])
        slope, intercept = np.polyfit(df['inv'], df['waste'], 1)
        stats[comp] = {'r': r, 'p': p, 'slope': slope}

        fig.add_trace(go.Scatter(
            x=df['inv'], y=df['waste'], mode='markers',
            marker=dict(size=4, color=color, opacity=0.45),
            name=label, showlegend=False,
            hovertemplate=('재고 %{x:,.0f}<br>폐기 %{y:,.0f}<extra></extra>'
                           if is_ko else 'Stock %{x:,.0f}<br>Waste %{y:,.0f}<extra></extra>'),
        ), row=1, col=i)
        # 회귀선
        xs = np.linspace(df['inv'].min(), df['inv'].max(), 50)
        fig.add_trace(go.Scatter(
            x=xs, y=slope * xs + intercept, mode='lines',
            line=dict(color=color, width=2.5), showlegend=False, hoverinfo='skip',
        ), row=1, col=i)
        # r값 annotation (row/col 지정 시 plotly가 올바른 축으로 매핑)
        sig = '***' if p < 0.001 else ('*' if p < 0.05 else ' n.s.')
        fig.add_annotation(
            text=f'r = {r:+.2f}{sig}',
            xref='x domain', yref='y domain',
            x=0.05, y=0.95, showarrow=False,
            font=dict(size=12, color=color, family='Arial Black'),
            row=1, col=i,
        )

    fig.update_layout(
        title=dict(text=('주간 평균 재고 vs 주간 폐기량 — 유통기한이 짧을수록 강한 결합'
                         if is_ko else 'Weekly Stock vs Waste — Shorter Shelf Life, Tighter Coupling'),
                   font=dict(size=13, color='#333')),
        height=340, margin=dict(l=0, r=10, t=70, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
    )
    for ann in fig['layout']['annotations'][:3]:
        ann['font'] = dict(size=11, color='#333')
    return fig, stats


def chart_waste_lag(lang='한국어'):
    """분석 C: 헌혈(공급) → 혈소판 폐기 시차 상관"""
    from scipy.stats import pearsonr
    is_ko = (lang == '한국어')

    don = load_donation_by_component_daily()
    wst = load_waste_by_component()
    d = don['whole_blood'].rolling(7).mean().dropna()
    w = wst['platelet'].rolling(7).mean().dropna()

    lags = list(range(0, 15))
    rs = []
    for lag in lags:
        a = d.shift(lag).dropna()
        common = a.index.intersection(w.index)
        r, _ = pearsonr(a[common], w[common])
        rs.append(r)

    colors = ['#c62828' if l <= 5 else '#90a4ae' for l in lags]
    fig = go.Figure(go.Bar(
        x=lags, y=rs, marker_color=colors,
        hovertemplate=('lag %{x}일: r=%{y:.3f}<extra></extra>'
                       if is_ko else 'lag %{x}d: r=%{y:.3f}<extra></extra>'),
    ))
    fig.add_vrect(x0=-0.5, x1=5.5, fillcolor='rgba(198,40,40,0.06)', line_width=0,
                  annotation_text=('혈소판 유통기한(5일) 구간' if is_ko else 'Platelet shelf life (5d)'),
                  annotation_position='top right', annotation_font_size=9)
    fig.update_layout(
        title=dict(text=('헌혈량 → 혈소판 폐기 시차 상관 (7일 이동평균)'
                         if is_ko else 'Donation → Platelet Waste Lag Correlation (7d MA)'),
                   font=dict(size=13, color='#333')),
        xaxis=dict(title=('시차 (일)' if is_ko else 'Lag (days)'), dtick=1),
        yaxis=dict(title='r', showgrid=True, gridcolor='#f0f0f0'),
        height=320, margin=dict(l=0, r=10, t=45, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
        showlegend=False,
    )
    return fig


# ──────────────────────────────────────────────────────────────────
#  What-If 최적화: Newsvendor 정책 비교 차트
# ──────────────────────────────────────────────────────────────────
def chart_optimization(opt_result, lang='한국어'):
    """캠페인 정책별 총비용(TC) + 부족/폐기 분해 막대 차트"""
    policies = opt_result['policies']
    best_name = opt_result['best']['name']

    names   = [p['name'] for p in policies]
    short_c = [p['E_shortage'] * opt_result['params']['Cu'] for p in policies]
    waste_c = [p['E_waste']    * opt_result['params']['Co'] for p in policies]

    short_lbl = '부족 비용' if lang == '한국어' else 'Shortage Cost'
    waste_lbl = '폐기 비용' if lang == '한국어' else 'Waste Cost'

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=short_c, name=short_lbl,
        marker_color='#d50000',
        hovertemplate='%{x}<br>' + short_lbl + ': <b>%{y:,.0f}</b><extra></extra>'
    ))
    fig.add_trace(go.Bar(
        x=names, y=waste_c, name=waste_lbl,
        marker_color='#ffb300',
        hovertemplate='%{x}<br>' + waste_lbl + ': <b>%{y:,.0f}</b><extra></extra>'
    ))

    # 최적 정책 표시 (총비용 위에 ★)
    for p in policies:
        tc = p['total_cost']
        is_best = (p['name'] == best_name)
        fig.add_annotation(
            x=p['name'], y=tc,
            text=f"★ {tc:,.0f}" if is_best else f"{tc:,.0f}",
            showarrow=False, yshift=14,
            font=dict(size=11 if is_best else 9,
                      color='#2e7d32' if is_best else '#666',
                      family='Arial Black' if is_best else 'Arial')
        )

    fig.update_layout(
        barmode='stack',
        title=dict(text=('캠페인 정책별 총비용 (TC = Cu·부족 + Co·폐기)'
                         if lang == '한국어' else
                         'Total Cost by Policy (TC = Cu·Short + Co·Waste)'),
                   font=dict(size=13, color='#333')),
        yaxis=dict(title='총비용' if lang == '한국어' else 'Total Cost',
                   showgrid=True, gridcolor='#f0f0f0'),
        legend=dict(orientation='h', y=-0.18),
        height=380, margin=dict(l=0, r=10, t=70, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# 언어 초기화 (세션 유지)
# ══════════════════════════════════════════════════════════════════
if 'lang' not in st.session_state:
    st.session_state['lang'] = '한국어'

# ══════════════════════════════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    # 언어 토글 (최상단)
    lang = st.radio(
        TEXT['한국어']['lang_label'],
        options=['한국어', 'English'],
        index=0 if st.session_state['lang'] == '한국어' else 1,
        horizontal=True,
    )
    if lang != st.session_state['lang']:
        st.session_state['lang'] = lang
        st.rerun()

    T = TEXT[lang]   # 현재 언어 텍스트 단축 참조

    st.markdown(f"### {T['sidebar_title']}")
    st.caption(f"{T['sidebar_caption']} · 노우찬 · 손준영 · 민예지")
    st.divider()
    run_btn = st.button(T['run_btn'], type="primary", use_container_width=True)
    st.divider()

    llm_status = "✅ gpt-4o-mini" if USE_LLM else "⚠️ Rule-based"
    st.caption(f"LLM: {llm_status}  |  📡 bloodinfo.net")

    with st.expander(T['sidebar_krc']):
        st.markdown(T['krc_table'])

    with st.expander(T['sidebar_pipeline']):
        st.markdown("🔵 Sensing → 🟡 Forecasting → 🟠 Risk → 🔴 Action → 📄 Report")

# ══════════════════════════════════════════════════════════════════
# 메인 UI
# ══════════════════════════════════════════════════════════════════
T = TEXT[st.session_state['lang']]   # 최신 언어 반영

st.markdown(f"## 🩸 {T['page_title']}")
st.caption(T['subtitle'])

# 실행
if run_btn:
    with st.status("...", expanded=True) as status:
        for msg in ["🔵 Sensing — bloodinfo.net",
                    "🟡 Forecasting", "🟠 Risk", "🔴 Action", "📄 Report"]:
            st.write(msg); time.sleep(0.25)
        result = run_pipeline()
        status.update(label="✅", state="complete", expanded=False)
    st.session_state['result'] = result
    st.rerun()

# 결과 없을 때
if 'result' not in st.session_state:
    st.info(T['no_result'], icon="👈")
    st.plotly_chart(chart_historical(3), use_container_width=True)
    st.stop()

# ── 결과 대시보드 ────────────────────────────────────────────────
result   = st.session_state['result']
risk     = result['risk_level']
src      = T['src_live'] if result.get('scrape_success') else T['src_csv']
min_days = min((r['days'] for r in result['forecast_14d']), default=0)

# KPI 카드
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(T['kpi_stock'],
              f"{result['current_inventory']:,} unit",
              f"{result['rbc_days']}{'일분' if lang=='한국어' else 'd'}  {src}")
with c2:
    st.metric(f"{RISK_EMOJI.get(risk,'❓')} {T['kpi_risk']}",
              risk, f"{'점수' if lang=='한국어' else 'Score'} {result['risk_score']}/100")
with c3:
    st.metric(T['kpi_min'],
              f"{min_days}{'일분' if lang=='한국어' else 'd'}", result['forecast_min_date'])
with c4:
    st.metric(T['kpi_action'],
              result.get('intervention_level', 'NONE'),
              result['current_season'])

st.caption(f"{result['historical_context']}  ·  {result['last_data_date']}")

# ── 위험 평가 근거 (AI 추론) ─────────────────────────────────────
reasoning = result.get('risk_reasoning', [])
if reasoning:
    method = result.get('risk_method', 'formula')
    method_label = T['risk_by_llm'] if method == 'LLM' else T['risk_by_formula']
    edge = RISK_COLOR.get(risk, '#888')
    items = ''.join(f'<li style="margin-bottom:3px;">{r}</li>' for r in reasoning)
    key   = result.get('risk_key_factor', '')
    key_html = (f'<div style="margin-top:8px;padding:6px 10px;background:rgba(0,0,0,0.04);'
                f'border-radius:6px;"><b>⚠️ {T["risk_key_factor"]}:</b> {key}</div>'
                if key else '')
    with st.expander(f"{T['risk_reasoning_title']}  ·  {method_label}", expanded=False):
        st.markdown(
            f'<div class="scenario-card" style="border-left:4px solid {edge};">'
            f'<ul style="margin:0;padding-left:20px;">{items}</ul>'
            f'{key_html}</div>',
            unsafe_allow_html=True
        )

st.divider()

# 메인 차트 2열
col_l, col_r = st.columns([3, 2])
with col_l:
    tab_plt, tab_fc, tab_comp, tab_btype = st.tabs(
        [T['tab_plt_fc'], T['tab_total_fc'], T['tab_comp_fc'], T['tab_btype_fc']])
    with tab_plt:
        st.plotly_chart(chart_platelet_forecast(result, lang), use_container_width=True)
        st.caption('유통기한 5일 — 가장 시급한 제제. 보유율 100% = 1일 소요량(4,572 unit). '
                   '농축혈소판 일별 데이터(2021–2025) 학습 + 오늘 실측 앵커링.'
                   if lang == '한국어' else
                   'Shelf life 5 days — the most urgent component. 100% rate = 1-day demand (4,572). '
                   'Trained on daily 2021–2025 data, anchored to today.')
    with tab_fc:
        st.plotly_chart(chart_forecast(result, lang), use_container_width=True)
        st.caption('Holt-Winters(damped, 주간 계절성) + 실시간 앵커링 · 14일 백테스트 MAPE 9.1%'
                   if lang == '한국어' else
                   'Holt-Winters (damped, weekly seasonality) + live anchoring · 14-day backtest MAPE 9.1%')
    with tab_comp:
        st.plotly_chart(chart_component_forecast(result, lang), use_container_width=True)
        st.caption('적십자사 정보공개 청구로 확보한 2021–2025 일별 데이터 기반 · RBC는 오늘 실측값 앵커링(★)'
                   if lang == '한국어' else
                   'Based on daily 2021–2025 data from KRC information disclosure · RBC anchored to today (★)')
    with tab_btype:
        st.plotly_chart(chart_blood_type_forecast(result, lang), use_container_width=True)
        st.caption('혈액형별 일별 보유량(전체 제제 합산, 2021–2025) 기반 14일 예측 · 정보공개 청구 데이터'
                   if lang == '한국어' else
                   'Blood-type daily inventory (all components, 2021–2025) from KRC disclosure data')
with col_r:
    tab1, tab2 = st.tabs([T['tab_blood_type'], T['tab_comp_stock']])
    with tab1:
        st.plotly_chart(chart_blood_types(result), use_container_width=True)
    with tab2:
        st.plotly_chart(chart_components(result), use_container_width=True)

st.divider()

# 운영 보고서 (현재 유력 / 시나리오별)
bg     = RISK_BG.get(risk, '#f8f9fa')
border = RISK_COLOR.get(risk, '#c62828')
st.markdown(f"## {T['report_title']}")

rep_tab1, rep_tab2 = st.tabs([T['report_tab_current'], T['report_tab_scenario']])

# ── 탭 1: 현재 유력 상황 보고서 ──────────────────────────────────
with rep_tab1:
    report_html = result.get("final_report", "").replace("\n", "<br>")
    st.markdown(
        f'<div class="report-box" style="border-left-color:{border};background:{bg};">'
        f'{report_html}'
        f'</div>',
        unsafe_allow_html=True
    )

# ── 탭 2: 시나리오별 대응 보고서 ─────────────────────────────────
with rep_tab2:
    st.caption(T['scenario_caption'])
    cards = build_scenario_reports(result)

    # 현재 등급을 맨 앞으로 정렬 (강조)
    cards_sorted = sorted(cards, key=lambda c: (not c['is_current']))

    for sc in cards_sorted:
        edge   = RISK_COLOR.get(sc['level'], '#888')
        cur_mark = f'  ◀ {T["scenario_current"]}' if sc['is_current'] else ''
        title = f'{sc["emoji"]} {sc["level"]} — RBC {sc["days"]} (~{sc["est_unit"]:,} unit){cur_mark}'

        with st.expander(title, expanded=sc['is_current']):
            measures_html = ''.join(f'<li style="margin-bottom:4px;">{m}</li>' for m in sc['measures'])
            extra = sc['extra_per_day']
            extra_line = (f'목표 헌혈 증가율 <b>{sc["boost"]}</b> 달성 시 '
                          f'<b>일일 약 +{extra:,} unit</b> 추가 확보 예상'
                          if extra > 0 else '추가 캠페인 불필요 (현 수준 유지)')

            st.markdown(
                f'<div class="scenario-card" style="border-left:4px solid {edge};">'
                f'<div style="display:grid;grid-template-columns:90px 1fr;gap:6px 12px;">'
                f'<b>발생 조건</b><span>{sc["trigger"]}</span>'
                f'<b>발령 채널</b><span>{sc["channel"]}</span>'
                f'<b>대상</b><span>{sc["target"]}</span>'
                f'<b>집행 시점</b><span>{sc["timing"]}</span>'
                f'</div>'
                f'<div style="margin-top:10px;"><b>세부 실행 조치</b>'
                f'<ul style="margin:6px 0 0 0;padding-left:20px;">{measures_html}</ul></div>'
                f'<div style="margin-top:10px;padding:8px 12px;background:rgba(0,0,0,0.04);'
                f'border-radius:6px;color:{edge} !important;">'
                f'🎯 {extra_line}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

# 에이전트 로그
with st.expander(T['log_title'], expanded=False):
    st.markdown("\n".join(result.get('agent_logs', [])))

# 심층 분석
st.divider()
st.markdown(f"## {T['analysis_title']}")

tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs(
    [T['tab_campaign'], T['tab_trend'], T['tab_gap'], T['tab_whatif'], T['tab_waste']])

with tab_a:
    st.markdown("#### 헌혈 캠페인 강도별 14일 보유량 변화 시뮬레이션")
    st.caption("캠페인으로 헌혈이 증가하면 14일 후 보유량이 얼마나 달라지는지 비교합니다.")
    st.plotly_chart(chart_campaign_sim(result), use_container_width=True)

    # D+14 비교 카드 (+ 실측 기울기 기반 예상 추가 폐기)
    rbc_days  = result.get('rbc_days', 0)
    daily_need = result['current_inventory'] / rbc_days if rbc_days > 0 else 5052
    fc_base_14 = pd.DataFrame(result['forecast_14d'])['forecast'].iloc[-1]
    slope_rbc  = get_waste_slopes()['RBC']   # 재고 1unit·1주당 폐기 전환 (실측)

    c1, c2, c3, c4 = st.columns(4)
    for col, (label, rate) in zip(
        [c1, c2, c3, c4],
        [("기준 (현재)", 0.0), ("+5% SMS", 0.05), ("+10% SNS", 0.10), ("+15% 긴급", 0.15)]
    ):
        val = fc_base_14 + daily_need * rate * 14
        days_val = round(val / daily_need, 1)
        # 추가 재고는 14일 동안 0→max로 선형 증가 → 평균은 절반, 기간 2주
        avg_extra  = daily_need * rate * 14 / 2
        waste_add  = slope_rbc * avg_extra * 2
        delta = (f"{days_val}일분 · 폐기 +{waste_add:,.0f}" if rate > 0
                 else f"{days_val}일분")
        col.metric(label, f"{val:,.0f} unit", delta,
                   delta_color=("inverse" if rate > 0 else "normal"))

    st.caption(
        f"🗑️ 예상 추가 폐기 = 실측 재고-폐기 회귀 기울기(RBC {slope_rbc:.4f}/주) × 평균 추가 재고 × 2주. "
        f"RBC는 유통기한 35일로 전환율이 낮음 — 혈소판이라면 동일 조건에서 약 "
        f"{get_waste_slopes()['platelet']/slope_rbc:.0f}배의 폐기가 발생합니다."
    )

with tab_b:
    st.markdown("#### 국민 헌혈률 장기 하락 추세와 구조적 위기 시점")
    st.caption("선형 회귀로 현재 추세를 연장해 헌혈률이 위험 수준에 도달하는 시기를 예측합니다.")

    try:
        from scipy.stats import linregress as _lr
        fig_trend, crisis_pts = chart_donation_trend()
        st.plotly_chart(fig_trend, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        df_yr = load_yearly_donors()
        c1.metric("2015년 헌혈률", f"{df_yr.iloc[0]['실제 국민 헌혈률 (%)']:.2f}%")
        c2.metric("2025년 헌혈률", f"{df_yr.iloc[-1]['실제 국민 헌혈률 (%)']:.2f}%",
                  f"{df_yr.iloc[-1]['실제 국민 헌혈률 (%)']-df_yr.iloc[0]['실제 국민 헌혈률 (%)']:+.2f}%p (10년)")
        crisis_3 = crisis_pts.get(3.0)
        c3.metric("3.0% 도달 예상", f"{crisis_3}년" if crisis_3 else "추세 내 없음",
                  "⚠️ 구조적 위기")
    except ImportError:
        st.warning("scipy가 설치되지 않았습니다. `pip install scipy`를 실행해주세요.")

with tab_c:
    st.markdown("#### 월별 헌혈(공급) vs RBC 보유량 변화(수요 초과 여부)")
    st.caption("보유량이 감소한 달(수요 > 공급)은 붉은 배경으로 표시됩니다.")

    fig_gap, surplus, deficit, worst = chart_supply_demand()
    st.plotly_chart(fig_gap, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("공급 > 수요 월 수", f"{surplus}개월", "보유량 증가")
    c2.metric("수요 > 공급 월 수", f"{deficit}개월", "보유량 감소", delta_color="inverse")
    c3.metric("최대 급감 월",
              worst['date'].strftime('%Y년 %m월'),
              f"{worst['delta']:+,.0f} unit")

# ── 탭 D: What-If 최적화 (Newsvendor) ────────────────────────────
with tab_d:
    is_ko = (lang == '한국어')
    st.markdown("#### " + ("혈액형별 부족 시나리오 → 최적 캠페인 도출 (Newsvendor 모델)"
                           if is_ko else
                           "Blood-Type Shortage Scenario → Optimal Campaign (Newsvendor)"))
    st.caption("가상 부족량을 입력하면, 부족·폐기 비용을 최소화하는 캠페인 강도를 계산합니다."
               if is_ko else
               "Enter a hypothetical shortage to compute the cost-minimizing campaign tier.")

    # ── 입력 폼 ──────────────────────────────────────────────────
    ci1, ci2, ci3, ci4 = st.columns(4)
    with ci1:
        wf_btype = st.selectbox("혈액형" if is_ko else "Blood Type",
                                ['A', 'B', 'O', 'AB'], index=1)
    with ci2:
        wf_comp = st.selectbox("제제" if is_ko else "Component",
                               ['RBC', 'PLT'], index=1,
                               format_func=lambda c: T['comp'][c])
    with ci3:
        wf_gap = st.number_input("예측 부족량 (unit)" if is_ko else "Shortage (unit)",
                                 min_value=10, max_value=10000, value=300, step=50)
    with ci4:
        wf_days = st.slider("대응 기간 (일)" if is_ko else "Response Days",
                            min_value=1, max_value=7, value=3)

    # 비용 비율 설정 (고급)
    with st.expander("⚙️ " + ("비용 파라미터 (Cu : Co)" if is_ko else "Cost Parameters (Cu : Co)")):
        cc1, cc2 = st.columns(2)
        wf_cu = cc1.number_input("Cu (부족 비용, 생명 직결)" if is_ko else "Cu (Underage)",
                                 min_value=10, max_value=500, value=100, step=10)
        wf_co = cc2.number_input("Co (폐기+캠페인 비용)" if is_ko else "Co (Overage)",
                                 min_value=1, max_value=200, value=10, step=5)

    # ── 최적화 실행 ──────────────────────────────────────────────
    total_need = {'RBC': 5052, 'PLT': 4572}[wf_comp]
    daily_need = estimate_daily_need(wf_btype, total_need)

    # 폐기 전환율 (실측 보정): 혈소판=1.0 기준, RBC는 재고-폐기 기울기 비율
    slopes = get_waste_slopes()
    if wf_comp == 'PLT':
        waste_conv = 1.0
    else:
        waste_conv = round(min(max(slopes['RBC'] / slopes['platelet'], 0.0), 1.0), 3)

    opt = optimize_response(
        shortage_gap=wf_gap, daily_need=daily_need,
        response_days=wf_days, Cu=wf_cu, Co=wf_co,
        waste_conversion=waste_conv,
    )
    best = opt['best']

    # ── 결과 KPI ─────────────────────────────────────────────────
    st.markdown("---")
    k1, k2, k3 = st.columns(3)
    k1.metric("✅ " + ("최적 캠페인" if is_ko else "Optimal Tier"),
              best['name'], best['channel'])
    k2.metric("💰 " + ("최소 총비용" if is_ko else "Min Total Cost"),
              f"{best['total_cost']:,.0f}",
              f"{'부족' if is_ko else 'Short'} {best['E_shortage']:.0f} / "
              f"{'폐기' if is_ko else 'Waste'} {best['E_waste']:.0f}")
    k3.metric("🎯 Critical Ratio",
              f"{opt['critical_ratio']:.3f}",
              f"{'목표 서비스' if is_ko else 'Service'} {opt['critical_ratio']*100:.0f}%")

    # ── 정책 비교 차트 ───────────────────────────────────────────
    st.plotly_chart(chart_optimization(opt, lang), use_container_width=True)

    # ── 정책 상세 테이블 ─────────────────────────────────────────
    import pandas as _pd
    rows = []
    for p in opt['policies']:
        rows.append({
            ('정책' if is_ko else 'Policy'):        p['name'],
            ('채널' if is_ko else 'Channel'):       p['channel'],
            ('기대 공급' if is_ko else 'Supply'):    f"{p['supply_mu']:,.0f}",
            ('E[부족]' if is_ko else 'E[Short]'):    f"{p['E_shortage']:.1f}",
            ('E[폐기]' if is_ko else 'E[Waste]'):    f"{p['E_waste']:.1f}",
            ('총비용' if is_ko else 'Total Cost'):   f"{p['total_cost']:,.0f}",
            ('서비스%' if is_ko else 'Service%'):    f"{p['service_level']*100:.0f}%",
        })
    df_opt = _pd.DataFrame(rows)
    st.dataframe(df_opt, use_container_width=True, hide_index=True)

    st.caption(
        f"📐 Newsvendor: TC = Cu·E[부족] + Co·E[폐기]  |  "
        f"Cu={wf_cu}, Co={wf_co}, Monte Carlo 5,000회  |  "
        f"{T['comp'][wf_comp]} {wf_btype}형 1일 소요 ≈ {daily_need:.0f} unit  |  "
        f"폐기 전환율 κ={waste_conv} ({'혈소판 5일 기준' if wf_comp == 'PLT' else 'RBC 재고-폐기 실측 기울기 비율 보정'})"
        if is_ko else
        f"📐 Newsvendor: TC = Cu·E[Short] + Co·E[Waste]  |  "
        f"Cu={wf_cu}, Co={wf_co}, 5,000 Monte Carlo runs  |  "
        f"waste conversion κ={waste_conv} (calibrated from 2021–2025 stock-waste regression)"
    )

# ── 탭 E: 폐기량 분석 (정보공개 데이터) ──────────────────────────
with tab_e:
    is_ko = (lang == '한국어')
    st.markdown("#### " + ("혈액 폐기 실태 분석 — 유통기한과 폐기의 구조적 결합"
                           if is_ko else
                           "Blood Disposal Analysis — Shelf Life & Waste Coupling"))
    st.caption("적십자사 정보공개 청구 데이터(2021–2025 일별 부적격 폐기량, 유효기간 경과 포함) 기반 분석입니다."
               if is_ko else
               "Based on KRC disclosure data (2021–2025 daily disposal, incl. expired units).")

    # ── 핵심 차트: 재고 vs 폐기 산점도 ───────────────────────────
    fig_scatter, w_stats = chart_waste_scatter(lang)

    # KPI 3개
    wst_y = load_waste_by_component().resample('YE').sum()
    don_y = load_donation_by_component_daily().resample('YE').sum()
    waste_2025 = wst_y.iloc[-1][['RBC', 'platelet', 'plasma']].sum()
    don_2025   = don_y.iloc[-1][['whole_blood','apheresis_platelet','platelet_plasma','plasma']].sum()
    plt_r = w_stats['platelet']['r']

    k1, k2, k3 = st.columns(3)
    k1.metric("🗑️ " + ("2025 연간 폐기량" if is_ko else "2025 Annual Waste"),
              f"{waste_2025:,.0f} unit",
              ("적혈구+혈소판+혈장" if is_ko else "RBC+PLT+Plasma"))
    k2.metric("📊 " + ("2025 폐기율" if is_ko else "2025 Waste Rate"),
              f"{waste_2025/don_2025*100:.2f}%",
              ("폐기 ÷ 헌혈" if is_ko else "Waste ÷ Donation"))
    k3.metric("🔗 " + ("혈소판 재고↔폐기 상관" if is_ko else "PLT Stock↔Waste"),
              f"r = {plt_r:+.2f}",
              "p < 0.001 ***")

    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(("유통기한 5일인 혈소판만 재고와 폐기가 유의하게 결합(r=%.2f***) — "
                "재고가 쌓일수록 폐기가 늘어나는 구조를 실데이터로 확인. "
                "적혈구(35일)·혈장(1년)은 무관." % plt_r)
               if is_ko else
               ("Only platelets (5-day shelf life) show significant stock-waste coupling (r=%.2f***). "
                "RBC (35d) and plasma (1y) show none." % plt_r))

    st.markdown("---")
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.plotly_chart(chart_waste_trend(lang), use_container_width=True)
        st.caption("2022년 폐기율 4.37% 급등 — 코로나 회복기 캠페인 과열 시기와 일치 (Reactive Panic의 실측 사례)"
                   if is_ko else
                   "2022 waste rate spiked to 4.37% — coinciding with post-COVID campaign surges (Reactive Panic).")
    with col_w2:
        st.plotly_chart(chart_waste_lag(lang), use_container_width=True)
        st.caption("헌혈 급증 후 5일(혈소판 유통기한) 내 폐기 상관이 집중 — 공급 스파이크가 폐기로 이어지는 경로"
                   if is_ko else
                   "Correlation concentrates within 5 days (platelet shelf life) after donation spikes.")

st.divider()
if st.button(T['refresh_btn']):
    del st.session_state['result']
    st.rerun()
