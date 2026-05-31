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

warnings.filterwarnings('ignore')
load_dotenv()

# ── 경로 설정 ─────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(BASE, 'processed')

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
    border-left: 4px solid #c62828;
    border-radius: 6px;
    padding: 16px 20px;
    font-size: 0.95rem;
    line-height: 1.7;
    white-space: pre-wrap;
}
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
    scrape_success: bool     # 스크래핑 성공 여부
    forecast_14d: list; forecast_min_value: int; forecast_min_date: str
    shortage_probability: float; days_until_warning: int
    risk_level: str; risk_score: int; component_risks: dict; historical_context: str
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
        plt_rate  = live['plt']['total_rate']
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
        daily_need_avg = 5052  # 역사적 평균 1일 소요량
        rbc_days = round(cur / daily_need_avg, 1)

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
        'scrape_success':     scrape_ok,
        'agent_logs':         logs,
    }

def forecasting_agent(state):
    logs = state.get('agent_logs', [])
    logs.append("🟡 **Forecasting Agent** — 오늘 실제값 기반 14일 예측")

    today_actual = state['current_inventory']
    rbc_days     = state['rbc_days']

    # 역사적 일별 패턴
    df  = load_daily_inventory()
    ts  = df.set_index('date')['inventory'].sort_index()
    today     = pd.Timestamp.today().normalize()
    today_doy = today.dayofyear
    daily_avg = ts.groupby(ts.index.dayofyear).mean()
    daily_std = ts.groupby(ts.index.dayofyear).std().fillna(0)
    hist_today  = daily_avg.get(today_doy, daily_avg.mean())
    daily_need  = (today_actual / rbc_days) if rbc_days > 0 else 5052

    # 14일 예측: 오늘 실제값 + 역사적 계절 변화량
    fc_list = []
    for k in range(1, 15):
        future_date = today + pd.Timedelta(days=k)
        future_doy  = future_date.dayofyear
        hist_future = daily_avg.get(future_doy, daily_avg.mean())
        delta       = hist_future - hist_today
        fc_val      = max(0, round(today_actual + delta))
        fc_std      = float(daily_std.get(future_doy, daily_std.mean()))
        fc_days     = round(fc_val / daily_need, 1) if daily_need > 0 else 0
        fc_list.append({
            'date':     future_date.strftime('%Y-%m-%d'),
            'forecast': fc_val,
            'lower_95': max(0, round(fc_val - 1.96 * fc_std)),
            'upper_95': round(fc_val + 1.96 * fc_std),
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

def risk_agent(state):
    logs = state.get('agent_logs', [])
    logs.append("🟠 **Risk Agent** — KRC 공식 기준 위험 점수 산출")

    rbc_days = state['rbc_days']      # 오늘 실제 보유일수
    fc_days_list = [r['days'] for r in state['forecast_14d']]
    fc_min_days  = min(fc_days_list) if fc_days_list else rbc_days
    d_warn   = state['days_until_warning']
    is_risk  = state['is_risk_season']
    sp       = state['shortage_probability']

    # ── 위험 점수 (KRC 일수 기준, 0-100) ─────────────────────────
    score = 0
    # 현재 보유일수 (0-40점)
    if   rbc_days < 1: score += 40
    elif rbc_days < 2: score += 30
    elif rbc_days < 3: score += 20
    elif rbc_days < 5: score += 10

    # 예측 최저 보유일수 (0-30점)
    if   fc_min_days < 1: score += 30
    elif fc_min_days < 2: score += 22
    elif fc_min_days < 3: score += 14
    elif fc_min_days < 5: score += 6

    # 경고 도달 여유 (0-20점)
    if   d_warn <= 2:  score += 20
    elif d_warn <= 5:  score += 15
    elif d_warn <= 10: score += 8

    # 부족 확률 (0-5점)
    score += int(sp * 5)
    # 위험 시즌 (0-5점)
    if is_risk: score += 5

    score = min(100, score)

    # KRC 공식 위험 등급 (일수 기반)
    lvl = classify_krc_risk(rbc_days)
    mbt = load_monthly_by_type()
    lm = mbt['date'].max(); ld = mbt[mbt['date']==lm]
    ha = mbt[mbt['date'].dt.month==lm.month].groupby('component_code')['inventory'].mean()
    comp={}
    for _,row in ld.iterrows():
        code=row['component_code']; val=row['inventory']
        avg=ha.get(code,val); ratio=val/avg if avg>0 else 1.0
        thr=0.85 if code in('PLT','SDP') else 0.75
        level='WARNING' if ratio<thr-0.10 else('CAUTION' if ratio<thr else 'NORMAL')
        comp[code]={'level':level,'ratio':round(ratio,3),'value':int(val)}
    df = load_daily_inventory(); ts = df.set_index('date')['inventory']
    try:
        prev=ts[ts.index.year==(ts.index.max().year-1)].iloc[-1]; yd=cur-int(prev)
        ctx=f'전년 동기 대비 {yd:+,} unit ({yd/prev*100:+.1f}%)'
    except: ctx='전년 동기 데이터 없음'
    logs.append(f"  → 위험 점수: **{score}/100**  |  등급: {RISK_EMOJI[lvl]} **{lvl}**")
    logs.append(f"  → 전년 동기: {ctx}")
    return {'risk_level':lvl,'risk_score':score,'component_risks':comp,'historical_context':ctx,'agent_logs':logs}

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
        prompt=f"""대한적십자사 혈액 관리 운영 보고서를 작성하세요.
현황: {state['last_data_date']} 보유량 {state['current_inventory']:,} unit, {state['trend_7d_direction']} ({state['trend_7d_rate']:+.0f}/일), {state['historical_context']}
위험: {state['risk_level']} ({state['risk_score']}/100), 제제: {comp_str}
예측: 14일 최저 {state['forecast_min_value']:,} unit ({state['forecast_min_date']}), 부족확률 {state['shortage_probability']*100:.0f}%
권고: {state.get('intervention_level','')} — {state.get('recommended_action','')}

[현황] → [예측] → [권고] 구조로 200자 내외 한국어 보고서 작성."""
        try:
            report=llm.invoke([HumanMessage(content=prompt)]).content.strip()
        except:
            report=_rule_report(state)
    else:
        report=_rule_report(state)
    logs.append("  → 보고서 생성 완료 ✅")
    return {'final_report':report,'agent_logs':logs}

def _rule_report(s):
    rl={'NORMAL':'정상','CAUTION':'주의','WARNING':'경고','CRITICAL':'위기'}
    am=_ACTION_RULES.get(s.get('risk_level','NORMAL'),_ACTION_RULES['NORMAL'])
    return (f"[현황] {s['last_data_date']} 기준 혈액 보유량 {s['current_inventory']:,} unit. "
            f"최근 7일 추세 {s['trend_7d_direction']} ({s['trend_7d_rate']:+.0f} unit/일). {s['historical_context']}.\n\n"
            f"[예측] 향후 14일 내 최저 보유량 {s['forecast_min_value']:,} unit 예상 ({s['forecast_min_date']}). "
            f"부족 확률 {s['shortage_probability']*100:.0f}%.\n\n"
            f"[권고] 위험 등급 {rl.get(s['risk_level'],s['risk_level'])} — {am[1]}")

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
        'intervention_level':  'NONE',
        'recommended_action':  '',
        'action_reasoning':    '',
        'final_report':'','agent_logs':[],
    }
    return app.invoke(init)

# ══════════════════════════════════════════════════════════════════
# 차트 함수
# ══════════════════════════════════════════════════════════════════
def chart_forecast(result):
    fc_df    = pd.DataFrame(result['forecast_14d'])
    fc_dates = pd.to_datetime(fc_df['date'])
    fc_vals  = fc_df['forecast'].values

    today      = pd.Timestamp.today().normalize()
    today_val  = result['current_inventory']

    # 역사적 데이터에서 최근 30일치만 추출 (같은 연도 내 데이터)
    daily  = load_daily_inventory()
    ts_all = daily.set_index('date')['inventory'].sort_index()
    cutoff = today - pd.Timedelta(days=30)
    recent = ts_all[ts_all.index >= cutoff]

    fig = go.Figure()

    # ── 최근 30일 역사 데이터 (있는 경우) ──────────────────────
    if len(recent) > 0:
        fig.add_trace(go.Scatter(
            x=recent.index, y=recent.values,
            name='실제 보유량 (최근 30일)',
            line=dict(color='#1565c0', width=2.0),
            hovertemplate='%{x|%Y-%m-%d}<br>%{y:,} unit<extra></extra>'
        ))

    # ── 오늘 스크래핑 값 (앵커 포인트) ──────────────────────────
    fig.add_trace(go.Scatter(
        x=[today], y=[today_val],
        name=f'오늘 실측 ({today.strftime("%m/%d")})',
        mode='markers',
        marker=dict(color='#1565c0', size=10, symbol='circle',
                    line=dict(color='white', width=2)),
        hovertemplate=f'{today.strftime("%Y-%m-%d")}<br>{today_val:,} unit<extra></extra>'
    ))

    # ── 오늘 → 예측 연결선 ───────────────────────────────────────
    connect_x = [today] + list(fc_dates)
    connect_y = [today_val] + list(fc_vals)
    fig.add_trace(go.Scatter(
        x=connect_x, y=connect_y,
        name='14일 예측',
        line=dict(color='#c62828', width=2.2, dash='dash'),
        mode='lines+markers',
        marker=dict(size=5, color='#c62828'),
        hovertemplate='%{x|%Y-%m-%d}<br>예측: %{y:,} unit<extra></extra>'
    ))

    # ── 신뢰구간 ────────────────────────────────────────────────
    if 'lower_95' in fc_df.columns:
        fig.add_trace(go.Scatter(
            x=list(fc_dates) + list(fc_dates[::-1]),
            y=list(fc_df['upper_95']) + list(fc_df['lower_95'])[::-1],
            fill='toself', fillcolor='rgba(198,40,40,0.10)',
            line=dict(color='rgba(0,0,0,0)'),
            name='95% 신뢰구간',
            hoverinfo='skip'
        ))

    # ── KRC 위험 임계선 ─────────────────────────────────────────
    x_start = (recent.index[0] if len(recent) > 0 else today - pd.Timedelta(days=5))
    x_end   = fc_dates.iloc[-1]
    daily_need = today_val / result['rbc_days'] if result.get('rbc_days', 0) > 0 else 5052

    for label, days, color, dash in [
        ('관심 (5일분)',  5, '#1565c0', 'dot'),
        ('주의 (3일분)',  3, '#ffb300', 'dot'),
        ('경계 (2일분)',  2, '#ff6d00', 'dash'),
        ('심각 (1일분)',  1, '#d50000', 'longdash'),
    ]:
        threshold_val = days * daily_need
        fig.add_shape(type='line', x0=x_start, x1=x_end,
                      y0=threshold_val, y1=threshold_val,
                      line=dict(color=color, width=1.2, dash=dash))
        fig.add_annotation(x=x_end, y=threshold_val, text=label,
                           showarrow=False, xanchor='right',
                           font=dict(size=9, color=color), yshift=6)

    # ── 예측 시작 구분선 ─────────────────────────────────────────
    fig.add_vline(x=today, line_width=1.3, line_dash='dash', line_color='#555')
    fig.add_annotation(
        x=today, y=max(connect_y) * 1.04,
        text='오늘 →', showarrow=False,
        font=dict(size=10, color='#555'), xanchor='left'
    )

    # ── x축 범위: 30일 전 ~ 14일 후 ─────────────────────────────
    x_min = today - pd.Timedelta(days=32)
    x_max = fc_dates.iloc[-1] + pd.Timedelta(days=1)

    fig.update_layout(
        title=dict(text='혈액 보유량 14일 예측 (오늘 실측 기준)', font=dict(size=14)),
        xaxis=dict(showgrid=False, range=[x_min, x_max]),
        yaxis=dict(tickformat=',d', title='보유량 (unit)', rangemode='tozero'),
        legend=dict(orientation='h', y=-0.22),
        height=380,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor='white', plot_bgcolor='#fafafa',
        hovermode='x unified',
    )
    return fig

def chart_components(result):
    comp = result['component_risks']
    names = list(comp.keys())
    ratios= [v['ratio']*100 for v in comp.values()]
    colors= [RISK_COLOR.get(v['level'],'#4caf50') for v in comp.values()]
    labels= {'RBC':'농축적혈구','PLT':'농축혈소판','FFP':'신선동결혈장','SDP':'성분채혈혈소판'}

    fig = go.Figure(go.Bar(
        x=[labels.get(n,n) for n in names], y=ratios,
        marker_color=colors, marker_line_color='white', marker_line_width=1.5,
        text=[f'{v:.0f}%' for v in ratios], textposition='outside',
        hovertemplate='%{x}<br>%{y:.1f}%<extra></extra>'
    ))
    for val, color, label in [
        (100, '#9e9e9e', '역사 평균'),
        (85,  '#ffb300', '혈소판 기준'),
        (75,  '#ff6d00', '일반 기준'),
    ]:
        fig.add_hline(y=val, line_dash='dot', line_color=color, line_width=1.3,
                      annotation_text=label, annotation_position='top right',
                      annotation_font_size=9)
    fig.update_layout(
        title=dict(text='제제별 보유량 (역사 평균 대비 %)', font=dict(size=13)),
        yaxis=dict(range=[0, 140], title='%'),
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='white', plot_bgcolor='#fafafa',
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

# ══════════════════════════════════════════════════════════════════
# 메인 UI
# ══════════════════════════════════════════════════════════════════

# ── 사이드바 ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩸 혈액 공급 운영 대시보드")
    st.markdown("**UNIST OM Term Project**  \nTeam 2: 노우찬 · 손준영 · 민예지")
    st.divider()

    run_btn = st.button("▶ 에이전트 파이프라인 실행",
                        type="primary", use_container_width=True)
    st.caption("5개 에이전트가 순차 실행되어 보고서를 생성합니다.")
    st.divider()

    with st.expander("⚙️ 시스템 정보"):
        llm_status = "✅ gpt-4o-mini (OpenAI)" if USE_LLM else "⚠️ Rule-based (API 키 없음)"
        st.markdown(f"**LLM**: {llm_status}")
        st.markdown("**모델**: Holt-Winters (MAPE 10.6%)")
        st.markdown("**임계값**")
        st.markdown(f"- 🟢 정상: ≥ 20,000 unit")
        st.markdown(f"- 🟡 주의: ≥ 15,000 unit")
        st.markdown(f"- 🟠 경고: ≥ 10,000 unit")
        st.markdown(f"- 🔴 위기: < 10,000 unit")

    with st.expander("📊 에이전트 구조"):
        st.markdown("""
```
START
 → 🔵 Sensing Agent
 → 🟡 Forecasting Agent
 → 🟠 Risk Agent
    ├─[CAUTION+]→ 🔴 Action Agent
    └─[NORMAL] ─┐
 → 📄 Report Agent
END
```""")

# ── 메인 ─────────────────────────────────────────────────────────
st.markdown("# 🩸 혈액 공급 운영 대시보드")
st.caption("혈액 부족 예측 및 헌혈 운영 최적화 — AI Multi-Agent System")
st.divider()

# 실행 버튼 처리
if run_btn:
    with st.status("🩸 에이전트 파이프라인 실행 중...", expanded=True) as status:
        st.write("🔵 Sensing Agent — 데이터 수집 중...")
        time.sleep(0.3)
        st.write("🟡 Forecasting Agent — 예측 로드 중...")
        time.sleep(0.3)
        st.write("🟠 Risk Agent — 위험 점수 산출 중...")
        time.sleep(0.3)
        st.write("🔴 Action Agent — 대응 방안 결정 중...")
        time.sleep(0.3)
        st.write("📄 Report Agent — 보고서 생성 중...")
        result = run_pipeline()
        status.update(label="✅ 파이프라인 완료!", state="complete", expanded=False)
    st.session_state['result'] = result
    st.rerun()

# 결과 표시
if 'result' not in st.session_state:
    # 초기 화면
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**: 왼쪽 사이드바에서\n'에이전트 파이프라인 실행' 클릭")
    with col2:
        st.info("**Step 2**: 5개 에이전트가 순차 실행되며\n혈액 현황을 분석합니다")
    with col3:
        st.info("**Step 3**: 14일 예측, 위험 등급,\n헌혈 캠페인 권고안이 생성됩니다")

    st.markdown("---")
    st.subheader("📈 역사적 혈액 보유량 추이 (사전 탐색)")
    st.plotly_chart(chart_historical(3), use_container_width=True)

else:
    result = st.session_state['result']
    risk   = result['risk_level']

    # ── KPI 카드 ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        src = "📡 실시간" if result.get('scrape_success') else "📂 CSV"
        st.metric("🩸 현재 보유량",
                  f"{result['current_inventory']:,} unit",
                  f"{result['rbc_days']}일분  ({src})")
    with c2:
        st.metric(f"{RISK_EMOJI.get(risk,'❓')} 위험 등급",
                  risk,
                  f"점수 {result['risk_score']}/100")
    with c3:
        min_days = min((r['days'] for r in result['forecast_14d']), default=0)
        st.metric("📉 14일 예측 최저",
                  f"{min_days}일분",
                  result['forecast_min_date'])
    with c4:
        lvl = result.get('intervention_level', 'NONE')
        st.metric("📢 권고 대응",
                  lvl,
                  result['current_season'])

    st.markdown(f"*{result['historical_context']}  |  기준일: {result['last_data_date']}*")
    st.divider()

    # ── 메인 차트 ────────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.plotly_chart(chart_forecast(result), use_container_width=True)
    with col_right:
        st.plotly_chart(chart_components(result), use_container_width=True)

    # ── 에이전트 로그 + 역사 비교 ─────────────────────────────────
    col_log, col_hist = st.columns([1, 1])
    with col_log:
        with st.expander("🤖 에이전트 실행 로그", expanded=True):
            log_text = "\n".join(result.get('agent_logs', []))
            st.markdown(log_text)
    with col_hist:
        st.plotly_chart(chart_historical(3), use_container_width=True)

    # ── 최종 보고서 ───────────────────────────────────────────────
    st.divider()
    st.subheader("📄 운영 보고서")
    report_text = result.get('final_report', '')
    bg = RISK_BG.get(risk, '#f8f9fa')
    border = RISK_COLOR.get(risk, '#c62828')
    st.markdown(
        f'<div class="report-box" style="border-left-color:{border};background:{bg};">'
        f'{report_text}'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── 재실행 버튼 ───────────────────────────────────────────────
    st.divider()
    if st.button("🔄 다시 실행", use_container_width=False):
        del st.session_state['result']
        st.rerun()
