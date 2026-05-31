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
    """오늘 실측값 → 14일 예측 차트 (오늘부터만 표시)"""
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
        title=dict(text=f'📅 오늘({today.strftime("%Y-%m-%d")}) 실측 기준  14일 예측',
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
        height=340, margin=dict(l=0, r=40, t=45, b=0),
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


# ══════════════════════════════════════════════════════════════════
# 메인 UI
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🩸 혈액 공급 운영 대시보드")
    st.caption("UNIST OM Team 2 · 노우찬 · 손준영 · 민예지")
    st.divider()
    run_btn = st.button("▶ 분석 실행", type="primary", use_container_width=True)
    st.divider()

    llm_status = "✅ gpt-4o-mini" if USE_LLM else "⚠️ Rule-based"
    st.caption(f"LLM: {llm_status}  |  데이터: 📡 bloodinfo.net")

    with st.expander("KRC 위험 단계 기준"):
        st.markdown("| 단계 | 적혈구 보유일수 |\n|------|----------------|\n| 🟢 정상 | 5일 이상 |\n| 🔵 관심 | 3~5일 |\n| 🟡 주의 | 2~3일 |\n| 🟠 경계 | 1~2일 |\n| 🔴 심각 | 1일 미만 |")

    with st.expander("에이전트 파이프라인"):
        st.markdown("🔵 Sensing → 🟡 Forecasting → 🟠 Risk → 🔴 Action → 📄 Report")

# ══════════════════════════════════════════════════════════════════
# 메인 UI
# ══════════════════════════════════════════════════════════════════
st.markdown("## 🩸 혈액 공급 운영 대시보드")
st.caption("대한적십자사 혈액정보 실시간 연동 · AI 멀티에이전트 예측 시스템")

# 실행
if run_btn:
    with st.status("에이전트 실행 중...", expanded=True) as status:
        for msg in ["🔵 Sensing — bloodinfo.net 스크래핑",
                    "🟡 Forecasting — 14일 예측 생성",
                    "🟠 Risk — KRC 위험 등급 판정",
                    "🔴 Action — 대응 방안 결정",
                    "📄 Report — 운영 보고서 작성"]:
            st.write(msg); time.sleep(0.25)
        result = run_pipeline()
        status.update(label="✅ 분석 완료", state="complete", expanded=False)
    st.session_state['result'] = result
    st.rerun()

# ── 결과 없을 때 초기 화면 ───────────────────────────────────────
if 'result' not in st.session_state:
    st.info("← 왼쪽 **분석 실행** 버튼을 눌러 에이전트를 시작하세요.", icon="👈")
    st.plotly_chart(chart_historical(3), use_container_width=True)
    st.stop()

# ── 결과 대시보드 ────────────────────────────────────────────────
result = st.session_state['result']
risk   = result['risk_level']
src    = "📡 실시간" if result.get('scrape_success') else "📂 CSV"
min_days = min((r['days'] for r in result['forecast_14d']), default=0)

# KPI 카드 4개
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🩸 현재 보유량",
              f"{result['current_inventory']:,} unit",
              f"{result['rbc_days']}일분  {src}")
with c2:
    st.metric(f"{RISK_EMOJI.get(risk,'❓')} 위험 등급",
              risk, f"점수 {result['risk_score']}/100")
with c3:
    st.metric("📉 14일 예측 최저",
              f"{min_days}일분", result['forecast_min_date'])
with c4:
    st.metric("📢 권고 대응",
              result.get('intervention_level', 'NONE'),
              result['current_season'])

st.caption(f"{result['historical_context']}  ·  기준: {result['last_data_date']}")
st.divider()

# 메인 차트 2열
col_l, col_r = st.columns([3, 2])
with col_l:
    st.plotly_chart(chart_forecast(result), use_container_width=True)
with col_r:
    tab1, tab2 = st.tabs(["🩸 혈액형별 보유일수", "📊 제제별 보유량"])
    with tab1:
        st.plotly_chart(chart_blood_types(result), use_container_width=True)
    with tab2:
        st.plotly_chart(chart_components(result), use_container_width=True)

st.divider()

# 운영 보고서 (풀 width)
bg     = RISK_BG.get(risk, '#f8f9fa')
border = RISK_COLOR.get(risk, '#c62828')
st.markdown("**📄 운영 보고서**")
st.markdown(
    f'<div class="report-box" style="border-left-color:{border};background:{bg};">'
    f'{result.get("final_report","")}'
    f'</div>',
    unsafe_allow_html=True
)

# 에이전트 로그 (collapsed)
with st.expander("🤖 에이전트 실행 로그", expanded=False):
    st.markdown("\n".join(result.get('agent_logs', [])))

# ── 심층 분석 섹션 ───────────────────────────────────────────────
st.divider()
st.markdown("## 🔬 심층 분석")

tab_a, tab_b, tab_c = st.tabs([
    "💉 캠페인 시뮬레이션",
    "📉 헌혈률 장기 추세",
    "⚖️ 공급-수요 갭",
])

with tab_a:
    st.markdown("#### 헌혈 캠페인 강도별 14일 보유량 변화 시뮬레이션")
    st.caption("캠페인으로 헌혈이 증가하면 14일 후 보유량이 얼마나 달라지는지 비교합니다.")
    st.plotly_chart(chart_campaign_sim(result), use_container_width=True)

    # D+14 비교 카드
    rbc_days  = result.get('rbc_days', 0)
    daily_need = result['current_inventory'] / rbc_days if rbc_days > 0 else 5052
    fc_base_14 = pd.DataFrame(result['forecast_14d'])['forecast'].iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    for col, (label, rate) in zip(
        [c1, c2, c3, c4],
        [("기준 (현재)", 0.0), ("+5% SMS", 0.05), ("+10% SNS", 0.10), ("+15% 긴급", 0.15)]
    ):
        val = fc_base_14 + daily_need * rate * 14
        days_val = round(val / daily_need, 1)
        col.metric(label, f"{val:,.0f} unit", f"{days_val}일분")

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

st.divider()
if st.button("🔄 다시 실행"):
    del st.session_state['result']
    st.rerun()
