"""
optimizer.py
Newsvendor 모델 기반 혈액 캠페인 대응 최적화

특정 혈액형·제제의 예상 부족량을 입력하면,
캠페인 강도(Tier)별 Monte Carlo 시뮬레이션으로
총비용(부족비용 + 폐기비용)을 최소화하는 정책을 도출한다.

OM 이론 근거:
- Newsvendor Model (단일 기간 재고 문제)
- 총비용  TC = Cu·E[Shortage] + Co·E[Waste]
- Critical Ratio  CR = Cu / (Cu + Co)  → 최적 서비스 수준
"""

import numpy as np

# 혈액형별 1일 소요량 비율 (대한적십자사 실측 기반, RBC 기준)
BLOOD_TYPE_RATIO = {'A': 0.34, 'B': 0.26, 'O': 0.28, 'AB': 0.12}

# 캠페인 정책 후보 (강도별)
CAMPAIGN_TIERS = [
    {'name': '무대응',       'boost': 0.00, 'channel': '모니터링만'},
    {'name': 'Tier 1 (+5%)', 'boost': 0.05, 'channel': 'SMS 정기헌혈자'},
    {'name': 'Tier 2 (+10%)','boost': 0.10, 'channel': 'SNS 집중 캠페인'},
    {'name': 'Tier 3 (+15%)','boost': 0.15, 'channel': '전채널 긴급 캠페인'},
]


def optimize_response(
    shortage_gap: float,
    daily_need: float,
    response_days: int = 5,
    Cu: float = 100.0,       # Underage cost (부족 비용, 생명 직결)
    Co: float = 10.0,        # Overage cost (폐기 + 캠페인 비용)
    demand_cv: float = 0.15, # 수요(부족분) 변동계수
    supply_cv: float = 0.25, # 캠페인 공급 변동계수
    n_sim: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Newsvendor + Monte Carlo로 최적 캠페인 정책 도출

    Parameters
    ----------
    shortage_gap : 예상 부족량 (unit)
    daily_need   : 해당 혈액형의 1일 소요량 (unit)
    response_days: 대응 기간 (일)
    Cu, Co       : 부족/과잉 단위 비용
    demand_cv    : 수요 불확실성 (변동계수)
    supply_cv    : 캠페인 공급 불확실성

    Returns
    -------
    dict : {policies, best, critical_ratio, params}
    """
    rng = np.random.default_rng(seed)

    # 수요(필요량) 분포: 부족분을 평균으로
    demand_mu    = float(shortage_gap)
    demand_sigma = max(demand_mu * demand_cv, 1e-6)

    results = []
    for tier in CAMPAIGN_TIERS:
        # 캠페인 기대 공급 = 강도 × 1일소요량 × 대응기간
        supply_mu    = tier['boost'] * daily_need * response_days
        supply_sigma = max(supply_mu * supply_cv, 1e-6)

        # Monte Carlo 시뮬레이션
        demand = rng.normal(demand_mu, demand_sigma, n_sim)
        demand = np.maximum(demand, 0)
        if supply_mu > 0:
            supply = rng.normal(supply_mu, supply_sigma, n_sim)
            supply = np.maximum(supply, 0)
        else:
            supply = np.zeros(n_sim)

        shortage = np.maximum(demand - supply, 0)   # 여전히 부족
        waste    = np.maximum(supply - demand, 0)   # 과잉 유입 → 폐기

        E_short = float(shortage.mean())
        E_waste = float(waste.mean())
        TC      = Cu * E_short + Co * E_waste

        # 서비스 수준 = 부족이 0인 시뮬레이션 비율
        service_level = float((shortage <= 0).mean())

        results.append({
            'name':          tier['name'],
            'boost':         tier['boost'],
            'channel':       tier['channel'],
            'supply_mu':     round(supply_mu, 0),
            'E_shortage':    round(E_short, 1),
            'E_waste':       round(E_waste, 1),
            'total_cost':    round(TC, 0),
            'service_level': round(service_level, 3),
            # 시뮬레이션 분포 일부 (시각화용, 500개만)
            '_shortage_dist': shortage[:500].tolist(),
        })

    best = min(results, key=lambda r: r['total_cost'])
    critical_ratio = Cu / (Cu + Co)

    return {
        'policies':       results,
        'best':           best,
        'critical_ratio': round(critical_ratio, 3),
        'params': {
            'shortage_gap':  shortage_gap,
            'daily_need':    daily_need,
            'response_days': response_days,
            'Cu':            Cu,
            'Co':            Co,
        },
    }


def estimate_daily_need(blood_type: str, total_daily_need: float = 5052.0) -> float:
    """혈액형별 1일 소요량 추정 (전체 소요량 × 혈액형 비율)"""
    return total_daily_need * BLOOD_TYPE_RATIO.get(blood_type, 0.25)


# ── 직접 실행 테스트 ──────────────────────────────────────────────
if __name__ == "__main__":
    # B형 혈소판 100 unit 부족 시나리오
    daily_need = estimate_daily_need('B', total_daily_need=4572)  # PLT 1일 소요
    out = optimize_response(shortage_gap=100, daily_need=daily_need, response_days=5)

    print(f"Critical Ratio = {out['critical_ratio']} "
          f"(목표 서비스수준 {out['critical_ratio']*100:.1f}%)\n")
    print(f"{'정책':<14}{'공급μ':>8}{'E[부족]':>10}{'E[폐기]':>10}{'TC':>12}{'서비스':>8}")
    for r in out['policies']:
        mark = ' ★' if r['name'] == out['best']['name'] else ''
        print(f"{r['name']:<14}{r['supply_mu']:>8.0f}{r['E_shortage']:>10.1f}"
              f"{r['E_waste']:>10.1f}{r['total_cost']:>12.0f}{r['service_level']*100:>7.0f}%{mark}")
    print(f"\n→ 최적 정책: {out['best']['name']} (TC={out['best']['total_cost']:.0f})")
