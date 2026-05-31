"""
scraper.py
bloodinfo.net 실시간 스크래핑 모듈

대한적십자사 혈액정보 사이트에서 오늘의 혈액 보유 현황을 수집합니다.
- 적혈구제제(RBC): 혈액형별 보유량 + 보유일수
- 혈소판제제(PLT): 혈액형별 보유량 + 보유율
- KRC 공식 위험 등급 자동 판정
"""

import re
import requests
from bs4 import BeautifulSoup
from datetime import date

BLOODINFO_URL = "https://www.bloodinfo.net/knrcbs/bi/info/bldHoldSttus.do"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    )
}

# KRC 공식 위험 단계 (적혈구 보유일수 기준)
KRC_RISK_LEVELS = [
    {"level": "CRITICAL", "label": "🔴 심각", "days_below": 1,
     "color": "#d50000", "action": "긴급 전 채널 캠페인 즉시 실행 (+15%)"},
    {"level": "WARNING",  "label": "🟠 경계", "days_below": 2,
     "color": "#ff6d00", "action": "SNS 캠페인 + 단체헌혈 협약기관 긴급 연락 (+10%)"},
    {"level": "CAUTION",  "label": "🟡 주의", "days_below": 3,
     "color": "#ffb300", "action": "정기헌혈자 SMS 발송 + 헌혈의집 이벤트 운영 (+5%)"},
    {"level": "WATCH",    "label": "🔵 관심", "days_below": 5,
     "color": "#1565c0", "action": "헌혈 독려 캠페인 운영 및 모니터링 강화"},
    {"level": "NORMAL",   "label": "🟢 정상", "days_below": None,
     "color": "#4caf50", "action": "정상 모니터링 유지"},
]


# ── 파싱 유틸리티 ─────────────────────────────────────────────────
def _to_float(s: str) -> float | None:
    """'29,018' → 29018.0 / '5.7일' → 5.7 / '200%' → 200.0"""
    try:
        return float(re.sub(r"[^\d.]", "", s))
    except (ValueError, TypeError):
        return None


def _classify_krc_risk(days: float) -> dict:
    """적혈구 보유일수로 KRC 공식 위험 등급 반환 (dict)"""
    for lvl in KRC_RISK_LEVELS:
        if lvl["days_below"] is None or days < lvl["days_below"]:
            return lvl
    return KRC_RISK_LEVELS[-1]

def classify_krc_risk(days: float) -> str:
    """적혈구 보유일수 → 위험 등급 문자열 반환 (app.py 등 외부 호출용)"""
    if days < 1: return "CRITICAL"
    if days < 2: return "WARNING"
    if days < 3: return "CAUTION"
    if days < 5: return "WATCH"
    return "NORMAL"


# ── 메인 스크래핑 함수 ────────────────────────────────────────────
def scrape_blood_inventory() -> dict:
    """
    bloodinfo.net에서 현재 혈액 보유 현황 수집

    Returns
    -------
    dict
        {
          'date'           : '2026-05-31',
          'rbc'            : {
              'total_units'      : 29018,    # 전체 RBC 보유량 (unit)
              'total_days'       : 5.7,      # 전체 보유일수
              'daily_need'       : 5052,     # 1일 소요량
              'units_by_type'    : {A/B/O/AB: unit},
              'days_by_type'     : {A/B/O/AB: days},
          },
          'plt'            : {
              'total_units'      : 1599,
              'total_rate'       : 240.0,    # 보유율 (%)
              'daily_need'       : 4572,
              'units_by_type'    : {A/B/O/AB: unit},
              'rate_by_type'     : {A/B/O/AB: %},
          },
          'krc_risk'       : {level / label / color / action / days_below},
        }
    """
    try:
        res = requests.get(BLOODINFO_URL, headers=HEADERS, timeout=10)
        res.raise_for_status()
    except Exception as e:
        raise ConnectionError(f"bloodinfo.net 접속 실패: {e}")

    soup   = BeautifulSoup(res.text, "html.parser")
    tables = soup.find_all("table")

    if len(tables) < 3:
        raise ValueError("페이지 구조가 변경되었습니다 — 스크래핑 로직 점검 필요")

    blood_types = ["A", "B", "O", "AB"]

    # ── 적혈구(RBC) ───────────────────────────────────────────────
    rbc = {}
    for row in tables[0].find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
        if not cells:
            continue
        if "1일 소요량" in cells[0]:
            rbc["daily_need"]          = _to_float(cells[1])
            rbc["daily_need_by_type"]  = {t: _to_float(cells[i + 2]) for i, t in enumerate(blood_types)}
        elif "현재 혈액보유량" in cells[0]:
            rbc["total_units"]         = _to_float(cells[1])
            rbc["units_by_type"]       = {t: _to_float(cells[i + 2]) for i, t in enumerate(blood_types)}
        elif "보유상태" in cells[0]:
            rbc["total_days"]          = _to_float(cells[1])
            rbc["days_by_type"]        = {t: _to_float(cells[i + 2]) for i, t in enumerate(blood_types)}

    # ── 혈소판(PLT) ───────────────────────────────────────────────
    plt_data = {}
    for row in tables[2].find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
        if not cells:
            continue
        if "농축혈소판" in cells[0] and "1일 소요량" in cells[1]:
            plt_data["daily_need"]         = _to_float(cells[2])
            plt_data["daily_need_by_type"] = {t: _to_float(cells[i + 3]) for i, t in enumerate(blood_types)}
        elif "현재 혈액보유량" in cells[0]:
            plt_data["total_units"]        = _to_float(cells[1])
            plt_data["units_by_type"]      = {t: _to_float(cells[i + 2]) for i, t in enumerate(blood_types)}
        elif "보유율" in cells[0]:
            plt_data["total_rate"]         = _to_float(cells[1])
            plt_data["rate_by_type"]       = {t: _to_float(cells[i + 2]) for i, t in enumerate(blood_types)}

    # ── KRC 위험 등급 ─────────────────────────────────────────────
    krc_risk = _classify_krc_risk(rbc.get("total_days", 999))

    return {
        "date":     str(date.today()),
        "rbc":      rbc,
        "plt":      plt_data,
        "krc_risk": krc_risk,
    }


# ── 직접 실행 테스트 ──────────────────────────────────────────────
if __name__ == "__main__":
    import json
    data = scrape_blood_inventory()
    print(f"=== bloodinfo.net 실시간 데이터 ({data['date']}) ===\n")
    print(f"[RBC 적혈구]")
    print(f"  전체 보유량 : {data['rbc']['total_units']:,.0f} unit")
    print(f"  보유일수    : {data['rbc']['total_days']} 일")
    print(f"  1일 소요량  : {data['rbc']['daily_need']:,.0f} unit")
    print(f"  혈액형별    : {data['rbc']['units_by_type']}")
    print(f"\n[PLT 혈소판]")
    print(f"  전체 보유량 : {data['plt']['total_units']:,.0f} 개")
    print(f"  보유율      : {data['plt']['total_rate']} %")
    print(f"\n[KRC 위험 등급]")
    print(f"  {data['krc_risk']['label']}  ({data['rbc']['total_days']}일분)")
    print(f"  권고 대응   : {data['krc_risk']['action']}")
