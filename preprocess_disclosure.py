"""
preprocess_disclosure.py
적십자사 정보공개 청구 엑셀(붙임2)을 분석용 CSV로 정제

원본: 한 시트에 [혈액형별 표 | 제제별 표] 좌우 분할, 3줄 병합 헤더
출력: processed/ 폴더에 깔끔한 long-format CSV 6개
"""

import os
import pandas as pd

BASE      = os.path.dirname(os.path.abspath(__file__))
XLSX      = os.path.join(BASE, '붙임2. 정보공개 청구(헌혈실적, 보유량, 폐기량 통계).xlsx')
PROCESSED = os.path.join(BASE, 'processed')
os.makedirs(PROCESSED, exist_ok=True)


def _read_sheet(sheet_name, blood_cols, comp_cols):
    """
    한 시트에서 혈액형별 표와 제제별 표를 각각 추출.
    blood_cols / comp_cols : {엑셀_컬럼_index(0-based): 출력_컬럼명}
    데이터는 4행(=index 3)부터 시작.
    """
    raw = pd.read_excel(XLSX, sheet_name=sheet_name, header=None)
    data = raw.iloc[3:].reset_index(drop=True)   # 헤더 3줄 제외

    # ── 혈액형별 표 ──────────────────────────────────────────────
    blood = pd.DataFrame()
    blood['date'] = pd.to_datetime(data.iloc[:, 0], errors='coerce')
    for idx, col in blood_cols.items():
        blood[col] = pd.to_numeric(data.iloc[:, idx], errors='coerce')
    blood = blood.dropna(subset=['date']).reset_index(drop=True)

    # ── 제제별 표 (별도 일자 컬럼 사용) ──────────────────────────
    comp = pd.DataFrame()
    comp['date'] = pd.to_datetime(data.iloc[:, 10], errors='coerce')   # col11 = index 10
    for idx, col in comp_cols.items():
        comp[col] = pd.to_numeric(data.iloc[:, idx], errors='coerce')
    comp = comp.dropna(subset=['date']).reset_index(drop=True)

    return blood, comp


# ── 컬럼 매핑 정의 ────────────────────────────────────────────────
# 혈액형별: col2~9 (index 1~8) = Rh(+) O/A/B/AB, Rh(-) O/A/B/AB
BLOOD_MAP = {
    1: 'O_pos', 2: 'A_pos', 3: 'B_pos', 4: 'AB_pos',
    5: 'O_neg', 6: 'A_neg', 7: 'B_neg', 8: 'AB_neg',
}

# 제제별 매핑 (시트마다 다름)
COMP_MAP_DONATION = {     # 헌혈실적: col12~15 (index 11~14)
    11: 'whole_blood', 12: 'apheresis_platelet',
    13: 'platelet_plasma', 14: 'plasma',
}
COMP_MAP_INV_WASTE = {    # 보유량/폐기량: col12~18 (index 11~17)
    11: 'RBC', 12: 'platelet', 13: 'plasma',
    14: 'F_platelet', 15: 'washed_platelet',
    16: 'apheresis_plasma', 17: 'etc',
}


def add_blood_type_totals(df):
    """Rh(+)와 Rh(-)를 합쳐 혈액형별 합계 컬럼 추가 + 전체 합계"""
    for bt in ['O', 'A', 'B', 'AB']:
        df[bt] = df[f'{bt}_pos'] + df[f'{bt}_neg']
    df['total'] = df[['O', 'A', 'B', 'AB']].sum(axis=1)
    return df


def main():
    print('=' * 60)
    print('  정보공개 청구 데이터 전처리')
    print('=' * 60)

    jobs = [
        ('헌혈실적',   'donation',  BLOOD_MAP, COMP_MAP_DONATION),
        ('혈액보유량', 'inventory', BLOOD_MAP, COMP_MAP_INV_WASTE),
        ('혈액폐기량', 'waste',     BLOOD_MAP, COMP_MAP_INV_WASTE),
    ]

    summary = []
    for sheet, prefix, bmap, cmap in jobs:
        blood, comp = _read_sheet(sheet, bmap, cmap)
        blood = add_blood_type_totals(blood)

        # 저장
        bpath = os.path.join(PROCESSED, f'{prefix}_by_type.csv')       # 혈액형별
        cpath = os.path.join(PROCESSED, f'{prefix}_by_component.csv')  # 제제별
        blood.to_csv(bpath, index=False, encoding='utf-8-sig')
        comp.to_csv(cpath, index=False, encoding='utf-8-sig')

        print(f'\n[{sheet}]')
        print(f'  기간: {blood["date"].min().date()} ~ {blood["date"].max().date()}')
        print(f'  - {prefix}_by_type.csv      ({len(blood):,}일 × {blood.shape[1]}열)')
        print(f'  - {prefix}_by_component.csv ({len(comp):,}일 × {comp.shape[1]}열)')
        summary.append((prefix, len(blood), blood['total'].mean()))

    print('\n' + '=' * 60)
    print('  검증 — 혈액형별 일평균')
    print('=' * 60)
    for prefix, n, avg in summary:
        print(f'  {prefix:<12}: {n:,}일,  전체 일평균 {avg:,.0f}')

    # 폐기량 핵심 지표 (발표용)
    print('\n' + '=' * 60)
    print('  * 폐기량 핵심 지표 (발표 검증용)')
    print('=' * 60)
    waste_c = pd.read_csv(os.path.join(PROCESSED, 'waste_by_component.csv'), parse_dates=['date'])
    inv_c   = pd.read_csv(os.path.join(PROCESSED, 'inventory_by_component.csv'), parse_dates=['date'])
    don_c   = pd.read_csv(os.path.join(PROCESSED, 'donation_by_component.csv'), parse_dates=['date'])

    # 혈소판 폐기율 = 폐기 / (보유 기준 근사). 연도별
    waste_c['year'] = waste_c['date'].dt.year
    plt_waste_yearly = waste_c.groupby('year')['platelet'].sum()
    rbc_waste_yearly = waste_c.groupby('year')['RBC'].sum()
    print('\n  혈소판(platelet) 연간 폐기량:')
    for yr, v in plt_waste_yearly.items():
        print(f'    {yr}: {v:,.0f}')
    print('\n  적혈구(RBC) 연간 폐기량:')
    for yr, v in rbc_waste_yearly.items():
        print(f'    {yr}: {v:,.0f}')

    print('\n[OK] 전처리 완료 — processed/ 폴더에 CSV 6개 저장')


if __name__ == '__main__':
    main()
