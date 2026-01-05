import pandas as pd
import numpy as np
from korean_lunar_calendar import KoreanLunarCalendar
import datetime

"""
개선된 피처 엔지니어링: 생산지별 기온 데이터 활용
- 계절별 주산지 기온을 선택적으로 사용
- 지역별 극한 기온 이벤트 추적
- 생육기간(60~90일 전) 주산지 기온 반영
"""

# 1. 통합 데이터 로드
try:
    df = pd.read_csv('양배추_통합데이터_최종.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)
    print(f"데이터 로드 완료: {df.shape}")
except FileNotFoundError:
    print("양배추_통합데이터_최종.csv 파일을 찾을 수 없습니다.")
    exit()

# -------------------------------------------------------------
# 2. 특수 이벤트 피처 생성
# -------------------------------------------------------------
calendar = KoreanLunarCalendar()

def is_holiday_season(date_obj):
    calendar.setSolarDate(date_obj.year, date_obj.month, date_obj.day)
    lunar_date = calendar.LunarIsoFormat()
    l_month = int(lunar_date[5:7])
    l_day = int(lunar_date[8:10])
    
    if (l_month == 12 and l_day >= 20) or (l_month == 1 and l_day <= 10):
        return 1
    if (l_month == 8 and l_day >= 5 and l_day <= 25):
        return 1
    return 0

def days_to_nearest_holiday(date_obj):
    """명절까지 남은 일수"""
    calendar.setSolarDate(date_obj.year, date_obj.month, date_obj.day)
    lunar_date = calendar.LunarIsoFormat()
    l_month = int(lunar_date[5:7])
    l_day = int(lunar_date[8:10])
    
    if l_month == 12:
        days_to_seol = (30 - l_day) + 1
        if days_to_seol <= 14:
            return days_to_seol
    elif l_month == 1 and l_day <= 10:
        return -l_day
    
    if l_month == 8:
        days_to_chuseok = 15 - l_day
        if abs(days_to_chuseok) <= 10:
            return days_to_chuseok
    
    return 999

# 명절 관련
df['is_holiday_season'] = df['DATE'].apply(is_holiday_season)
df['days_to_holiday'] = df['DATE'].apply(days_to_nearest_holiday)
df['is_pre_holiday'] = (df['days_to_holiday'] > 0) & (df['days_to_holiday'] <= 7)
df['is_pre_holiday'] = df['is_pre_holiday'].astype(int)

# 김장철
df['is_kimjang_season'] = df['DATE'].apply(
    lambda x: 1 if (x.month == 11) or (x.month == 12 and x.day <= 20) else 0
)
df['is_pre_kimjang'] = (df['DATE'].dt.month == 10).astype(int)

# COVID-19
df['is_covid'] = df['DATE'].apply(
    lambda x: 1 if (datetime.date(2020, 2, 1) <= x.date() <= datetime.date(2022, 4, 30)) else 0
)

# 계절 정의
df['month'] = df['DATE'].dt.month
df['is_summer_highland'] = df['month'].isin([6, 7, 8]).astype(int)  # 강원 고랭지
df['is_winter_jeju'] = df['month'].isin([12, 1, 2]).astype(int)      # 제주
df['is_spring_south'] = df['month'].isin([3, 4, 5]).astype(int)      # 남부
df['is_fall_south'] = df['month'].isin([9, 10, 11]).astype(int)      # 가을 남부

print(f"\n✓ 이벤트 피처 생성 완료")

# -------------------------------------------------------------
# 3. ★ 개선: 생산지별 기온 데이터 활용
# -------------------------------------------------------------

# 3-1. 계절별 주산지 기온 선택
# 여름(6-8월): 강원 고랭지 (홍천, 태백)
# 겨울(12-2월): 제주
# 봄/가을: 남부 (목포)

def get_seasonal_temp(row):
    """계절별 주산지의 평균 기온 반환"""
    month = row['month']
    
    if month in [6, 7, 8]:  # 여름 - 고랭지
        return (row['홍천_평균기온'] + row['태백_평균기온']) / 2
    elif month in [12, 1, 2]:  # 겨울 - 제주
        return row['제주_평균기온']
    else:  # 봄/가을 - 남부
        return row['목포_평균기온']

def get_seasonal_temp_max(row):
    """계절별 주산지의 최고 기온 반환"""
    month = row['month']
    
    if month in [6, 7, 8]:
        return max(row['홍천_최고기온'], row['태백_최고기온'])
    elif month in [12, 1, 2]:
        return row['제주_최고기온']
    else:
        return row['목포_최고기온']

def get_seasonal_temp_min(row):
    """계절별 주산지의 최저 기온 반환"""
    month = row['month']
    
    if month in [6, 7, 8]:
        return min(row['홍천_최저기온'], row['태백_최저기온'])
    elif month in [12, 1, 2]:
        return row['제주_최저기온']
    else:
        return row['목포_최저기온']

# 주산지 기온 피처 생성
df['주산지_평균기온'] = df.apply(get_seasonal_temp, axis=1)
df['주산지_최고기온'] = df.apply(get_seasonal_temp_max, axis=1)
df['주산지_최저기온'] = df.apply(get_seasonal_temp_min, axis=1)
df['주산지_일교차'] = df['주산지_최고기온'] - df['주산지_최저기온']

print(f"✓ 계절별 주산지 기온 피처 생성 완료")

# 3-2. 지역별 극한 기온 이벤트
# 각 지역의 폭염/한파 여부를 개별적으로 추적
df['홍천_폭염'] = (df['홍천_최고기온'] >= 33).astype(int)
df['태백_폭염'] = (df['태백_최고기온'] >= 33).astype(int)
df['제주_폭염'] = (df['제주_최고기온'] >= 33).astype(int)
df['목포_폭염'] = (df['목포_최고기온'] >= 33).astype(int)

df['홍천_한파'] = (df['홍천_최저기온'] <= -10).astype(int)
df['태백_한파'] = (df['태백_최저기온'] <= -10).astype(int)
df['제주_한파'] = (df['제주_최저기온'] <= -5).astype(int)  # 제주는 기준 완화
df['목포_한파'] = (df['목포_최저기온'] <= -5).astype(int)

# 주산지 극한 기온 (계절별로 해당 지역만)
df['주산지_폭염'] = 0
df.loc[df['is_summer_highland'] == 1, '주산지_폭염'] = (
    (df['홍천_폭염'] == 1) | (df['태백_폭염'] == 1)
).astype(int)
df.loc[df['is_winter_jeju'] == 1, '주산지_폭염'] = df['제주_폭염']
df.loc[(df['is_spring_south'] == 1) | (df['is_fall_south'] == 1), '주산지_폭염'] = df['목포_폭염']

df['주산지_한파'] = 0
df.loc[df['is_summer_highland'] == 1, '주산지_한파'] = (
    (df['홍천_한파'] == 1) | (df['태백_한파'] == 1)
).astype(int)
df.loc[df['is_winter_jeju'] == 1, '주산지_한파'] = df['제주_한파']
df.loc[(df['is_spring_south'] == 1) | (df['is_fall_south'] == 1), '주산지_한파'] = df['목포_한파']

print(f"✓ 지역별 극한 기온 이벤트 생성 완료")

# 3-3. 생육기간(60~90일 전) 주산지 기온
# 현재 출하되는 양배추는 60~90일 전에 재배된 것
# 그 시기의 주산지 기온이 중요!

df['주산지_기온_lag60'] = df['주산지_평균기온'].shift(60)
df['주산지_기온_lag75'] = df['주산지_평균기온'].shift(75)
df['주산지_기온_lag90'] = df['주산지_평균기온'].shift(90)

# 생육기간 평균 기온
df['생육기_평균기온'] = df['주산지_평균기온'].shift(60).rolling(30).mean()
df['생육기_최고기온'] = df['주산지_최고기온'].shift(60).rolling(30).max()
df['생육기_최저기온'] = df['주산지_최저기온'].shift(60).rolling(30).min()

# 생육기 극한 일수
df['생육기_폭염일수'] = df['주산지_폭염'].shift(60).rolling(30).sum()
df['생육기_한파일수'] = df['주산지_한파'].shift(60).rolling(30).sum()

# 생육기 온도 변동성 (스트레스 지표)
df['생육기_기온변동성'] = df['주산지_평균기온'].shift(60).rolling(30).std()

print(f"✓ 생육기간 기온 피처 생성 완료")

# 3-4. 강수량 처리
rain_cols = [c for c in df.columns if '강수량' in c]
df['avg_rain_5d_sum'] = df[rain_cols].mean(axis=1).rolling(window=5).sum().fillna(0)

# 생육기 강수
df['생육기_강수량'] = df[rain_cols].mean(axis=1).shift(60).rolling(30).sum()

# 3-5. 최근 30일 극한 기온 누적
df['최근30일_폭염일수'] = df['주산지_폭염'].rolling(window=30).sum()
df['최근30일_한파일수'] = df['주산지_한파'].rolling(window=30).sum()

# 원본 지역별 기온 컬럼 제거 (파생 피처만 유지)
drop_weather = [c for c in df.columns if any(x in c for x in ['목포_', '제주_', '태백_', '홍천_']) and '기온' in c]
drop_weather += [c for c in df.columns if '습도' in c]
drop_weather += rain_cols
df = df.drop(columns=drop_weather)

print(f"✓ 날씨 파생 완료 (지역별 특성 반영)")

# -------------------------------------------------------------
# 4. 시계열 & 기술적 지표
# -------------------------------------------------------------
targets = ['특_가격', '상_가격']

for col in targets:
    # Lag
    df[f'{col}_lag7'] = df[col].shift(7)
    df[f'{col}_lag14'] = df[col].shift(14)
    df[f'{col}_lag21'] = df[col].shift(21)
    df[f'{col}_lag28'] = df[col].shift(28)
    df[f'{col}_lag365'] = df[col].shift(365)
    
    # 이동평균
    df[f'{col}_MA7'] = df[col].rolling(window=7).mean()
    df[f'{col}_MA30'] = df[col].rolling(window=30).mean()
    
    # 변동성
    df[f'{col}_std7'] = df[col].rolling(window=7).std()
    df[f'{col}_std30'] = df[col].rolling(window=30).std()
    
    # 가격 변화율
    df[f'{col}_pct_change_7d'] = df[col].pct_change(7)
    df[f'{col}_pct_change_14d'] = df[col].pct_change(14)
    df[f'{col}_yoy_change'] = (df[col] - df[f'{col}_lag365']) / (df[f'{col}_lag365'] + 1e-10)
    
    # RSI
    delta = df[col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df[f'{col}_RSI'] = 100 - (100 / (1 + rs))
    
    # 볼린저 밴드
    ma20 = df[col].rolling(20).mean()
    std20 = df[col].rolling(20).std()
    bb_upper = ma20 + (std20 * 2)
    bb_lower = ma20 - (std20 * 2)
    df[f'{col}_BB_position'] = (df[col] - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    # 가격 레벨
    high_30 = df[col].rolling(30).max()
    low_30 = df[col].rolling(30).min()
    df[f'{col}_level_30d'] = (df[col] - low_30) / (high_30 - low_30 + 1e-10)

print(f"✓ 가격 피처 생성 완료")

# -------------------------------------------------------------
# 5. 외부 변수 가공
# -------------------------------------------------------------
df['volume_MA7'] = df['총반입량(ton)'].rolling(window=7).mean()
df['volume_MA30'] = df['총반입량(ton)'].rolling(window=30).mean()
df['volume_pct_change'] = df['총반입량(ton)'].pct_change(7)
df['supply_demand_ratio'] = df['총반입량(ton)'] / (df['volume_MA30'] + 1e-5)
df['supply_pressure'] = (df['총반입량(ton)'] - df['volume_MA30']) / (df['volume_MA30'] + 1e-5)

total_supply = df['총반입량(ton)'] + df['수입중량(ton)']
df['import_dependency'] = df['수입중량(ton)'] / (total_supply + 1e-5)

import_ma = df['수입중량(ton)'].rolling(30).mean()
df['import_surge'] = (df['수입중량(ton)'] > import_ma * 1.5).astype(int)

if '유가_종가(USD)' in df.columns:
    df['oil_pct_change'] = df['유가_종가(USD)'].pct_change(7)

print(f"✓ 외부 변수 가공 완료")

# -------------------------------------------------------------
# 6. 상호작용 피처
# -------------------------------------------------------------
df['holiday_supply'] = df['is_holiday_season'] * df['supply_pressure']
df['summer_supply'] = df['is_summer_highland'] * df['총반입량(ton)']
df['winter_import'] = df['is_winter_jeju'] * df['수입중량(ton)']
df['price_volume_특'] = df['특_가격'] * df['총반입량(ton)']

# ★ 새로운 상호작용: 생육기 기온 × 계절
df['생육기온_여름고랭지'] = df['생육기_평균기온'] * df['is_summer_highland']
df['생육기온_겨울제주'] = df['생육기_평균기온'] * df['is_winter_jeju']

# 극한 기온 × 공급
df['폭염_공급압력'] = df['생육기_폭염일수'] * df['supply_pressure']
df['한파_공급압력'] = df['생육기_한파일수'] * df['supply_pressure']

print(f"✓ 상호작용 피처 생성 완료")

# -------------------------------------------------------------
# 7. 계절성
# -------------------------------------------------------------
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df['week'] = df['DATE'].dt.isocalendar().week
df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)

df['day_of_week'] = df['DATE'].dt.dayofweek
df['is_monday'] = (df['day_of_week'] == 0).astype(int)

df = df.drop(columns=['month', 'week', 'day_of_week'])

print(f"✓ 계절성 인코딩 완료")

# -------------------------------------------------------------
# 8. 타겟 생성 & 최종 정리
# -------------------------------------------------------------
df['target_high'] = df['특_가격'].shift(-7)
df['target_mid'] = df['상_가격'].shift(-7)

# 결측치 제거
df = df.dropna()

# 최종 통계
feature_cols = [c for c in df.columns if c not in ['DATE', 'target_high', 'target_mid']]
print("\n" + "="*60)
print("📊 최종 데이터 요약")
print("="*60)
print(f"데이터 크기: {df.shape}")
print(f"총 피처 개수: {len(feature_cols)}개")
print(f"기간: {df['DATE'].min()} ~ {df['DATE'].max()}")

# 주요 피처 카테고리
print("\n주요 피처 카테고리:")
print(f"  - 가격 관련: {len([c for c in feature_cols if '가격' in c])}개")
print(f"  - 주산지 기온: {len([c for c in feature_cols if '주산지' in c or '생육기' in c])}개")
print(f"  - 날씨 관련: {len([c for c in feature_cols if any(x in c for x in ['rain', 'temp', '폭염', '한파', '최근30일'])])}개")
print(f"  - 공급 관련: {len([c for c in feature_cols if any(x in c for x in ['volume', 'supply', 'import'])])}개")
print(f"  - 이벤트: {len([c for c in feature_cols if any(x in c for x in ['holiday', 'kimjang', 'covid', 'summer', 'winter'])])}개")

# 저장
output_file = '양배추_학습데이터_지역별.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n💾 저장 완료: {output_file}")

print("\n" + "="*60)
print("✨ 주요 개선사항 (생산지 기온 활용)")
print("="*60)
print("1. ✅ 계절별 주산지 기온 선택 (여름=고랭지, 겨울=제주, 봄가을=남부)")
print("2. ✅ 생육기간(60~90일 전) 주산지 기온 반영")
print("3. ✅ 지역별 극한 기온 이벤트 추적 (폭염/한파)")
print("4. ✅ 생육기 기온 변동성 (스트레스 지표)")
print("5. ✅ 생육기 극한 일수 (폭염일수, 한파일수)")
print("6. ✅ 계절 × 생육기 기온 상호작용")
print("7. ✅ 극한 기온 × 공급 압력 상호작용")
print("="*60)
