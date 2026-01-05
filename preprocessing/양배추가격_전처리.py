
import pandas as pd
import numpy as np

# 데이터 로드
try:
    df = pd.read_csv('양배추 가격.csv')
    print("데이터 로드 성공")
except FileNotFoundError:
    print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

df['DATE'] = pd.to_datetime(df['DATE'])

# 특/상 분리
df_high = df[df['등급명'] == '특'].copy().sort_values('DATE')
df_mid = df[df['등급명'] == '상'].copy().sort_values('DATE')

# 전체 날짜 범위 생성 (빈 날짜 채우기 위함)
full_idx = pd.date_range(start=df['DATE'].min(), end=df['DATE'].max(), freq='D')

def process_price(df_subset, col_name):
    # 날짜 인덱스 설정 및 리인덱싱 (빈 날짜 생성)
    df_subset = df_subset.set_index('DATE').reindex(full_idx)
    df_subset.index.name = 'DATE'
    df_subset = df_subset.reset_index()
    
    # '전일' 컬럼이 없는 경우를 대비해 처리 (reindex로 인해 NaN이 생길 수 있음)
    if '전일' in df_subset.columns:
        # 0값 처리 1단계: '전일' 값으로 대체 (평균가격이 0 또는 NaN이고, 전일 가격이 0보다 큰 경우)
        # reindex로 인해 새로 생긴 행은 '전일' 컬럼도 NaN일 수 있음 -> 이 경우는 ffill로 넘어감
        mask = ((df_subset['평균가격'] == 0) | df_subset['평균가격'].isna()) & (df_subset['전일'] > 0)
        df_subset.loc[mask, '평균가격'] = df_subset.loc[mask, '전일']
    
    # 0값 처리 2단계: 여전히 0인 값을 NaN으로 변환 후 ffill
    df_subset['평균가격'] = df_subset['평균가격'].replace(0, np.nan)
    
    # ffill 적용 (직전 값으로 채움)
    df_subset[col_name] = df_subset['평균가격'].ffill()
    
    # 맨 앞부분에 결측치가 남은 경우 bfill (1단계에서 전일 가격으로도 못 채운 경우 대비)
    df_subset[col_name] = df_subset[col_name].bfill()
    
    return df_subset[['DATE', col_name]]

# 각각 처리
print("특 등급 데이터 처리 중...")
df_high_clean = process_price(df_high, '특_가격')
print("상 등급 데이터 처리 중...")
df_mid_clean = process_price(df_mid, '상_가격')

# 병합
print("데이터 병합 중...")
df_final = pd.merge(df_high_clean, df_mid_clean, on='DATE', how='outer')

# 결과 확인
print('='*50)
print('전체 행 수:', len(df_final))
print('결측치 수:')
print(df_final.isnull().sum())
print('='*50)
print('샘플 데이터 (최근 5일):')
print(df_final.tail(5))

# 저장
output_file = '양배추_가격_전처리완료.csv'
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f'\n저장 완료: {output_file}')
