
import pandas as pd
import numpy as np

# 1. 기존 통합 데이터(반입량까지) 로드
df_main = pd.read_csv('양배추_통합_반입량까지.csv') # 생성하실 파일명
df_main['DATE'] = pd.to_datetime(df_main['DATE'])
print(f"기존 데이터 로드: {len(df_main)}행")

# 2. 유가 데이터 로드
df_oil = pd.read_csv('WTI유 선물 과거 데이터.csv')
df_oil['DATE'] = pd.to_datetime(df_oil['날짜'])
df_oil = df_oil[['DATE', '종가']].copy()
df_oil.rename(columns={'종가': '유가_종가(USD)'}, inplace=True) # 컬럼명 변경

# 중복 날짜 제거 (같은 날짜면 평균 또는 첫번째 값)
df_oil = df_oil.groupby('DATE', as_index=False)['유가_종가(USD)'].mean()

# 날짜 정렬
df_oil = df_oil.sort_values('DATE')

# 3. 데이터 병합 (Left Join)
df_final = pd.merge(df_main, df_oil, on='DATE', how='left')

# 4. 결측치 처리 (ffill: 직전 종가 유지)
# 유가는 장이 열리지 않는 날(주말/휴일)은 직전 거래일 가격 유지
df_final['유가_종가(USD)'] = df_final['유가_종가(USD)'].ffill()

# 맨 앞부분 결측치 처리 (bfill: 최초 유가 데이터 이전 날짜들)
df_final['유가_종가(USD)'] = df_final['유가_종가(USD)'].bfill()

print(f"유가 병합 및 처리 완료: {len(df_final)}행")
print(f"유가 결측치 수: {df_final['유가_종가(USD)'].isnull().sum()}")

# 5. 결과 확인
print('='*50)
print('샘플 데이터 (최근 5일):')
print(df_final.tail(5))

# 6. 저장
output_file = '양배추_통합_유가까지.csv'
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f'\n저장 완료: {output_file}')
