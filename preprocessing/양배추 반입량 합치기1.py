
import pandas as pd
import numpy as np

# 1. 전처리된 가격 데이터 로드
df_price = pd.read_csv('양배추_가격_전처리완료.csv')
df_price['DATE'] = pd.to_datetime(df_price['DATE'])
print(f"가격 데이터 로드: {len(df_price)}행")

# 2. 반입량 데이터 로드 및 전처리
df_volume = pd.read_csv('양배추 반입량.csv')
df_volume['DATE'] = pd.to_datetime(df_volume['DATE'])

# 필요한 컬럼만 선택 ('DATE', '총반입량')
df_volume = df_volume[['DATE', '총반입량']].copy()
df_volume.rename(columns={'총반입량': '총반입량(ton)'}, inplace=True)

# 중복 날짜 제거 (혹시 모를 중복 대비, 같은 날짜면 평균값 사용)
df_volume = df_volume.groupby('DATE', as_index=False)['총반입량(ton)'].mean()

# 3. 데이터 병합 (Left Join: 가격 데이터 날짜 기준)
df_merged = pd.merge(df_price, df_volume, on='DATE', how='left')

# 4. 반입량 결측치 처리 (선형 보간 - Interpolate)
# 0값을 NaN으로 변환 후 보간 (0이 '없음'을 의미한다면 보간 대상이 되어야 함)
df_merged['총반입량(ton)'] = df_merged['총반입량(ton)'].replace(0, np.nan)

# 선형 보간 적용
df_merged['총반입량(ton)'] = df_merged['총반입량(ton)'].interpolate(method='linear')

# 보간으로도 안 채워진 앞/뒤 부분 채우기 (ffill, bfill)
df_merged['총반입량(ton)'] = df_merged['총반입량(ton)'].ffill().bfill()

print(f"병합 및 보간 완료: {len(df_merged)}행")
print('결측치 수:\n', df_merged.isnull().sum())

# 5. 결과 확인
print('='*50)
print('샘플 데이터 (최근 5일):')
print(df_merged.tail(5))

# 6. 저장
output_file = '양배추_통합_반입량까지.csv'
df_merged.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f'\n저장 완료: {output_file}')
