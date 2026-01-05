
import pandas as pd
import numpy as np

# 1. 기존 데이터 로드 (반입량 + 유가 완료된 파일)
try:
    df_main = pd.read_csv('양배추_통합_유가까지.csv') # 유가까지 완료된 파일
    df_main['DATE'] = pd.to_datetime(df_main['DATE'])
    print(f"기존 데이터 로드: {len(df_main)}행")
except FileNotFoundError:
    print("양배추_통합_유가까지.csv 파일을 찾을 수 없습니다. 이전 단계를 확인해주세요.")
    # 테스트를 위해 이전 단계 파일 사용 가능성 고려 (실제 사용시엔 주석 처리)
    # try:
    #     df_main = pd.read_csv('양배추_통합_반입량까지.csv')
    #     df_main['DATE'] = pd.to_datetime(df_main['DATE'])
    #     print("대체 파일(반입량까지) 로드")
    # except:
    exit()

# 2. 수입 데이터 처리
df_import = pd.read_csv('양배추수입.csv')
# 'DATE' 컬럼이 202511 같은 숫자형일 수 있으므로 문자열 변환
df_import['YM'] = df_import['DATE'].astype(str)

# 월별 합계 집계
df_import_group = df_import.groupby('YM')[['금액', '중량']].sum().reset_index()
df_import_group['수입중량(ton)'] = df_import_group['중량'] / 1000  # kg -> ton 변환
df_import_group = df_import_group.rename(columns={'금액': '수입금액(USD)'})
df_import_group = df_import_group[['YM', '수입금액(USD)', '수입중량(ton)']]

# 3. 수출 데이터 처리
df_export = pd.read_csv('양배추수출.csv')
df_export['YM'] = df_export['DATE'].astype(str)

# 월별 합계 집계
df_export_group = df_export.groupby('YM')[['금액', '중량']].sum().reset_index()
df_export_group['수출중량(ton)'] = df_export_group['중량'] / 1000  # kg -> ton 변환
df_export_group = df_export_group.rename(columns={'금액': '수출금액(USD)'})
df_export_group = df_export_group[['YM', '수출금액(USD)', '수출중량(ton)']]

# 4. 데이터 병합 (매핑)
# 일별 데이터에 'YM' (년월) 컬럼 생성
df_main['YM'] = df_main['DATE'].dt.strftime('%Y%m')

# 수입 데이터 병합
df_main = pd.merge(df_main, df_import_group, on='YM', how='left')

# 수출 데이터 병합
df_main = pd.merge(df_main, df_export_group, on='YM', how='left')

# 5. 결측치 처리 (ffill -> bfill)
# 월별 데이터가 없는 경우 전월 데이터 유지
cols_trade = ['수입금액(USD)', '수입중량(ton)', '수출금액(USD)', '수출중량(ton)']
df_main[cols_trade] = df_main[cols_trade].ffill().bfill().fillna(0) # 앞뒤 다 없으면 0

# 임시 컬럼 제거
df_main = df_main.drop('YM', axis=1)

print(f"무역 데이터 병합 완료: {len(df_main)}행")
print('결측치 확인:\n', df_main.isnull().sum())

# 6. 결과 확인
print('='*50)
print('샘플 데이터 (최근 5일):')
print(df_main.tail(5))

# 7. 저장
output_file = '양배추_통합_최종_1차.csv' # 날씨 제외한 모든 데이터
df_main.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f'\n저장 완료: {output_file}')
