
import pandas as pd
import numpy as np
import glob
import os

# 1. 메인 데이터 로드 (최종 1차: 반입량+유가+수출입 완료된 파일)
main_file = '양배추_통합_최종_1차.csv'
try:
    df_main = pd.read_csv(main_file)
    df_main['DATE'] = pd.to_datetime(df_main['DATE'])
    print(f"메인 데이터 로드: {len(df_main)}행 ({main_file})")
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {main_file}")
    exit()

# 2. 날씨 데이터 처리 함수 (지역별)
def process_weather_condition(region_name):
    print(f"--- {region_name} 날씨 데이터 처리 중 ---")
    
    # 파일 경로 패턴 (날씨 데이터 폴더 내)
    # 예: 날씨 데이터/목포기온_정리.csv
    base_path = '날씨 데이터/'
    file_temp = f"{base_path}{region_name}기온_정리.csv"
    file_humid = f"{base_path}{region_name}습도_정리.csv"
    file_rain = f"{base_path}{region_name}강수량_정리.csv"
    
    # 2-1. 기온 (평균, 최고, 최저)
    try:
        df_temp = pd.read_csv(file_temp)
        # 컬럼명 확인 및 변경 ('일시' -> 'DATE')
        df_temp = df_temp.rename(columns={'일시': 'DATE'})
        df_temp['DATE'] = pd.to_datetime(df_temp['DATE'])
        
        # 필요한 컬럼만 선택 및 이름 변경
        df_temp = df_temp[['DATE', '평균기온', '최고기온', '최저기온']].copy()
        df_temp.columns = ['DATE', f'{region_name}_평균기온', f'{region_name}_최고기온', f'{region_name}_최저기온']
    except Exception as e:
        print(f"[{region_name}] 기온 파일 로드 실패: {e}")
        return None

    # 2-2. 습도 (평균습도)
    try:
        df_humid = pd.read_csv(file_humid)
        df_humid = df_humid.rename(columns={'일시': 'DATE'})
        df_humid['DATE'] = pd.to_datetime(df_humid['DATE'])
        
        # '평균습도(%rh)' 컬럼 확인
        target_col = [c for c in df_humid.columns if '평균습도' in c][0]
        df_humid = df_humid[['DATE', target_col]].copy()
        df_humid.columns = ['DATE', f'{region_name}_평균습도']
    except Exception as e:
        print(f"[{region_name}] 습도 파일 로드 실패 (또는 컬럼 없음): {e}")
        # 습도 파일이 없어도 진행할 수 있도록 빈 프레임 반환하거나 처리
        df_humid = pd.DataFrame(columns=['DATE', f'{region_name}_평균습도'])

    # 2-3. 강수량 (일강수량)
    try:
        df_rain = pd.read_csv(file_rain)
        df_rain = df_rain.rename(columns={'일시': 'DATE'})
        df_rain['DATE'] = pd.to_datetime(df_rain['DATE'])
        
        # '강수량(mm)' 컬럼 확인
        target_col = [c for c in df_rain.columns if '강수량' in c][0]
        df_rain = df_rain[['DATE', target_col]].copy()
        df_rain.columns = ['DATE', f'{region_name}_강수량']
    except Exception as e:
        print(f"[{region_name}] 강수량 파일 로드 실패: {e}")
        df_rain = pd.DataFrame(columns=['DATE', f'{region_name}_강수량'])

    # 2-4. 지역 날씨 병합 (기온 + 습도 + 강수량)
    df_region = pd.merge(df_temp, df_humid, on='DATE', how='outer')
    df_region = pd.merge(df_region, df_rain, on='DATE', how='outer')
    
    # 2-5. 전처리 (보간 및 0채우기)
    # 날짜 정렬
    df_region = df_region.sort_values('DATE')
    
    # 기온, 습도 -> Linear Interpolate
    cols_interp = [c for c in df_region.columns if ('기온' in c) or ('습도' in c)]
    df_region[cols_interp] = df_region[cols_interp].interpolate(method='linear').ffill().bfill()
    
    # 강수량 -> 0 채우기 (NaN은 비 안 온 것)
    col_rain = f'{region_name}_강수량'
    if col_rain in df_region.columns:
        df_region[col_rain] = df_region[col_rain].fillna(0)
        
    return df_region

# 3. 전체 지역 통합
regions = ['목포', '제주', '태백', '홍천']
df_weather_final = df_main.copy()

for region in regions:
    df_region_weather = process_weather_condition(region)
    if df_region_weather is not None:
        # 중복 날짜 처리 (혹시 모를 중복 제거)
        df_region_weather = df_region_weather.drop_duplicates(subset=['DATE'])
        
        # 메인 데이터에 병합 (Left Join)
        df_weather_final = pd.merge(df_weather_final, df_region_weather, on='DATE', how='left')
        
        # 병합 후 생기는 결측치 (메인 데이터 날짜에는 있지만 날씨 데이터가 없는 경우) 처리
        # 기온/습도 -> 보간, 강수량 -> 0
        # (이미 지역별 처리에서 했지만, 병합 과정에서 날짜 범위 차이로 생길 수 있음)
        cols_interp = [c for c in df_weather_final.columns if (region in c) and (('기온' in c) or ('습도' in c))]
        df_weather_final[cols_interp] = df_weather_final[cols_interp].interpolate(method='linear').ffill().bfill()
        
        col_rain = f'{region}_강수량'
        if col_rain in df_weather_final.columns:
            df_weather_final[col_rain] = df_weather_final[col_rain].fillna(0)

print("="*50)
print(f"최종 통합 완료: {len(df_weather_final)}행")
print(f"컬럼 차원: {df_weather_final.shape}")
print("컬럼 목록:", df_weather_final.columns.tolist())
print('결측치 확인:\n', df_weather_final.isnull().sum())

# 4. 최종 저장
output_file = '양배추_통합데이터_최종.csv'
df_weather_final.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f'\n[완료] 모든 데이터 통합 저장 완료: {output_file}')
