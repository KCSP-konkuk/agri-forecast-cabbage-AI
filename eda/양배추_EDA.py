"""
양배추 도매가격 예측 - 탐색적 데이터 분석 (EDA)
로컬 환경용 Python 스크립트
"""

# ==========================================
# 1. 환경 설정 및 한글 폰트
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import platform

# 한글 폰트 설정 (로컬 환경)
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # Windows
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')    # Mac
else:
    plt.rc('font', family='NanumBarunGothic')  # Linux
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

print("=" * 60)
print("양배추 도매가격 예측 - 탐색적 데이터 분석")
print("=" * 60)

# ==========================================
# 2. 데이터 로드 (로컬 환경)
# ==========================================
print("\n[1단계] 데이터 로드 중...")
base_path = "./"  # 현재 폴더

try:
    price = pd.read_excel(base_path + "양배추 가격.xlsx")
    volume = pd.read_excel(base_path + "양배추 반입량.xlsx")
    
    print("✓ 데이터 로드 완료")
    print(f"  - 가격 데이터: {len(price):,} rows")
    print(f"  - 반입량 데이터: {len(volume):,} rows")
except FileNotFoundError as e:
    print(f"✗ 오류: 파일을 찾을 수 없습니다 - {e}")
    exit(1)

# 날짜 변환 및 정렬
price['DATE'] = pd.to_datetime(price['DATE'])
volume['DATE'] = pd.to_datetime(volume['DATE'])
price = price.sort_values('DATE')
volume = volume.sort_values('DATE')

print(f"  - 가격 데이터 기간: {price['DATE'].min()} ~ {price['DATE'].max()}")
print(f"  - 반입량 데이터 기간: {volume['DATE'].min()} ~ {volume['DATE'].max()}")

# ==========================================
# 3. 데이터 전처리
# ==========================================
print("\n[2단계] 데이터 전처리 중...")

# '특' 등급만 필터링
df = price[price['등급명'] == '특'].copy()
print(f"  - '특' 등급 데이터: {len(df):,} rows")

# 가격이 0인 경우 처리
# ffill()로 전일 가격을 가져오되, 맨 첫날이 0이면 bfill()로 다음날 가격을 가져옴
zero_count = (df['평균가격'] == 0).sum()
df['평균가격'] = df['평균가격'].replace(0, np.nan).ffill().bfill()
print(f"  - 가격 0인 데이터 보정: {zero_count}개")

# 반입량 데이터와 병합
df = pd.merge(df, volume[['DATE', '총반입량']], on='DATE', how='left')

# 반입량이 없는 날(NaN)은 0으로 처리
nan_count = df['총반입량'].isna().sum()
df['총반입량'] = df['총반입량'].fillna(0)
print(f"  - 반입량 결측치 처리: {nan_count}개")

print("✓ 전처리 완료")

# ==========================================
# 4. 기본 통계 정보
# ==========================================
print("\n[3단계] 기본 통계 정보")
print("-" * 60)
print(df[['평균가격', '총반입량']].describe())

# ==========================================
# 5. 시각화: 전체 추이 확인
# ==========================================
print("\n[4단계] 시각화 생성 중...")

plt.figure(figsize=(15, 6))
plt.plot(df['DATE'], df['평균가격'], label='가격(특)', color='blue', alpha=0.6)
plt.title('양배추 일별 가격 추이 (결측치 보정)', fontsize=16)
plt.xlabel('날짜', fontsize=12)
plt.ylabel('평균가격 (원)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('양배추_가격추이.png', dpi=150, bbox_inches='tight')
print("  - 저장: 양배추_가격추이.png")
plt.show()

# ==========================================
# 6. 계절성 분석 (월별 Boxplot)
# ==========================================
df['Month'] = df['DATE'].dt.month

plt.figure(figsize=(12, 5))
sns.boxplot(x='Month', y='평균가격', data=df, palette='Set3')
plt.title('월별 양배추 가격 분포 (계절성 확인)', fontsize=16)
plt.xlabel('월', fontsize=12)
plt.ylabel('평균가격 (원)', fontsize=12)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('양배추_월별분포.png', dpi=150, bbox_inches='tight')
print("  - 저장: 양배추_월별분포.png")
plt.show()

# ==========================================
# 7. 자기상관(ACF) 확인
# ==========================================
plt.figure(figsize=(12, 5))
plot_acf(df['평균가격'].dropna(), lags=30, ax=plt.gca())
plt.title('가격 데이터의 자기상관성 (ACF)', fontsize=16)
plt.xlabel('Lag', fontsize=12)
plt.ylabel('자기상관계수', fontsize=12)
plt.tight_layout()
plt.savefig('양배추_ACF.png', dpi=150, bbox_inches='tight')
print("  - 저장: 양배추_ACF.png")
plt.show()

# ==========================================
# 8. 월별 평균 가격 계산
# ==========================================
print("\n[5단계] 월별 평균 가격")
print("-" * 60)
monthly_avg = df.groupby('Month')['평균가격'].agg(['mean', 'std', 'min', 'max'])
monthly_avg.columns = ['평균', '표준편차', '최소', '최대']
print(monthly_avg.round(2))

print("\n" + "=" * 60)
print("EDA 완료!")
print("=" * 60)
