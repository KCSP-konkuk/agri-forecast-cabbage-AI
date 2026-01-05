"""
피처 분석 및 최적화
- 타겟과 상관관계 분석
- 다중공선성 제거
- LSTM 최적 피처 선택
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트
font_path = 'C:/Windows/Fonts/malgun.ttf'
if fm.FontProperties(fname=font_path):
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("🔍 피처 분석 및 최적화")
print("=" * 60)

# 데이터 로드
df = pd.read_csv('양배추_학습데이터_지역별.csv')
print(f"데이터 로드: {df.shape}")

# 현재 LSTM 피처 (12개)
current_features = [
    '특_가격', '상_가격',
    '특_가격_lag7',
    '특_가격_MA7', '특_가격_MA30',
    '상_가격_lag7', '상_가격_MA7',
    '총반입량(ton)',
    '유가_종가(USD)',
    'month_sin', 'month_cos',
    '주산지_평균기온',
]
current_features = [f for f in current_features if f in df.columns]

# 전체 후보 피처
all_features = [
    '특_가격', '상_가격',
    '특_가격_lag7', '특_가격_lag14',
    '특_가격_MA7', '특_가격_MA30',
    '특_가격_RSI',
    '상_가격_lag7', '상_가격_MA7',
    '총반입량(ton)', '수입중량(ton)',
    'supply_pressure', 'import_dependency',
    '유가_종가(USD)',
    'month_sin', 'month_cos',
    'is_holiday_season', 'is_kimjang_season',
    'is_summer_highland', 'is_winter_jeju',
    '주산지_평균기온', '주산지_일교차',
    '생육기_평균기온', '생육기_최고기온', '생육기_최저기온',
    '생육기_폭염일수', '생육기_한파일수', '생육기_기온변동성',
    '최근30일_폭염일수', '최근30일_한파일수',
    'avg_rain_5d_sum', '생육기_강수량',
]
all_features = [f for f in all_features if f in df.columns]

print(f"\n현재 LSTM 피처: {len(current_features)}개")
print(f"전체 후보 피처: {len(all_features)}개")

# ===========================================
# 1. 타겟과 상관관계 분석
# ===========================================
print("\n" + "="*60)
print("📊 1. 타겟과 상관관계 분석")
print("="*60)

corr_high = df[all_features + ['target_high']].corr()['target_high'].drop('target_high')
corr_mid = df[all_features + ['target_mid']].corr()['target_mid'].drop('target_mid')

# 상관관계 정렬
corr_high_sorted = corr_high.abs().sort_values(ascending=False)
corr_mid_sorted = corr_mid.abs().sort_values(ascending=False)

print("\n[특 등급 target_high] 상관관계 Top 15:")
for i, (feat, val) in enumerate(corr_high_sorted.head(15).items(), 1):
    in_current = "✓" if feat in current_features else " "
    print(f"  {i:2d}. [{in_current}] {feat:25s}: {corr_high[feat]:+.4f}")

print("\n[상 등급 target_mid] 상관관계 Top 15:")
for i, (feat, val) in enumerate(corr_mid_sorted.head(15).items(), 1):
    in_current = "✓" if feat in current_features else " "
    print(f"  {i:2d}. [{in_current}] {feat:25s}: {corr_mid[feat]:+.4f}")

# ===========================================
# 2. 다중공선성 분석 (VIF 대신 상관관계 사용)
# ===========================================
print("\n" + "="*60)
print("📊 2. 피처 간 다중공선성 분석")
print("="*60)

feature_corr = df[current_features].corr()

# 높은 상관관계 쌍 찾기 (0.9 이상)
high_corr_pairs = []
for i in range(len(current_features)):
    for j in range(i+1, len(current_features)):
        corr_val = abs(feature_corr.iloc[i, j])
        if corr_val > 0.9:
            high_corr_pairs.append((
                current_features[i], 
                current_features[j], 
                corr_val
            ))

if high_corr_pairs:
    print("\n⚠️ 높은 상관관계 피처 쌍 (|r| > 0.9):")
    for f1, f2, val in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
        print(f"  {f1} ↔ {f2}: {val:.4f}")
else:
    print("\n✅ 다중공선성 문제 없음 (모든 피처 쌍 |r| < 0.9)")

# 0.8 이상도 체크
print("\n참고: 중간 상관관계 피처 쌍 (0.8 < |r| < 0.9):")
for i in range(len(current_features)):
    for j in range(i+1, len(current_features)):
        corr_val = abs(feature_corr.iloc[i, j])
        if 0.8 < corr_val <= 0.9:
            print(f"  {current_features[i]} ↔ {current_features[j]}: {corr_val:.4f}")

# ===========================================
# 3. 최적 피처 제안
# ===========================================
print("\n" + "="*60)
print("📊 3. 최적 피처 제안")
print("="*60)

# 현재 피처 중 상관관계 낮은 것
low_corr_current = []
for feat in current_features:
    avg_corr = (abs(corr_high[feat]) + abs(corr_mid[feat])) / 2
    if avg_corr < 0.3:
        low_corr_current.append((feat, avg_corr))

if low_corr_current:
    print("\n⚠️ 현재 피처 중 타겟과 상관관계 낮은 것 (|r| < 0.3):")
    for feat, val in low_corr_current:
        print(f"  ❌ {feat}: {val:.4f} → 제거 고려")

# 현재 없지만 상관관계 높은 피처
missing_high_corr = []
for feat in all_features:
    if feat not in current_features:
        avg_corr = (abs(corr_high[feat]) + abs(corr_mid[feat])) / 2
        if avg_corr > 0.4:
            missing_high_corr.append((feat, avg_corr))

if missing_high_corr:
    print("\n💡 현재 없지만 상관관계 높은 피처 (|r| > 0.4):")
    for feat, val in sorted(missing_high_corr, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  ✅ {feat}: {val:.4f} → 추가 고려")

# ===========================================
# 4. 최적화된 피처셋 제안
# ===========================================
print("\n" + "="*60)
print("📊 4. 최적화된 LSTM 피처셋 제안")
print("="*60)

# 상관관계 기준 상위 12개 선택
avg_corr = {}
for feat in all_features:
    avg_corr[feat] = (abs(corr_high[feat]) + abs(corr_mid[feat])) / 2

sorted_features = sorted(avg_corr.items(), key=lambda x: x[1], reverse=True)

# 다중공선성 제거하면서 상위 피처 선택
optimized_features = []
for feat, corr_val in sorted_features:
    # 이미 선택된 피처와 상관관계 체크
    is_redundant = False
    for existing in optimized_features:
        if abs(df[feat].corr(df[existing])) > 0.85:
            is_redundant = True
            break
    
    if not is_redundant:
        optimized_features.append(feat)
    
    if len(optimized_features) >= 12:
        break

print("\n🎯 최적화된 LSTM 피처 (12개):")
for i, feat in enumerate(optimized_features, 1):
    corr_val = avg_corr[feat]
    in_current = "✓" if feat in current_features else "★"
    print(f"  {i:2d}. [{in_current}] {feat:25s} (상관관계: {corr_val:.4f})")

# 변경 사항
print("\n📝 현재 대비 변경:")
removed = set(current_features) - set(optimized_features)
added = set(optimized_features) - set(current_features)

if removed:
    print(f"  제거: {list(removed)}")
if added:
    print(f"  추가: {list(added)}")
if not removed and not added:
    print("  변경 없음 - 현재 피처셋이 최적!")

# ===========================================
# 5. 시각화
# ===========================================
print("\n" + "="*60)
print("📊 5. 시각화 저장")
print("="*60)

# 상관관계 히트맵
plt.figure(figsize=(12, 10))
sns.heatmap(feature_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('현재 LSTM 피처 간 상관관계')
plt.tight_layout()
plt.savefig('feature_correlation_heatmap.png', dpi=150)
plt.close()
print("  ✅ feature_correlation_heatmap.png 저장")

# 타겟 상관관계 바 차트
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 특 등급
corr_high_sorted.head(15).plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('특 등급 (target_high) 상관관계 Top 15')
axes[0].set_xlabel('상관관계 (절대값)')

# 상 등급
corr_mid_sorted.head(15).plot(kind='barh', ax=axes[1], color='coral')
axes[1].set_title('상 등급 (target_mid) 상관관계 Top 15')
axes[1].set_xlabel('상관관계 (절대값)')

plt.tight_layout()
plt.savefig('feature_target_correlation.png', dpi=150)
plt.close()
print("  ✅ feature_target_correlation.png 저장")

print("\n" + "="*60)
print("✅ 피처 분석 완료!")
print("="*60)

# 최종 추천 피처 출력
print("\n📋 복사용 최적화 피처 리스트:")
print("optimized_features = [")
for feat in optimized_features:
    print(f"    '{feat}',")
print("]")
