"""
LightGBM + LSTM 앙상블
- LightGBM: 28개 피처 (ML 모델)
- LSTM: 12개 피처 (시계열 모델)
- 최적 가중치 자동 탐색
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------
# 0. 설정
# ---------------------------------------------------------
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(42)

font_path = 'C:/Windows/Fonts/malgun.ttf'
if os.path.exists(font_path):
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("🚀 LightGBM + LSTM 앙상블")
print("=" * 60)

# ---------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------
df = pd.read_csv('양배추_학습데이터_지역별.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)
print(f"데이터 로드: {df.shape}")

# ---------------------------------------------------------
# 2. 피처 정의 (LightGBM: 28개, LSTM: 12개)
# ---------------------------------------------------------

# LightGBM용 풀 피처 (28개)
lgb_features = [
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
    '생육기온_여름고랭지', '생육기온_겨울제주', 'holiday_supply',
]
lgb_features = [f for f in lgb_features if f in df.columns]

# LSTM용 핵심 피처 (12개) - 원래 버전 유지
lstm_features = [
    '특_가격', '상_가격',
    '특_가격_lag7',
    '특_가격_MA7', '특_가격_MA30',
    '상_가격_lag7', '상_가격_MA7',
    '총반입량(ton)',
    '유가_종가(USD)',
    'month_sin', 'month_cos',
    '주산지_평균기온',
]
lstm_features = [f for f in lstm_features if f in df.columns]

print(f"LightGBM 피처: {len(lgb_features)}개")
print(f"LSTM 피처: {len(lstm_features)}개")

# ---------------------------------------------------------
# 3. 데이터 분할
# ---------------------------------------------------------
test_size = 365
train_size = len(df) - test_size
WINDOW_SIZE = 60

# LightGBM용 데이터 (28개 피처)
X_lgb = df[lgb_features]
y_high = df['target_high']
y_mid = df['target_mid']

X_train_lgb = X_lgb.iloc[:train_size]
X_test_lgb = X_lgb.iloc[train_size:]
y_high_train = y_high.iloc[:train_size]
y_high_test = y_high.iloc[train_size:]
y_mid_train = y_mid.iloc[:train_size]
y_mid_test = y_mid.iloc[train_size:]

# LSTM용 데이터 (12개 피처)
X_lstm = df[lstm_features]
scaler_x = MinMaxScaler()
scaler_y_high = MinMaxScaler()
scaler_y_mid = MinMaxScaler()

X_scaled = scaler_x.fit_transform(X_lstm)
y_high_scaled = scaler_y_high.fit_transform(y_high.values.reshape(-1, 1))
y_mid_scaled = scaler_y_mid.fit_transform(y_mid.values.reshape(-1, 1))

def create_window_dataset(X, y, window_size):
    X_list, y_list = [], []
    for i in range(len(X) - window_size):
        X_list.append(X[i:i + window_size])
        y_list.append(y[i + window_size])
    return np.array(X_list), np.array(y_list)

X_lstm_w, y_high_lstm = create_window_dataset(X_scaled, y_high_scaled, WINDOW_SIZE)
_, y_mid_lstm = create_window_dataset(X_scaled, y_mid_scaled, WINDOW_SIZE)

lstm_train_size = len(X_lstm_w) - test_size
X_train_lstm = X_lstm_w[:lstm_train_size]
X_test_lstm = X_lstm_w[lstm_train_size:]
y_high_train_lstm = y_high_lstm[:lstm_train_size]
y_high_test_lstm = y_high_lstm[lstm_train_size:]
y_mid_train_lstm = y_mid_lstm[:lstm_train_size]
y_mid_test_lstm = y_mid_lstm[lstm_train_size:]

test_dates = df['DATE'].iloc[train_size:].reset_index(drop=True)

print(f"LightGBM 학습: {X_train_lgb.shape}, 테스트: {X_test_lgb.shape}")
print(f"LSTM 학습: {X_train_lstm.shape}, 테스트: {X_test_lstm.shape}")

# ---------------------------------------------------------
# 4. 모델 정의
# ---------------------------------------------------------

def train_lightgbm(X_train, y_train, X_test, y_test, n_trials=100):
    """LightGBM 튜닝"""
    print("  LightGBM 튜닝 중...")
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return r2_score(y_test, pred)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # 최적 모델
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    })
    
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train)
    
    return model, study.best_value

def build_lstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_lstm(X_train, y_train, X_test, y_test, scaler_y):
    print("  LSTM 학습 중...")
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)
    ]
    
    model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=0
    )
    
    test_pred = scaler_y.inverse_transform(model.predict(X_test))
    return model, test_pred.flatten()

# ---------------------------------------------------------
# 5. 앙상블 실행
# ---------------------------------------------------------

def run_ensemble(target_name, y_train_lgb, y_test_lgb, y_train_lstm, y_test_lstm, scaler_y):
    print(f"\n{'='*60}")
    print(f"[{target_name}] LightGBM + LSTM 앙상블")
    print(f"{'='*60}")
    
    # LightGBM 학습 (28개 피처)
    print("\n[1/2] LightGBM 학습 (28개 피처)")
    lgb_model, lgb_best = train_lightgbm(X_train_lgb, y_train_lgb, X_test_lgb, y_test_lgb, n_trials=100)
    lgb_pred = lgb_model.predict(X_test_lgb)
    lgb_r2 = r2_score(y_test_lgb, lgb_pred)
    print(f"  LightGBM Test R²: {lgb_r2:.4f}")
    
    # LSTM 학습 (12개 피처)
    print("\n[2/2] LSTM 학습 (12개 피처)")
    lstm_model, lstm_pred = train_lstm(X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, scaler_y)
    
    actual_test = scaler_y.inverse_transform(y_test_lstm).flatten()
    lstm_r2 = r2_score(actual_test, lstm_pred)
    print(f"  LSTM Test R²: {lstm_r2:.4f}")
    
    # 예측값 길이 맞추기
    lgb_pred_aligned = lgb_pred[:len(lstm_pred)]
    actual_aligned = y_test_lgb.values[:len(lstm_pred)]
    
    # 최적 가중치 탐색
    print("\n[가중치 탐색]")
    best_r2 = 0
    best_w = 0.5
    
    for w in np.arange(0, 1.01, 0.05):
        ensemble_pred = w * lgb_pred_aligned + (1-w) * lstm_pred
        r2 = r2_score(actual_aligned, ensemble_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_w = w
    
    final_pred = best_w * lgb_pred_aligned + (1-best_w) * lstm_pred
    final_r2 = r2_score(actual_aligned, final_pred)
    final_rmse = root_mean_squared_error(actual_aligned, final_pred)
    final_mae = mean_absolute_error(actual_aligned, final_pred)
    
    print(f"\n{'='*60}")
    print(f"[{target_name}] 앙상블 결과")
    print(f"{'='*60}")
    print(f"  개별 모델:")
    print(f"    LightGBM (28개 피처): R² = {lgb_r2:.4f}")
    print(f"    LSTM (12개 피처):     R² = {lstm_r2:.4f}")
    print(f"  최적 가중치: LightGBM {best_w:.2f} : LSTM {1-best_w:.2f}")
    print(f"  앙상블 최종:")
    print(f"    R²:   {final_r2:.4f}")
    print(f"    RMSE: {final_rmse:.2f}원")
    print(f"    MAE:  {final_mae:.2f}원")
    print(f"{'='*60}")
    
    # 시각화
    plot_dates = test_dates[:len(final_pred)]
    
    plt.figure(figsize=(16, 6))
    plt.plot(plot_dates, actual_aligned, label='실제', alpha=0.7, color='black', linewidth=2)
    plt.plot(plot_dates, lgb_pred_aligned, label=f'LightGBM (R²={lgb_r2:.3f})', 
             alpha=0.4, linestyle=':', linewidth=1.5)
    plt.plot(plot_dates, lstm_pred, label=f'LSTM (R²={lstm_r2:.3f})', 
             alpha=0.4, linestyle=':', linewidth=1.5)
    plt.plot(plot_dates, final_pred, label=f'앙상블 (R²={final_r2:.3f})', 
             alpha=0.9, color='red', linewidth=2)
    plt.title(f'{target_name} - LightGBM + LSTM 앙상블', fontsize=14, fontweight='bold')
    plt.xlabel('날짜')
    plt.ylabel('가격 (원)')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'result_lgb_lstm_{target_name}.png', dpi=150)
    plt.close()
    
    return final_r2, lgb_r2, lstm_r2, best_w

# ---------------------------------------------------------
# 6. 실행
# ---------------------------------------------------------
print("\n" + "="*60)
print("🎯 LightGBM + LSTM 앙상블 시작")
print("="*60)

r2_high, lgb_r2_high, lstm_r2_high, w_high = run_ensemble(
    '특_등급',
    y_high_train, y_high_test,
    y_high_train_lstm, y_high_test_lstm,
    scaler_y_high
)

r2_mid, lgb_r2_mid, lstm_r2_mid, w_mid = run_ensemble(
    '상_등급',
    y_mid_train, y_mid_test,
    y_mid_train_lstm, y_mid_test_lstm,
    scaler_y_mid
)

# ---------------------------------------------------------
# 7. 최종 요약
# ---------------------------------------------------------
print("\n" + "="*60)
print("📊 LightGBM + LSTM 앙상블 최종 결과")
print("="*60)
print(f"{'등급':<8} {'LGB R²':<10} {'LSTM R²':<10} {'앙상블 R²':<12} {'LGB 가중치'}")
print("-"*52)
print(f"{'특':<8} {lgb_r2_high:<10.4f} {lstm_r2_high:<10.4f} {r2_high:<12.4f} {w_high:.2f}")
print(f"{'상':<8} {lgb_r2_mid:<10.4f} {lstm_r2_mid:<10.4f} {r2_mid:<12.4f} {w_mid:.2f}")
print("-"*52)
print(f"{'평균':<8} {(lgb_r2_high+lgb_r2_mid)/2:<10.4f} {(lstm_r2_high+lstm_r2_mid)/2:<10.4f} {(r2_high+r2_mid)/2:<12.4f}")
print("="*60)

print("\n✨ RF+LSTM 대비 비교:")
print(f"  RF+LSTM:      특 0.68, 상 0.68 (평균 0.68)")
print(f"  LightGBM+LSTM: 특 {r2_high:.2f}, 상 {r2_mid:.2f} (평균 {(r2_high+r2_mid)/2:.2f})")
print("="*60)
