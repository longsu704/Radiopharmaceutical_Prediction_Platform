import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import warnings
import xgboost as xgb

try:
    from category_encoders import TargetEncoder
except ImportError:
    print("请先安装category_encoders库：pip install category_encoders")
    exit()

warnings.filterwarnings('ignore')

# 加载数据
train_file_path = r'D:\wordjihe\Biodistribution_data_radiopharmaceuticals.xlsx'  # 请修改为你的路径
df_train = pd.read_excel(train_file_path, na_values=['NA', 'na', 'N/A', 'n/a', ''])

X_train_origin = df_train.iloc[:, 0:13].copy()
y_train_origin = df_train.iloc[:, 13].copy()

# 异常值处理
y_train_origin = y_train_origin.dropna()
X_train_origin = X_train_origin.loc[y_train_origin.index].reset_index(drop=True)
y_train_origin = y_train_origin.reset_index(drop=True)

q_low = y_train_origin.quantile(0.025)
q_high = y_train_origin.quantile(0.975)
mask = (y_train_origin >= q_low) & (y_train_origin <= q_high)
X_train_origin = X_train_origin[mask].reset_index(drop=True)
y_train_origin = y_train_origin[mask].reset_index(drop=True)

# 预处理
X_categorical = X_train_origin.iloc[:, 0:7].copy()
X_numerical = X_train_origin.iloc[:, 7:13].copy()

# 数值列处理
for col in X_numerical.columns:
    X_numerical[col] = pd.to_numeric(X_numerical[col], errors='coerce')

num_imputer = SimpleImputer(strategy='median', missing_values=np.nan)
X_numerical_imputed = pd.DataFrame(num_imputer.fit_transform(X_numerical), columns=X_numerical.columns)

scaler = StandardScaler()
X_numerical_scaled = pd.DataFrame(scaler.fit_transform(X_numerical_imputed), columns=X_numerical.columns)

# 特征交互
num_cols = X_numerical_scaled.columns.tolist()
for i in range(len(num_cols)):
    for j in range(i+1, min(i+3, len(num_cols))):
        col1, col2 = num_cols[i], num_cols[j]
        X_numerical_scaled[f'{col1}_mul_{col2}'] = X_numerical_scaled[col1] * X_numerical_scaled[col2]

for col in num_cols[:3]:
    X_numerical_scaled[f'{col}_sq'] = X_numerical_scaled[col] **2
    X_numerical_scaled[f'{col}_log'] = np.log1p(np.abs(X_numerical_scaled[col]))

if len(num_cols) >= 2:
    X_numerical_scaled[f'{num_cols[0]}_div_{num_cols[1]}'] = X_numerical_scaled[num_cols[0]] / (X_numerical_scaled[num_cols[1]] + 1e-6)

# 类别列处理
X_categorical = X_categorical.fillna('NA').astype(str)
te = TargetEncoder()
X_categorical_encoded = te.fit_transform(X_categorical, y_train_origin)

# 合并特征
X_processed_temp = pd.concat([X_categorical_encoded, X_numerical_scaled], axis=1)
X_processed_temp = X_processed_temp.fillna(0)

# 特征选择
temp_cat = CatBoostRegressor(objective='RMSE', learning_rate=0.05, n_estimators=500, depth=6, random_state=42, verbose=0)
temp_cat.fit(X_processed_temp, y_train_origin)

feat_importance = pd.DataFrame({'feature': X_processed_temp.columns, 'importance': temp_cat.get_feature_importance()}).sort_values('importance', ascending=False)
threshold = feat_importance['importance'].quantile(0.2)
selected_feats = feat_importance[feat_importance['importance'] >= threshold]['feature'].tolist()
X_processed = X_processed_temp[selected_feats].copy()

# 训练模型
# XGBoost
xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.021,
    'n_estimators': 1450,
    'max_depth': 4,
    'min_child_weight': 28,
    'gamma': 0.25,
    'subsample': 0.62,
    'colsample_bytree': 0.62,
    'reg_alpha': 0.70,
    'reg_lambda': 0.70,
    'random_state': 42,
    'verbosity': 0,
    'sampling_method': 'uniform',
}

final_model = xgb.XGBRegressor(**xgb_params)
final_model.fit(X_processed, y_train_origin)

# 保存所有组件
prefix = 'efm_xgboost'
joblib.dump(num_imputer, f'{prefix}_num_imputer.pkl')
joblib.dump(scaler, f'{prefix}_scaler.pkl')
joblib.dump(te, f'{prefix}_te.pkl')
joblib.dump(selected_feats, f'{prefix}_selected_feats.pkl')
joblib.dump(final_model, f'{prefix}_model.pkl')
joblib.dump(num_cols, f'{prefix}_num_cols.pkl')
joblib.dump(X_train_origin.columns.tolist(), f'{prefix}_feature_names.pkl')

# 保存类别列唯一值（排序）
cat_unique_vals = {}
for col in X_categorical.columns:
    cat_unique_vals[col] = sorted(X_categorical[col].unique().tolist())
joblib.dump(cat_unique_vals, f'{prefix}_cat_unique_vals.pkl')

print(f"EFM模型组件保存完成！")