# -*- coding: utf-8 -*-
"""
Script Function: XGBoost prediction for punching shear capacity (Output: Real-Scale RMSE).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import os
import joblib

# =======================================================================
# 1. Configuration & Hyperparameters
# =======================================================================
# --- Paths ---
DATA_PATH = '../data'
OUTPUT_DIR = '../results'

# --- Model Settings ---
USE_K_FOLD = True
K_FOLDS = 5
RANDOM_SEED = 50
TRAIN_SET_SIZE = 36

# --- XGBoost Params ---
XGB_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.005,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'n_jobs': 1,
    'random_state': RANDOM_SEED
}
EARLY_STOPPING_ROUNDS = 50

# Apply Seed
np.random.seed(RANDOM_SEED)


# =======================================================================
# 2. Data Loading
# =======================================================================
def load_data():
    print(">>> Loading data...")
    features = ['L1', 'h0', 'c/R', 'CKB', 'rho', 'fcu', 't_bot', 'fy_bot', 'fu_bot', 't_top', 'fy_top', 'fu_top',
                'stud_space', 'stud_D', 'stud_height']
    target = 'Vu'

    file_path = os.path.join(DATA_PATH, 'single_plate_slabs.xlsx')

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None, None, None

    df = pd.read_excel(file_path)
    df[target] = pd.to_numeric(df[target], errors='coerce')
    df.dropna(subset=[target], inplace=True)

    X = df[features]
    y = df[target]
    return X, y, df


# =======================================================================
# 3. Training Function
# =======================================================================
def train_xgb(X_train, y_train, X_val, y_val):
    """
    Trains XGBoost and evaluates on Real Scale data.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = XGB_PARAMS.copy()
    num_boost_round = params.pop('n_estimators')
    params['eval_metric'] = 'rmse'

    if 'random_state' in params:
        params['seed'] = params.pop('random_state')

    watchlist = [(dtrain, 'train'), (dval, 'eval')]

    # Version compatibility for Early Stopping
    try:
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=watchlist,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False
        )
    except TypeError:
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=watchlist,
            callbacks=[xgb.callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS)],
            verbose_eval=False
        )

    # Prediction compatibility
    try:
        best_iteration = model.best_iteration
        preds = model.predict(dval, iteration_range=(0, best_iteration + 1))
    except (AttributeError, TypeError):
        try:
            preds = model.predict(dval, ntree_limit=model.best_ntree_limit)
        except:
            preds = model.predict(dval)

    # Metrics (Real Scale)
    mse_real = mean_squared_error(y_val, preds)
    rmse_real = np.sqrt(mse_real)
    r2_real = r2_score(y_val, preds)

    return preds, mse_real, rmse_real, r2_real, model


# =======================================================================
# 4. Main Execution
# =======================================================================
if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- Running XGBoost (Real Scale RMSE) ---")
    X, y, df_full = load_data()

    if X is None:
        exit()

    X_np = X.values
    y_np = y.values

    if USE_K_FOLD:
        print(f"Mode: K-Fold CV (K={K_FOLDS})")
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        results = []
        all_preds = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_np)):
            print(f"Processing Fold {fold + 1}...")

            X_tr, X_va = X_np[train_idx], X_np[val_idx]
            y_tr, y_va = y_np[train_idx], y_np[val_idx]

            # Standardize Features only
            scaler_X = StandardScaler()
            X_tr_s = scaler_X.fit_transform(X_tr)
            X_va_s = scaler_X.transform(X_va)

            preds, mse_real, rmse_real, r2, _ = train_xgb(X_tr_s, y_tr, X_va_s, y_va)

            print(f"  Fold {fold + 1} -> R2: {r2:.4f} | RMSE (Real): {rmse_real:.2f} kN")

            results.append({
                'Fold': fold + 1,
                'R2': r2,
                'MSE_Real': mse_real,
                'RMSE_Real': rmse_real
            })

            fold_df = pd.DataFrame({
                'Fold': fold + 1,
                'True_Value_kN': y_va,
                'Predicted_Value_kN': preds
            })
            all_preds.append(fold_df)

        # Save Results
        df_res = pd.DataFrame(results)
        df_res.to_excel(os.path.join(OUTPUT_DIR, 'xgboost_kfold_metrics.xlsx'), index=False)

        df_preds = pd.concat(all_preds, ignore_index=True)
        df_preds.to_excel(os.path.join(OUTPUT_DIR, 'xgboost_kfold_predictions.xlsx'), index=False)

        print(f"\nAverage R2: {df_res['R2'].mean():.4f}")
        print(f"Average RMSE (Real): {df_res['RMSE_Real'].mean():.2f} kN")
        print(f"Results saved to {OUTPUT_DIR}")

    else:
        # Single Split Mode
        if len(X_np) <= TRAIN_SET_SIZE:
            raise ValueError(f"Total data ({len(X_np)}) is too small for requested train size ({TRAIN_SET_SIZE})")

        X_tr, X_va, y_tr, y_va = train_test_split(
            X_np, y_np,
            train_size=TRAIN_SET_SIZE,
            random_state=RANDOM_SEED
        )
        print(f"Train samples: {len(X_tr)}, Validation samples: {len(X_va)}")

        scaler_X = StandardScaler()
        X_tr_s = scaler_X.fit_transform(X_tr)
        X_va_s = scaler_X.transform(X_va)

        joblib.dump(scaler_X, os.path.join(OUTPUT_DIR, 'xgboost_std_scaler_X.joblib'))

        preds, mse_real, rmse_real, r2, model = train_xgb(X_tr_s, y_tr, X_va_s, y_va)

        print(f"Result -> R2: {r2:.4f} | RMSE (Real): {rmse_real:.2f} kN")

        pd.DataFrame([{
            'R2': r2,
            'MSE_Real': mse_real,
            'RMSE_Real': rmse_real
        }]).to_excel(os.path.join(OUTPUT_DIR, 'xgboost_std_metrics.xlsx'), index=False)

        pd.DataFrame({
            'True_Value_kN': y_va,
            'Predicted_Value_kN': preds
        }).to_excel(os.path.join(OUTPUT_DIR, 'xgboost_std_predictions.xlsx'), index=False)

    print("--- Done ---")