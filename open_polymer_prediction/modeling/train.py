from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

import open_polymer_prediction.config as cfg

# Ensure models directory exists
cfg.MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Load processed data
df = pd.read_parquet(cfg.PROCESSED_DIR / 'train.parquet')
# enforce boolean for external flag if present
if 'is_external' in df:
    df['is_external'] = df['is_external'].fillna(False).astype(bool)

# Single target
target = cfg.TARGET
# Feature columns
features = [c for c in df.columns if c not in cfg.DROP_COLS]

# Filter only rows where target is not null
df_train = df.dropna(subset=[target]).reset_index(drop=True)
X = df_train[features]
y = df_train[target].values

# Prepare CV
gkf = KFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)

# DataFrame to collect OOF predictions
oof = pd.Series(np.nan, index=df_train.index, name=target)

# Loop models
for model_name in cfg.MODELS_TO_RUN:
    print(f"==> {model_name} on {target}")
    oof_model = oof.copy()
    fold_scores = []
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X, y)):
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y[trn_idx], y[val_idx]

        if model_name == 'autogluon':
            train_fold = pd.concat([
                X_tr.reset_index(drop=True),
                pd.Series(y_tr, name=target)
            ], axis=1)
            predictor = TabularPredictor(
                label=target,
                eval_metric='mae',
                path=str(cfg.MODELS_DIR / f"ag_{target}_fold{fold}")
            ).fit(
                train_data=train_fold,
                time_limit=cfg.MODEL_PARAMS['autogluon']['time_limit'],
                presets=cfg.MODEL_PARAMS['autogluon']['presets'],
                ag_args_fit=cfg.MODEL_PARAMS['autogluon']['ag_args_fit']
            )
            preds = predictor.predict(X_val)
        else:
            if model_name == 'lgb':
                mdl = lgb.LGBMRegressor(**cfg.MODEL_PARAMS['lgb'])
            else:
                mdl = xgb.XGBRegressor(**cfg.MODEL_PARAMS['xgb'])
            mdl.fit(X_tr, y_tr)
            joblib.dump(mdl, cfg.MODELS_DIR / f"{model_name}_{target}_fold{fold}.pkl")
            preds = mdl.predict(X_val)

        # store OOF and score
        oof_model.iloc[val_idx] = preds
        score = mean_absolute_error(y_val, preds)
        fold_scores.append(score)
        print(f" Fold {fold} MAE: {score:.5f}")

    mean_score = np.mean(fold_scores)
    std_score  = np.std(fold_scores)
    print(f"{model_name} CV MAE: {mean_score:.5f} ± {std_score:.5f}\n")

    # save OOF
    oof_df = pd.DataFrame({ 'index': df_train.index, target: oof_model })
    oof_df.to_csv(cfg.MODELS_DIR / f"oof_{model_name}_{target}.csv", index=False)

# Train final models on full dataset
for model_name in cfg.MODELS_TO_RUN:
    print(f"Training full {model_name} on {target}")
    if model_name == 'autogluon':
        df_full = pd.concat([X, pd.Series(y, name=target)], axis=1)
        TabularPredictor(
            label=target,
            eval_metric='mae',
            path=str(cfg.MODELS_DIR / f"ag_{target}_full")
        ).fit(
            train_data=df_full,
            time_limit=cfg.MODEL_PARAMS['autogluon']['time_limit'],
            presets=cfg.MODEL_PARAMS['autogluon']['presets'],
            ag_args_fit=cfg.MODEL_PARAMS['autogluon']['ag_args_fit']
        )
    else:
        if model_name == 'lgb':
            mdl = lgb.LGBMRegressor(**cfg.MODEL_PARAMS['lgb'])
        else:
            mdl = xgb.XGBRegressor(**cfg.MODEL_PARAMS['xgb'])
        mdl.fit(X, y)
        joblib.dump(mdl, cfg.MODELS_DIR / f"{model_name}_{target}_full.pkl")

print("Training for {target} complete.")