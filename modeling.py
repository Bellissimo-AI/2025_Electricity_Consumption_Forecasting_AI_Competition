import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from config import RANDOM_SEED, KFOLD_SPLITS
from utils import smape, weighted_mse, custom_smape

def train_xgb(X, Y, test_X, group_col, max_depth_dict):
    """
    그룹별 학습 (building_type, building_number, cluster 등 단위 학습)
    """
    kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    answer_df = pd.DataFrame(index=test_X.index, columns=["answer"], dtype=float)
    pred_df   = pd.DataFrame(index=X.index, columns=["pred"], dtype=float)

    groups = X[group_col].unique()

    for group in groups:
        x  = X[X[group_col] == group].copy()
        y  = Y[Y[group_col] == group]['power_consumption'].copy()
        xt = test_X[test_X[group_col] == group].copy()

        # one-hot
        if "building_number" in x.columns:
            x  = pd.get_dummies(x,  columns=["building_number"], drop_first=False)
            xt = pd.get_dummies(xt, columns=["building_number"], drop_first=False)
            xt = xt.reindex(columns=x.columns, fill_value=0)

        drop_cols = [group_col]
        x  = x.drop(columns=drop_cols, errors="ignore")
        xt = xt.drop(columns=drop_cols, errors="ignore")

        preds_valid = pd.Series(index=y.index, dtype=float)
        preds_test  = []

        x_values = x.values
        y_values = y.values

        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(x_values), 1):
            X_tr, X_va = x_values[tr_idx], x_values[va_idx]
            y_tr, y_va = y_values[tr_idx], y_values[va_idx]

            y_tr_log = np.log(y_tr)
            y_va_log = np.log(y_va)

            model = XGBRegressor(
                learning_rate     = 0.05,
                n_estimators      = 5000,
                max_depth         = max_depth_dict.get(group, 10),
                subsample         = 0.7,
                colsample_bytree  = 0.5,
                min_child_weight  = 3,
                random_state      = RANDOM_SEED,
                objective         = weighted_mse(3),
                tree_method       = "gpu_hist",
                gpu_id            = 0,
                early_stopping_rounds = 100,
            )

            model.fit(
                X_tr, y_tr_log,
                eval_set=[(X_va, y_va_log)],
                eval_metric=custom_smape,
                verbose=False,
            )

            va_pred = np.exp(model.predict(X_va))
            preds_valid.iloc[va_idx] = va_pred

            fold_smape = smape(y_va, va_pred)
            fold_scores.append(fold_smape)

            preds_test.append(np.exp(model.predict(xt.values)))

        pred_df.loc[preds_valid.index, "pred"] = preds_valid
        answer_df.loc[xt.index, "answer"] = np.mean(preds_test, axis=0)

        print(f"{group_col} = {group} : XGB SMAPE = {np.mean(fold_scores):.4f}")

    return answer_df, pred_df


def train_global(X, Y, test_X, max_depth=10):
    """
    전체 데이터(global) 단위 학습
    """
    kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    preds_valid = pd.Series(index=Y.index, dtype=float)
    preds_test  = []
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X.values), 1):
        X_tr, X_va = X.values[tr_idx], X.values[va_idx]
        y_tr, y_va = Y.values[tr_idx], Y.values[va_idx]

        y_tr_log = np.log(y_tr)
        y_va_log = np.log(y_va)

        model = XGBRegressor(
            learning_rate     = 0.05,
            n_estimators      = 5000,
            max_depth         = max_depth,
            subsample         = 0.7,
            colsample_bytree  = 0.5,
            min_child_weight  = 3,
            random_state      = RANDOM_SEED,
            objective         = weighted_mse(3),
            tree_method       = "gpu_hist",
            gpu_id            = 0,
            early_stopping_rounds = 100,
        )

        model.fit(
            X_tr, y_tr_log,
            eval_set=[(X_va, y_va_log)],
            eval_metric=custom_smape,
            verbose=False,
        )

        va_pred = np.exp(model.predict(X_va))
        preds_valid.iloc[va_idx] = va_pred

        fold_smape = smape(y_va, va_pred)
        fold_scores.append(fold_smape)

        preds_test.append(np.exp(model.predict(test_X.values)))

    print(f"Global Model : XGB SMAPE = {np.mean(fold_scores):.4f}")

    pred_df = pd.DataFrame({"pred": preds_valid})
    answer_df = pd.DataFrame({"answer": np.mean(preds_test, axis=0)})

    return answer_df, pred_df