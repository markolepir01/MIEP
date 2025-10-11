from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import tensorflow as tf
import tensorflow_decision_forests as tfdf

def get_model(name="ridge"):
    name = name.lower()

    if name == "ridge":
        return Ridge(alpha=10.0)
    elif name == "lasso":
        return Lasso(alpha=0.001)
    elif name == "elastic":
        return ElasticNet(alpha=0.001, l1_ratio=0.5)
    elif name == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
    elif name == "xgb":
        return xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
    elif name == "lgbm":
        return lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42
        )
    elif name == "catboost":
        return cb.CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=False
        )
    elif name == "tfdf":
        return tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
    else:
        raise ValueError(f"Unknown model name: {name}")