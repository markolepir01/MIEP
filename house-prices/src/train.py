import pandas as pd
import numpy as np
import argparse
import joblib
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from config import TRAIN_PATH, MODEL_DIR, TARGET, RANDOM_STATE, N_FOLDS
from preprocess import get_preprocessor
from models import get_model
from utils import rmse_log, log_transform
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def main(model_name):
    df = pd.read_csv(TRAIN_PATH)
    y = log_transform(df[TARGET])
    X = df.drop(columns=[TARGET, "Id"])

    model = get_model(model_name)
    preprocessor = get_preprocessor(X)
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        rmse = rmse_log(y_val, y_pred)
        scores.append(rmse)
        print(f"[{model_name}] Fold {fold+1}: RMSE(log) = {rmse:.5f}")

    print(f"\n[{model_name}] Mean CV RMSE(log): {np.mean(scores):.5f}")

    joblib.dump(pipe, f"{MODEL_DIR}/{model_name}_model.pkl")
    print(f"Model saved → {MODEL_DIR}/{model_name}_model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ridge",
                        help="Model name: ridge, lasso, elastic, rf, xgb, lgbm, catboost")
    args = parser.parse_args()
    main(args.model)