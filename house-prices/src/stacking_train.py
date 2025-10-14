import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
from joblib import dump
from models import get_model
from preprocess import preprocess_data

def rmse_log(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

BASE_MODELS = ["ridge", "lasso", "elastic", "rf", "xgb", "lgbm", "catboost"]
META_MODEL = "xgb"
N_FOLDS = 5
RANDOM_STATE = 42

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X_train, X_test, y_train = preprocess_data(train, test)
y_train = np.log1p(y_train)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_predictions = np.zeros((len(X_train), len(BASE_MODELS)))
test_predictions = np.zeros((len(X_test), len(BASE_MODELS)))

for i, name in enumerate(BASE_MODELS):
    print(f"\nTraining the base model: {name.upper()}")
    model = get_model(name)

    oof_fold_preds = np.zeros(len(X_train))
    test_fold_preds = np.zeros((len(X_test), N_FOLDS))

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        oof_fold_preds[valid_idx] = y_pred
        test_fold_preds[:, fold] = model.predict(X_test)

        print(f"  Fold {fold + 1}: RMSE(log) = {rmse_log(y_val, y_pred):.5f}")

    oof_predictions[:, i] = oof_fold_preds
    test_predictions[:, i] = test_fold_preds.mean(axis=1)

meta_model = get_model(META_MODEL)
meta_model.fit(oof_predictions, y_train)

meta_oof_pred = meta_model.predict(oof_predictions)
meta_rmse = rmse_log(y_train, meta_oof_pred)

final_predictions = np.expm1(meta_model.predict(test_predictions))

submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": final_predictions
})

submission.to_csv("submissions/submission_stacking.csv", index=False)
dump(meta_model, "models/meta_model.pkl")

print("\nSaved meta_model.pkl and submission_stacking.csv")