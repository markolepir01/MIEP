import pandas as pd
import joblib
import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from config import TEST_PATH, SUBMISSION_DIR, ID_COL
from utils import inverse_log

def main(model_name):
    model_path = f"models/{model_name}_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Run train.py first!")

    pipe = joblib.load(model_path)
    test_df = pd.read_csv(TEST_PATH)

    ids = test_df[ID_COL]
    X_test = test_df.drop(columns=[ID_COL])

    preds = pipe.predict(X_test)
    preds = inverse_log(preds)

    submission = pd.DataFrame({
        ID_COL: ids,
        "SalePrice": preds
    })
    out_path = os.path.join(SUBMISSION_DIR, f"submission_{model_name}.csv")
    submission.to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ridge",
                        help="Model name used in train.py")
    args = parser.parse_args()
    main(args.model)