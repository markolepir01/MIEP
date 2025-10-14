import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def get_preprocessor(df):
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor

def preprocess_data(train_df, test_df):
    y_train = train_df["SalePrice"]
    X_train = train_df.drop(columns=["SalePrice"])
    X_test = test_df.copy()

    combined = pd.concat([X_train, X_test], axis=0)
    preprocessor = get_preprocessor(combined)

    processed = preprocessor.fit_transform(combined)

    num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    cat_features = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out()
    feature_names = np.concatenate([num_features, cat_features])

    X_train_transformed = processed[:len(X_train)]
    X_test_transformed = processed[len(X_train):]

    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    return X_train_df, X_test_df, y_train