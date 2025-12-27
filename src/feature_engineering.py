import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def separate_features_target(df: pd.DataFrame, target_column: str):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def encode_categorical_features(X: pd.DataFrame):
    # Drop high-cardinality / text columns
    drop_cols = ["description", "amenities", "name", "thumbnail_url"]
    X = X.drop(columns=[col for col in drop_cols if col in X.columns])

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )

    encoded_data = encoder.fit_transform(X[categorical_cols])

    encoded_df = pd.DataFrame.sparse.from_spmatrix(
        encoded_data,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    final_df = pd.concat(
        [
            X[numerical_cols].reset_index(drop=True),
            encoded_df.reset_index(drop=True)
        ],
        axis=1
    )

    return final_df, encoder
