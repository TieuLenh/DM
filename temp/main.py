import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def preprocessing(file_path: str, option: dict | None = None):
    default_option = {
        "delimiter": ",",
        "skipinitialspace": True,
        "standardize_sym_col": LabelEncoder,
        "standardize_num_col": MinMaxScaler,
    }
    cfg = {**default_option, **(option or {})}

    df = pd.read_csv(
        file_path,
        delimiter=cfg["delimiter"],
        skipinitialspace=cfg["skipinitialspace"],
    )

    num_cols = df.select_dtypes(include="number").columns.tolist()
    sym_cols = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    sym_encoder_cls = cfg["standardize_sym_col"] or LabelEncoder
    num_scaler_cls = cfg["standardize_num_col"] or MinMaxScaler

    if sym_cols:
        df[sym_cols] = df[sym_cols].apply(
            lambda col: sym_encoder_cls().fit_transform(col)
        ).astype(int)

    if num_cols:
        df[num_cols] = num_scaler_cls().fit_transform(df[num_cols])

    return df, {"sym_cols": sym_cols, "num_cols": num_cols}


df, cols = preprocessing('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

print(df, cols)

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Giả sử cột mục tiêu là 'species'
X = df.drop(columns=["species"])
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # 2 lớp ẩn: 64 và 32 nơ-ron
    activation="relu",
    max_iter=500,
    random_state=42,
)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))