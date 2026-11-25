
# THUẬT TOÁN AI: ID3 (ITERATIVE DICHOTOMISER 3)


from collections import Counter
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42



# 1. SỬ DỤNG THƯ VIỆN (CHEFBOOST) & HÀM THAM KHẢO


def entropy_from_series(y):
    probabilities = y.value_counts(normalize=True).to_numpy()
    return -np.sum(probabilities * np.log2(probabilities + 1e-12))


def information_gain_from_dataframe(df, feature, target):
    parent_entropy = entropy_from_series(df[target])
    weighted_child_entropy = 0.0

    for _, subset in df.groupby(feature):
        weighted_child_entropy += (
            len(subset) / len(df)
        ) * entropy_from_series(subset[target])

    return parent_entropy - weighted_child_entropy


def run_id3_chefboost(train_df, X_test, target_label="Play"):
    from chefboost import Chefboost as chef

    # Ép kiểu object để tránh lỗi StringDtype trên các phiên bản pandas mới
    df_train_obj = train_df.astype(object)
    config = {"algorithm": "ID3"}
    model_lib = chef.fit(df_train_obj.copy(), config=config, target_label=target_label)

    pred_lib = []
    for _, row in X_test.iterrows():
        prediction = chef.predict(model_lib, row.tolist())
        if isinstance(prediction, (list, tuple, np.ndarray)):
            prediction = prediction[0]
        pred_lib.append(prediction)

    return model_lib, pred_lib



# 2. TỰ CÀI ĐẶT THUẬT TOÁN ID3 (DẠNG HÀM THƯỜNG)


def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total)
                for count in counts.values())


def information_gain(data, feature, target):
    parent_entropy = entropy(data[target].tolist())
    total = len(data)
    weighted_entropy = 0.0

    for _, subset in data.groupby(feature, dropna=False):
        weighted_entropy += (len(subset) / total) * entropy(subset[target].tolist())

    return parent_entropy - weighted_entropy


def majority_class(labels):
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def build_id3_tree(data, features, target):
    labels = data[target].tolist()

    # Trường hợp 1: tất cả mẫu cùng một lớp.
    if len(set(labels)) == 1:
        return labels[0]

    # Trường hợp 2: không còn thuộc tính để chia.
    if not features:
        return majority_class(labels)

    gains = {feature: information_gain(data, feature, target)
             for feature in features}
    best_feature = max(gains, key=gains.get)

    # Không có lợi ích thông tin đáng kể: tạo lá theo lớp đa số.
    if gains[best_feature] <= 1e-12:
        return majority_class(labels)

    tree = {
        "feature": best_feature,
        "default": majority_class(labels),
        "branches": {}
    }

    remaining_features = [f for f in features if f != best_feature]

    for value, subset in data.groupby(best_feature, dropna=False):
        tree["branches"][value] = build_id3_tree(
            subset.drop(columns=[best_feature]),
            remaining_features,
            target
        )

    return tree


def predict_one(tree, sample):
    if not isinstance(tree, dict):
        return tree

    feature = tree["feature"]
    value = sample.get(feature)

    if value not in tree["branches"]:
        return tree["default"]

    return predict_one(tree["branches"][value], sample)


def predict_id3(tree, X):
    return [predict_one(tree, row) for row in X.to_dict(orient="records")]


def run_id3_custom(train_df, X_test, target="Play"):
    # Hàm gọi nhanh huấn luyện và dự đoán bằng ID3 tự cài đặt
    features = [col for col in train_df.columns if col != target]
    tree = build_id3_tree(train_df.copy(), features, target=target)
    preds = predict_id3(tree, X_test)
    return tree, preds


def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "-> " + str(tree))
        return

    feature = tree["feature"]
    print(indent + f"[{feature}]")
    for value, child in tree["branches"].items():
        print(indent + f"  {feature} = {value}")
        print_tree(child, indent + "    ")



# 3. DATA MOCK VÀ KIỂM THỬ HAI THUẬT TOÁN


if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_STATE)
    n_samples = 500

    weather_values = ["Sunny", "Cloudy", "Rainy"]
    temperature_values = ["Hot", "Mild", "Cool"]
    humidity_values = ["High", "Normal"]
    wind_values = ["Strong", "Weak"]

    rows = []
    for _ in range(n_samples):
        weather = rng.choice(weather_values, p=[0.38, 0.35, 0.27])
        temperature = rng.choice(temperature_values, p=[0.30, 0.45, 0.25])
        humidity = rng.choice(humidity_values, p=[0.48, 0.52])
        wind = rng.choice(wind_values, p=[0.35, 0.65])

        if weather == "Rainy":
            play = "No"
        elif weather == "Sunny" and humidity == "High":
            play = "No"
        elif weather == "Cloudy":
            play = "Yes"
        elif temperature == "Cool" and wind == "Weak":
            play = "Yes"
        else:
            play = "Yes" if rng.random() < 0.55 else "No"

        if rng.random() < 0.04:
            play = "No" if play == "Yes" else "Yes"

        rows.append({
            "Weather": weather,
            "Temperature": temperature,
            "Humidity": humidity,
            "Wind": wind,
            "Play": play
        })

    df = pd.DataFrame(rows)
    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=df["Play"]
    )

    X_train = train_df.drop(columns=["Play"])
    X_test = test_df.drop(columns=["Play"])
    y_test = test_df["Play"]

    # Kiểm thử ID3 tự cài đặt
    tree_custom, pred_custom = run_id3_custom(train_df, X_test, target="Play")
    print("Cây quyết định ID3 tự cài đặt:")
    print_tree(tree_custom)

    acc_custom = accuracy_score(y_test, pred_custom)
    print(f"\nAccuracy (ID3 tự viết): {acc_custom:.4f}")
    print(classification_report(y_test, pred_custom, zero_division=0))

    # Kiểm thử Chefboost nếu đã cài đặt
    try:
        model_lib, pred_lib = run_id3_chefboost(train_df, X_test, target_label="Play")
        acc_lib = accuracy_score(y_test, pred_lib)
        print(f"Accuracy (Chefboost ID3): {acc_lib:.4f}")
    except ImportError:
        print("Thư viện chefboost chưa được cài đặt, bỏ qua bước so sánh với chefboost.")
