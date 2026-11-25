
# MỤC 1: SỬ DỤNG THƯ VIỆN (SCIKIT-LEARN) CHO THUẬT TOÁN NAIVE BAYES


import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification


def train_and_predict_sklearn(X_train, y_train, X_test):
    # Khởi tạo mô hình Gaussian Naive Bayes từ thư viện sklearn
    model = GaussianNB()

    # Huấn luyện mô hình trên tập dữ liệu huấn luyện
    model.fit(X_train, y_train)

    # Dự đoán nhãn lớp và xác suất rơi vào từng lớp
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    return model, y_pred, y_proba



# MỤC 2: TỰ VIẾT CODE THUẬT TOÁN GAUSSIAN NAIVE BAYES TỪ ĐẦU (DẠNG HÀM THƯỜNG)


def custom_gaussian_nb_fit(X, y):
    # Lấy số lượng mẫu (n_samples) và số lượng đặc trưng (n_features)
    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    mean = np.zeros((n_classes, n_features), dtype=np.float64)
    var = np.zeros((n_classes, n_features), dtype=np.float64)
    priors = np.zeros(n_classes, dtype=np.float64)

    # Duyệt qua từng lớp để tính các tham số thống kê
    for idx, c in enumerate(classes):
        X_c = X[y == c]
        mean[idx, :] = X_c.mean(axis=0)
        var[idx, :] = X_c.var(axis=0) + 1e-9
        priors[idx] = X_c.shape[0] / float(n_samples)

    return {
        "classes": classes,
        "mean": mean,
        "var": var,
        "priors": priors,
    }


def gaussian_pdf(class_idx, x, params):
    # Tính hàm mật độ xác suất phân phối chuẩn (Gaussian PDF) cho vector đặc trưng x tại lớp class_idx
    mean = params["mean"][class_idx]
    var = params["var"][class_idx]
    numerator = np.exp(-((x - mean) ** 2) / (2 * var))
    denominator = np.sqrt(2 * np.pi * var)
    return numerator / denominator


def custom_gaussian_nb_predict_single(x, params):
    # Tính xác suất hậu nghiệm cho từng lớp theo định lý Bayes
    posteriors = []
    for idx, _ in enumerate(params["classes"]):
        prior = np.log(params["priors"][idx])
        class_conditional = np.sum(np.log(gaussian_pdf(idx, x, params) + 1e-12))
        posterior = prior + class_conditional
        posteriors.append(posterior)

    return params["classes"][np.argmax(posteriors)]


def custom_gaussian_nb_predict(X, params):
    # Áp dụng hàm dự đoán cho từng mẫu trong tập dữ liệu X
    return np.array([custom_gaussian_nb_predict_single(x, params) for x in X])


def train_and_predict_custom_nb(X_train, y_train, X_test):
    # Hàm gọi nhanh huấn luyện và dự đoán bằng Gaussian Naive Bayes tự viết
    params = custom_gaussian_nb_fit(X_train, y_train)
    y_pred = custom_gaussian_nb_predict(X_test, params)
    return params, y_pred



# MỤC 3: TẠO DATA MOCK VÀ KIỂM THỬ SO SÁNH 2 THUẬT TOÁN (THƯ VIỆN VS TỰ VIẾT)


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Kích thước tập huấn luyện: {X_train.shape}, Tập kiểm thử: {X_test.shape}\n")

    # Bước 2: Chạy thử nghiệm với mô hình từ thư viện Scikit-Learn
    sklearn_model, y_pred_sklearn, y_proba_sklearn = train_and_predict_sklearn(X_train, y_train, X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

    # Bước 3: Chạy thử nghiệm với mô hình tự viết dạng hàm
    custom_params, y_pred_custom = train_and_predict_custom_nb(X_train, y_train, X_test)
    acc_custom = accuracy_score(y_test, y_pred_custom)

    # Bước 4: In kết quả so sánh giữa 2 phương pháp
    print("=== KẾT QUẢ ĐÁNH GIÁ & SO SÁNH ===")
    print(f"Độ chính xác (Accuracy) - Thư viện Sklearn : {acc_sklearn * 100:.2f}%")
    print(f"Độ chính xác (Accuracy) - Thuật toán tự viết: {acc_custom * 100:.2f}%")

    match_rate = np.mean(y_pred_sklearn == y_pred_custom) * 100
    print(f"Tỷ lệ dự đoán trùng khớp giữa 2 mô hình      : {match_rate:.2f}%\n")

    print("Ma trận nhầm lẫn (Confusion Matrix - Custom Model):")
    print(confusion_matrix(y_test, y_pred_custom))
    print("\nBáo cáo phân loại chi tiết (Classification Report - Custom Model):")
    print(classification_report(y_test, y_pred_custom))

    print("Đối chiếu 5 mẫu đầu tiên trên tập test (Thực tế | Sklearn | Tự viết):")
    for i in range(5):
        print(f"Mẫu {i + 1}: Nhãn thực tế = {y_test[i]} | Sklearn = {y_pred_sklearn[i]} | Tự viết = {y_pred_custom[i]}")
