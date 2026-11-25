
# MỤC 1: SỬ DỤNG THƯ VIỆN (SCIKIT-LEARN) CHO THUẬT TOÁN K-MEANS CLUSTERING


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.datasets import make_blobs


def run_kmeans_sklearn(X, n_clusters=3, max_iter=300, random_state=42):
    # Khởi tạo mô hình KMeans với số cụm k (n_clusters), số vòng lặp tối đa và hạt giống ngẫu nhiên
    model = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=max_iter,
        random_state=random_state
    )

    # Huấn luyện mô hình và gán nhãn cụm trực tiếp cho từng điểm dữ liệu trong X
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_
    inertia = model.inertia_

    return model, labels, centroids, inertia



# MỤC 2: TỰ VIẾT CODE THUẬT TOÁN K-MEANS TỪ ĐẦU (DẠNG HÀM THƯỜNG)


def initialize_centroids(X, n_clusters, random_state=42):
    # Đặt hạt giống ngẫu nhiên và chọn ngẫu nhiên k chỉ số điểm không trùng lặp làm tâm cụm ban đầu
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    random_indices = rng.choice(n_samples, size=n_clusters, replace=False)
    return X[random_indices].copy()


def compute_kmeans_distances(X, centroids):
    # Tính khoảng cách Euclid từ mỗi điểm dữ liệu trong X tới từng tâm cụm
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    return distances


def assign_kmeans_clusters(distances):
    # Gán mỗi điểm dữ liệu vào cụm có tâm gần nhất
    return np.argmin(distances, axis=1)


def update_kmeans_centroids(X, labels, n_clusters, old_centroids):
    # Cập nhật tọa độ các tâm cụm mới dựa trên trung bình cộng các điểm thuộc mỗi cụm
    new_centroids = np.zeros((n_clusters, X.shape[1]), dtype=np.float64)
    for k in range(n_clusters):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            new_centroids[k] = cluster_points.mean(axis=0)
        else:
            new_centroids[k] = old_centroids[k]
    return new_centroids


def compute_kmeans_inertia(X, labels, centroids, n_clusters):
    # Tính tổng bình phương khoảng cách (WCSS / Inertia) từ mỗi điểm đến tâm cụm của nó
    inertia = 0.0
    for k in range(n_clusters):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            inertia += np.sum((cluster_points - centroids[k]) ** 2)
    return inertia


def custom_kmeans_fit(X, n_clusters=3, max_iter=300, tol=1e-4, random_state=42):
    # Bước 1: Khởi tạo tọa độ các tâm cụm ban đầu
    centroids = initialize_centroids(X, n_clusters, random_state=random_state)
    labels = None

    # Vòng lặp tối ưu hóa luân phiên giữa Gán cụm và Cập nhật tâm
    for _ in range(max_iter):
        distances = compute_kmeans_distances(X, centroids)
        labels = assign_kmeans_clusters(distances)
        new_centroids = update_kmeans_centroids(X, labels, n_clusters, centroids)

        centroid_shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if centroid_shift < tol:
            break

    inertia = compute_kmeans_inertia(X, labels, centroids, n_clusters)
    return labels, centroids, inertia


def custom_kmeans_predict(X, centroids):
    # Dự đoán nhãn cụm cho dữ liệu mới dựa trên các tâm cụm đã học được
    distances = compute_kmeans_distances(X, centroids)
    return assign_kmeans_clusters(distances)



# MỤC 3: TẠO DATA MOCK VÀ KIỂM THỬ SO SÁNH 2 THUẬT TOÁN (THƯ VIỆN VS TỰ VIẾT)


if __name__ == "__main__":
    X, y_true = make_blobs(
        n_samples=500,
        n_features=2,
        centers=4,
        cluster_std=1.1,
        random_state=42
    )

    print(f"Kích thước tập dữ liệu Mock: {X.shape}, Số cụm thực tế: {len(np.unique(y_true))}\n")

    # Bước 2: Chạy thử nghiệm phân cụm với thư viện Scikit-Learn
    sklearn_model, labels_sklearn, centroids_sklearn, inertia_sklearn = run_kmeans_sklearn(
        X, n_clusters=4, random_state=42
    )

    # Bước 3: Chạy thử nghiệm phân cụm với thuật toán tự viết dạng hàm
    labels_custom, centroids_custom, inertia_custom = custom_kmeans_fit(
        X, n_clusters=4, max_iter=300, random_state=42
    )

    # Bước 4: Đánh giá và so sánh hiệu năng giữa 2 phương pháp
    ari_between_models = adjusted_rand_score(labels_sklearn, labels_custom)
    ari_sklearn_vs_true = adjusted_rand_score(y_true, labels_sklearn)
    ari_custom_vs_true = adjusted_rand_score(y_true, labels_custom)

    sil_sklearn = silhouette_score(X, labels_sklearn)
    sil_custom = silhouette_score(X, labels_custom)

    print("=== KẾT QUẢ ĐÁNH GIÁ & SO SÁNH K-MEANS ===")
    print(f"Inertia (WCSS) - Thư viện Sklearn  : {inertia_sklearn:.4f}")
    print(f"Inertia (WCSS) - Thuật toán tự viết : {inertia_custom:.4f}\n")

    print(f"Silhouette Score - Thư viện Sklearn : {sil_sklearn:.4f}")
    print(f"Silhouette Score - Thuật toán tự viết: {sil_custom:.4f}\n")

    print(f"Adjusted Rand Index (so với nhãn gốc) - Sklearn : {ari_sklearn_vs_true:.4f}")
    print(f"Adjusted Rand Index (so với nhãn gốc) - Tự viết : {ari_custom_vs_true:.4f}")
    print(f"Độ tương đồng phân cụm giữa 2 mô hình (ARI)     : {ari_between_models * 100:.2f}%\n")

    sorted_centroids_sklearn = centroids_sklearn[np.argsort(centroids_sklearn[:, 0])]
    sorted_centroids_custom = centroids_custom[np.argsort(centroids_custom[:, 0])]

    print("Tọa độ 4 tâm cụm (đã sắp xếp) - Thư viện Sklearn:")
    print(np.round(sorted_centroids_sklearn, 4))
    print("\nTọa độ 4 tâm cụm (đã sắp xếp) - Thuật toán tự viết:")
    print(np.round(sorted_centroids_custom, 4))

    new_samples = np.array([[-8.0, 7.0], [4.5, 2.0], [-2.5, 9.0]])
    print("\nDự đoán cụm cho 3 điểm dữ liệu mới:")
    print("Sklearn dự đoán :", sklearn_model.predict(new_samples))
    print("Tự viết dự đoán :", custom_kmeans_predict(new_samples, centroids_custom))
