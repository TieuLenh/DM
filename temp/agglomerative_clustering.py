
# MỤC 1: SỬ DỤNG THƯ VIỆN (SCIKIT-LEARN) CHO THUẬT TOÁN AGGLOMERATIVE CLUSTERING


import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.datasets import make_blobs


def run_agglomerative_sklearn(X, n_clusters=3, linkage='single'):
    # Khởi tạo mô hình AgglomerativeClustering:
    # n_clusters: số lượng cụm cuối cùng muốn dừng lại (cắt cây Dendrogram)
    # metric='euclidean': sử dụng khoảng cách Euclid giữa các điểm dữ liệu
    # linkage: phương pháp đo khoảng cách giữa 2 cụm ('single', 'complete', 'average', 'ward')
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='euclidean',
        linkage=linkage
    )

    # Thực hiện phân cụm phân cấp từ dưới lên (bottom-up) và trả về nhãn cụm cho từng điểm dữ liệu
    labels = model.fit_predict(X)

    # Trả về mô hình đã huấn luyện và mảng nhãn cụm dự đoán
    return model, labels



# MỤC 2: TỰ VIẾT CODE THUẬT TOÁN AGGLOMERATIVE CLUSTERING (DẠNG HÀM THƯỜNG)


def compute_pairwise_distances(X):
    # Tính ma trận khoảng cách Euclid đôi một giữa tất cả các điểm dữ liệu ban đầu
    # Sử dụng broadcasting để tính nhanh ma trận kích thước (n_samples, n_samples)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist_matrix


def compute_cluster_distance(cluster_a, cluster_b, point_dist_matrix, linkage='single'):
    # Tính khoảng cách giữa 2 cụm (cluster_a và cluster_b) dựa trên danh sách chỉ số điểm của từng cụm
    # Trích xuất ma trận con chứa khoảng cách giữa mọi cặp điểm (i thuộc A, j thuộc B)
    sub_distances = point_dist_matrix[np.ix_(cluster_a, cluster_b)]

    if linkage == 'single':
        # Liên kết đơn (Single Linkage): khoảng cách ngắn nhất giữa 2 điểm bất kỳ thuộc 2 cụm
        return np.min(sub_distances)
    if linkage == 'complete':
        # Liên kết đầy đủ (Complete Linkage): khoảng cách xa nhất giữa 2 điểm bất kỳ thuộc 2 cụm
        return np.max(sub_distances)
    if linkage == 'average':
        # Liên kết trung bình (Average Linkage): trung bình cộng khoảng cách mọi cặp điểm giữa 2 cụm
        return np.mean(sub_distances)
    raise ValueError(f"Linkage không hợp lệ: {linkage}")


def custom_agglomerative_fit_predict(X, n_clusters=3, linkage='single'):
    # Lấy tổng số điểm dữ liệu ban đầu
    n_samples = X.shape[0]

    # Bước 1: Tính sẵn ma trận khoảng cách Euclid giữa tất cả các cặp điểm dữ liệu
    point_dist_matrix = compute_pairwise_distances(X)

    # Bước 2: Khởi tạo ban đầu mỗi điểm dữ liệu là một cụm độc lập (tổng cộng n_samples cụm)
    # Dùng dictionary ánh xạ cluster_id -> danh sách các chỉ số điểm thuộc cụm đó
    active_clusters = {i: [i] for i in range(n_samples)}

    # Khởi tạo ma trận khoảng cách hiện hành giữa các cụm đang hoạt động
    cluster_dist = point_dist_matrix.copy()
    # Đặt đường chéo chính bằng vô cùng (np.inf) để một cụm không tự gộp với chính nó
    np.fill_diagonal(cluster_dist, np.inf)

    # Biến lưu lịch sử các bước gộp cụm (tương tự cấu trúc cây Dendrogram)
    merge_history = []
    next_cluster_id = n_samples

    # Bước 3: Vòng lặp phân tụ - liên tục gộp 2 cụm gần nhau nhất cho đến khi còn đúng n_clusters cụm
    while len(active_clusters) > n_clusters:
        # Tìm vị trí (i, j) có khoảng cách nhỏ nhất trong ma trận khoảng cách cụm hiện tại
        flat_idx = np.argmin(cluster_dist)
        idx_i, idx_j = np.unravel_index(flat_idx, cluster_dist.shape)
        min_dist = cluster_dist[idx_i, idx_j]

        # Đảm bảo idx_i < idx_j để thao tác xóa và cập nhật chỉ số nhất quán
        if idx_i > idx_j:
            idx_i, idx_j = idx_j, idx_i

        # Lấy danh sách ID các cụm đang hoạt động hiện tại
        current_ids = list(active_clusters.keys())
        id_i = current_ids[idx_i]
        id_j = current_ids[idx_j]

        # Gộp các điểm dữ liệu của cụm id_j vào cụm id_i tạo thành cụm mới
        merged_points = active_clusters[id_i] + active_clusters[id_j]

        # Ghi lại lịch sử bước gộp (cụm 1, cụm 2, khoảng cách gộp, số lượng điểm sau gộp)
        merge_history.append((id_i, id_j, min_dist, len(merged_points)))

        # Cập nhật dictionary các cụm hoạt động: xóa 2 cụm cũ và thêm cụm mới gộp
        del active_clusters[id_i]
        del active_clusters[id_j]
        active_clusters[next_cluster_id] = merged_points
        next_cluster_id += 1

        # Cập nhật lại ma trận khoảng cách cụm:
        # Xóa hàng/cột của 2 cụm cũ (idx_j trước rồi idx_i để không lệch chỉ số)
        cluster_dist = np.delete(cluster_dist, [idx_i, idx_j], axis=0)
        cluster_dist = np.delete(cluster_dist, [idx_i, idx_j], axis=1)

        # Tính khoảng cách từ cụm mới gộp tới tất cả các cụm còn lại
        remaining_ids = list(active_clusters.keys())[:-1]
        if len(remaining_ids) > 0:
            new_dists = np.array([
                compute_cluster_distance(merged_points, active_clusters[cid], point_dist_matrix, linkage=linkage)
                for cid in remaining_ids
            ])
            # Mở rộng thêm 1 hàng và 1 cột vào ma trận khoảng cách cho cụm mới
            cluster_dist = np.vstack([cluster_dist, new_dists])
            new_col = np.append(new_dists, np.inf).reshape(-1, 1)
            cluster_dist = np.hstack([cluster_dist, new_col])

    # Bước 4: Gán nhãn cụm cuối cùng (từ 0 đến n_clusters - 1) cho từng điểm dữ liệu ban đầu
    labels = np.zeros(n_samples, dtype=int)
    for cluster_label, (_, point_indices) in enumerate(active_clusters.items()):
        for pt_idx in point_indices:
            labels[pt_idx] = cluster_label

    return labels, merge_history



# MỤC 3: TẠO DATA MOCK VÀ KIỂM THỬ SO SÁNH 2 THUẬT TOÁN (THƯ VIỆN VS TỰ VIẾT)

if __name__ == "__main__":
    X, y_true = make_blobs(
        n_samples=150,
        n_features=2,
        centers=3,
        cluster_std=0.9,
        random_state=42
    )

    print(f"Kích thước tập dữ liệu Mock: {X.shape}, Số cụm thực tế: {len(np.unique(y_true))}\n")

    # Thử nghiệm so sánh trên cả 3 phương pháp liên kết phổ biến: 'single', 'complete', 'average'
    for linkage_method in ['single', 'complete', 'average']:
        # Chạy thuật toán từ thư viện Scikit-Learn
        sklearn_model, labels_sklearn = run_agglomerative_sklearn(
            X, n_clusters=3, linkage=linkage_method
        )

        # Chạy thuật toán tự viết dạng hàm
        labels_custom, merge_history = custom_agglomerative_fit_predict(
            X, n_clusters=3, linkage=linkage_method
        )

        # Đánh giá và so sánh hiệu năng giữa 2 phương pháp
        ari_match = adjusted_rand_score(labels_sklearn, labels_custom)
        ari_vs_true = adjusted_rand_score(y_true, labels_custom)
        sil_sklearn = silhouette_score(X, labels_sklearn)
        sil_custom = silhouette_score(X, labels_custom)

        print(f"--- Linkage: {linkage_method} ---")
        print(f"ARI trùng khớp (Sklearn vs Tự viết): {ari_match:.4f}")
        print(f"ARI so với nhãn thật               : {ari_vs_true:.4f}")
        print(f"Silhouette (Sklearn / Tự viết)     : {sil_sklearn:.4f} / {sil_custom:.4f}\n")
