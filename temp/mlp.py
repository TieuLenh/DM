import numpy as np


def init_mlp_params(layer_dims: list[int], seed: int = 42) -> dict[str, np.ndarray]:
    """Khởi tạo trọng số (He initialization) và bias cho các lớp."""
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {}
    for l in range(1, len(layer_dims)):
        n_in = layer_dims[l - 1]
        n_out = layer_dims[l]
        params[f"W{l}"] = rng.standard_normal((n_out, n_in)) * np.sqrt(2.0 / n_in)
        params[f"b{l}"] = np.zeros((n_out, 1))
    return params


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(np.float64)


def softmax(z: np.ndarray) -> np.ndarray:
    # Trừ max theo cột để tránh tràn số (overflow) khi tính exp
    shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def to_one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = y.astype(int).ravel()
    one_hot = np.zeros((num_classes, y.size), dtype=np.float64)
    one_hot[y, np.arange(y.size)] = 1.0
    return one_hot


def compute_mlp_loss(
    y_hat: np.ndarray,
    y_one_hot: np.ndarray,
    params: dict[str, np.ndarray],
    layer_dims: list[int],
    reg_lambda: float = 0.0,
) -> float:
    m = y_one_hot.shape[1]
    eps = 1e-12
    ce_loss = -np.sum(y_one_hot * np.log(y_hat + eps)) / m
    l2_loss = 0.0
    if reg_lambda > 0.0:
        L = len(layer_dims) - 1
        l2_loss = 0.5 * reg_lambda * sum(
            np.sum(params[f"W{l}"] ** 2) for l in range(1, L + 1)
        ) / m
    return float(ce_loss + l2_loss)


def mlp_forward(
    X: np.ndarray,
    params: dict[str, np.ndarray],
    layer_dims: list[int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    X có dạng (n_features, m_samples).
    Trả về xác suất đầu ra (n_classes, m_samples) và cache các giá trị trung gian.
    """
    L = len(layer_dims) - 1
    cache: dict[str, np.ndarray] = {"A0": X}
    a_prev = X

    for l in range(1, L):
        W = params[f"W{l}"]
        b = params[f"b{l}"]
        z = W @ a_prev + b
        a = relu(z)
        cache[f"Z{l}"] = z
        cache[f"A{l}"] = a
        a_prev = a

    # Lớp cuối cùng: Softmax
    WL = params[f"W{L}"]
    bL = params[f"b{L}"]
    zL = WL @ a_prev + bL
    aL = softmax(zL)
    cache[f"Z{L}"] = zL
    cache[f"A{L}"] = aL

    return aL, cache


def mlp_backward(
    y_one_hot: np.ndarray,
    cache: dict[str, np.ndarray],
    params: dict[str, np.ndarray],
    layer_dims: list[int],
    reg_lambda: float = 0.0,
) -> dict[str, np.ndarray]:
    """Tính gradient cho toàn bộ trọng số và bias."""
    L = len(layer_dims) - 1
    m = y_one_hot.shape[1]
    grads: dict[str, np.ndarray] = {}

    # Gradient tại lớp output (Softmax + Cross-Entropy)
    dz = (cache[f"A{L}"] - y_one_hot) / m

    for l in range(L, 0, -1):
        a_prev = cache[f"A{l - 1}"]
        W = params[f"W{l}"]

        dW = dz @ a_prev.T
        if reg_lambda > 0.0:
            dW += (reg_lambda / m) * W
        db = np.sum(dz, axis=1, keepdims=True)

        grads[f"dW{l}"] = dW
        grads[f"db{l}"] = db

        if l > 1:
            da_prev = W.T @ dz
            dz = da_prev * relu_grad(cache[f"Z{l - 1}"])

    return grads


def update_mlp_params(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    layer_dims: list[int],
    learning_rate: float,
) -> None:
    L = len(layer_dims) - 1
    for l in range(1, L + 1):
        params[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        params[f"b{l}"] -= learning_rate * grads[f"db{l}"]


def mlp_predict_proba(
    X: np.ndarray,
    params: dict[str, np.ndarray],
    layer_dims: list[int],
) -> np.ndarray:
    """Trả về ma trận xác suất dạng (m_samples, n_classes)."""
    probs, _ = mlp_forward(X.T, params, layer_dims)
    return probs.T


def mlp_predict(
    X: np.ndarray,
    params: dict[str, np.ndarray],
    layer_dims: list[int],
) -> np.ndarray:
    """Trả về nhãn dự đoán dạng (m_samples,)."""
    return np.argmax(mlp_predict_proba(X, params, layer_dims), axis=1)


def fit_mlp(
    X: np.ndarray,
    y: np.ndarray,
    layer_dims: list[int],
    learning_rate: float = 0.01,
    reg_lambda: float = 0.0,
    epochs: int = 500,
    batch_size: int = 32,
    verbose_every: int = 100,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], list[float]]:
    """
    Huấn luyện mô hình MLP bằng các hàm thuần túy.
    X: (m_samples, n_features)
    y: (m_samples,) chứa nhãn số nguyên từ 0 .. C-1
    Trả về (params, loss_history).
    """
    rng = np.random.default_rng(seed)
    params = init_mlp_params(layer_dims, seed=seed)

    X_T = X.T  # chuyển sang (n_features, m_samples)
    m = X_T.shape[1]
    num_classes = layer_dims[-1]
    y_oh_all = to_one_hot(y, num_classes)
    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        indices = rng.permutation(m)
        for start in range(0, m, batch_size):
            batch_idx = indices[start : start + batch_size]
            Xb = X_T[:, batch_idx]
            yb_oh = y_oh_all[:, batch_idx]

            y_hat, cache = mlp_forward(Xb, params, layer_dims)
            grads = mlp_backward(yb_oh, cache, params, layer_dims, reg_lambda=reg_lambda)
            update_mlp_params(params, grads, layer_dims, learning_rate)

        full_probs, _ = mlp_forward(X_T, params, layer_dims)
        epoch_loss = compute_mlp_loss(full_probs, y_oh_all, params, layer_dims, reg_lambda=reg_lambda)
        loss_history.append(epoch_loss)

        if verbose_every and (epoch % verbose_every == 0 or epoch == 1):
            preds = np.argmax(full_probs, axis=0)
            acc = np.mean(preds == y.ravel())
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {acc:.4f}")

    return params, loss_history



# HÀM GỌI TIỆN LỢI (TRAIN + EVAL + PREDICT)

def run_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    hidden_layers: list[int] | None = None,
    learning_rate: float = 0.05,
    epochs: int = 500,
    batch_size: int = 32,
    reg_lambda: float = 0.0,
    seed: int = 42,
) -> dict:
    if hidden_layers is None:
        hidden_layers = [32, 16]

    n_features = X_train.shape[1]
    num_classes = int(np.max(y_train)) + 1
    layer_dims = [n_features, *hidden_layers, num_classes]

    params, losses = fit_mlp(
        X=X_train,
        y=y_train,
        layer_dims=layer_dims,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        epochs=epochs,
        batch_size=batch_size,
        verbose_every=max(1, epochs // 5),
        seed=seed,
    )

    train_preds = mlp_predict(X_train, params, layer_dims)
    result: dict = {
        "params": params,
        "layer_dims": layer_dims,
        "losses": losses,
        "train_acc": float(np.mean(train_preds == y_train.ravel())),
    }

    if X_test is not None:
        y_pred_test = mlp_predict(X_test, params, layer_dims)
        result["test_preds"] = y_pred_test
        if y_test is not None:
            result["test_acc"] = float(np.mean(y_pred_test == y_test.ravel()))

    return result


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    # Tạo dữ liệu mẫu: 300 mẫu, 4 đặc trưng, 3 lớp (0, 1, 2)
    X = rng.standard_normal((300, 4))
    # Gán nhãn có quy luật phi tuyến để mạng có thể học
    scores = np.column_stack([
        X[:, 0] ** 2 - X[:, 1],
        X[:, 2] + 0.5 * X[:, 3] ** 2,
        -X[:, 0] + X[:, 1] - X[:, 2],
    ])
    y = np.argmax(scores, axis=1)

    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]

    out = run_mlp(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        hidden_layers=[24, 12],
        learning_rate=0.05,
        epochs=400,
        batch_size=32,
        reg_lambda=1e-3,
        seed=42,
    )
    print(f"Train Acc: {out['train_acc']:.4f} | Test Acc: {out.get('test_acc'):.4f}")