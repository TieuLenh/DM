import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def to_one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = y.astype(int).ravel()
    oh = np.zeros((y.size, num_classes), dtype=np.float64)
    oh[np.arange(y.size), y] = 1.0
    return oh


def cross_entropy_loss(probs: np.ndarray, y_oh: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.mean(np.sum(y_oh * np.log(probs + eps), axis=1)))


def init_cnn_params(
    input_shape: tuple[int, int, int],
    num_classes: int,
    conv_filters: list[int],
    hidden_dim: int = 64,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    pool_size: int = 2,
    pool_stride: int = 2,
    seed: int = 42,
) -> dict:
    """Khởi tạo trọng số và cấu hình cho mạng CNN dạng dictionary."""
    rng = np.random.default_rng(seed)
    c_in, h, w = input_shape
    conv_layers = []

    for f_out in conv_filters:
        fan_in = c_in * kernel_size * kernel_size
        W = rng.standard_normal((f_out, c_in, kernel_size, kernel_size)) * np.sqrt(2.0 / fan_in)
        b = np.zeros((f_out,), dtype=np.float64)
        conv_layers.append({
            "W": W,
            "b": b,
            "stride": stride,
            "padding": padding,
            "pool_size": pool_size,
            "pool_stride": pool_stride,
        })

        h_conv = (h + 2 * padding - kernel_size) // stride + 1
        w_conv = (w + 2 * padding - kernel_size) // stride + 1
        h = (h_conv - pool_size) // pool_stride + 1
        w = (w_conv - pool_size) // pool_stride + 1
        c_in = f_out

    if h <= 0 or w <= 0:
        raise ValueError("Kích thước ảnh quá nhỏ so với số lớp conv/pool.")

    flat_dim = c_in * h * w
    W1 = rng.standard_normal((flat_dim, hidden_dim)) * np.sqrt(2.0 / flat_dim)
    b1 = np.zeros((1, hidden_dim), dtype=np.float64)
    W2 = rng.standard_normal((hidden_dim, num_classes)) * np.sqrt(1.0 / hidden_dim)
    b2 = np.zeros((1, num_classes), dtype=np.float64)

    return {
        "conv_layers": conv_layers,
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
    }


def conv_relu_forward(
    X: np.ndarray, W: np.ndarray, b: np.ndarray, stride: int, padding: int
) -> tuple[np.ndarray, dict]:
    """Tích chập 2D + ReLU. X: (N, C_in, H, W) -> A: (N, C_out, H_out, W_out)."""
    N, _, H, W_in = X.shape
    _, _, kh, kw = W.shape
    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W_in + 2 * padding - kw) // stride + 1

    X_pad = np.pad(
        X,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    Z = np.zeros((N, W.shape[0], H_out, W_out), dtype=np.float64)

    for i in range(H_out):
        hs = i * stride
        for j in range(W_out):
            ws = j * stride
            patch = X_pad[:, :, hs : hs + kh, ws : ws + kw]
            Z[:, :, i, j] = np.tensordot(patch, W, axes=([1, 2, 3], [1, 2, 3])) + b

    A = np.maximum(0.0, Z)
    cache = {"X": X, "X_pad": X_pad, "Z": Z, "W": W, "stride": stride, "padding": padding}
    return A, cache


def conv_relu_backward(dA: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lan truyền ngược qua ReLU + Conv2D."""
    X = cache["X"]
    X_pad = cache["X_pad"]
    Z = cache["Z"]
    W = cache["W"]
    stride = cache["stride"]
    padding = cache["padding"]

    _, _, kh, kw = W.shape
    _, _, H_out, W_out = dA.shape

    dZ = dA * (Z > 0.0)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 2, 3))
    dX_pad = np.zeros_like(X_pad)

    for i in range(H_out):
        hs = i * stride
        for j in range(W_out):
            ws = j * stride
            patch = X_pad[:, :, hs : hs + kh, ws : ws + kw]
            dz_ij = dZ[:, :, i, j]
            dW += np.tensordot(dz_ij, patch, axes=([0], [0]))
            dX_pad[:, :, hs : hs + kh, ws : ws + kw] += np.tensordot(dz_ij, W, axes=([1], [0]))

    if padding > 0:
        dX = dX_pad[:, :, padding:-padding, padding:-padding]
    else:
        dX = dX_pad

    return dX, dW, db


def maxpool_forward(X: np.ndarray, pool_size: int, stride: int) -> tuple[np.ndarray, dict]:
    """Max Pooling 2D. X: (N, C, H, W) -> out: (N, C, H_out, W_out)."""
    N, C, H, W = X.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    out = np.zeros((N, C, H_out, W_out), dtype=np.float64)

    for i in range(H_out):
        hs = i * stride
        for j in range(W_out):
            ws = j * stride
            window = X[:, :, hs : hs + pool_size, ws : ws + pool_size]
            out[:, :, i, j] = np.max(window, axis=(2, 3))

    cache = {"X": X, "pool_size": pool_size, "stride": stride, "H_out": H_out, "W_out": W_out}
    return out, cache


def maxpool_backward(dout: np.ndarray, cache: dict) -> np.ndarray:
    """Lan truyền ngược qua MaxPool2D."""
    X = cache["X"]
    ph = cache["pool_size"]
    pw = cache["pool_size"]
    stride = cache["stride"]
    H_out = cache["H_out"]
    W_out = cache["W_out"]

    dX = np.zeros_like(X)
    for i in range(H_out):
        hs = i * stride
        for j in range(W_out):
            ws = j * stride
            window = X[:, :, hs : hs + ph, ws : ws + pw]
            max_val = np.max(window, axis=(2, 3), keepdims=True)
            mask = (window == max_val).astype(np.float64)
            mask_sum = np.sum(mask, axis=(2, 3), keepdims=True)
            norm_mask = mask / np.maximum(mask_sum, 1.0)
            dX[:, :, hs : hs + ph, ws : ws + pw] += dout[:, :, i, j][:, :, None, None] * norm_mask

    return dX


def cnn_forward(X: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    """Lan truyền tiến toàn bộ mạng CNN."""
    feat = X
    block_caches = []

    for layer in params["conv_layers"]:
        a_conv, conv_cache = conv_relu_forward(
            feat, layer["W"], layer["b"], layer["stride"], layer["padding"]
        )
        feat, pool_cache = maxpool_forward(a_conv, layer["pool_size"], layer["pool_stride"])
        block_caches.append((conv_cache, pool_cache))

    N = feat.shape[0]
    feat_shape = feat.shape
    flat = feat.reshape(N, -1)

    z1 = flat @ params["W1"] + params["b1"]
    a1 = np.maximum(0.0, z1)
    logits = a1 @ params["W2"] + params["b2"]
    probs = softmax(logits)

    cache = {
        "block_caches": block_caches,
        "feat_shape": feat_shape,
        "flat": flat,
        "z1": z1,
        "a1": a1,
        "probs": probs,
    }
    return probs, cache


def cnn_backward(y_oh: np.ndarray, params: dict, cache: dict) -> dict:
    """Lan truyền ngược toàn bộ mạng CNN, trả về dict gradients."""
    N = y_oh.shape[0]
    probs = cache["probs"]
    a1 = cache["a1"]
    z1 = cache["z1"]
    flat = cache["flat"]
    feat_shape = cache["feat_shape"]

    d_logits = (probs - y_oh) / N
    dW2 = a1.T @ d_logits
    db2 = np.sum(d_logits, axis=0, keepdims=True)

    da1 = d_logits @ params["W2"].T
    dz1 = da1 * (z1 > 0.0)
    dW1 = flat.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    d_flat = dz1 @ params["W1"].T
    d_feat = d_flat.reshape(feat_shape)

    conv_grads = []
    for conv_cache, pool_cache in reversed(cache["block_caches"]):
        d_pool = maxpool_backward(d_feat, pool_cache)
        d_feat, dW_c, db_c = conv_relu_backward(d_pool, conv_cache)
        conv_grads.append({"dW": dW_c, "db": db_c})

    conv_grads.reverse()
    return {
        "conv_grads": conv_grads,
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
    }


def update_cnn_params(params: dict, grads: dict, learning_rate: float) -> None:
    """Cập nhật trọng số tại chỗ bằng Gradient Descent."""
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]

    for layer, g_conv in zip(params["conv_layers"], grads["conv_grads"]):
        layer["W"] -= learning_rate * g_conv["dW"]
        layer["b"] -= learning_rate * g_conv["db"]


def train_cnn(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    learning_rate: float = 0.01,
    epochs: int = 20,
    batch_size: int = 16,
    verbose_every: int = 5,
    seed: int = 42,
) -> list[float]:
    """Vòng lặp huấn luyện mini-batch cho CNN."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    num_classes = params["b2"].shape[1]
    y_oh = to_one_hot(y, num_classes)
    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        idx = rng.permutation(N)
        for start in range(0, N, batch_size):
            b_idx = idx[start : start + batch_size]
            _, cache = cnn_forward(X[b_idx], params)
            grads = cnn_backward(y_oh[b_idx], params, cache)
            update_cnn_params(params, grads, learning_rate)

        full_probs, _ = cnn_forward(X, params)
        loss = cross_entropy_loss(full_probs, y_oh)
        loss_history.append(loss)

        if verbose_every and (epoch % verbose_every == 0 or epoch == 1):
            preds = np.argmax(full_probs, axis=1)
            acc = np.mean(preds == y.ravel())
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss:.4f} | Acc: {acc:.4f}")

    return loss_history


def predict_cnn(X: np.ndarray, params: dict) -> np.ndarray:
    """Dự đoán nhãn lớp cho dữ liệu đầu vào X."""
    probs, _ = cnn_forward(X, params)
    return np.argmax(probs, axis=1)

def run_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    conv_filters: list[int] | None = None,
    hidden_dim: int = 64,
    learning_rate: float = 0.01,
    epochs: int = 20,
    batch_size: int = 16,
    seed: int = 42,
) -> dict:
    """Hàm gọi gọn để khởi tạo, huấn luyện, đánh giá và dự đoán bằng CNN."""
    if conv_filters is None:
        conv_filters = [8]

    _, c, h, w = X_train.shape
    num_classes = int(np.max(y_train)) + 1

    params = init_cnn_params(
        input_shape=(c, h, w),
        num_classes=num_classes,
        conv_filters=conv_filters,
        hidden_dim=hidden_dim,
        seed=seed,
    )
    losses = train_cnn(
        X=X_train,
        y=y_train,
        params=params,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        verbose_every=max(1, epochs // 4),
        seed=seed,
    )

    train_preds = predict_cnn(X_train, params)
    result: dict = {
        "params": params,
        "losses": losses,
        "train_acc": float(np.mean(train_preds == y_train.ravel())),
    }

    if X_test is not None:
        test_preds = predict_cnn(X_test, params)
        result["test_preds"] = test_preds
        if y_test is not None:
            result["test_acc"] = float(np.mean(test_preds == y_test.ravel()))

    return result


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    N, C, H, W = 120, 1, 8, 8
    X = rng.normal(0.0, 0.3, size=(N, C, H, W))
    y = np.zeros(N, dtype=int)

    for i in range(N):
        if i % 2 == 0:
            X[i, 0, H // 2, :] += 1.5
            y[i] = 0
        else:
            X[i, 0, :, W // 2] += 1.5
            y[i] = 1

    out = run_cnn(
        X_train=X[:96],
        y_train=y[:96],
        X_test=X[96:],
        y_test=y[96:],
        conv_filters=[6],
        hidden_dim=32,
        learning_rate=0.02,
        epochs=25,
        batch_size=16,
    )
    print(f"Train Acc: {out['train_acc']:.4f} | Test Acc: {out.get('test_acc'):.4f}")