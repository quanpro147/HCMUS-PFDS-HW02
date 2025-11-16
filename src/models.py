import numpy as np


class LabelEncoder:
    def __init__(self):
        self.classes_ = None
        self.class_to_index = None

    def fit(self, array_1d: np.ndarray):
        # Chuyển về string (an toàn cho mixed types)
        array_1d = array_1d.astype(str)
        # Tìm các giá trị duy nhất, sắp xếp
        self.classes_ = np.unique(array_1d)
        # Tạo mapping value -> index
        self.class_to_index = {val: i for i, val in enumerate(self.classes_)}
        return self

    def transform(self, array_1d: np.ndarray) -> np.ndarray:
        array_1d = array_1d.astype(str)
        if self.class_to_index is None:
            raise ValueError("LabelEncoder chưa được fit trước")
        # Map từng giá trị sang index
        return np.array([self.class_to_index[val] for val in array_1d], dtype=int)

    def fit_transform(self, array_1d: np.ndarray) -> np.ndarray:
        self.fit(array_1d)
        return self.transform(array_1d)

    def inverse_transform(self, index_array: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("LabelEncoder chưa được fit trước")
        return np.array([self.classes_[i] for i in index_array])


class SMOTE:
    """
    Simple SMOTE implementation using NumPy.
    Tăng dữ liệu cho class thiểu số bằng cách tạo synthetic samples.
    """
    def __init__(self, k_neighbors=5, random_state=None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def _nearest_neighbors(self, X):
        """
        Tính khoảng cách Euclidean giữa tất cả các điểm
        và trả về index của k nearest neighbors cho mỗi điểm
        """
        n_samples = X.shape[0]
        neighbors_idx = []

        # Tính khoảng cách Euclidean
        for i in range(n_samples):
            diff = X - X[i]
            dist = np.sqrt(np.sum(diff**2, axis=1))
            dist[i] = np.inf  # loại bỏ chính nó
            # Lấy k smallest distances
            nn_idx = np.argsort(dist)[:self.k_neighbors]
            neighbors_idx.append(nn_idx)
        return np.array(neighbors_idx)

    def fit_resample(self, X, y):
        """
        Args:
            X: np.ndarray, shape (n_samples, n_features)
            y: np.ndarray, shape (n_samples,)
        Returns:
            X_resampled: np.ndarray
            y_resampled: np.ndarray
        """
        X_resampled = X.copy()
        y_resampled = y.copy()

        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        X_minority = X[y == minority_class]

        n_minority = X_minority.shape[0]
        n_majority = max(counts)

        # Số lượng sample synthetic cần tạo
        n_synthetic = n_majority - n_minority
        if n_synthetic == 0:
            return X_resampled, y_resampled

        # Tính k nearest neighbors
        neighbors_idx = self._nearest_neighbors(X_minority)

        synthetic_samples = []
        for _ in range(n_synthetic):
            # Chọn ngẫu nhiên 1 điểm minority
            idx = np.random.randint(0, n_minority)
            # Chọn 1 neighbor ngẫu nhiên
            nn = neighbors_idx[idx, np.random.randint(0, self.k_neighbors)]
            diff = X_minority[nn] - X_minority[idx]
            gap = np.random.rand()
            synthetic = X_minority[idx] + gap * diff
            synthetic_samples.append(synthetic)

        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.array([minority_class] * n_synthetic)

        # Ghép với dữ liệu gốc
        X_resampled = np.vstack([X_resampled, X_synthetic])
        y_resampled = np.hstack([y_resampled, y_synthetic])

        return X_resampled, y_resampled


class KNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        y_pred = []
        for x in X:
            # Euclidean distance
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            nn_idx = np.argsort(distances)[:self.n_neighbors]
            nn_labels = self.y_train[nn_idx]
            # Majority vote
            values, counts = np.unique(nn_labels, return_counts=True)
            y_pred.append(values[np.argmax(counts)])
        return np.array(y_pred)



class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None
        self.n_classes_ = None

    def _entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p + 1e-9))

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None, None

        parent_entropy = self._entropy(y)
        best_gain = 0
        best_feature = None
        best_threshold = None

        for f in range(n_features):
            thresholds = np.unique(X[:, f])
            for t in thresholds:
                left_mask = X[:, f] <= t
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                e_left = self._entropy(y[left_mask])
                e_right = self._entropy(y[right_mask])

                gain = parent_entropy - (
                    (left_mask.sum() / n_samples) * e_left +
                    (right_mask.sum() / n_samples) * e_right
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = t

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        num_samples = len(y)
        num_classes = len(np.unique(y))

        # 1) Điều kiện dừng
        if (
            num_classes == 1 or
            num_samples < self.min_samples_split or
            (self.max_depth is not None and depth >= self.max_depth)
        ):
            value = np.bincount(y).argmax()
            return DecisionTreeNode(value=value)

        # 2) Tìm split tốt nhất
        feature, threshold = self._best_split(X, y)

        # 3) Nếu không có split tốt → leaf
        if feature is None:
            value = np.bincount(y).argmax()
            return DecisionTreeNode(value=value)

        # 4) Chia dữ liệu
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        # 5) Nếu child rỗng → leaf
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            value = np.bincount(y).argmax()
            return DecisionTreeNode(value=value)

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionTreeNode(
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child
        )

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.tree_ = self._build_tree(X, y)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])