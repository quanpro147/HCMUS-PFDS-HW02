import numpy as np


def train_test_split(X, y, test_size=0.2, train_size=None, shuffle=True, random_state=None):
    """
    Chia dữ liệu thành tập train và test.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    n_samples = X.shape[0]

    # Xử lý random_state
    if random_state is not None:
        np.random.seed(random_state)

    # Xác định số lượng test
    if isinstance(test_size, float):
        n_test = int(n_samples * test_size)
    elif isinstance(test_size, int):
        n_test = test_size
    else:
        raise ValueError("test_size must be float or int")

    # Xác định số lượng train
    if train_size is None:
        n_train = n_samples - n_test
    elif isinstance(train_size, float):
        n_train = int(n_samples * train_size)
    elif isinstance(train_size, int):
        n_train = train_size
    else:
        raise ValueError("train_size must be float or int")

    # Shuffle indices
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    # Chia train-test
    test_idx = indices[:n_test]
    train_idx = indices[n_test:n_test + n_train]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def accuracy(y_true, y_pred):
    """
    Accuracy = số dự đoán đúng / tổng số mẫu
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)


def precision(y_true, y_pred, pos_label=1):
    """
    Precision = TP / (TP + FP)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = np.sum((y_pred == pos_label) & (y_true == pos_label))
    FP = np.sum((y_pred == pos_label) & (y_true != pos_label))

    if TP + FP == 0:
        return 0.0
    return TP / (TP + FP)


def recall(y_true, y_pred, pos_label=1):
    """
    Recall = TP / (TP + FN)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = np.sum((y_pred == pos_label) & (y_true == pos_label))
    FN = np.sum((y_pred != pos_label) & (y_true == pos_label))

    if TP + FN == 0:
        return 0.0
    return TP / (TP + FN)


def f1(y_true, y_pred, pos_label=1):
    """
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    precision_score = precision(y_true, y_pred, pos_label)
    recall_score = recall(y_true, y_pred, pos_label)

    if precision_score + recall_score == 0:
        return 0.0
    return 2 * (precision_score * recall_score) / (precision_score + recall_score)

def cal_all_metrics(y_true, y_pred, pos_label=1):
    
    accuracy_score = accuracy(y_true, y_pred)
    precision_score = precision(y_true, y_pred)
    recall_score = recall(y_true, y_pred)
    f1_score = f1(y_true, y_pred)
    print(f"Accuracy: {accuracy_score}")
    print(f"Precision: {precision_score}")
    print(f"Recall: {recall_score}")
    print(f"F1 Score: {f1_score}")

    return accuracy_score, precision_score, recall_score, f1_score
