import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def try_convert_to_numeric(column):
    try:
        numeric_column = column.astype(float)
        if np.issubdtype(column.dtype, np.number) or not np.any(np.isnan(numeric_column)):
            return numeric_column, True
        return column, False
    except (ValueError, TypeError):
        return column, False

def visualize_features(X, y, feature_names=None):

    n_features = X.shape[1]
    for i in range(n_features):
        feature_data = X[:, i]
        converted_feature, is_numeric = try_convert_to_numeric(feature_data)

        if is_numeric:
            continue
        else:
            _plot_categorical_feature(feature_data, y, feature_names[i])

def _plot_categorical_feature(feature, y, feature_name):
    feature = feature.astype(str)
    categories = np.unique(feature)

    if len(categories) > 15:
        print(f"Skipping {feature_name} - too many categories ({len(categories)}). Showing top 10.")
        values, counts = np.unique(feature, return_counts=True)
        top_idx = np.argsort(-counts)[:10]
        categories = values[top_idx]

        mask = np.isin(feature, categories)
        feature = feature[mask]
        y_filtered = y[mask]
    else:
        y_filtered = y

    labels = np.unique(y_filtered)

    plt.figure(figsize=(max(8, len(categories)*0.8), 6))
    bar_width = 0.8 / len(labels)
    positions = np.arange(len(categories))

    for idx, label in enumerate(labels):
        counts = [np.sum((feature == cat) & (y_filtered == label)) for cat in categories]
        plt.bar(positions + idx * bar_width, counts, width=bar_width, label=f"Label={label}")

    plt.xticks(positions + bar_width * (len(labels)-1)/2, categories, rotation=45, ha="right")
    plt.xlabel(feature_name)
    plt.ylabel("Count")
    plt.title(f"Feature: {feature_name}")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_features(X, feature_names=None):
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    for i in range(n_features):
        feature_data = X[:, i]
        original_dtype = feature_data.dtype

        converted_feature, is_numeric = try_convert_to_numeric(feature_data)

        if is_numeric:
            print(f"✓ {feature_names[i]}: NUMERIC | "
                  f"min={np.nanmin(converted_feature):.2f}, "
                  f"max={np.nanmax(converted_feature):.2f}, "
                  f"mean={np.nanmean(converted_feature):.2f}")
        else:
            unique_vals = np.unique(feature_data.astype(str))
            print(f"○ {feature_names[i]}: CATEGORICAL | {len(unique_vals)} categories")

def bar_plot(col_data, col_name, title=None, color='skyblue'):
    """
    Vẽ bar chart từ numpy array.
    - col_data: numpy array (có thể là category/string hoặc số)
    - col_name: tên cột (string)
    """
    # Lấy các category duy nhất và số lần xuất hiện
    categories, counts = np.unique(col_data, return_counts=True)

    plt.figure(figsize=(8,5))
    plt.bar(categories, counts, color=color)
    plt.title(title if title else f"Bar plot of {col_name}")
    plt.xlabel(col_name)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def pie_plot(col_data, col_name, title=None, colors=None, explode=None):
    """
    Vẽ pie chart từ numpy array.
    - col_data: numpy array (category)
    - col_name: tên cột (string)
    """
    labels, counts = np.unique(col_data, return_counts=True)

    plt.figure(figsize=(6,6))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=colors, explode=explode, shadow=True)
    plt.title(title if title else f"Pie chart of {col_name}")
    plt.axis('equal')
    plt.show()


def line_plot(col_data, col_name, title=None, color='blue', marker='o'):
    """
    Vẽ line chart từ numpy array.
    - col_data: numpy array (numeric)
    - col_name: tên cột (string)
    """
    x = np.arange(len(col_data))
    y = col_data

    plt.figure(figsize=(8,5))
    plt.plot(x, y, marker=marker, color=color, linestyle='-', linewidth=2)
    plt.title(title if title else f"Line plot of {col_name}")
    plt.xlabel("Index")
    plt.ylabel(col_name)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def confusion_matrix_plot(y_true, y_pred, labels=None, normalize=False, cmap="Blues", title="Confusion Matrix"):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    n_labels = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    # Tạo ma trận confusion
    cm = np.zeros((n_labels, n_labels), dtype=int)
    for t, p in zip(y_true, y_pred):
        i = label_to_index[t]
        j = label_to_index[p]
        cm[i, j] += 1
    
    if normalize:
        cm = cm.astype(float)
        cm = cm / cm.sum(axis=1, keepdims=True)
    
    # Vẽ heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()


def visualize_model_results_comparison(model1_name, model1_results, model2_name, model2_results):
    """
    So sánh và visualize hiệu suất của hai mô hình bằng biểu đồ cột nhóm.
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    scores1 = np.array(model1_results)
    scores2 = np.array(model2_results)

    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    rects1 = plt.bar(x - width/2, scores1, width, label=model1_name, color='skyblue')
    rects2 = plt.bar(x + width/2, scores2, width, label=model2_name, color='lightcoral')
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.3f}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3), 
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    plt.title(f'So sánh Hiệu suất của {model1_name} và {model2_name}', fontsize=16)
    plt.ylabel('Giá trị Chỉ số (Score)', fontsize=14)
    plt.xticks(x, metrics)
    plt.ylim(np.min([scores1, scores2]) * 0.9, np.max([scores1, scores2]) * 1.1)
    
    plt.legend(title='Mô hình')
    plt.tight_layout()
    plt.show()


def non_missing_count_plot(data_array, column_names=None, title="Non-missing Values per Column"):
    """
    Vẽ biểu đồ cột thể hiện số giá trị không phải missing cho mỗi cột.
    - Với cột số: loại bỏ np.nan
    - Với cột object/string: loại bỏ chuỗi rỗng hoặc chỉ có khoảng trắng
    """
    # Nếu là structured array
    if hasattr(data_array, "dtype") and data_array.dtype.names is not None:
        cols = data_array.dtype.names
        X = np.column_stack([data_array[col] for col in cols])
        if column_names is None:
            column_names = cols
    else:
        X = data_array
        if column_names is None:
            column_names = [f"Col {i}" for i in range(X.shape[1])]
    
    non_missing_counts = []

    for i in range(X.shape[1]):
        col = X[:, i]
        # Kiểm tra dtype
        if np.issubdtype(col.dtype, np.number):
            count = np.sum(~np.isnan(col.astype(float)))
        else:
            # object/string: loại bỏ chuỗi rỗng hoặc chỉ khoảng trắng
            count = np.sum([bool(str(x).strip()) for x in col])
        non_missing_counts.append(count)

    # Vẽ bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(column_names, non_missing_counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Non-missing Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()

