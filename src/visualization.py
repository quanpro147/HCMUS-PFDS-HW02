import numpy as np
import matplotlib.pyplot as plt

def try_convert_to_numeric(column):
    try:
        numeric_column = column.astype(float)
        if np.issubdtype(column.dtype, np.number) or not np.any(np.isnan(numeric_column)):
            return numeric_column, True
        return column, False
    except (ValueError, TypeError):
        return column, False

def visualize_features(X, y, feature_names=None):
    feature_names_map = {
        "f0": "enrollee_id",
        "f1": "city",
        "f2": "city_development_index",
        "f3": "gender",
        "f4": "relevent_experience",
        "f5": "enrolled_university",
        "f6": "education_level",
        "f7": "major_discipline",
        "f8": "experience",
        "f9": "company_size",
        "f10": "company_type",
        "f11": "last_new_job",
        "f12": "training_hours",
        "f13": "target"
    }

    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [feature_names_map.get(f"f{i}", f"f{i}") for i in range(n_features)]

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
