import numpy as np
from .models import LabelEncoder


# ĐỌC VÀ LOAD DỮ LIỆU
def load_data(file_path: str, delimiter: str = ',') -> np.ndarray:
    """
    Doc du lieu tu file CSV/TXT va chuyen thanh numpy ndarray.
    """
    try:
        data = np.genfromtxt(
            file_path,
            delimiter=delimiter,
            skip_header=1,
            dtype=None,
            encoding='utf-8',
            missing_values=['NA', 'NaN', ' '],
        )
        return data
    except Exception as e:
        print(f"Lỗi khi load dữ liệu bằng NumPy: {e}")
        return np.array([])

def clean_data():
    pass

# XỬ LÝ MISSING VALUES
def process_missing_value(data_array: np.ndarray) -> np.ndarray:
    """
    Điền giá trị thiếu (NaN) cho cả numeric và categorical.
    - Numeric -> mean
    - Categorical -> mode (giá trị xuất hiện nhiều nhất)
    """
    fields = data_array.dtype.fields

    for col, (dtype, _) in fields.items():
        # Xử lý các cột number
        if np.issubdtype(dtype, np.number):
            col_vals = data_array[col].astype(float)
            mean_val = np.nanmean(col_vals)
            mask = np.isnan(col_vals)
            col_vals[mask] = mean_val
            data_array[col] = col_vals

        # Xử lý các cột category
        else:
            col_vals = data_array[col].astype(str)
            mask = (col_vals == '') | (col_vals == 'nan')

            # Lấy giá trị mode bằng NumPy
            valid_vals = col_vals[~mask]
            if len(valid_vals) > 0:
                # np.unique trả về sorted unique values và counts
                uniq, counts = np.unique(valid_vals, return_counts=True)
                mode_val = uniq[np.argmax(counts)]
            else:
                mode_val = ""  

            col_vals[mask] = mode_val
            data_array[col] = col_vals

    return data_array



# XỬ LÝ OUTLIERS
def handle_outliers(data_array: np.ndarray, threshold: float = 1.5) -> np.ndarray:
    """
    Handle outliers using IQR for structured array (mixed numeric + string).
    """
    # Lọc các cột có value là number
    numeric_cols = [
        name for name, (dtype, _) in data_array.dtype.fields.items()
        if np.issubdtype(dtype, np.number)
    ]

    # Xác định ngưỡng Outliers
    row_mask = np.ones(len(data_array), dtype=bool)

    for col in numeric_cols:
        values = data_array[col].astype(float)

        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1

        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        valid = (values >= lower) & (values <= upper)

        row_mask &= valid

    return data_array[row_mask]


# CHUẨN HÓA DỮ LIỆU
def normalize_min_max(data_array: np.ndarray) -> np.ndarray:
    """
    Thực hiện Min-Max Normalization cho các cột numeric của structured array.
    Các cột string/categorical được giữ nguyên.
    """
    fields = data_array.dtype.fields

    for col, (dtype, _) in fields.items():
        if np.issubdtype(dtype, np.number):
            col_vals = data_array[col].astype(float)
            min_val = np.min(col_vals)
            max_val = np.max(col_vals)
            range_val = max_val - min_val if max_val - min_val != 0 else 1.0

            col_vals = (col_vals - min_val) / range_val
            data_array[col] = col_vals

    return data_array



def standardize_z_score(data_array: np.ndarray) -> np.ndarray:
    """
    Thuc hien dieu chuan Z-score (Standardization) [6, 8].
    Dua du lieu ve trung binh 0 va phuong sai 1. 
    Can thiet truoc khi dung thuat toan dua tren gradient [6, 8].
    
    Args:
        data_array: Mảng NumPy (chi chua du lieu so).
        
    Returns:
        Mảng NumPy đã được điều chuẩn.
    """
    # Tinh Trung binh (mean) va Do lech chuan (std) tren tung cot (axis=0)
    mean_vals = np.mean(data_array, axis=0)
    std_vals = np.std(data_array, axis=0)
    
    # np.where de tranh chia cho 0 (neu std = 0)
    standardized_array = (data_array - mean_vals) / np.where(std_vals != 0, std_vals, 1)
    
    return standardized_array


# ENCODING dữ liệu text
def encode_data(data_array: np.ndarray, col_names: list) -> np.ndarray:
    """
    Áp dụng LabelEncoder cho nhiều cột trong structured array.
    
    Args:
        data_array: structured array
        col_names: list tên cột categorical cần encode
    
    Returns:
        structured array mới với các cột được encode thành số nguyên
    """
    new_dtype = []
    for name, (dtype, _) in data_array.dtype.fields.items():
        if name in col_names:
            new_dtype.append((name, 'i8'))
        else:
            new_dtype.append((name, dtype))

    new_array = np.empty(data_array.shape, dtype=new_dtype)

    for name, _ in new_dtype:
        if name in col_names:
            le = LabelEncoder()
            encoded_col = le.fit_transform(data_array[name])
            new_array[name] = encoded_col
        else:
            new_array[name] = data_array[name]

    return new_array