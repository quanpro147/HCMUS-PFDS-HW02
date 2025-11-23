import numpy as np
from .models import LabelEncoder, KNNImputer


# ĐỌC VÀ LOAD DỮ LIỆU
def load_data(file_path: str, delimiter: str = ',', header = None) -> np.ndarray:
    """
    Doc du lieu tu file CSV/TXT va chuyen thanh numpy ndarray.
    """
    try:
        data = np.genfromtxt(
            file_path,
            delimiter=delimiter,
            names=header,
            dtype=None,
            skip_header=1,
            encoding='utf-8',
        )
        return data
    except Exception as e:
        print(f"Lỗi khi load dữ liệu bằng NumPy: {e}")
        return np.array([])

#     """
#     Điền giá trị thiếu (NaN) cho cả numeric và categorical.
#     - Numeric -> mean
#     - Categorical -> mode (giá trị xuất hiện nhiều nhất)
#     """
#     fields = data_array.dtype.fields

#     for col, (dtype, _) in fields.items():
#         # Xử lý các cột number
#         if np.issubdtype(dtype, np.number):
#             col_vals = data_array[col].astype(float)
#             mean_val = np.nanmean(col_vals)
#             mask = np.isnan(col_vals)
#             col_vals[mask] = mean_val
#             data_array[col] = col_vals

#         # Xử lý các cột category
#         else:
#             col_vals = data_array[col].astype(str)
#             mask = (col_vals == '') | (col_vals == 'nan')

#             # Lấy giá trị mode bằng NumPy
#             valid_vals = col_vals[~mask]
#             if len(valid_vals) > 0:
#                 # np.unique trả về sorted unique values và counts
#                 uniq, counts = np.unique(valid_vals, return_counts=True)
#                 mode_val = uniq[np.argmax(counts)]
#             else:
#                 mode_val = ""  

#             col_vals[mask] = mode_val
#             data_array[col] = col_vals

#     return data_array

def process_missing_value(data_array: np.ndarray, n_neighbors=5) -> np.ndarray:
    """
    Xử lý missing value cho array numeric bằng KNNImputer.
    """
    # Lấy tên cột và chuyển sang 2D numeric array
    cols = data_array.dtype.names
    X = np.column_stack([data_array[col].astype(float) for col in cols])

    # Áp dụng KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_filled = imputer.fit_transform(X)

    # Chuyển lại về structured array
    data_filled = np.empty(data_array.shape, dtype=data_array.dtype)
    for i, col in enumerate(cols):
        # Nếu cột gốc là int, round và convert về int
        if np.issubdtype(data_array[col].dtype, np.integer):
            data_filled[col] = np.rint(X_filled[:, i]).astype(data_array[col].dtype)
        else:
            data_filled[col] = X_filled[:, i]
    
    return data_filled

# ENCODING dữ liệu text
def encode_data(data_array: np.ndarray, col_names: list) -> np.ndarray:
    """
    Áp dụng LabelEncoder cho nhiều cột trong structured array.
    """
    new_dtype = []
    for name, (dtype, _) in data_array.dtype.fields.items():
        if name in col_names:
            new_dtype.append((name, 'f8'))
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