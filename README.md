# Tiêu đề Project
Decision Tree Classifier Implementation with NumPy

## Mô tả ngắn gọn
Dự án này triển khai một Decision Tree Classifier hoàn toàn bằng NumPy, nhằm phân loại dữ liệu, trực quan hóa cây quyết định và đánh giá kết quả thông qua các biểu đồ.

---

## Mục lục
1. [Giới thiệu](#giới-thiệu)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Results](#results)
7. [Project Structure](#project-structure)
8. [Challenges & Solutions](#challenges--solutions)
9. [Future Improvements](#future-improvements)
10. [Contributors](#contributors)
11. [Thông tin tác giả](#thông-tin-tác-giả)
12. [Contact](#contact)
13. [License](#license)

---

## Giới thiệu
### Mô tả bài toán
Triển khai một Decision Tree Classifier từ đầu sử dụng NumPy, không phụ thuộc vào thư viện học máy bên ngoài.

### Động lực và ứng dụng thực tế
Decision Tree là thuật toán cơ bản trong học máy, được sử dụng rộng rãi trong phân loại dữ liệu y tế, marketing, phân loại sản phẩm, dự đoán rủi ro, v.v.

### Mục tiêu cụ thể
- Hiểu cách xây dựng Decision Tree từ đầu.  
- Triển khai các thuật toán phân tách và tính entropy/giảm thông tin.  
- Trực quan hóa cây và kết quả phân loại.  
- Đánh giá độ chính xác và loss.

---

## Dataset
### Nguồn dữ liệu
Sử dụng dataset mẫu [tên dataset, ví dụ: Iris] từ nguồn mở UCI hoặc tự tạo.

### Mô tả các features
- `feature_1`: Mô tả  
- `feature_2`: Mô tả  
- …  

### Kích thước và đặc điểm dữ liệu
- Số samples: 150  
- Số features: 4  
- Classes: 3  

---

## Method
### Quy trình xử lý dữ liệu
1. Chuẩn hóa dữ liệu nếu cần.  
2. Chia train/test.  
3. Huấn luyện Decision Tree bằng thuật toán ID3/Entropy.

### Thuật toán sử dụng
- **Entropy**:  
\[
H(Y) = -\sum_{i=1}^{n} p_i \log_2 p_i
\]

- **Information Gain**:  
\[
IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)
\]

- **Split Node**: Chọn feature và threshold cho gain cao nhất.

### Giải thích cách implement bằng NumPy
- Tính entropy, split, chọn feature tốt nhất sử dụng các hàm `np.unique`, `np.bincount`, boolean indexing.  
- Duy trì cây bằng class `DecisionTreeNode` và đệ quy `_build_tree`.  

---

## Installation & Setup
```bash
# Clone repository
git clone <repo_url>
cd <repo_folder>

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```
## Usage
Chỉ cần chạy 3 file notebook theo thứ tự là được

## Project Structure
HR-ANALYTICS/
│
├── data/
│   ├── raw/                # Dữ liệu gốc chưa xử lý
│   └── processed/          # Dữ liệu sau xử lý, clean, feature engineering
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Phân tích dữ liệu (EDA)
│   ├── 02_data_preprocessing.ipynb   # Làm sạch và xử lý dữ liệu
│   └── 03_data_modeling.ipynb        # Huấn luyện và đánh giá mô hình
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Hàm load, clean, encode dữ liệu
│   ├── models.py             # Cài đặt Decision Tree + các model khác
│   ├── utils.py              # Hàm hỗ trợ (metrics, check duplicate, split data,…)
│   └── visualization.py      # Hàm vẽ biểu đồ (bar_plot, line_plot, pie_plot)
│
├── requirements.txt          # Danh sách thư viện cần cài đặt
└── README.md                 # Tài liệu mô tả project

## Challenges & Solutions

Khó khăn: Implement thuật toán ID3 bằng NumPy, xử lý dữ liệu category và numeric.

Giải pháp: Sử dụng np.unique, np.bincount, boolean indexing để tính entropy và gain hiệu quả.

## Future Improvements

Hỗ trợ pruning để giảm overfitting.

Thêm Random Forest và Gradient Boosting.

Tối ưu tốc độ cho dataset lớn.

Thêm trực quan hóa cây bằng matplotlib trực tiếp.

## Contributors

Quân Phan Ngọc
Contact:
- Email: quanphanpq147@gmail.com