# Explore HR Analytics Dataset with Numpy

## Mô tả ngắn gọn
Dự án này triển khai một quy trình exploratory data analysis (EDA), preprocessing, xây dựng mô hình Decision Tree và K-Nearest Neighbors (KNN), và dự đoán nhãn hoàn toàn bằng NumPy, không phụ thuộc vào thư viện học máy bên ngoài.

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
Trong bối cảnh nhu cầu tuyển dụng Data Scientist ngày càng tăng, các công ty thường gặp khó khăn trong việc duy trì đội ngũ nhân sự chất lượng. Nhiều ứng viên Data Science đăng ký các khóa đào tạo, tham gia thi tuyển, nhưng khó dự đoán liệu họ có thực sự muốn chuyển việc hay không. Điều này dẫn đến:
- Lãng phí chi phí tuyển dụng
- Lãng phí thời gian phỏng vấn
- Khó tối ưu nguồn lực đào tạo và hỗ trợ
- Giảm hiệu quả trong chiến lược nhân sự
Bài toán đặt mục tiêu xây dựng một mô hình dự đoán xem ứng viên có đang tìm kiếm cơ hội việc làm mới hay không dựa trên hồ sơ cá nhân, kỹ năng, kinh nghiệm và tương tác của họ với chương trình đào tạo.

### Động lực và ứng dụng thực tế
- Hỗ trợ HR và các công ty tối ưu hóa chiến lược tuyển dụng.
- Dự đoán sớm ứng viên có nguy cơ nghỉ việc giúp giữ chân nhân sự quan trọng.
- Phân tích các yếu tố ảnh hưởng đến quyết định thay đổi công việc.

### 📊 Mục tiêu cụ thể

#### 1. Phân tích nhân khẩu học
- **Xác định các biến nhân khẩu học** ảnh hưởng đến quyết định thay đổi công việc
- **Phân tích mối tương quan** giữa đặc điểm cá nhân và ý định nghỉ việc
- **Đánh giá tác động** của các yếu tố như trình độ học vấn, kinh nghiệm, giới tính, v.v.

#### 2. Dự đoán nhị phân
- **Xây dựng mô hình dự đoán** kết quả nhị phân:
  - `0` - Không tìm kiếm việc làm
  - `1` - Đang tìm kiếm việc làm (có ý định thay đổi)
- **Đánh giá hiệu suất** mô hình với các metrics phù hợp

#### 3. Áp dụng quy trình Khoa học Dữ liệu
**Triển khai quy trình KDD (Knowledge Discovery in Databases):**
Thực hiện đầy đủ các bước: Cleaning → Preprocessing → Feature Encoding → Modeling → Evaluation
---

## Dataset
### Nguồn dữ liệu
[hr-analytics-job-change-of-data-scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)

### Mô tả thuộc tính dataset

- **enrollee_id**: ID duy nhất của ứng viên
- **city**: Mã thành phố  
- **city_development_index**: Chỉ số phát triển của thành phố (đã được điều chỉnh tỷ lệ)
- **gender**: Giới tính của ứng viên
- **relevent_experience**: Kinh nghiệm liên quan của ứng viên
- **enrolled_university**: Loại khóa học đại học (nếu có đăng ký)
- **education_level**: Trình độ học vấn của ứng viên
- **major_discipline**: Chuyên ngành học chính
- **experience**: Tổng số năm kinh nghiệm của ứng viên
- **company_size**: Số lượng nhân viên trong công ty hiện tại
- **company_type**: Loại hình công ty hiện tại
- **lastnewjob**: Khoảng thời gian (năm) giữa công việc trước và công việc hiện tại
- **training_hours**: Số giờ đào tạo đã hoàn thành
- **target**: Kết quả (nhãn)
  - `0` - Không tìm kiếm việc làm
  - `1` - Đang tìm kiếm việc làm

### Kích thước và đặc điểm dữ liệu
- Số samples: 19158
- Số features: 14
- Classes: 2

---

## Method
### Quy trình xử lý dữ liệu
1. Cleaning Data 
2. Missing value processing
3. Encode Category Data

### Thuật toán sử dụng

#### 1. K-Nearest Neighbors (KNN)
1. Chuẩn hóa dữ liệu nếu cần.  
2. Tính khoảng cách giữa mẫu test và toàn bộ training set (ví dụ: Euclidean).  
3. Sắp xếp các khoảng cách và chọn `K` láng giềng gần nhất.  
4. Thực hiện "majority vote" để quyết định nhãn dự đoán.  
5. Trả về nhãn và (tùy chọn) tỉ lệ phiếu làm độ tin cậy.

#### 2. Decision Tree (Cây quyết định)
**Metríc sử dụng**
- **Entropy**  

  $$
  H(Y) = -\sum_{i=1}^{n} p_i \log_2 p_i
  $$

- **Information Gain (IG)**  

  $$
  IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)
  $$

**Quy trình xây dựng cây**
1. Tại mỗi node, duyệt từng feature và các ngưỡng (threshold) có thể.  
2. Với mỗi split: chia dữ liệu thành các nhánh và tính entropy cho từng phần.  
3. Tính Information Gain → chọn split có IG lớn nhất.  
4. Dừng khi:
   - tất cả mẫu cùng nhãn, hoặc  
   - số mẫu quá nhỏ, hoặc  
   - đạt max depth.  
5. Node lá gán nhãn bằng nhãn xuất hiện nhiều nhất.

---

## Installation & Setup
```bash
# Clone repository
git clone https://github.com/quanpro147/HCMUS-PFDS-HW02.git
cd HCMUS-PFDS-HW02
# Cài đặt môi trường (optional) - conda
conda create --name hw02 python=3.10
conda activate hw02

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```
---

## Run
Chạy lần lượt 3 notebook trong thư mục `notebooks/`:

1. `01_data_exploration.ipynb`
2. `02_data_preprocessing.ipynb`
3. `03_data_modeling.ipynb`

---

## Project Structure
```
HR-ANALYTICS/
│
├── data/
│   ├── raw/                # Dữ liệu gốc chưa xử lý
│   └── processed/          # Dữ liệu sau xử lý, clean
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
```
## Challenges & Solutions

Khó khăn: Implement thuật toán ID3 hoàn toàn bằng NumPy, đặc biệt là xử lý
categorical features, tính entropy/gain và tìm split tối ưu.

Giải pháp: Tận dụng np.unique, np.bincount, và boolean indexing để giảm độ phức tạp,
tăng tốc tính toán; chuẩn hóa pipeline encode → split → compute gain nhất quán.

---

## Future Improvements
- Hỗ trợ pruning để giảm overfitting.
- Thêm Random Forest và Gradient Boosting.
- Tối ưu tốc độ cho dataset lớn.
- Thêm trực quan hóa cây bằng matplotlib trực tiếp.

---

## Contributors
- **Phan Ngọc Quân**

## Contact
- Email: quanphanpq147@gmail.com

---

### License
MIT License

Copyright (c) 2025 Phan Ngọc Quân

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.