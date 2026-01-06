# Stock Short-Term Prediction with Transformer & Gemini Alphas

Dự án dự báo giá cổ phiếu **ngắn hạn (next-day prediction)** sử dụng mô hình **Transformer Encoder-Decoder** kết hợp với các **alpha công thức** được sinh tự động bởi **Gemini AI**.

### Mục tiêu
- Dự đoán **giá đóng cửa ngày kế tiếp** (`adjClose`) cho từng cổ phiếu
- Sử dụng context lịch sử **20 ngày** gần nhất (window_size = 20)
- Kết hợp các chỉ báo kỹ thuật + 5 công thức alpha sáng tạo từ Gemini
- Đánh giá bằng RMSE, MAE và vẽ biểu đồ so sánh thực tế vs dự đoán

## Yêu cầu hệ thống (Windows)
- Windows 10/11
- Python 3.10 hoặc 3.11 (khuyến nghị)
- RAM ≥ 8GB
- GPU NVIDIA + CUDA (tùy chọn, để train nhanh hơn)

## Hướng dẫn cài đặt và chạy trên Windows

### 1. Tạo môi trường ảo (virtual environment)
Mở **PowerShell** (nhấn Windows → gõ PowerShell → chạy với quyền bình thường):

```powershell
python -m venv venv
```

### 2. Kích hoạt môi trường ảo
Nếu gặp lỗi "running scripts is disabled", chạy lệnh này một lần (với quyền Administrator):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Sau đó kích hoạt venv:

```powershell
venv\Scripts\activate
```

### 3. Cài đặt thư viện cần thiết
```powershell
pip install -r requirements.txt
```

### 4. Thiết lập API Key cho Gemini (để sinh alpha tự động)
Truy cập: https://aistudio.google.com/app/apikey → tạo key miễn phí
Trong PowerShell (đang ở thư mục dự án và venv đã active), chạy:

```powershell
$env:GEMINI_API_KEY = "your_api_key_here"
```
### 5. Chạy dự án
```powershell
python main.py
```

### Quy trình sẽ diễn ra tự động:

- Tải dữ liệu cổ phiếu từ Google Drive
- Chia train/test (trước/sau năm 2023)
- Thêm các chỉ báo kỹ thuật (SMA, EMA, RSI, MACD, Bollinger Bands, OBV…)
- Gọi Gemini AI sinh 5 công thức alpha mới
- Scale dữ liệu
- Huấn luyện Transformer với context 20 ngày
- Đánh giá dự đoán ngày kế tiếp bằng MSE, RMSE, MAE
- In ví dụ dự đoán cụ thể
- Vẽ biểu đồ so sánh giá thực tế (xanh) và giá dự đoán (đỏ đứt nét) cho từng cổ phiếu

### Kết quả đầu ra

- Metrics: MSE, RMSE, MAE
- Ví dụ dự đoán chi tiết
- Biểu đồ matplotlib cho từng cổ phiếu
- Model tốt nhất được lưu tại best_model.pth

### Tùy chỉnh (nếu muốn)

- Thay đổi số ngày context: sửa window_size = 20 trong main.py
- Thay đổi số epoch/learning rate: chỉnh trong hàm train_model() ở model.py

### Lưu ý quan trọng

- Lần đầu chạy sẽ mất khoảng 10-30 phút tùy cấu hình máy (do training model).
- Nếu không set Gemini key → vẫn chạy nhưng không có alpha (kết quả kém hơn).
- Mỗi lần mở PowerShell mới cần chạy lại lệnh activate venv và set GEMINI_API_KEY.

Chúc bạn dự báo ngắn hạn chính xác và có kết quả tốt! 📈