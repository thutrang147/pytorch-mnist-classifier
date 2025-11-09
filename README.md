# 🔢 Nhận Diện Số Viết Tay với PyTorch

Dự án Deep Learning nhận diện chữ số viết tay (0-9) sử dụng PyTorch và MNIST Dataset, với giao diện web Streamlit thân thiện.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Mục Lục

- [Giới thiệu](#-giới-thiệu)
- [Tính năng](#-tính-năng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Kiến trúc mô hình](#-kiến-trúc-mô-hình)
- [Kết quả](#-kết-quả)

## 🎯 Giới thiệu

Dự án này xây dựng một hệ thống nhận diện chữ số viết tay hoàn chỉnh từ đầu đến cuối:
- **Huấn luyện**: Train mô hình Neural Network trên MNIST dataset
- **Dự đoán**: Module để dự đoán số từ ảnh mới
- **Web App**: Giao diện người dùng đẹp mắt với Streamlit

## ✨ Tính năng

### 🎨 Giao diện Web (Streamlit)
- ✅ Upload ảnh từ máy tính
- ✅ Chụp ảnh trực tiếp qua webcam
- ✅ Hiển thị kết quả nhận diện với độ tin cậy
- ✅ Biểu đồ xác suất cho 10 chữ số (0-9)
- ✅ Xử lý ảnh tự động (resize, grayscale, invert)
- ✅ Giao diện responsive, thân thiện

### 🧠 Mô hình AI
- Neural Network 3 tầng
- Độ chính xác: ~97-98% trên MNIST test set
- Tốc độ dự đoán: < 100ms
- Hỗ trợ GPU (CUDA) nếu có

## 📁 Cấu trúc dự án

```
mnist_recognition_app/
│
├── app.py                 # Ứng dụng Streamlit chính
├── train.py              # Script huấn luyện mô hình
├── requirements.txt      # Dependencies
├── README.md            # File này
│
├── utils/
│   ├── model.py         # Định nghĩa kiến trúc Neural Network
│   └── predict.py       # Module dự đoán
│
├── models/
│   └── mnist_model.pth  # Mô hình đã train (tạo sau khi train)
│
├── data/
│   └── MNIST/           # Dataset (tự động tải về)
│
└── uploads/             # Thư mục lưu ảnh upload (tùy chọn)
```

## 🚀 Cài đặt

### 1. Clone hoặc tải dự án

```bash
cd mnist_recognition_app
```

### 2. Tạo môi trường ảo (khuyến nghị)

**Với venv:**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# hoặc
venv\Scripts\activate  # Windows
```

**Với conda:**
```bash
conda create -n mnist python=3.10
conda activate mnist
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý:** 
- Nếu có GPU NVIDIA, cài PyTorch với CUDA từ [pytorch.org](https://pytorch.org)
- Ví dụ: `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

## 💻 Sử dụng

### Bước 1: Huấn luyện mô hình

Chạy script train để tải MNIST dataset và huấn luyện mô hình:

```bash
python train.py
```

**Output:**
- Mô hình sẽ train trong 5 epochs (~2-3 phút trên CPU, ~30 giây trên GPU)
- Mô hình được lưu tại: `models/mnist_model.pth`
- Độ chính xác sẽ được in ra màn hình

### Bước 2: Chạy ứng dụng Web

```bash
streamlit run app.py
```

**Hoặc:**
```bash
python -m streamlit run app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt tại: `http://localhost:8501`

### Bước 3: Sử dụng ứng dụng

1. **Upload ảnh**: Click "Browse files" hoặc kéo thả ảnh vào
2. **Hoặc chụp ảnh**: Sử dụng webcam để chụp trực tiếp
3. **Xem kết quả**: 
   - Số được nhận diện
   - Độ tin cậy (%)
   - Biểu đồ xác suất cho 10 chữ số
   - Top 3 dự đoán

## 🏗️ Kiến trúc mô hình

### Neural Network 3 tầng

```
Input Layer:    784 neurons  (28x28 pixels flattened)
                    ↓
Hidden Layer:   128 neurons  (ReLU activation)
                    ↓
Output Layer:   10 neurons   (0-9 digits)
```

### Hyperparameters

| Tham số | Giá trị |
|---------|---------|
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Batch Size | 64 |
| Epochs | 5 |

### Code mô hình

```python
class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
```

## 📊 Kết quả

### Độ chính xác
- **Train set**: ~99%
- **Test set**: ~97-98%

### Demo Screenshots

*(Bạn có thể thêm ảnh chụp màn hình ứng dụng ở đây)*

## 🎨 Tính năng nâng cao (Tùy chỉnh)

### Thêm xử lý ảnh
File `utils/predict.py` có hàm `preprocess_image()` để:
- Tự động phát hiện và đảo màu (nếu nền trắng)
- Resize về 28x28
- Chuyển sang grayscale

### Tùy chỉnh giao diện
File `app.py` có CSS tùy chỉnh, bạn có thể thay đổi:
- Màu sắc
- Font chữ
- Layout

## 🔧 Test mô hình từ command line

```bash
python utils/predict.py
```

Sẽ test mô hình với 5 ảnh từ MNIST test set và lưu kết quả vào `test_predictions.png`

## 🐛 Troubleshooting

### Lỗi: "Không tìm thấy mô hình"
➡️ Hãy chạy `python train.py` trước để train và lưu mô hình

### Lỗi: "Import torch could not be resolved"
➡️ Cài đặt PyTorch: `pip install torch torchvision`

### Lỗi: "CUDA out of memory"
➡️ Giảm `batch_size` trong `train.py` hoặc train trên CPU

### Ảnh nhận diện sai
➡️ Đảm bảo:
- Chữ số rõ ràng, không bị mờ
- Nền đơn giản (trắng hoặc đen)
- Chỉ có 1 chữ số trong ảnh

## 📚 Tài nguyên tham khảo

- [PyTorch Documentation](https://pytorch.org/docs/)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📝 License

MIT License - Bạn tự do sử dụng, chỉnh sửa và phân phối