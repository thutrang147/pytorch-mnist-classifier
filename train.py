import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from utils.model import NeuralNet


def train_model():
    """Huấn luyện mô hình MNIST"""
    
    # ==================== THIẾT LẬP ====================
    
    # Thiết lập device (ưu tiên GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    # Hyperparameters
    input_size = 784      # 28x28 pixel
    hidden_size = 128     # Số nơ-ron ở tầng ẩn
    num_classes = 10      # 10 chữ số (0-9)
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 5        # Học 5 lần qua toàn bộ dữ liệu
    
    # ==================== CHUẨN BỊ DỮ LIỆU ====================
    
    print("\n--- ĐANG TẢI DỮ LIỆU MNIST ---")
    
    # Định nghĩa phép biến đổi với Data Augmentation để mô hình robust hơn
    transform = transforms.Compose([
        transforms.RandomRotation(10),      # Xoay ngẫu nhiên ±10 độ
        transforms.RandomAffine(             # Biến dạng nhẹ (shift, scale)
            degrees=0, 
            translate=(0.1, 0.1),           # Dịch chuyển 10%
            scale=(0.9, 1.1)                # Scale 90%-110%
        ),
        transforms.ToTensor(),
        transforms.RandomInvert(p=0.5),     # 50% ảnh bị đảo màu (nền trắng/chữ đen)
    ])
    
    # Tải dữ liệu train và test
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transform
    )
    
    # Tạo DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"Số lượng ảnh train: {len(train_dataset)}")
    print(f"Số lượng ảnh test: {len(test_dataset)}")
    
    # ==================== XÂY DỰNG MÔ HÌNH ====================
    
    print("\n--- KHỞI TẠO MÔ HÌNH ---")
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    print(model)
    
    # Loss function và Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # ==================== HUẤN LUYỆN ====================
    
    print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
    
    total_steps = len(train_loader)
    
    for epoch in range(num_epochs):
        model.train()  # Chuyển sang chế độ train
        
        for i, (images, labels) in enumerate(train_loader):
            # Đẩy dữ liệu lên device
            images = images.to(device)
            labels = labels.to(device)
            
            # ---- 5 BƯỚC KINH ĐIỂN ----
            
            # 1. Forward pass
            outputs = model(images)
            
            # 2. Calculate Loss
            loss = criterion(outputs, labels)
            
            # 3. Zero gradients
            optimizer.zero_grad()
            
            # 4. Backward pass
            loss.backward()
            
            # 5. Update weights
            optimizer.step()
            
            # In kết quả mỗi 100 batch
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{i+1}/{total_steps}], '
                      f'Loss: {loss.item():.4f}')
    
    print("--- HUẤN LUYỆN HOÀN TẤT ---")
    
    # ==================== ĐÁNH GIÁ ====================
    
    print("\n--- ĐÁNH GIÁ MÔ HÌNH ---")
    model.eval()  # Chuyển sang chế độ eval
    
    with torch.no_grad():
        correct = 0
        total = 0
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Độ chính xác trên {len(test_dataset)} ảnh test: {accuracy:.2f}%')
    
    # ==================== LƯU MÔ HÌNH ====================
    
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'mnist_model.pth')
    
    # Lưu toàn bộ state của mô hình
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_classes': num_classes,
        'accuracy': accuracy
    }, model_path)
    
    print(f"\n--- MÔ HÌNH ĐÃ ĐƯỢC LƯU TẠI: {model_path} ---")
    print(f"Độ chính xác: {accuracy:.2f}%")
    
    return model, accuracy


if __name__ == "__main__":
    train_model()
