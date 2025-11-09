"""
Định nghĩa kiến trúc Neural Network cho MNIST
"""
import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    """
    Mô hình Neural Network 3 tầng để phân loại chữ số MNIST
    
    Kiến trúc:
    - Input: 784 (28x28 pixels)
    - Hidden Layer: 128 neurons + ReLU
    - Output: 10 classes (0-9)
    """
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Tầng 1: 784 -> 128
        self.relu = nn.ReLU()                          # Hàm kích hoạt
        self.fc2 = nn.Linear(hidden_size, num_classes)  # Tầng 2: 128 -> 10

    def forward(self, x):
        """
        Forward pass: dữ liệu đi qua mô hình
        
        Args:
            x: Tensor có shape [batch_size, 1, 28, 28] hoặc [batch_size, 784]
            
        Returns:
            Tensor có shape [batch_size, 10] - xác suất cho mỗi lớp
        """
        # Làm phẳng ảnh từ [batch_size, 1, 28, 28] -> [batch_size, 784]
        x = x.reshape(-1, 28 * 28)
        
        # Luồng đi: fc1 -> relu -> fc2
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out
