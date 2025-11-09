"""
Module dự đoán số từ ảnh
"""
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from utils.model import NeuralNet


class MNISTPredictor:
    """Class để load mô hình và dự đoán số từ ảnh"""
    
    def __init__(self, model_path='models/mnist_model.pth'):
        """
        Khởi tạo predictor
        
        Args:
            model_path: Đường dẫn đến file mô hình đã train
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Sử dụng thiết bị: {self.device}")
        
        # Load mô hình
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Khởi tạo mô hình với cùng architecture
        self.model = NeuralNet(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_classes=checkpoint['num_classes']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Chuyển sang chế độ eval
        
        print(f"✅ Đã load mô hình từ {model_path}")
        print(f"Độ chính xác: {checkpoint.get('accuracy', 'N/A'):.2f}%")
        
        # Định nghĩa transform để xử lý ảnh đầu vào
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Chuyển về grayscale
            transforms.Resize((28, 28)),                   # Resize về 28x28
            transforms.ToTensor(),                         # Chuyển sang tensor
            # transforms.Normalize((0.1307,), (0.3081,))  # Normalize như MNIST
        ])
    
    def predict(self, image):
        """
        Dự đoán số từ ảnh
        
        Args:
            image: PIL Image hoặc numpy array hoặc đường dẫn file
            
        Returns:
            tuple: (predicted_digit, probabilities)
                - predicted_digit: Số được dự đoán (0-9)
                - probabilities: List xác suất cho mỗi chữ số
        """
        # Xử lý input
        if isinstance(image, str):
            # Nếu là đường dẫn file
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            # Nếu là numpy array
            image = Image.fromarray(image)
        
        # Đảm bảo image là PIL Image
        if not isinstance(image, Image.Image):
            raise ValueError("Image phải là PIL Image, numpy array, hoặc đường dẫn file")
        
        # Áp dụng transform
        img_tensor = self.transform(image)
        
        # Thêm batch dimension: [1, 28, 28] -> [1, 1, 28, 28]
        img_batch = img_tensor.unsqueeze(0).to(self.device)
        
        # Dự đoán
        with torch.no_grad():
            output = self.model(img_batch)
            
            # Tính xác suất (softmax)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            probabilities = probabilities.cpu().numpy()[0]
            
            # Lấy dự đoán
            _, predicted = torch.max(output.data, 1)
            predicted_digit = predicted.item()
        
        return predicted_digit, probabilities
    
    def predict_from_path(self, image_path):
        """
        Dự đoán từ đường dẫn file ảnh
        
        Args:
            image_path: Đường dẫn đến file ảnh
            
        Returns:
            tuple: (predicted_digit, probabilities)
        """
        return self.predict(image_path)


def main():
    """Hàm test predictor"""
    import matplotlib.pyplot as plt
    
    # Khởi tạo predictor
    predictor = MNISTPredictor()
    
    # Test với ảnh từ MNIST dataset
    import torchvision
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )
    
    # Lấy 5 ảnh ngẫu nhiên để test
    indices = [0, 100, 500, 1000, 5000]
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    for idx, ax in zip(indices, axes):
        img, true_label = test_dataset[idx]
        
        # Chuyển tensor thành PIL Image
        img_pil = transforms.ToPILImage()(img)
        
        # Dự đoán
        predicted, probs = predictor.predict(img_pil)
        
        # Hiển thị
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f'Thật: {true_label}\nĐoán: {predicted}\nXS: {probs[predicted]:.2%}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_predictions.png')
    print("\n✅ Đã lưu kết quả test vào test_predictions.png")
    plt.show()


if __name__ == "__main__":
    main()
