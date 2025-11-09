import streamlit as st
import torch
from PIL import Image, ImageOps
HEIF_SUPPORTED = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORTED = True
except Exception:
    HEIF_SUPPORTED = False
import numpy as np
import os
from utils.predict import MNISTPredictor
import matplotlib.pyplot as plt
import io


# Cấu hình trang
st.set_page_config(
    page_title="Nhận Diện Số Viết Tay",
    page_icon="🔢",
    layout="wide"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 3rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-number {
        font-size: 5rem;
        color: #1565C0;
        font-weight: bold;
    }
    .confidence {
        font-size: 1.5rem;
        color: #555;
    }
    .info-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
        color: #424242;
    }
    .info-box strong {
        color: #D84315;
    }
    .info-box ul {
        color: #424242;
    }
    .info-box li {
        color: #424242;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load mô hình (cache để không phải load lại mỗi lần)"""
    model_path = 'models/mnist_model.pth'
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy mô hình tại {model_path}. Vui lòng chạy train.py trước!")
        st.stop()
    return MNISTPredictor(model_path)


def preprocess_image(image):
    """
    Xử lý ảnh để phù hợp với mô hình MNIST
    - Chuyển về grayscale
    - Đảm bảo nền đen, chữ trắng như MNIST
    - Resize về 28x28
    """
    # Chuyển về RGB trước (tránh lỗi với ảnh RGBA)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Chuyển về grayscale
    image = ImageOps.grayscale(image)   # L mode
    
    # Resize về 28x28 TRƯỚC KHI xử lý màu để giữ thông tin tốt hơn
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert sang array để phân tích
    img_array = np.array(image).astype(float)
    
    # Kiểm tra xem nền là trắng hay đen bằng cách xem pixel ở 4 góc
    # (Giả định: nền chiếm phần lớn ảnh)
    corners = [
        img_array[0, 0], img_array[0, -1], 
        img_array[-1, 0], img_array[-1, -1]
    ]
    avg_corner = np.mean(corners)
    
    # Nếu góc sáng (>128) => nền sáng, cần đảo ngược
    # Nếu góc tối (<128) => nền tối, giữ nguyên
    if avg_corner > 128:
        # Nền trắng (sáng) -> Đảo ngược để có nền đen
        img_array = 255 - img_array
    
    # Normalize và tăng độ tương phản bằng histogram stretching
    # Tìm min/max thực tế của ảnh (bỏ qua outliers)
    p2, p98 = np.percentile(img_array, (2, 98))
    
    # Stretch histogram: kéo giá trị từ [p2, p98] về [0, 255]
    img_array = np.clip((img_array - p2) * 255.0 / (p98 - p2), 0, 255)
    
    # Convert back to PIL Image
    image = Image.fromarray(img_array.astype('uint8'))
    
    return image


def create_probability_chart(probabilities):
    """Tạo biểu đồ xác suất cho 10 chữ số"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    digits = list(range(10))
    colors = ['#EF5350' if p == max(probabilities) else '#42A5F5' for p in probabilities]
    
    bars = ax.bar(digits, probabilities, color=colors, alpha=0.8)
    ax.set_xlabel('Chữ số', fontsize=12)
    ax.set_ylabel('Xác suất', fontsize=12)
    ax.set_title('Phân bố xác suất cho các chữ số', fontsize=14, fontweight='bold')
    ax.set_xticks(digits)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Thêm giá trị lên trên mỗi cột
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.1%}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🔢 Nhận Diện Số Viết Tay</h1>', unsafe_allow_html=True)

    # Load mô hình
    with st.spinner('Đang load mô hình...'):
        predictor = load_model()
    
    # Tạo 2 cột
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Ảnh")
        
        # Hướng dẫn
        st.markdown("""
        <div class="info-box">
        <strong>💡 Hướng dẫn:</strong>
        <ul>
            <li>Chọn hoặc kéo thả ảnh chứa chữ số viết tay (0-9)</li>
            <li>Ảnh nên có nền trắng hoặc đen, chữ số rõ ràng</li>
            <li>Định dạng: JPG, PNG, JPEG</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload file
        uploaded_file = st.file_uploader(
            "Chọn ảnh...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload ảnh chứa chữ số viết tay"
        )
        
        # Hoặc sử dụng camera
        st.markdown("### 📷 Hoặc chụp ảnh:")
        camera_image = st.camera_input("Chụp ảnh chữ số viết tay")
        
        # Chọn nguồn ảnh
        image_source = uploaded_file if uploaded_file else camera_image
        
        if image_source:
            # Đọc ảnh một cách an toàn: lấy bytes rồi mở bằng PIL qua BytesIO
            try:
                # Cố gắng đọc bytes một cách an toàn từ các loại đối tượng khác nhau
                if hasattr(image_source, "getvalue"):
                    # Một số đối tượng hỗ trợ getvalue()
                    image_bytes = image_source.getvalue()
                elif hasattr(image_source, "read"):
                    # Thường là Streamlit UploadedFile or SpooledTemporaryFile
                    try:
                        image_source.seek(0)
                    except Exception:
                        pass
                    image_bytes = image_source.read()
                elif isinstance(image_source, (bytes, bytearray)):
                    image_bytes = bytes(image_source)
                else:
                    # Thử coi đó là đường dẫn tới file
                    try:
                        with open(str(image_source), "rb") as f:
                            image_bytes = f.read()
                    except Exception:
                        st.error("❌ Không thể chuyển đổi nguồn ảnh thành bytes.")
                        return

                # Debug info (commented out for production)
                # info_lines = []
                # if hasattr(image_source, "name"):
                #     info_lines.append(f"filename={getattr(image_source, 'name')}")
                # if hasattr(image_source, "type"):
                #     info_lines.append(f"type={getattr(image_source, 'type')}")
                # st.write("Debug:", ", ".join(info_lines))

                # Detect HEIC/HEIF signatures (common when iPhone photos are HEIC)
                is_heic = False
                try:
                    head = image_bytes[:32]
                    if b'ftyp' in head and (b'heic' in head or b'heif' in head or b'heix' in head or b'hevc' in head):
                        is_heic = True
                except Exception:
                    is_heic = False

                if is_heic:
                    # Try to use pillow_heif if installed
                    try:
                        import pillow_heif
                        pillow_heif.register_heif_opener()
                        image = Image.open(io.BytesIO(image_bytes))
                        image.load()
                    except Exception as heic_err:
                        st.error(
                            "❌ Ảnh có vẻ ở định dạng HEIC/HEIF mà PIL mặc định không hỗ trợ.\n"
                            "Gợi ý khắc phục: 1) Lưu ảnh dưới dạng JPG/PNG trước khi upload; 2) Cài thêm `pillow-heif` (ví dụ: `pip install pillow-heif`) để ứng dụng có thể mở HEIC.\n"
                            f"(chi tiết lỗi mở HEIC: {heic_err})"
                        )
                        return
                else:
                    image = Image.open(io.BytesIO(image_bytes))
                    image.load()  # force decode to surface errors
            except Exception as e:
                # Nếu có khả năng HEIC nhưng chưa cài pillow-heif, gợi ý rõ ràng
                head_hex = ""
                try:
                    head_hex = image_bytes[:32].hex()
                except Exception:
                    head_hex = ""

                looks_like_heic = False
                if head_hex:
                    if '6674797068656963' in head_hex or 'ftypheic' in head_hex:
                        looks_like_heic = True
                    # broader check: ftyp... with heic/heif variants
                    if '66747970' in head_hex and any(k in head_hex for k in ('68656963','68656966','68656978','68657663')):
                        looks_like_heic = True

                if looks_like_heic:
                    if not HEIF_SUPPORTED:
                        st.error("❌ Ảnh có vẻ ở định dạng HEIC/HEIF mà PIL mặc định không hỗ trợ.")
                        st.markdown("**Khắc phục:** cài `pillow-heif` và thư viện hệ thống `libheif` trên máy chủ. Ví dụ (macOS):<br>`brew install libheif` và `pip install pillow-heif`", unsafe_allow_html=True)
                    else:
                        st.error(f"❌ Không thể đọc ảnh (HEIC): {e}")
                else:
                    st.error(f"❌ Không thể đọc ảnh: {e}")

                return

            # Hiển thị ảnh gốc
            st.image(image, caption='Ảnh gốc', use_container_width=True)

            # Xử lý ảnh
            processed_image = preprocess_image(image)

            # Hiển thị ảnh đã xử lý (phóng to để dễ nhìn)
            # Tạo figure với 2 subplots: ảnh và histogram
            fig_processed, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            
            # Hiển thị ảnh 28x28
            ax1.imshow(processed_image, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
            ax1.set_title('Ảnh sau xử lý (28×28)', fontsize=11, fontweight='bold')
            ax1.axis('off')
            
            # Hiển thị histogram để phân tích phân bố pixel
            img_arr = np.array(processed_image)
            ax2.hist(img_arr.flatten(), bins=50, color='gray', alpha=0.7, edgecolor='black')
            ax2.set_title('Phân bố giá trị pixel', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Giá trị pixel (0=đen, 255=trắng)')
            ax2.set_ylabel('Số lượng pixel')
            ax2.grid(axis='y', alpha=0.3)
            
            # Hiển thị thông tin thống kê
            mean_val = img_arr.mean()
            min_val = img_arr.min()
            max_val = img_arr.max()
            ax2.text(0.5, 0.95, f'TB: {mean_val:.1f}\nMin: {min_val}\nMax: {max_val}', 
                    transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            st.pyplot(fig_processed)
            plt.close(fig_processed)
            
            # Gợi ý nếu ảnh có vấn đề
            if mean_val < 10:
                st.warning("⚠️ Ảnh gần như toàn đen - có thể cần điều chỉnh độ sáng hoặc ảnh gốc không phù hợp")
            elif mean_val > 245:
                st.warning("⚠️ Ảnh gần như toàn trắng - có thể cần điều chỉnh độ tương phản hoặc ảnh gốc không phù hợp")
    
    with col2:
        st.header("🎯 Kết Quả Nhận Diện")
        
        if image_source:
            # Dự đoán
            with st.spinner('Đang phân tích...'):
                predicted_digit, probabilities = predictor.predict(processed_image)
            
            # Hiển thị kết quả
            st.markdown(f"""
            <div class="prediction-box">
                <p style="font-size: 1.5rem; margin: 0;">Số được nhận diện là:</p>
                <p class="prediction-number">{predicted_digit}</p>
                <p class="confidence">Độ tin cậy: {probabilities[predicted_digit]:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Hiển thị biểu đồ xác suất
            st.markdown("### 📊 Phân bố xác suất")
            fig = create_probability_chart(probabilities)
            st.pyplot(fig)
            
            # Hiển thị top 3 dự đoán
            st.markdown("### 🏆 Top 3 Dự Đoán")
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            
            for i, idx in enumerate(top_3_indices, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                st.markdown(f"{emoji} **Số {idx}**: {probabilities[idx]:.2%}")
        
        else:
            st.info("👆 Vui lòng upload hoặc chụp ảnh ở cột bên trái")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
