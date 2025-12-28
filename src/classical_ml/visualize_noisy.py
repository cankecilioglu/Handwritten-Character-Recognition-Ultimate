import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- AYARLAR ---
TEST_IMAGE_PATH = 'data/Img/img001-001.png' # İlk '0' harfi (veya herhangi bir resim yolu)
NOISE_LEVEL = 200 # Gürültü şiddeti (Sigma)

def add_gaussian_noise(image, sigma):
    """Görüntüye (uint8) Gauss gürültüsü ekler."""
    mean = 0
    gauss = np.random.normal(mean, sigma, image.shape)
    # Gürültüyü ekle ve 0-255 aralığında kırp (clip)
    noisy_image = np.clip(image.astype(float) + gauss, 0, 255)
    return noisy_image.astype(np.uint8)

def main():
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Hata: {TEST_IMAGE_PATH} bulunamadı.")
        return

    # 1. Resmi Oku
    image = cv2.imread(TEST_IMAGE_PATH)
    if image is None:
        print("Resim okunamadı.")
        return
        
    # 2. Gri Tonlama
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. Gürültü Ekle
    noisy_image = add_gaussian_noise(gray_image, sigma=NOISE_LEVEL)

    # 4. Yan Yana Göster
    plt.figure(figsize=(10, 5))
    
    # Orijinal
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Orijinal Görüntü')
    plt.axis('off')
    
    # Gürültülü
    plt.subplot(1, 2, 2)
    plt.imshow(noisy_image, cmap='gray')
    plt.title(f'Gürültülü Görüntü (Sigma={NOISE_LEVEL})')
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    main()