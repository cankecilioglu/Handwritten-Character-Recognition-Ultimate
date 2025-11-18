import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
import joblib
import time
import os

# --- AYARLAR ---
# Dosya yolları Linux formatına uygundur
CSV_FILE = 'english.csv'
IMG_FOLDER = 'Img'  # Resim klasörünün adı
TARGET_SIZE = (64, 64)
MODEL_FILE = "model_exp4_noise_amd_cpu.joblib"
NOISE_LEVEL = 20  # Gürültü şiddeti (Sigma)

# Ryzen 5 7600 CPU için optimize edilmiş parametreler
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.05,
    'objective': 'multi:softmax',
    'random_state': 42,
    'n_jobs': -1,        # Tüm çekirdekleri kullan
    'tree_method': 'hist' # CPU için en hızlı yöntem
}

def add_gaussian_noise(image, sigma):
    """Görüntüye (uint8) Gauss gürültüsü ekler."""
    mean = 0
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image.astype(float) + gauss, 0, 255)
    return noisy_image.astype(np.uint8)

def main():
    print(f"--- DENEY 4: GÜRÜLTÜLÜ VERİ SETİ (AMD CPU: Ryzen 5 7600) ---")
    print(f"İşlem başlıyor... PID: {os.getpid()}")

    # --- 1. Veri Yükleme ve Gürültü Ekleme ---
    if not os.path.exists(CSV_FILE):
        print(f"HATA: {CSV_FILE} bulunamadı!")
        return

    df = pd.read_csv(CSV_FILE)
    print(f"'{CSV_FILE}' yüklendi. Toplam {len(df)} kayıt.")

    data_list = []
    label_list = []

    print("Görüntüler yükleniyor ve gürültü ekleniyor...")
    # tqdm ile ilerleme çubuğu
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # CSV'deki yol 'Img/img001-001.png' formatındaysa ve
        # biz zaten proje klasöründeysek, yol doğrudur.
        # Ancak Linux'ta '/' ayracı garanti olsun diye os.path.join kullanabiliriz.
        # CSV içeriğiniz 'Img/resim.png' ise direkt kullanabiliriz.
        image_path = row['image'] 
        
        label = row['label']
        
        # Dosya var mı kontrolü
        if not os.path.exists(image_path):
             # Bazen CSV'de 'Img/...' yazar ama klasör adı 'img' olabilir (Linux büyük/küçük harf duyarlıdır)
             # Basit bir düzeltme denemesi:
             if os.path.exists(image_path.replace('Img/', 'img/')):
                 image_path = image_path.replace('Img/', 'img/')
             else:
                 continue # Dosya yoksa atla

        try:
            image = cv2.imread(image_path)
            if image is None:
                continue

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            # --- GÜRÜLTÜ EKLEME ---
            noisy_image = add_gaussian_noise(resized_image, sigma=NOISE_LEVEL)
            
            data_list.append(noisy_image)
            label_list.append(label)
        except Exception as e:
            print(f"Hata: {image_path} - {e}")

    print(f"Toplam {len(data_list)} görüntü başarıyla işlendi.")

    X_raw = np.array(data_list)
    y_raw = np.array(label_list)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    print(f"Sınıflar kodlandı. Toplam Sınıf: {len(label_encoder.classes_)}")

    # --- 2. HOG Öznitelik Çıkarımı ---
    print("\n--- HOG Öznitelik Çıkarımı Başlatıldı ---")
    hog_features_list = []
    for image in tqdm(X_raw, desc="HOG Hesaplanıyor"):
        features = hog(
            image, orientations=8, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), visualize=False, feature_vector=True
        )
        hog_features_list.append(features)
    
    X_hog = np.array(hog_features_list)
    print(f"HOG Tamamlandı. Veri Şekli: {X_hog.shape}")

    # --- 3. Veri Bölme ---
    print("\n--- Veri Seti Bölünüyor (80/20) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_hog, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Eğitim Seti: {X_train.shape}")
    print(f"Test Seti:   {X_test.shape}")

    # --- 4. Model Eğitimi ---
    print("\n--- XGBoost Modeli Eğitiliyor ---")
    print(f"Kullanılan Parametreler: {XGB_PARAMS}")
    
    model = xgb.XGBClassifier(**XGB_PARAMS, num_class=len(label_encoder.classes_))

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"\n✅ EĞİTİM TAMAMLANDI!")
    print(f"Süre: {(end_time - start_time):.2f} saniye ({(end_time - start_time)/60:.2f} dakika)")

    # --- 5. Kaydetme ve Rapor ---
    data_to_save = {'model': model, 'encoder': label_encoder}
    joblib.dump(data_to_save, MODEL_FILE)
    print(f"Model kaydedildi: {MODEL_FILE}")

    print("\n--- PERFORMANS DEĞERLENDİRMESİ ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Doğruluğu (Accuracy): %{accuracy * 100:.2f}")

    print("\n--- Detaylı Rapor ---")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

if __name__ == "__main__":
    main()