import pandas as pd
import cv2
import numpy as np
import os
import time
import joblib
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog

# --- AYARLAR ---
CSV_FILE = 'english.csv'
MODEL_FILE = "model_exp2_filtered_final.joblib"
TARGET_SIZE = (64, 64)

# Ryzen 5 7600 CPU için Parametreler
XGB_PARAMS = {
    'n_estimators': 300,      
    'max_depth': 8,           
    'learning_rate': 0.05,
    'objective': 'multi:softmax',
    'random_state': 42,
    'n_jobs': -1,             
    'tree_method': 'hist',
    'eval_metric': 'merror'  # <-- DÜZELTME: Parametre buraya taşındı
}

# ÇIKARILACAK KARAKTERLER LİSTESİ (Potansiyel Hata Kaynakları)
BLACKLIST_CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
    'o', 'O', 'i', 'I', 'l', 
    's', 'S', 'z', 'Z', 'c', 'C', 'p', 'P', 'k', 'K', 
    'v', 'V', 'w', 'W', 'x', 'X', 'u', 'U', 'j', 'J'
]

def main():
    print(f"--- DENEY 2: FİLTRELİ VERİ SETİ (Versiyon 3 - Final) ---")
    print(f"İşlem PID: {os.getpid()}")
    
    # --- 1. Veri Yükleme ve Filtreleme ---
    if not os.path.exists(CSV_FILE):
        print(f"HATA: {CSV_FILE} bulunamadı!")
        return

    df = pd.read_csv(CSV_FILE)
    original_count = len(df)
    print(f"Veri seti yüklendi. Orijinal kayıt: {original_count}")

    # FİLTRELEME İŞLEMİ
    df = df[~df['label'].isin(BLACKLIST_CHARS)]
    filtered_count = len(df)
    print(f"Temizlik bitti. Kalan kayıt: {filtered_count} (Silinen: {original_count - filtered_count})")

    data_list = []
    label_list = []

    print("Görüntüler işleniyor...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Resim Okuma"):
        image_path = row['image']
        label = row['label']
        
        if not os.path.exists(image_path):
             if os.path.exists(image_path.replace('Img/', 'img/')):
                 image_path = image_path.replace('Img/', 'img/')
             else:
                 continue 

        try:
            image = cv2.imread(image_path)
            if image is None: continue

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            data_list.append(resized_image)
            label_list.append(label)
        except Exception as e:
            pass

    X_raw = np.array(data_list)
    y_raw = np.array(label_list)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    print(f"İşlenen Veri: {X_raw.shape}")
    print(f"Sınıflar ({len(label_encoder.classes_)}): {label_encoder.classes_}")

    # --- 2. HOG Öznitelik Çıkarımı ---
    print("\n--- HOG Öznitelik Çıkarımı ---")
    hog_features_list = []
    for image in tqdm(X_raw, desc="HOG Hesaplanıyor"):
        features = hog(
            image, orientations=8, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), visualize=False, feature_vector=True
        )
        hog_features_list.append(features)
    
    X_hog = np.array(hog_features_list)
    print(f"HOG Tamamlandı. Öznitelik Matrisi: {X_hog.shape}")

    # --- 3. Veri Bölme ---
    print("\n--- Veri Seti Bölünüyor (80/20) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_hog, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Eğitim Seti: {X_train.shape}")
    print(f"Test Seti:   {X_test.shape}")

    # --- 4. Model Eğitimi ---
    print("\n--- XGBoost Eğitimi Başlıyor ---")
    print("Not: İlerleme, her 10 ağaçta bir hata oranı (merror) olarak gösterilecektir.")
    print("merror ne kadar düşerse o kadar iyidir (0 = hatasız, 1 = tamamen hatalı).")
    
    model = xgb.XGBClassifier(**XGB_PARAMS, num_class=len(label_encoder.classes_))

    start_time = time.time()
    
    # EĞİTİM
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)], 
        verbose=10 # Her 10 ağaçta bir rapor ver (Canlı takip)
    )
    
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\n✅ EĞİTİM BAŞARIYLA TAMAMLANDI!")
    print(f"Geçen Süre: {duration:.2f} saniye ({duration/60:.2f} dakika)")

    # --- 5. Kaydetme ve Rapor ---
    data_to_save = {'model': model, 'encoder': label_encoder}
    joblib.dump(data_to_save, MODEL_FILE)
    print(f"Model diske kaydedildi: {MODEL_FILE}")

    print("\n--- PERFORMANS RAPORU ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"NİHAİ DOĞRULUK (ACCURACY): %{accuracy * 100:.2f}")

    print("\n--- Detaylı Sınıflandırma Raporu ---")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

if __name__ == "__main__":
    main()