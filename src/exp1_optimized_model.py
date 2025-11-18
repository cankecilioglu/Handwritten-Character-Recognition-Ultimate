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
MODEL_FILE = "model_exp1_optimized_brain.joblib"
TARGET_SIZE = (64, 64)

# OPTİMİZE EDİLMİŞ MODEL PARAMETRELERİ (Ryzen CPU Gücü)
XGB_PARAMS = {
    'n_estimators': 500,      # 500 Ağaç
    'max_depth': 8,           # Derinlik 8
    'learning_rate': 0.05,    
    'subsample': 0.8,         
    'colsample_bytree': 0.8,  
    'objective': 'multi:softmax',
    'random_state': 42,
    'n_jobs': -1,             
    'tree_method': 'hist',
    'eval_metric': 'merror'   # Hata oranı metriği
}

def main():
    print(f"--- İYİLEŞTİRME 1: MODEL OPTİMİZASYONU (Tüm Veri) ---")
    
    # 1. Veri Yükleme
    if not os.path.exists(CSV_FILE): 
        print("Hata: CSV dosyası bulunamadı.")
        return
        
    df = pd.read_csv(CSV_FILE)
    print(f"Veri seti: {len(df)} kayıt. (FİLTRELEME YOK)")

    data_list, label_list = [], []
    print("Görüntüler işleniyor...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            img_path = row['image']
            # Linux dosya yolu düzeltmesi
            if not os.path.exists(img_path):
                if os.path.exists(img_path.replace('Img/', 'img/')):
                    img_path = img_path.replace('Img/', 'img/')
                else: continue
            
            image = cv2.imread(img_path)
            if image is None: continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            data_list.append(resized)
            label_list.append(row['label'])
        except: pass

    X_raw = np.array(data_list)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(np.array(label_list))

    # 2. STANDART HOG (8x8)
    print("\n--- Standart HOG (8x8) Çıkarılıyor ---")
    hog_feats = []
    for img in tqdm(X_raw):
        f = hog(img, orientations=8, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=False, feature_vector=True)
        hog_feats.append(f)
    X_hog = np.array(hog_feats)
    print(f"Öznitelik Sayısı: {X_hog.shape[1]}")

    # 3. Eğitim
    X_train, X_test, y_train, y_test = train_test_split(X_hog, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    print("\n--- XGBoost Modeli Eğitiliyor ---")
    print(f"Parametreler: {XGB_PARAMS}")
    
    model = xgb.XGBClassifier(**XGB_PARAMS, num_class=len(label_encoder.classes_))
    
    start = time.time()
    # Callbacks yerine verbose kullanıyoruz (Hatasız yöntem)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50 # Her 50 ağaçta bir durum raporu yazdır
    )
    end = time.time()
    
    print(f"\n✅ Eğitim Bitti! Süre: {(end-start):.2f} sn")
    
    # 4. Sonuç
    joblib.dump({'model': model, 'encoder': label_encoder}, MODEL_FILE)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\nNİHAİ DOĞRULUK: %{acc*100:.2f}")
    print(classification_report(y_test, model.predict(X_test), target_names=label_encoder.classes_, zero_division=0))

if __name__ == "__main__":
    main()