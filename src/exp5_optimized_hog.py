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
MODEL_FILE = "model_exp1_optimized_eyes.joblib"
TARGET_SIZE = (64, 64)

# HOG OPTİMİZASYONU İÇİN DENGELİ MODEL PARAMETRELERİ
XGB_PARAMS = {
    'n_estimators': 200,      
    'max_depth': 6,           
    'learning_rate': 0.1,     
    'objective': 'multi:softmax',
    'random_state': 42,
    'n_jobs': -1,             
    'tree_method': 'hist',
    'eval_metric': 'merror'
}

def main():
    print(f"--- İYİLEŞTİRME 2: HOG (DETAYLI) OPTİMİZASYONU ---")
    
    if not os.path.exists(CSV_FILE): return
    df = pd.read_csv(CSV_FILE)
    print(f"Veri seti: {len(df)} kayıt.")

    data_list, label_list = [], []
    print("Görüntüler işleniyor...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            img_path = row['image']
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

    # 2. DETAYLI HOG (4x4)
    print("\n--- DETAYLI HOG (4x4) Çıkarılıyor ---")
    print("Bu işlem daha uzun sürecek çünkü öznitelik sayısı artıyor...")
    hog_feats = []
    for img in tqdm(X_raw):
        # pixels_per_cell=(4, 4) yaparak detayı 4 kat artırıyoruz
        f = hog(img, orientations=8, pixels_per_cell=(4, 4),
                cells_per_block=(2, 2), visualize=False, feature_vector=True)
        hog_feats.append(f)
    X_hog = np.array(hog_feats)
    print(f"YENİ Öznitelik Sayısı: {X_hog.shape[1]} (Eski: ~1500)")

    # 3. Eğitim
    X_train, X_test, y_train, y_test = train_test_split(X_hog, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    print("\n--- XGBoost Modeli Eğitiliyor ---")
    model = xgb.XGBClassifier(**XGB_PARAMS, num_class=len(label_encoder.classes_))
    
    start = time.time()
    # Callbacks yerine verbose kullanıyoruz
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=20 # Her 20 ağaçta bir durum raporu
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