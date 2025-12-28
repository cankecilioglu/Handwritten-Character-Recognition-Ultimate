import pandas as pd
import cv2
import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from skimage.feature import hog

# --- AYARLAR ---
CSV_FILE = 'data/english.csv'
RESULTS_FILE = "tuning_results.csv" # Sonuçların kaydedileceği dosya
TARGET_SIZE = (64, 64)

# Varsayılan Sabit Değerler (Base Code Ayarları)
DEFAULT_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.3, # XGBoost varsayılanı genelde 0.3'tür, biz değiştireceğiz
    'objective': 'multi:softmax',
    'random_state': 42,
    'n_jobs': -1,             # Ryzen 5 7600 Tüm Çekirdekler
    'tree_method': 'hist',    # CPU Hızlandırma
    'eval_metric': 'merror'
}

# --- DENEY LİSTELERİ ---
# 1. Deney: Sadece Learning Rate değişir
learning_rates_to_test = [0.3, 0.1, 0.05, 0.01, 0.001, 0.0001]

# 2. Deney: Sadece N_Estimators değişir (LR=0.1 sabit varsayalım)
n_estimators_to_test = [50, 100, 300, 500, 800]

# 3. Deney: Sadece Max_Depth değişir (LR=0.1, N=100 sabit varsayalım)
max_depths_to_test = [3, 6, 8, 10, 12]

def load_and_process_data():
    """Veriyi yükler ve HOG işlemini BİR KEZ yapar (Hız için)"""
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"{CSV_FILE} bulunamadı! Lütfen data klasörünü kontrol edin.")
        
    df = pd.read_csv(CSV_FILE)
    print(f"Veri seti yüklendi: {len(df)} kayıt.")
    
    data_list = []
    label_list = []
    
    print("Resimler işleniyor ve HOG çıkarılıyor...")
    
    # Başarısız dosya sayacı
    missing_count = 0
    
    # Resim okuma ve HOG çıkarma döngüsü
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        img_path = row['image'] # Örn: Img/img001-001.png
        
        # --- DÜZELTME BURADA YAPILDI ---
        # os.path.basename() kullanmıyoruz, çünkü 'Img/' klasör ismini korumalıyız.
        # data/ + Img/img001-001.png şeklinde birleştiriyoruz.
        full_path = os.path.join('data', img_path)
        
        # Linux yol düzeltmesi (Büyük/Küçük harf duyarlılığı)
        if not os.path.exists(full_path):
            # Alternatif yolları dene
            if os.path.exists(full_path.replace('Img', 'img')):
                full_path = full_path.replace('Img', 'img')
            elif os.path.exists(os.path.join('data', os.path.basename(img_path))):
                # Belki resimler direkt data klasöründedir
                full_path = os.path.join('data', os.path.basename(img_path))
            else:
                missing_count += 1
                continue

        try:
            image = cv2.imread(full_path)
            if image is None: 
                missing_count += 1
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            # HOG Çıkarımı (Standart 8x8)
            fd = hog(resized, orientations=8, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), visualize=False, feature_vector=True)
            
            data_list.append(fd)
            label_list.append(row['label'])
        except Exception as e:
            missing_count += 1
            pass

    if missing_count > 0:
        print(f"UYARI: {missing_count} resim bulunamadı veya okunamadı.")
        
    if len(data_list) == 0:
        raise ValueError("HİÇBİR RESİM İŞLENEMEDİ! Lütfen 'data/Img' klasörünün dolu olduğundan emin olun.")

    return np.array(data_list), np.array(label_list)

def run_experiment(X_train, X_test, y_train, y_test, num_class, param_name, param_value):
    """Tek bir parametre setiyle eğitimi çalıştırır ve sonucu döner"""
    
    # Varsayılan parametreleri kopyala
    current_params = DEFAULT_PARAMS.copy()
    current_params['num_class'] = num_class
    
    # Değişen parametreyi güncelle
    if param_name in current_params:
        current_params[param_name] = param_value
    
    # Model Eğitimi (Verbose kapalı, sadece sonuç odaklı)
    model = xgb.XGBClassifier(**current_params)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start_time
    
    # Tahmin
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return acc, duration

def main():
    # 1. Veriyi Hazırla (Sadece 1 kez)
    try:
        X, y = load_and_process_data()
    except Exception as e:
        print(f"KRİTİK HATA: {e}")
        return
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    results = []
    
    print("\n--- TEST 1: LEARNING RATE ETKİSİ ---")
    for lr in learning_rates_to_test:
        print(f"Testing Learning Rate: {lr} ...", end=" ", flush=True)
        acc, dur = run_experiment(X_train, X_test, y_train, y_test, num_classes, 'learning_rate', lr)
        print(f"Accuracy: %{acc*100:.2f} ({dur:.2f}s)")
        results.append({'Experiment': 'Learning Rate', 'Value': lr, 'Accuracy': acc, 'Time': dur})

    print("\n--- TEST 2: N_ESTIMATORS ETKİSİ ---")
    # LR'yi 0.1'e sabitleyelim (Base değer olarak)
    DEFAULT_PARAMS['learning_rate'] = 0.1 
    for n in n_estimators_to_test:
        print(f"Testing N_Estimators: {n} ...", end=" ", flush=True)
        acc, dur = run_experiment(X_train, X_test, y_train, y_test, num_classes, 'n_estimators', n)
        print(f"Accuracy: %{acc*100:.2f} ({dur:.2f}s)")
        results.append({'Experiment': 'N Estimators', 'Value': n, 'Accuracy': acc, 'Time': dur})

    print("\n--- TEST 3: MAX_DEPTH ETKİSİ ---")
    # N=100, LR=0.1 sabit
    DEFAULT_PARAMS['n_estimators'] = 100
    for d in max_depths_to_test:
        print(f"Testing Max Depth: {d} ...", end=" ", flush=True)
        acc, dur = run_experiment(X_train, X_test, y_train, y_test, num_classes, 'max_depth', d)
        print(f"Accuracy: %{acc*100:.2f} ({dur:.2f}s)")
        results.append({'Experiment': 'Max Depth', 'Value': d, 'Accuracy': acc, 'Time': dur})

    # Sonuçları Kaydet
    df_res = pd.DataFrame(results)
    df_res.to_csv(RESULTS_FILE, index=False)
    print(f"\n✅ Tüm deneyler bitti! Sonuçlar '{RESULTS_FILE}' dosyasına kaydedildi.")
    print(df_res)

if __name__ == "__main__":
    main()