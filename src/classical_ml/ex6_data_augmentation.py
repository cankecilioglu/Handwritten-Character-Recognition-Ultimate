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

# --- CONFIGURATION ---
CSV_FILE = 'data/english.csv'
MODEL_FILE = "models/model_exp7_moderate_aug.joblib"
TARGET_SIZE = (64, 64)

# SADECE MAKUL GÜRÜLTÜ SEVİYELERİ (Sigma 25 ve 50)
# Toplam Veri = Orijinal + (Sigma 25) + (Sigma 50) = 3x Veri
SIGMA_VALUES = [25, 50] 

# XGBoost Parameters (Updated as requested)
XGB_PARAMS = {
    'n_estimators': 500,      # <-- 500'e çıkarıldı (Daha güçlü model)
    'max_depth': 8,           
    'learning_rate': 0.05,    # <-- 0.05'e düşürüldü (Daha hassas öğrenme)
    'subsample': 0.8,         
    'colsample_bytree': 0.8,  
    'objective': 'multi:softmax',
    'random_state': 42,
    'n_jobs': -1,             
    'tree_method': 'hist',    
    'eval_metric': 'merror',
    'early_stopping_rounds': 15
}

def add_gaussian_noise(image, sigma):
    """Adds Gaussian noise to the image."""
    mean = 0
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image.astype(float) + gauss, 0, 255)
    return noisy_image.astype(np.uint8)

def main():
    print(f"--- EXPERIMENT 7: MODERATE AUGMENTATION (1+2 Versions) ---")
    print(f"Sigmas: {SIGMA_VALUES}")
    print(f"Model: n_estimators=500, lr=0.05")
    print(f"Process PID: {os.getpid()}")
    
    # --- 1. Data Loading ---
    if not os.path.exists(CSV_FILE): 
        print(f"Error: CSV file not found at {CSV_FILE}")
        return
        
    df = pd.read_csv(CSV_FILE)
    print(f"Original Dataset: {len(df)} records.")

    data_list = []
    label_list = []
    
    print("Processing images and Expanding Dataset (x3)...")
    
    missing_count = 0
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        try:
            img_path = row['image']
            label = row['label']
            
            # --- PATH FIX (Kesin Çözüm) ---
            # data/ + Img/img001-001.png
            full_image_path = os.path.join('data', img_path)
            
            if not os.path.exists(full_image_path):
                # Linux case-sensitivity check
                if os.path.exists(full_image_path.replace('Img', 'img')):
                    full_image_path = full_image_path.replace('Img', 'img')
                elif os.path.exists(img_path): 
                     full_image_path = img_path
                else:
                    missing_count += 1
                    continue
            
            image = cv2.imread(full_image_path)
            if image is None: 
                missing_count += 1
                continue
            
            # Preprocessing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            # --- MODERATE AUGMENTATION LOOP ---
            
            # 1. Original Image
            data_list.append(resized)
            label_list.append(label)
            
            # 2. Noisy Images (Sigma 25, 50)
            for s in SIGMA_VALUES:
                noisy_img = add_gaussian_noise(resized, sigma=s)
                data_list.append(noisy_img)
                label_list.append(label)
            
        except Exception as e:
            missing_count += 1
            pass

    if missing_count > 0:
        print(f"Warning: {missing_count} images could not be loaded.")

    if len(data_list) == 0:
        print("CRITICAL ERROR: No images loaded.")
        return

    X_raw = np.array(data_list)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(np.array(label_list))
    
    print(f"\n--- Augmentation Complete ---")
    print(f"Original Count: {len(df)}")
    print(f"New Total Count: {len(X_raw)} (Should be ~10,230)")

    # --- 2. HOG Feature Extraction ---
    print("\n--- Extracting HOG Features (8x8) ---")
    hog_feats = []
    for img in tqdm(X_raw, desc="Calculating HOG"):
        f = hog(img, orientations=8, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=False, feature_vector=True)
        hog_feats.append(f)
    X_hog = np.array(hog_feats)
    print(f"Feature Matrix Shape: {X_hog.shape}")

    # --- 3. Advanced Split (75% Train / 10% Val / 15% Test) ---
    print("\n--- Splitting Dataset (75/10/15) ---")
    
    # 1. Separate Test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_hog, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    
    # 2. Separate Validation (10% of total)
    val_ratio = 0.10 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )
    
    print(f"Training Set:   {X_train.shape[0]} samples")
    print(f"Validation Set: {X_val.shape[0]} samples")
    print(f"Test Set:       {X_test.shape[0]} samples")
    
    # --- 4. Training ---
    print("\n--- Training XGBoost Model ---")
    print(f"Parameters: {XGB_PARAMS}")
    
    model = xgb.XGBClassifier(**XGB_PARAMS, num_class=len(label_encoder.classes_))
    
    start = time.time()
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)], 
        verbose=50 
    )
    end = time.time()
    
    duration = end - start
    print(f"\n✅ Training Completed! Duration: {duration:.2f} sec ({duration/60:.2f} min)")
    
    # --- 5. Results ---
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    
    joblib.dump({'model': model, 'encoder': label_encoder}, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")
    
    print("\n--- FINAL TEST EVALUATION ---")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"TEST ACCURACY: %{acc*100:.2f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

if __name__ == "__main__":
    main()