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
MODEL_FILE = "models/model_exp3_validation.joblib"
TARGET_SIZE = (64, 64)

# XGBoost Parameters (Baseline for Validation Test)
XGB_PARAMS = {
    'n_estimators': 100,      # Standard baseline
    'max_depth': 6,           # Standard depth
    'learning_rate': 0.05,
    'objective': 'multi:softmax',
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
    'eval_metric': 'mlogloss' # Log loss is better for early stopping monitoring
}

def main():
    print(f"--- EXPERIMENT 3: VALIDATION SPLIT ANALYSIS ---")
    print(f"Process PID: {os.getpid()}")
    
    # --- 1. Data Loading ---
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: {CSV_FILE} not found!")
        return

    df = pd.read_csv(CSV_FILE)
    print(f"Dataset loaded. Total records: {len(df)}")

    data_list = []
    label_list = []

    print("Processing images...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Reading Images"):
        try:
            image_rel_path = row['image']
            label = row['label']
            
            # Robust path construction
            full_image_path = os.path.join('data', os.path.basename(image_rel_path))
            
            if not os.path.exists(full_image_path):
                if os.path.exists(full_image_path.replace('Img', 'img')):
                    full_image_path = full_image_path.replace('Img', 'img')
                else:
                    if os.path.exists(image_rel_path):
                         full_image_path = image_rel_path
                    else:
                        continue

            image = cv2.imread(full_image_path)
            if image is None: continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            data_list.append(resized)
            label_list.append(label)
        except Exception as e:
            pass

    X_raw = np.array(data_list)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(np.array(label_list))
    
    print(f"Total Unique Classes: {len(label_encoder.classes_)}")

    # --- 2. HOG Feature Extraction ---
    print("\n--- Extracting HOG Features ---")
    hog_feats = []
    for img in tqdm(X_raw, desc="Calculating HOG"):
        f = hog(img, orientations=8, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=False, feature_vector=True)
        hog_feats.append(f)
    X_hog = np.array(hog_feats)
    print(f"Feature Matrix Shape: {X_hog.shape}")

    # --- 3. Advanced Data Splitting (Train/Val/Test) ---
    print("\n--- Splitting Dataset (75% Train / 10% Val / 15% Test) ---")
    
    # Step 1: Separate Test Set (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_hog, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    
    # Step 2: Separate Validation Set from the remaining 85%
    # We want Validation to be ~10% of the TOTAL original data.
    # Since we have 85% left, we need (10 / 85) = 0.1176 split ratio.
    val_split_ratio = 0.10 / 0.85
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_split_ratio, random_state=42, stratify=y_train_val
    )
    
    print(f"Training Set:   {X_train.shape} samples")
    print(f"Validation Set: {X_val.shape} samples")
    print(f"Test Set:       {X_test.shape} samples")

    # --- 4. Training with Validation Monitoring ---
    print("\n--- Training XGBoost with Early Stopping ---")
    print("The model will stop training if validation score doesn't improve for 10 rounds.")
    
    model = xgb.XGBClassifier(**XGB_PARAMS, num_class=len(label_encoder.classes_))
    
    start = time.time()
    
    # Fitting with eval_set
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)], # Monitor Train and Val
        verbose=10, # Print log every 10 trees
        early_stopping_rounds=10 # Stop if Validation Metric stops improving
    )
    end = time.time()
    
    duration = end - start
    print(f"\n✅ Training Completed! Duration: {duration:.2f} sec")
    
    if model.best_iteration:
        print(f"Best Iteration: {model.best_iteration}")

    # --- 5. Saving and Evaluation ---
    # Ensure models directory exists
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    
    joblib.dump({'model': model, 'encoder': label_encoder}, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")
    
    # Final Evaluation is ALWAYS done on the Test Set (Unseen Data)
    print("\n--- FINAL TEST EVALUATION ---")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"TEST ACCURACY: %{acc*100:.2f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

if __name__ == "__main__":
    main()