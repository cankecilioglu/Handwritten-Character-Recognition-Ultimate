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

# --- SETTINGS ---
CSV_FILE = 'data/english.csv'
MODEL_FILE = "models/model_exp1_optimized_eyes.joblib"
TARGET_SIZE = (64, 64)

# BALANCED MODEL PARAMETERS FOR HOG OPTIMIZATION
# Since feature count increases significantly (7000+), we use a balanced model configuration.
XGB_PARAMS = {
    'n_estimators': 200,      # Balanced number of trees
    'max_depth': 6,           # Standard depth
    'learning_rate': 0.1,     # Standard learning rate
    'objective': 'multi:softmax',
    'random_state': 42,
    'n_jobs': -1,             # Use all CPU cores
    'tree_method': 'hist',    # Fast histogram method
    'eval_metric': 'merror'   # Classification error rate
}

def main():
    print(f"--- IMPROVEMENT 2: HOG (DETAILED) OPTIMIZATION ---")
    
    # 1. Data Loading
    if not os.path.exists(CSV_FILE): 
        print(f"Error: CSV file not found at {CSV_FILE}")
        return
        
    df = pd.read_csv(CSV_FILE)
    print(f"Dataset: {len(df)} records.")

    data_list, label_list = [], []
    print("Processing images...")
    
    # Image processing loop
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Reading Images"):
        try:
            img_path = row['image']
            
            # Robust path handling for Linux/Project structure
            full_image_path = os.path.join('data', os.path.basename(img_path))
            
            if not os.path.exists(full_image_path):
                # Fallback for 'Img' vs 'img' case sensitivity
                if os.path.exists(full_image_path.replace('Img', 'img')):
                    full_image_path = full_image_path.replace('Img', 'img')
                else:
                    # Try raw path if outside data folder
                    if os.path.exists(img_path):
                         full_image_path = img_path
                    else:
                        continue
            
            image = cv2.imread(full_image_path)
            if image is None: continue
            
            # Grayscale and Resize
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            data_list.append(resized)
            label_list.append(row['label'])
        except Exception as e:
            pass

    X_raw = np.array(data_list)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(np.array(label_list))

    # 2. DETAILED HOG (4x4) - CRITICAL CHANGE
    print("\n--- Extracting DETAILED HOG Features (4x4) ---")
    print("This process will take longer due to increased feature count...")
    
    hog_feats = []
    for img in tqdm(X_raw, desc="Calculating HOG"):
        # Decreasing pixels_per_cell to (4, 4) increases detail by 4x
        f = hog(img, orientations=8, pixels_per_cell=(4, 4),
                cells_per_block=(2, 2), visualize=False, feature_vector=True)
        hog_feats.append(f)
    
    X_hog = np.array(hog_feats)
    print(f"NEW Feature Count: {X_hog.shape[1]} (Old: ~1500)")

    # 3. Training
    print("\n--- Splitting Data (80/20) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_hog, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print("\n--- Training XGBoost Model ---")
    model = xgb.XGBClassifier(**XGB_PARAMS, num_class=len(label_encoder.classes_))
    
    start = time.time()
    
    # Training with verbose logging
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=20 # Report status every 20 trees
    )
    end = time.time()
    
    duration = end - start
    print(f"\n✅ Training Completed! Duration: {duration:.2f} sec ({duration/60:.2f} min)")
    
    # 4. Results and Saving
    # Ensure models directory exists
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    
    joblib.dump({'model': model, 'encoder': label_encoder}, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\nFINAL ACCURACY: %{acc*100:.2f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, model.predict(X_test), target_names=label_encoder.classes_, zero_division=0))

if __name__ == "__main__":
    main()