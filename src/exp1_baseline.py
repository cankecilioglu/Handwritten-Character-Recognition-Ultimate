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
CSV_FILE = 'data/english.csv'  # Updated for folder structure
MODEL_FILE = "models/model_exp1_optimized_brain.joblib"
TARGET_SIZE = (64, 64)

# OPTIMIZED MODEL PARAMETERS (For AMD Ryzen CPU Power)
XGB_PARAMS = {
    'n_estimators': 500,      # Increased to 500 Trees
    'max_depth': 8,           # Increased Depth to 8
    'learning_rate': 0.05,    # Slower learning rate for better convergence
    'subsample': 0.8,         # Data subsampling to prevent overfitting
    'colsample_bytree': 0.8,  # Feature subsampling
    'objective': 'multi:softmax',
    'random_state': 42,
    'n_jobs': -1,             # Use all CPU cores
    'tree_method': 'hist',    # Fast histogram method
    'eval_metric': 'merror'   # Classification error rate
}

def main():
    print(f"--- IMPROVEMENT 1: MODEL OPTIMIZATION (Full Dataset) ---")
    
    # 1. Data Loading
    if not os.path.exists(CSV_FILE): 
        print(f"Error: CSV file not found at {CSV_FILE}")
        return
        
    df = pd.read_csv(CSV_FILE)
    print(f"Dataset: {len(df)} records. (NO FILTERING)")

    data_list, label_list = [], []
    print("Processing images...")
    
    # Processing loop with progress bar
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Reading Images"):
        try:
            img_path = row['image']
            
            # Linux/Path compatibility check
            # Adjusts path if running from root directory vs src directory
            full_image_path = os.path.join('data', os.path.basename(img_path))
            
            if not os.path.exists(full_image_path):
                # Fallback for 'Img' vs 'img' folder case sensitivity
                if os.path.exists(full_image_path.replace('Img', 'img')):
                    full_image_path = full_image_path.replace('Img', 'img')
                else:
                    # Try checking raw path if above logic fails
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

    # 2. STANDARD HOG (8x8)
    print("\n--- Extracting Standard HOG Features (8x8) ---")
    hog_feats = []
    for img in tqdm(X_raw, desc="Calculating HOG"):
        f = hog(img, orientations=8, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=False, feature_vector=True)
        hog_feats.append(f)
    X_hog = np.array(hog_feats)
    print(f"Feature Count: {X_hog.shape[1]}")

    # 3. Training
    print("\n--- Splitting Data (80/20) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_hog, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print("\n--- Training XGBoost Model ---")
    print(f"Parameters: {XGB_PARAMS}")
    
    model = xgb.XGBClassifier(**XGB_PARAMS, num_class=len(label_encoder.classes_))
    
    start = time.time()
    
    # Training with verbose logging instead of callbacks for compatibility
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50 # Print status report every 50 trees
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