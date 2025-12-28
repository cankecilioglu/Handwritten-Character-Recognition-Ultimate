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

# --- SETTINGS ---
# File paths adapted for the project structure
CSV_FILE = 'data/english.csv'
MODEL_FILE = "models/model_exp4_noise_test.joblib"
TARGET_SIZE = (64, 64)
NOISE_LEVEL = 20  # Noise intensity (Sigma)

# XGBoost Parameters Optimized for Ryzen 5 7600 CPU
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.05,
    'objective': 'multi:softmax',
    'random_state': 42,
    'n_jobs': -1,        # Use all cores
    'tree_method': 'hist', # Fastest method for CPU
    'eval_metric': 'merror'
}

def add_gaussian_noise(image, sigma):
    """Adds Gaussian noise to the image (uint8)."""
    mean = 0
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image.astype(float) + gauss, 0, 255)
    return noisy_image.astype(np.uint8)

def main():
    print(f"--- EXPERIMENT 4: NOISE ROBUSTNESS TEST (AMD CPU: Ryzen 5 7600) ---")
    print(f"Process started... PID: {os.getpid()}")

    # --- 1. Data Loading and Adding Noise ---
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: {CSV_FILE} not found!")
        return

    df = pd.read_csv(CSV_FILE)
    print(f"Dataset loaded. Total records: {len(df)}")

    data_list = []
    label_list = []

    print(f"Loading images and adding Gaussian Noise (Sigma={NOISE_LEVEL})...")
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Images"):
        image_rel_path = row['image'] 
        label = row['label']
        
        # Construct full path (assuming script is run from project root)
        full_image_path = os.path.join('data', os.path.basename(image_rel_path))
        
        # Linux Case-Sensitivity Check
        if not os.path.exists(full_image_path):
             if os.path.exists(full_image_path.replace('Img', 'img')):
                 full_image_path = full_image_path.replace('Img', 'img')
             else:
                 # Fallback: check if path exists as is (if running inside data folder)
                 if os.path.exists(image_rel_path):
                     full_image_path = image_rel_path
                 else:
                     continue 

        try:
            image = cv2.imread(full_image_path)
            if image is None:
                continue

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            # --- ADD NOISE ---
            noisy_image = add_gaussian_noise(resized_image, sigma=NOISE_LEVEL)
            
            data_list.append(noisy_image)
            label_list.append(label)
        except Exception as e:
            print(f"Error: {full_image_path} - {e}")

    print(f"Successfully processed {len(data_list)} images.")

    X_raw = np.array(data_list)
    y_raw = np.array(label_list)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    print(f"Classes encoded. Total Classes: {len(label_encoder.classes_)}")

    # --- 2. HOG Feature Extraction ---
    print("\n--- Starting HOG Feature Extraction ---")
    hog_features_list = []
    for image in tqdm(X_raw, desc="Calculating HOG"):
        features = hog(
            image, orientations=8, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), visualize=False, feature_vector=True
        )
        hog_features_list.append(features)
    
    X_hog = np.array(hog_features_list)
    print(f"HOG Complete. Data Shape: {X_hog.shape}")

    # --- 3. Data Split ---
    print("\n--- Splitting Dataset (80/20) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_hog, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Training Set: {X_train.shape}")
    print(f"Test Set:     {X_test.shape}")

    # --- 4. Model Training ---
    print("\n--- Training XGBoost Model ---")
    print(f"Parameters: {XGB_PARAMS}")
    
    model = xgb.XGBClassifier(**XGB_PARAMS, num_class=len(label_encoder.classes_))

    start_time = time.time()
    
    # Training with verbose output for tracking
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=10
    )
    
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\n✅ TRAINING COMPLETED!")
    print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

    # --- 5. Saving Model and Report ---
    # Ensure models directory exists
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    
    data_to_save = {'model': model, 'encoder': label_encoder}
    joblib.dump(data_to_save, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")

    print("\n--- PERFORMANCE EVALUATION ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: %{accuracy * 100:.2f}")

    print("\n--- Detailed Report ---")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

if __name__ == "__main__":
    main()