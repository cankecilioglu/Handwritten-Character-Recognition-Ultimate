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
CSV_FILE = 'data/english.csv'  # Path to the CSV file
MODEL_FILE = "models/final_model_filtered.joblib" # Path to save the model
TARGET_SIZE = (64, 64)

# XGBoost Hyperparameters (Optimized for AMD Ryzen 5 7600 CPU)
XGB_PARAMS = {
    'n_estimators': 300,      # Number of trees
    'max_depth': 8,           # Tree depth
    'learning_rate': 0.05,    # Learning rate (eta)
    'objective': 'multi:softmax',
    'random_state': 42,
    'n_jobs': -1,             # Use all CPU cores
    'tree_method': 'hist',    # Histogram-based method (Fastest for CPU)
    'eval_metric': 'merror'   # Multiclass classification error rate
}

# CHARACTERS TO REMOVE (Potential Sources of Confusion)
# Removing numbers and visually identical upper/lower case letters
BLACKLIST_CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
    'o', 'O', 'i', 'I', 'l', 
    's', 'S', 'z', 'Z', 'c', 'C', 'p', 'P', 'k', 'K', 
    'v', 'V', 'w', 'W', 'x', 'X', 'u', 'U', 'j', 'J'
]

def main():
    print(f"--- EXPERIMENT 2: FILTERED DATASET (Final Version) ---")
    print(f"Process PID: {os.getpid()}")
    
    # --- 1. Data Loading and Filtering ---
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: {CSV_FILE} not found!")
        return

    df = pd.read_csv(CSV_FILE)
    original_count = len(df)
    print(f"Dataset loaded. Original records: {original_count}")

    # FILTERING PROCESS
    df = df[~df['label'].isin(BLACKLIST_CHARS)]
    filtered_count = len(df)
    print(f"Filtering complete. Remaining records: {filtered_count} (Removed: {original_count - filtered_count})")

    data_list = []
    label_list = []

    print("Processing images...")
    # Loop through images with progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Reading Images"):
        # Relative path from CSV
        image_rel_path = row['image'] 
        label = row['label']
        
        # Construct full path (assuming script is run from project root)
        # Adjust this based on your folder structure. 
        # If CSV contains 'Img/img.png', and folder is 'data/Img', we handle it here:
        full_image_path = os.path.join('data', os.path.basename(image_rel_path))
        
        # Linux Case-Sensitivity Check
        if not os.path.exists(full_image_path):
             # Fallback: Check if folder is named 'img' instead of 'Img'
             if os.path.exists(full_image_path.replace('Img', 'img')):
                 full_image_path = full_image_path.replace('Img', 'img')
             else:
                 # Fallback 2: Try raw path if files are not in 'data' folder
                 if os.path.exists(image_rel_path):
                     full_image_path = image_rel_path
                 else:
                     continue 

        try:
            image = cv2.imread(full_image_path)
            if image is None: continue

            # Convert to Grayscale and Resize
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
    print(f"Processed Data Shape: {X_raw.shape}")
    print(f"Unique Classes ({len(label_encoder.classes_)}): {label_encoder.classes_}")

    # --- 2. HOG Feature Extraction ---
    print("\n--- Starting HOG Feature Extraction ---")
    hog_features_list = []
    for image in tqdm(X_raw, desc="Calculating HOG"):
        # Standard HOG parameters (8x8 cells)
        features = hog(
            image, orientations=8, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), visualize=False, feature_vector=True
        )
        hog_features_list.append(features)
    
    X_hog = np.array(hog_features_list)
    print(f"HOG Complete. Feature Matrix: {X_hog.shape}")

    # --- 3. Train/Test Split ---
    print("\n--- Splitting Dataset (80/20) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_hog, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Training Set: {X_train.shape}")
    print(f"Test Set:     {X_test.shape}")

    # --- 4. Model Training ---
    print("\n--- Starting XGBoost Training ---")
    print("Note: Progress will be reported every 10 trees as 'merror' (Multiclass Error Rate).")
    print("Lower merror is better (0 = Perfect, 1 = All Wrong).")
    
    model = xgb.XGBClassifier(**XGB_PARAMS, num_class=len(label_encoder.classes_))

    start_time = time.time()
    
    # Training
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)], 
        verbose=10 # Report status every 10 trees
    )
    
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

    # --- 5. Saving Model and Reporting ---
    # Ensure models directory exists
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

    data_to_save = {'model': model, 'encoder': label_encoder}
    joblib.dump(data_to_save, MODEL_FILE)
    print(f"Model saved to disk: {MODEL_FILE}")

    print("\n--- PERFORMANCE REPORT ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"FINAL ACCURACY: %{accuracy * 100:.2f}")

    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

if __name__ == "__main__":
    main()