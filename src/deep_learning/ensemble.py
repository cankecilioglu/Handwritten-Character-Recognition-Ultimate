import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
BATCH_SIZE = 64
IMG_SIZE = 64  # If all models use 64x64. Change if your models expect 128x128, etc.

# Load models
model_paths = [
    'swish_cnn_model.keras',
    'optimized_cnn_model.keras',
    'nuclear_model.keras'
]

models = []
print('Loading models...')
for path in model_paths:
    if os.path.exists(path):
        print(f'Loading: {path}')
        models.append(load_model(path))
    else:
        print(f'File not found: {path}')

if len(models) == 0:
    print('ERROR: No models could be loaded.')
    exit()

# Data preparation (test set only)
# This script assumes `X_test` and `y_test` already exist in memory.
# If they don't, load/prepare them before running.

print(f'\nStarting ensemble inference ({len(models)} models)...')

# Collect predictions from each model
all_preds = []
for i, model in enumerate(models):
    print(f'Model {i + 1} predicting...')

    # 1) Direct prediction
    pred = model.predict(X_test, verbose=0)

    # 2) Optional: TTA prediction (can be disabled if too slow)
    datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1)
    pred_tta = model.predict(datagen.flow(X_test, batch_size=BATCH_SIZE, shuffle=False), verbose=0)
    pred = (pred + pred_tta) / 2

    all_preds.append(pred)

# Voting (probability averaging)
ensemble_pred_probs = np.mean(all_preds, axis=0)
ensemble_classes = np.argmax(ensemble_pred_probs, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Compute accuracy
acc = accuracy_score(true_classes, ensemble_classes)

print("\n" + "="*40)
print(f'ENSEMBLE ACCURACY: {acc * 100:.2f}%')
print(f"="*40)