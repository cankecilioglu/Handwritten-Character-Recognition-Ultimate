import os
import zipfile
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    BatchNormalization, Activation, Input, GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Configuration
ZIP_FILE = 'archive.zip'
CSV_FILE = 'english.csv'
IMG_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 100

# GPU check
if tf.config.list_physical_devices('GPU'):
    print("GPU available: training will use GPU if configured.")
else:
    print("No GPU detected: training may be slow.")

# Prepare data: extract archive if needed and load CSV
if not os.path.exists('Img') and os.path.exists(ZIP_FILE):
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall('.')

if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    data = []
    labels = []
    base_path = '.'

    print('Loading images...')
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(base_path, row['image'])
        if not os.path.exists(img_path):
            # keep original path if change attempts fail
            pass
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(row['label'])
        except Exception:
            # ignore individual file errors
            continue

    X = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    y_categorical = to_categorical(y_encoded)
    num_classes = len(le.classes_)

    # Split into train / val / test (75/10/15 approx)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_categorical, test_size=0.15, random_state=42, stratify=y_encoded
    )
    val_split = 0.10 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=42, stratify=np.argmax(y_temp, axis=1)
    )

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        shear_range=0.15,
        fill_mode='nearest'
    )
    train_datagen.fit(X_train)

    # Cosine decay scheduler with warmup
    def cosine_decay_with_warmup(epoch):
        lr_start = 1e-5
        lr_max = 1e-3
        lr_min = 1e-5
        warmup_epochs = 10
        if epoch < warmup_epochs:
            return lr_start + ((lr_max - lr_start) / warmup_epochs) * epoch
        progress = (epoch - warmup_epochs) / max(1, (EPOCHS - warmup_epochs))
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))

    lr_scheduler = LearningRateScheduler(cosine_decay_with_warmup, verbose=0)

    # Model definition (Swish activation + L2 regularization)
    reg = l2(1e-4)
    model = Sequential()
    model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 1)))

    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.4))

    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.4))

    # Head
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    # Training
    checkpoint = ModelCheckpoint('swish_cnn_model.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    print('Starting training...')
    model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, lr_scheduler],
        verbose=1
    )

    # Test with TTA (Test Time Augmentation)
    print('\nRunning TTA evaluation...')
    model.load_weights('swish_cnn_model.keras')
    tta_datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest'
    )
    preds_tta = []
    for _ in range(5):
        preds_tta.append(model.predict(tta_datagen.flow(X_test, batch_size=BATCH_SIZE, shuffle=False), verbose=0))
    preds_tta.append(model.predict(X_test, verbose=0))

    final_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(np.mean(preds_tta, axis=0), axis=1))
    print(f'Swish model TTA accuracy: {final_acc * 100:.2f}%')

    # If running in an environment that supports file download, user can download the saved model manually.