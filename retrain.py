"""
retrain.py — CropGuard AI · Train from scratch with ImageNet weights
=====================================================================
Previous approach failed because the old resnet.h5 head has wrong
weight shapes (2048->1024 vs expected 256->10), so loading it by_name
silently skips all the important layers and trains from random init.

This version loads ResNet50V2 with official ImageNet weights (downloads
automatically ~100MB) and trains a fresh classifier head on top.
Expected final accuracy: 85-95% on PlantVillage data.

Run:
    python retrain.py

Output:
    models/resnet_best.keras   <- best checkpoint (use this)
    models/resnet_final.keras  <- final epoch weights
"""

import os, json, logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger('retrain')

# ── CONFIG ────────────────────────────────────────────────
TRAIN_DIR       = 'data/train'
VAL_DIR         = 'data/val'
OUTPUT_BEST     = 'models/resnet_best.keras'
OUTPUT_FINAL    = 'models/resnet_final.keras'
CLASS_JSON      = 'class_names.json'

IMG_SIZE        = 224
BATCH_SIZE      = 16        # lower to 8 if you get memory errors
PHASE1_EPOCHS   = 10       # head only
PHASE2_EPOCHS   = 20       # fine-tune top layers
PHASE1_LR       = 1e-3
PHASE2_LR       = 1e-5
UNFREEZE_LAYERS = 50        # more layers = better accuracy, slower

# ImageNet normalisation — must match predict_pipeline.py
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
# ─────────────────────────────────────────────────────────

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from tensorflow.keras.applications import ResNet50V2

log.info('TensorFlow %s | GPU: %s', tf.__version__,
         bool(tf.config.list_physical_devices('GPU')))

with open(CLASS_JSON) as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)
log.info('%d classes: %s', NUM_CLASSES, CLASS_NAMES)

# ── AUGMENTATION ──────────────────────────────────────────
augment = keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.3),
    layers.RandomZoom((-0.25, 0.25)),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.3),
    layers.RandomContrast(0.3),
], name='augmentation')


def normalize(image):
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
    std  = tf.constant(IMAGENET_STD,  dtype=tf.float32)
    return (image - mean) / std


def load_dataset(directory, training):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'Not found: {directory}')
    ds = keras.utils.image_dataset_from_directory(
        directory,
        labels      = 'inferred',
        label_mode  = 'categorical',
        class_names = CLASS_NAMES,
        image_size  = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        shuffle     = training,
        seed        = 42,
    )
    if training:
        ds = ds.map(lambda x, y: (augment(normalize(x), training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (normalize(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


log.info('Loading datasets...')
train_ds = load_dataset(TRAIN_DIR, training=True)
val_ds   = load_dataset(VAL_DIR,   training=False)
log.info('Datasets ready.')

# ── BUILD MODEL FROM IMAGENET WEIGHTS ─────────────────────
# This is the key fix: use weights='imagenet' so the base
# starts with proven features, not random noise.
log.info('Building ResNet50V2 with ImageNet weights...')
base = ResNet50V2(
    weights     = 'imagenet',   # downloads ~100MB automatically
    include_top = False,
    input_shape = (IMG_SIZE, IMG_SIZE, 3),
)
base.trainable = False  # freeze entire base for Phase 1
log.info('Base loaded. %d layers.', len(base.layers))

# Classifier head — clean simple design for best accuracy
x   = base.output
x   = layers.GlobalAveragePooling2D()(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dense(512, activation='relu',
                   kernel_regularizer=regularizers.l2(1e-4))(x)
x   = layers.Dropout(0.4)(x)
x   = layers.Dense(256, activation='relu',
                   kernel_regularizer=regularizers.l2(1e-4))(x)
x   = layers.Dropout(0.3)(x)
out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = keras.Model(inputs=base.input, outputs=out)

total      = len(model.layers)
trainable  = sum(1 for l in model.layers if l.trainable)
log.info('Model: %d total layers, %d trainable (head only)', total, trainable)

# ── CALLBACKS ─────────────────────────────────────────────
def make_callbacks(monitor='val_accuracy'):
    return [
        callbacks.ModelCheckpoint(
            OUTPUT_BEST, monitor=monitor,
            save_best_only=True, verbose=1),
        callbacks.EarlyStopping(
            monitor=monitor, patience=5,
            restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-9, verbose=1),
        callbacks.CSVLogger('training_log.csv', append=False),
    ]

# ── PHASE 1: HEAD ONLY ────────────────────────────────────
log.info('='*60)
log.info('PHASE 1: Train head only (base frozen)')
log.info('Epochs: %d  |  LR: %s', PHASE1_EPOCHS, PHASE1_LR)
log.info('='*60)

model.compile(
    optimizer = keras.optimizers.Adam(PHASE1_LR),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy',
                 keras.metrics.TopKCategoricalAccuracy(k=3, name='top3')],
)

h1 = model.fit(
    train_ds, epochs=PHASE1_EPOCHS,
    validation_data=val_ds,
    callbacks=make_callbacks(),
    verbose=1,
)
best1 = max(h1.history.get('val_accuracy', [0]))
log.info('Phase 1 done. Best val_accuracy: %.2f%%', best1 * 100)

# ── PHASE 2: UNFREEZE TOP LAYERS ──────────────────────────
log.info('='*60)
log.info('PHASE 2: Unfreeze top %d layers, fine-tune', UNFREEZE_LAYERS)
log.info('Epochs: %d  |  LR: %s', PHASE2_EPOCHS, PHASE2_LR)
log.info('='*60)

# Unfreeze top N layers of the base
for layer in model.layers:
    layer.trainable = False
for layer in model.layers[-UNFREEZE_LAYERS:]:
    layer.trainable = True

trainable = sum(1 for l in model.layers if l.trainable)
log.info('Trainable layers: %d / %d', trainable, len(model.layers))

model.compile(
    optimizer = keras.optimizers.Adam(PHASE2_LR),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy',
                 keras.metrics.TopKCategoricalAccuracy(k=3, name='top3')],
)

h2 = model.fit(
    train_ds, epochs=PHASE2_EPOCHS,
    validation_data=val_ds,
    callbacks=make_callbacks(),
    verbose=1,
)

# ── SAVE FINAL ────────────────────────────────────────────
model.save(OUTPUT_FINAL)

best_acc = max(
    h1.history.get('val_accuracy', [0]) +
    h2.history.get('val_accuracy', [0])
) * 100

log.info('='*60)
log.info('TRAINING COMPLETE')
log.info('Best val_accuracy : %.2f%%', best_acc)
log.info('Best model saved  : %s', OUTPUT_BEST)
log.info('Final model saved : %s', OUTPUT_FINAL)
log.info('')
log.info('Next step:')
log.info('  The app will auto-load %s', OUTPUT_BEST)
log.info('  Just restart app.py')
log.info('='*60)
