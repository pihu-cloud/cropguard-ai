"""
CropGuard AI — Flask Application
=================================
ResNet50V2-only backend.
Model file: resnet.h5  (place in the same folder as this script OR in models/)
"""

import json
import logging
import os

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────────────────
# App configuration
# ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.update(
    SECRET_KEY         = 'cropguard-production-secret-2025',
    UPLOAD_FOLDER      = 'static/uploads',
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024,   # 16 MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'},
)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(name)s — %(message)s')
log = logging.getLogger('cropguard.app')

# ─────────────────────────────────────────────────────────
# Static data  (class_names.json + disease_info.json live
# next to app.py — generated automatically from the model)
# ─────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_BASE, 'class_names.json')) as f:
    CLASS_NAMES: list = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)

with open(os.path.join(_BASE, 'disease_info.json')) as f:
    DISEASE_INFO: dict = json.load(f)


def _make_fallback(label: str) -> dict:
    """Metadata for any class label not in disease_info.json."""
    parts      = label.split('___')
    crop       = parts[0].replace('_', ' ')
    disease_raw = parts[1] if len(parts) > 1 else 'Unknown'
    is_healthy = 'healthy' in disease_raw.lower()
    name       = ('Healthy ' + crop) if is_healthy else disease_raw.replace('_', ' ')
    return {
        'name':        name,
        'crop':        crop,
        'description': 'No disease detected.' if is_healthy else 'Consult your local extension service.',
        'symptoms':    ['No visible symptoms.'] if is_healthy else ['Consult a plant pathologist.'],
        'treatment':   ['Continue standard care.'] if is_healthy else ['Consult your local agricultural extension service.'],
        'severity':    'None' if is_healthy else 'Medium',
    }


# ─────────────────────────────────────────────────────────
# Model registry  (ResNet only)
# ─────────────────────────────────────────────────────────
models: dict = {}


def _find_model_file() -> str | None:
    """
    Search for model file in priority order:
      1. models/resnet_best.keras   (output of new retrain.py)
      2. models/resnet_final.keras
      3. models/resnet.h5           (original model)
      4. resnet.h5 in root folder
    Returns the first path found, or None.
    """
    candidates = [
        os.path.join(_BASE, 'models', 'resnet_best.keras'),
        os.path.join(_BASE, 'models', 'resnet_final.keras'),
        os.path.join(_BASE, 'models', 'resnet.h5'),
        os.path.join(_BASE, 'resnet.h5'),
    ]
    for p in candidates:
        if os.path.exists(p):
            log.info('Found model at: %s', p)
            return p
    return None


def _build_resnet_architecture(num_classes: int):
    """
    Rebuild ResNet50V2 architecture from scratch so we can load
    weights manually — avoids the keras.models.load_model hang on
    Windows caused by optimizer reconstruction from old HDF5 files.
    Architecture matches the saved model exactly:
      ResNet50V2 base → GAP → BN → Dense(1024,relu) → Drop(0.4)
                      → Dense(512,relu) → Drop(0.3)
                      → Dense(256,relu) → Drop(0.2)
                      → Dense(num_classes, softmax)
    """
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50V2
    from tensorflow.keras import layers, models as km, regularizers

    base = ResNet50V2(
        weights      = None,
        include_top  = False,
        input_shape  = (224, 224, 3),
    )
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1024, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return km.Model(inputs=base.input, outputs=out)


def load_models() -> None:
    global models
    models = {}

    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        log.warning('TensorFlow not installed — running in DEMO mode')
        return

    path = _find_model_file()
    if path is None:
        log.warning('resnet.h5 not found in %s or %s/models/ — DEMO mode',
                    _BASE, _BASE)
        return

    log.info('Loading ResNet50V2 from %s ...', path)

    # .keras format loads fine on all platforms (no optimizer hang issue)
    # Only .h5 legacy files hang on Windows.
    import platform
    is_keras_format = path.endswith('.keras')
    on_windows = platform.system() == 'Windows'

    m = None

    # Strategy 1: standard load
    # Always works for .keras files; Linux/Mac only for .h5
    if is_keras_format or not on_windows:
        try:
            log.info('Strategy 1: keras.models.load_model ...')
            m = keras.models.load_model(path, compile=False)
            log.info('Strategy 1 succeeded.')
        except Exception as e1:
            log.warning('Strategy 1 failed: %s', e1)

    # Strategy 2: rebuild architecture + load weights by name
    # Primary path on Windows; fallback on other OS.
    if m is None:
        try:
            log.info('Strategy 2: rebuild arch + load_weights by_name ...')
            m = _build_resnet_architecture(NUM_CLASSES)
            m.load_weights(path, by_name=True, skip_mismatch=True)
            log.info('Strategy 2 succeeded.')
        except Exception as e2:
            log.warning('Strategy 2 failed: %s', e2)
            m = None

    # Strategy 3: rebuild architecture + load weights positionally
    if m is None:
        try:
            log.info('Strategy 3: rebuild arch + load_weights positional ...')
            m = _build_resnet_architecture(NUM_CLASSES)
            m.load_weights(path)
            log.info('Strategy 3 succeeded.')
        except Exception as e3:
            log.warning('Strategy 3 failed: %s', e3)
            m = None

    # Strategy 4: h5py direct weight injection (last resort)
    if m is None:
        try:
            import h5py
            log.info('Strategy 4: h5py direct weight injection ...')
            m = _build_resnet_architecture(NUM_CLASSES)
            with h5py.File(path, 'r') as hf:
                wg = hf['model_weights']
                for layer in m.layers:
                    if layer.name in wg:
                        grp = wg[layer.name]
                        wnames = grp.attrs.get('weight_names', [])
                        weights = [grp[w][()] for w in wnames]
                        if weights:
                            layer.set_weights(weights)
            log.info('Strategy 4 succeeded.')
        except Exception as e4:
            log.warning('Strategy 4 failed: %s', e4)
            m = None

    if m is None:
        log.error('All loading strategies failed -- DEMO mode active')
        return

    out_classes = m.output_shape[-1]
    if out_classes != NUM_CLASSES:
        log.error(
            'Model output has %d classes but class_names.json has %d — '
            'update class_names.json to match your model.',
            out_classes, NUM_CLASSES
        )

    # Warm up: run one dummy forward pass so first real prediction is fast
    try:
        import numpy as np
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        m.predict(dummy, verbose=0)
        log.info('Model warm-up complete.')
    except Exception as ew:
        log.warning('Warm-up failed (non-fatal): %s', ew)

    models['resnet'] = m
    log.info('ResNet50V2 ready — %d output classes', out_classes)

    if not models:
        log.info('No valid model loaded — DEMO mode active')


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS'])


def _run_predict(image_path: str, use_seg: bool) -> dict:
    """Thin wrapper that binds app-level data into predict_disease()."""
    from predict_pipeline import predict_disease  # local import avoids TF at module load
    result = predict_disease(
        image_path       = image_path,
        models           = models,
        class_names      = CLASS_NAMES,
        disease_info     = DISEASE_INFO,
        make_fallback_fn = _make_fallback,
        use_segmentation = use_seg,
    )
    result['image_url'] = '/' + image_path.replace('\\', '/')
    return result


def _app_stats() -> dict:
    disease_labels = [k for k in CLASS_NAMES if 'healthy' not in k.lower()]
    crops = sorted(set(
        k.split('___')[0].replace('_', ' ').rstrip(',') for k in CLASS_NAMES
    ))
    return {
        'total_classes':        NUM_CLASSES,
        'disease_classes':      len(disease_labels),
        'healthy_classes':      NUM_CLASSES - len(disease_labels),
        'crops':                len(crops),
        'crop_list':            crops,
        'models_loaded':        len(models),
        'model_names':          list(models.keys()),
        'demo_mode':            not bool(models),
        'tta_views':            5,
        'confidence_threshold': 30.0,
    }


# ─────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('home.html', stats=_app_stats())


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'GET':
        return render_template('detect.html', result=None, error=None)

    if 'image' not in request.files:
        return render_template('detect.html', result=None,
                               error='No file part in request.')

    file = request.files['image']
    if not file or file.filename == '':
        return render_template('detect.html', result=None, error='No file selected.')
    if not allowed_file(file.filename):
        return render_template('detect.html', result=None,
                               error='Unsupported format. Use JPG, PNG, or WEBP.')

    save_path = os.path.join(
        app.config['UPLOAD_FOLDER'], secure_filename(file.filename)
    )
    file.save(save_path)

    try:
        use_seg = request.form.get('segmentation', 'true').lower() != 'false'
        result  = _run_predict(save_path, use_seg)
        return render_template('detect.html', result=result, error=None)
    except Exception as exc:
        log.exception('Prediction failed')
        return render_template('detect.html', result=None,
                               error=f'Prediction failed: {exc}')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    JSON API endpoint.

    POST multipart/form-data:
        image        : image file (required)
        segmentation : 'true'/'false'  (default 'true')
    """
    if 'image' not in request.files:
        return jsonify({'error': "No 'image' field in request"}), 400

    file = request.files['image']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    save_path = os.path.join(
        app.config['UPLOAD_FOLDER'], secure_filename(file.filename)
    )
    file.save(save_path)

    try:
        use_seg = request.form.get('segmentation', 'true').lower() != 'false'
        return jsonify(_run_predict(save_path, use_seg))
    except Exception as exc:
        log.exception('API prediction failed')
        return jsonify({'error': str(exc)}), 500


@app.route('/api/stats')
def api_stats():
    return jsonify(_app_stats())


@app.route('/diseases')
def diseases():
    crop_filter = request.args.get('crop', 'all')
    crops       = sorted(set(v['crop'] for v in DISEASE_INFO.values()))
    filtered    = (DISEASE_INFO if crop_filter == 'all'
                   else {k: v for k, v in DISEASE_INFO.items()
                         if v['crop'] == crop_filter})
    return render_template('diseases.html', diseases=filtered,
                           crops=crops, selected_crop=crop_filter)


@app.route('/about')
def about():
    model_path   = _find_model_file() or 'Not found'
    model_status = {
        'resnet': {
            'loaded':      'resnet' in models,
            'file_exists': _find_model_file() is not None,
            'path':        model_path,
        }
    }
    return render_template('about.html', model_status=model_status,
                           class_names=CLASS_NAMES, disease_count=NUM_CLASSES,
                           stats=_app_stats())


# ─────────────────────────────────────────────────────────
# Load models at startup (works for both direct run and gunicorn)
load_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
