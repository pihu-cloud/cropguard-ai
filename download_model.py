"""
download_model.py
=================
Downloads resnet_best.keras from Google Drive if not present locally.
Called automatically at app startup on Render.

Set env var: GDRIVE_MODEL_ID = your_google_drive_file_id
"""

import os
import logging

log = logging.getLogger('model_downloader')

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'resnet_best.keras')
GDRIVE_ID  = os.environ.get('GDRIVE_MODEL_ID', '')


def ensure_model() -> bool:
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        log.info('Model already present (%.1f MB): %s', size_mb, MODEL_PATH)
        return True

    if not GDRIVE_ID:
        log.warning('Model not found and GDRIVE_MODEL_ID not set. Running in Demo Mode.')
        return False

    try:
        import gdown
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        url = f'https://drive.google.com/uc?id={GDRIVE_ID}'
        log.info('Downloading model from Google Drive (ID: %s)...', GDRIVE_ID)
        gdown.download(url, MODEL_PATH, quiet=False)
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        log.info('Downloaded: %.1f MB', size_mb)
        return True
    except Exception as e:
        log.error('Download failed: %s', e)
        return False


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    ensure_model()
