import numpy as np
import tensorflow as tf
from pathlib import Path

SAVED_MODEL_DIR = 'tf_model'
OUTPUT_PATH = Path('tflite_actors/model_int8.tflite')
STATES_PATH = Path('states_eval.npy')

if not STATES_PATH.exists():
    raise SystemExit('missing states_eval.npy')

states = np.load(STATES_PATH).astype('float32')

def representative_dataset():
    for row in states[:1000]:
        yield [row.reshape(1, -1)]

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR, signature_keys=['serving_default'])
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

try:
    tflite_model = converter.convert()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_bytes(tflite_model)
    print(f'[OK] Saved {OUTPUT_PATH}')
except Exception as exc:
    import traceback
    print(f'[X] Conversion failed: {exc}')
    traceback.print_exc()
