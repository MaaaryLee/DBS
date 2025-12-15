import numpy as np
import tensorflow as tf
from pathlib import Path

SAVED_MODEL_DIR = 'tf_model'
OUTPUT_PATH = Path('tflite_actors/model_int8.tflite')
STATES_PATH = Path('states_eval.npy')

def _ensure_states(saved_model_dir: str, path: Path) -> np.ndarray:
    """
    Ensure a states_eval.npy exists. If missing, synthesize states that match the
    SavedModel input dimension (batch ignored).
    """
    loaded = tf.saved_model.load(saved_model_dir)
    func = loaded.signatures.get("serving_default")
    if func is None:
        raise SystemExit("SavedModel has no 'serving_default' signature to infer input shape.")

    # Assume single input named 'observation' or take first key.
    input_tensor = None
    for name, t in func.structured_input_signature[1].items():
        input_tensor = t
        break
    if input_tensor is None:
        raise SystemExit("Could not infer input tensor from SavedModel signature.")

    shape = input_tensor.shape.as_list()
    if len(shape) != 2:
        raise SystemExit(f"Expected 2D input [batch, features], got shape={shape}")
    obs_dim = int(shape[1])

    rng = np.random.default_rng(seed=0)
    states = rng.normal(loc=0.0, scale=1.0, size=(1000, obs_dim)).astype("float32")

    if path.exists():
        existing = np.load(path).astype("float32")
        if existing.ndim == 2 and existing.shape[1] == obs_dim:
            return existing

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, states)
    print(f"[OK] Synthesized states_eval.npy with shape={states.shape} for INT8 calibration")
    return states

states = _ensure_states(SAVED_MODEL_DIR, STATES_PATH)

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
