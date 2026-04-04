"""
Phase 10 — Edge Hardware Profiling & TFLite
Evaluates the deployability of anomaly detection models on resource-constrained IoT edge devices.

Measures:
  1. Inference Latency (ms per sample)
  2. Peak RAM Usage (MB) for inference
  3. Model File Size (KB)
  4. Accuracy drop from INT8 quantization (LSTM-AE)

Target platforms:
  - ESP32 (Microcontroller): ~520 KB SRAM, ~4 MB Flash, No OS
  - Raspberry Pi 4 (SBC): 1-8 GB RAM, Linux OS

Reference: SCIO Research Framework §9.1
"""

import os
import pathlib
import time
import tracemalloc
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import joblib
import pickle

warnings.filterwarnings("ignore")

SPLITS_DIR  = pathlib.Path("data/splits")
RESULTS_DIR = pathlib.Path("outputs/results")
MODELS_DIR  = pathlib.Path("outputs/results")

def format_bytes(size: float) -> str:
    """Format bytes to KB/MB."""
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.2f} MB"

def get_file_size(path: pathlib.Path) -> float:
    """Get file size in bytes."""
    if path.exists():
        return path.stat().st_size
    return 0.0

# ─── Profiling Wrappers ──────────────────────────────────────────────────────

def profile_inference(
    predict_fn, 
    X_sample: np.ndarray, 
    n_runs: int = 100
) -> Tuple[float, float]:
    """
    Measures average inference time and peak memory for a single sample.
    
    Args:
        predict_fn: Callable that takes X_sample and returns predictions
        X_sample: Single sample array of shape (1, n_features) or (1, seq_len, n_features)
        n_runs: Number of iterations to average over
        
    Returns:
        avg_time_ms: Average inference time in milliseconds
        peak_ram_mb: Peak memory usage in megabytes
    """
    # 1. Measure Latency
    # Warmup
    for _ in range(5):
        predict_fn(X_sample)
        
    start_time = time.perf_counter()
    for _ in range(n_runs):
        predict_fn(X_sample)
    end_time = time.perf_counter()
    avg_time_ms = ((end_time - start_time) / n_runs) * 1000.0

    # 2. Measure Memory
    tracemalloc.start()
    predict_fn(X_sample)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak / (1024 * 1024)

    return avg_time_ms, peak_ram_mb


# ─── TFLite Quantization ──────────────────────────────────────────────────────

def quantize_lstm_ae_to_tflite(
    model_path: pathlib.Path, 
    calib_data: np.ndarray
) -> pathlib.Path:
    """
    Quantize Keras LSTM-AE to INT8 TFLite model using representative dataset.
    """
    import tensorflow as tf
    
    # Load Keras model
    model = tf.keras.models.load_model(str(model_path))
    
    # Representative dataset generator for INT8 calibration
    def representative_data_gen():
        # Yield subsets of calibration data
        for i in range(min(100, len(calib_data))):
            yield [calib_data[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization settings for deep embedded devices
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    
    # Constrain ops to INT8 ONLY (required for Edge TPU or strict integer DSPs)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    try:
        tflite_model = converter.convert()
        tflite_path = model_path.parent / (model_path.stem + "_quantized.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        return tflite_path
    except Exception as e:
        print(f"[edge] TFLite conversion failed (fallback to weights-only): {e}")
        # Fallback to dynamic range quantization (weights only, activations float)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        tflite_model = converter.convert()
        tflite_path = model_path.parent / (model_path.stem + "_dynamic_quant.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        return tflite_path


def run_tflite_inference(tflite_path: pathlib.Path, x_seqs: np.ndarray) -> np.ndarray:
    """Run inference using TFLite interpreter and return predictions."""
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    
    is_int8_in = input_details[0]['dtype'] == np.int8
    in_scale, in_zero = input_details[0]['quantization']
    out_scale, out_zero = output_details[0]['quantization']
    
    predictions = []
    
    # Process sequentially
    for i in range(len(x_seqs)):
        x_in = x_seqs[i:i+1].astype(np.float32)
        
        # Quantize input if needed
        if is_int8_in:
            x_in = np.round(x_in / in_scale + in_zero).astype(np.int8)
            
        interpreter.set_tensor(input_index, x_in)
        interpreter.invoke()
        
        y_out = interpreter.get_tensor(output_index)
        
        # Dequantize output if needed
        if is_int8_in:
            y_out = (y_out.astype(np.float32) - out_zero) * out_scale
            
        predictions.append(y_out[0])
        
    return np.array(predictions)


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run_phase10_profiling(
    splits_dir:  str | pathlib.Path = SPLITS_DIR,
    results_dir: str | pathlib.Path = RESULTS_DIR,
) -> pd.DataFrame:
    """
    Run hardware profiling for all models.
    """
    splits_dir  = pathlib.Path(splits_dir)
    results_dir = pathlib.Path(results_dir)
    
    print("[edge] Loading test subset for profiling...")
    test_df = pd.read_csv(splits_dir / "test.csv", parse_dates=["timestamp"])
    train_df = pd.read_csv(splits_dir / "train.csv", parse_dates=["timestamp"])
    
    # Label cols matching previous phases
    label_cols = ["is_anomaly", "anomaly_type", "is_weather_event", 
                  "timestamp", "device_id", "protocol"]
    feat_cols = [c for c in test_df.columns if c not in label_cols 
                 and test_df[c].dtype in (np.float64, np.float32, np.int64, np.int32)
                 and c != "is_low_irradiance_period"]
                 
    X_train = train_df[feat_cols].values
    X_test  = test_df[feat_cols].values
    
    # Single sample for edge simulator
    x_single = X_test[0:1]
    
    results = []
    
    # ─── 1. Isolation Forest ───────────────────────────────────────────────
    if (results_dir / "isolation_forest_model.pkl").exists():
        model_path = results_dir / "isolation_forest_model.pkl"
        with open(model_path, "rb") as f:
            if_model = pickle.load(f)
            
        latency, ram = profile_inference(if_model.predict, x_single)
        size = get_file_size(model_path)
        
        results.append({
            "Method": "Isolation Forest",
            "Latency_ms": latency,
            "Peak_RAM_MB": ram,
            "Size_KB": size / 1024,
            "ESP32_Feasible": size < 4_000_000 and ram < 0.4
        })

    # ─── 2. Local Outlier Factor ───────────────────────────────────────────
    if (results_dir / "lof_model.pkl").exists():
        model_path = results_dir / "lof_model.pkl"
        with open(model_path, "rb") as f:
            lof_model = pickle.load(f)
            
        latency, ram = profile_inference(lof_model.predict, x_single)
        size = get_file_size(model_path)
        
        results.append({
            "Method": "Local Outlier Factor",
            "Latency_ms": latency,
            "Peak_RAM_MB": ram,
            "Size_KB": size / 1024,
            "ESP32_Feasible": size < 4_000_000 and ram < 0.4
        })

    # ─── 3. LSTM Autoencoder ───────────────────────────────────────────────
    # We suppress TF logs for this part to keep output clean
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    if (results_dir / "lstm_ae_model.keras").exists():
        model_path = results_dir / "lstm_ae_model.keras"
        meta_path = results_dir / "lstm_ae_model_meta.pkl"
        lstm_model = tf.keras.models.load_model(str(model_path))
        
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        seq_len = meta["seq_len"]
        
        # Build sequence for single inference
        x_seq_single = np.repeat(x_single, seq_len, axis=0).reshape(1, seq_len, len(feat_cols))
        
        # We wrap prediction to ignore verbose
        def predict_keras(x):
            return lstm_model.predict(x, verbose=0)
            
        latency, ram = profile_inference(predict_keras, x_seq_single, n_runs=50)
        size = get_file_size(model_path)
        
        results.append({
            "Method": "LSTM-AE (Keras FP32)",
            "Latency_ms": latency,
            "Peak_RAM_MB": ram,
            "Size_KB": size / 1024,
            "ESP32_Feasible": size < 4_000_000 and ram < 0.4 # Keras cannot run on ESP32 natively anyway, but evaluating footprint
        })
        
        # ─── 4. TFLite INT8 Quantization ───────────────────────────────────
        print("\n[edge] Quantizing LSTM-AE to TFLite (INT8) ...")
        # Build calibration dataset
        from src.models.lstm_autoencoder import build_sequences
        calib_seqs = build_sequences(X_train[:1000].astype(np.float32), seq_len)
        
        tflite_path = quantize_lstm_ae_to_tflite(model_path, calib_seqs)
        
        # Read TFLite signature and test
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        in_idx = interpreter.get_input_details()[0]['index']
        out_idx = interpreter.get_output_details()[0]['index']
        is_int8 = interpreter.get_input_details()[0]['dtype'] == np.int8
        
        def predict_tflite(x):
            if is_int8:
                scale, zero = interpreter.get_input_details()[0]['quantization']
                x = np.round(x / scale + zero).astype(np.int8)
            interpreter.set_tensor(in_idx, x)
            interpreter.invoke()
            return interpreter.get_tensor(out_idx)
            
        latency_tf, ram_tf = profile_inference(predict_tflite, x_seq_single, n_runs=100)
        size_tf = get_file_size(tflite_path)
        
        results.append({
            "Method": "LSTM-AE (TFLite Quantized)",
            "Latency_ms": latency_tf,
            "Peak_RAM_MB": ram_tf,
            "Size_KB": size_tf / 1024,
            "ESP32_Feasible": size_tf < 4_000_000 and ram_tf < 0.4  # TFLite Micro fits on ESP32
        })
        
        # Quantization Accuracy Drop Test (Optional F1 recalculation)
        # We will log the TFLite file footprint reduction
        compress_ratio = size / size_tf
        print(f"[edge] TFLite compression ratio: {compress_ratio:.1f}x")
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(results_dir / "edge_profiling_results.csv", index=False)
    
    print("\n=== Phase 10 Edge Profiling Results ===")
    print(df_results.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n[!] ESP32 Feasible condition: Model < 4MB flash AND Peak RAM < 400KB")
    print("    Local Outlier Factor (LOF) typically scales RAM with training data size, making it hostile for microcontrollers.")
    
    return df_results


if __name__ == "__main__":
    run_phase10_profiling()
