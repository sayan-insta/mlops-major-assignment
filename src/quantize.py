import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from utils import load_data, split_data, evaluate_model


def quantize_to_uint8(weights):
    """Symmetric per-weight quantization using uint8 encoding."""
    scales = np.zeros_like(weights, dtype=np.float32)
    q_weights = np.zeros_like(weights, dtype=np.uint8)

    for i, w in enumerate(weights):
        max_val = max(abs(w), 1e-8)  # Prevent divide-by-zero
        scale = max_val / 127.0
        q_val = int(round(w / scale)) + 128
        q_val = np.clip(q_val, 0, 255)
        q_weights[i] = q_val
        scales[i] = scale

    return q_weights, scales


def dequantize_from_uint8(q_weights, scales):
    """Dequantize from uint8 back to float using scales."""
    return (q_weights.astype(np.int16) - 128) * scales


def memory_kb(arr):
    return arr.nbytes / 1024


def main():
    # Load model and extract weights
    model = joblib.load("model.joblib")
    coef = model.coef_.astype(np.float32)
    intercept = np.array([model.intercept_], dtype=np.float32)

    # Save original weights
    joblib.dump({"coef": coef, "intercept": intercept}, "unquant_params.joblib")

    # Quantize weights
    q_coef, coef_scales = quantize_to_uint8(coef)
    q_intercept, intercept_scales = quantize_to_uint8(intercept)

    # Save quantized weights
    joblib.dump({
        "q_coef": q_coef,
        "coef_scales": coef_scales,
        "q_intercept": q_intercept,
        "intercept_scales": intercept_scales
    }, "quant_params.joblib")

    # Load data and evaluate original model
    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)
    r2_orig, mse_orig = evaluate_model(model, X_test, y_test)

    # Reconstruct model from quantized weights
    deq_coef = dequantize_from_uint8(q_coef, coef_scales)
    deq_intercept = dequantize_from_uint8(q_intercept, intercept_scales)[0]
    preds = X_test @ deq_coef + deq_intercept

    # Evaluate quantized model
    r2_q = r2_score(y_test, preds)
    mse_q = mean_squared_error(y_test, preds)

    # Memory comparison
    orig_mem = memory_kb(coef) + memory_kb(intercept)
    quant_mem = memory_kb(q_coef) + memory_kb(q_intercept)

    # Print results
    print("\n Coefficient Comparison (first 5 values):")
    for i in range(min(5, len(coef))):
        print(f"Original: {coef[i]:.4f} → Quantized: {q_coef[i]} → Dequantized: {deq_coef[i]:.4f}")

    print("\n R² & MSE Comparison:")
    print(f"Original Model → R²: {r2_orig:.4f}, MSE: {mse_orig:.4f}")
    print(f"Quantized Model → R²: {r2_q:.4f}, MSE: {mse_q:.4f}")

    print("\n Memory Usage:")
    print(f"Original weights:  {orig_mem:.2f} KB")
    print(f"Quantized weights: {quant_mem:.2f} KB")

    print("\n Sample Predictions:", preds[:5])


if __name__ == "__main__":
    main()

