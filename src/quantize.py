import joblib
import numpy as np
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Load data and model
X, y = fetch_california_housing(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = joblib.load("model.joblib")

# Inference with original model
original_preds = model.predict(X_test)
original_r2 = r2_score(y_test, original_preds)
original_size = os.path.getsize("model.joblib") / 1024  # KB

# Save original coefficients
params = {'coef': model.coef_, 'intercept': model.intercept_}
joblib.dump(params, "unquant_params.joblib")

#  Fixed-point quantization
scaling_factor = 1000
quantized = np.round(model.coef_ * scaling_factor).astype(np.int16)

# Save quantized model
quant_params = {
    'coef': quantized,
    'intercept': model.intercept_,
    'scale': scaling_factor
}
joblib.dump(quant_params, "quant_params.joblib")
quant_size = os.path.getsize("quant_params.joblib") / 1024  # KB

#  Dequantization
dequantized = quantized.astype(float) / scaling_factor

# Inference using dequantized model
dequant_model = LinearRegression()
dequant_model.coef_ = dequantized
dequant_model.intercept_ = model.intercept_
dequant_preds = dequant_model.predict(X_test)
dequant_r2 = r2_score(y_test, dequant_preds)

# 🔍 Print Comparison
print("\n Coefficient Comparison (first 5 values):")
for i in range(5):
    print(f"Original: {model.coef_[i]:.4f} → Quantized: {quantized[i]} → Dequantized: {dequantized[i]:.4f}")

print("\n R² Score Comparison:")
print(f"Original Model R²:    {original_r2:.4f}")
print(f"Dequantized Model R²: {dequant_r2:.4f}")

print("\n File Size Comparison:")
print(f"Original model.joblib:         {original_size:.2f} KB")
print(f"Quantized quant_params.joblib: {quant_size:.2f} KB")

