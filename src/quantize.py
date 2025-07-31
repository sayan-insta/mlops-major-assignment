import joblib
import numpy as np

model = joblib.load("model.joblib")
params = {'coef': model.coef_, 'intercept': model.intercept_}
joblib.dump(params, "unquant_params.joblib")

quant_coef = np.round((params['coef'] - params['coef'].min()) / 
                      (params['coef'].max() - params['coef'].min()) * 255).astype(np.uint8)

joblib.dump({'coef': quant_coef}, "quant_params.joblib")

# De-quantize and test
dequant_coef = quant_coef.astype(float) / 255 * (params['coef'].max() - params['coef'].min()) + params['coef'].min()
print("Sample Dequantized Coefficients:", dequant_coef[:5])

