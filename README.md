**--  MLOps Linear Regression Pipeline --**

This project implements a complete MLOps pipeline using Linear Regression on the California Housing Dataset. It follows the full development lifecycle: training, testing, quantization, Dockerization, and CI/CD — all in a single main branch, without any manual web uploads.


##  Project Structure
--**src/ contains:
    **train.py: Trains the model and saves it.
    **predict.py: Loads and runs predictions using the trained model.
    **quantize.py: Manually quantizes model coefficients.
    **utils.py: Helper functions (e.g., save/load model).
--**tests/ contains:
    **test_train.py: Unit tests for training pipeline.
- **.github/workflows/ci.yml: GitHub Actions pipeline with test, train, and build stages.
- **Dockerfile: Builds the container image for model inference.
- **requirements.txt: Python dependencies.
- **model.joblib: Trained model (if generated).
- **README.md: Project documentation.
---

##  Model Overview

- **Model Used:** `LinearRegression` from scikit-learn
- **Dataset:** California Housing (`sklearn.datasets`)
- **Split:** 80% Train / 20% Test
- **Evaluation Metrics:**
  - R² Score (goodness of fit)
  - Mean Squared Error (loss)

---

## ️ Docker Integration

This project includes a `Dockerfile` that runs `src/predict.py` inside a container. This validates the model inside an isolated environment for reproducible deployment.

---

##  CI/CD – GitHub Actions

On every push to `main`, GitHub Actions runs:

1. ** test-suite:** Unit tests via `pytest`
2. ** train-and-quantize:** Model training and quantization
3. ** build-and-test-container:** Docker image build + inference test

All jobs must succeed for the pipeline to pass.

---

##  Model Performance

###  R² Score and File Size Comparison

| Metric                      | Original Model      | Quantized Model (uint8) |
|----------------------------|---------------------|--------------------------|
| **R² Score**               | 0.5758              | 0.5758                  |
| **Mean Squared Error (MSE)** | 0.5559            | 0.5559                  |
| **Model File Size**        | 0.04 KB             | 0.01 KB                 |
| **Quantization Method**    | —                   | Per-weight, `uint8` (0–255) |

>  Quantization successfully reduces model size without degrading accuracy.


##  Quantization Method

- **Approach:** Per-weight symmetric quantization
- **Type:** `uint8` (0–255)
- **Scaling:** Each weight is scaled individually for precision
- **Dequantization:** Restores float coefficients for evaluation


##  Constraints Followed

- ️ Only `main` branch used
- ️ All Git operations via CLI (no web uploads)
- ️ Docker tested locally with `predict.py`
- ️ CI/CD using GitHub Actions
- ️ Code structured under `src/`, `tests/`, and `utils/`
- ️ Fully modular and reproducible
-  Final `README.md` and reports auto-generated

##  Final Notes
The pipeline ensures reproducibility, automation, and containerized deployment. It reflects real-world MLOps practices and satisfies all assignment constraints.
