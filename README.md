#  MLOps Linear Regression Pipeline

This project implements a complete MLOps pipeline using **Linear Regression** on the **California Housing Dataset**. It follows the full development lifecycle: training, testing, quantization, Dockerization, and CI/CD — all in a single `main` branch, without any manual web uploads.

##  Project Structure

- `src/` contains:
  - `train.py`: Trains the model and saves it.
  - `predict.py`: Loads and runs predictions using the trained model.
  - `quantize.py`: Manually quantizes model coefficients.
  - `utils.py`: Helper functions (e.g., save/load model).
- `tests/` contains:
  - `test_train.py`: Unit tests for training pipeline.
- `.github/workflows/ci.yml`: GitHub Actions pipeline with test, train, and build stages.
- `Dockerfile`: Builds the container image for model inference.
- `requirements.txt`: Python dependencies.
- `model.joblib`: Trained model (if generated).
- `README.md`: Project documentation.

##  Model Overview

- **Model Used**: `LinearRegression` from `scikit-learn`
- **Dataset**: California Housing (from `sklearn.datasets`)
- **Split**: Train/Test with 80:20 ratio
- **Metrics**:
  - R² Score to evaluate goodness of fit
  - Mean Squared Error to measure loss

##  Docker Integration

The project includes a Dockerfile that encapsulates dependencies and runs inference via `src/predict.py`. Once built, the Docker image can be used to verify predictions and portability of the trained model.

##  CI/CD (GitHub Actions)

The repository includes a GitHub Actions pipeline that triggers on every push to `main`. It runs the following jobs in sequence:

1. **test-suite**: Runs unit tests using `pytest` to validate model training.
2. **train-and-quantize**: Trains the model and performs quantization.
3. **build-and-test-container**: Builds the Docker image and runs it to ensure predictions work correctly inside a container.

Each stage must pass for the pipeline to be considered successful.

##  Model Comparison Table
##  R² Score and File Size Comparison

| Metric                     | Before Quantization | After Quantization |
|----------------------------|---------------------|--------------------|
| R² Score                   | 0.5758              | 0.5747             |
| Model File Size (joblib)   | 0.68 KB             | 0.37 KB            |
| Quantization Method        | —                   | Fixed-point (×1000, int16) |



##  Constraints Followed

- Only `main` branch used.
- No GitHub web uploads (all CLI-based).
- Dockerized workflow tested and working.
- CI/CD fully configured with GitHub Actions.
- Code modularized into `src/`, `tests/`, and utils.
- No hardcoded outputs or plagiarized code.

##  Final Notes

The pipeline ensures reproducibility, automation, and containerized deployment. It reflects real-world MLOps practices and satisfies all assignment constraints.

