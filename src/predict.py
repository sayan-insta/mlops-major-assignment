import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

model = joblib.load("model.joblib")
X, _ = fetch_california_housing(return_X_y=True)
_, X_test = train_test_split(X, test_size=0.2)
pred = model.predict(X_test[:5])
print("Sample Predictions:", pred)

