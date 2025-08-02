import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


def save_model(model, path):
    joblib.dump(model, path)


def load_data():
    """Fetch the California housing dataset."""
    X, y = fetch_california_housing(return_X_y=True)
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split the dataset into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(model, X_test, y_test):
    """Evaluate a regression model and return R² and MSE."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return r2, mse

