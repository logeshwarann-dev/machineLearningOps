# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn
import psutil
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set the MLflow tracking URI to 'http'
mlflow.set_tracking_uri("http://localhost:5000")

# Function for data preprocessing
def preprocess_data(data: pd.DataFrame):
    """Preprocess the data by splitting features (X) and target (y)."""
    X = data.drop('cardio', axis=1)
    y = data['cardio']
    return X, y

# Function to perform hyperparameter tuning using GridSearchCV
def hyperparameter_tuning(model, param_grid, X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Function to train a RandomForest model with hyperparameter tuning
def train_rf_model(X_train, y_train):
    """Train a RandomForest model using GridSearchCV for hyperparameter tuning."""
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'n_estimators': [100, 200, 300]
    }
    return hyperparameter_tuning(rf, param_grid, X_train, y_train)

# Function to train a LogisticRegression model with hyperparameter tuning
def train_lr_model(X_train, y_train):
    """Train a LogisticRegression model using GridSearchCV for hyperparameter tuning."""
    lr = LogisticRegression(random_state=42, max_iter=1000)
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }
    return hyperparameter_tuning(lr, param_grid, X_train, y_train)

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return key metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    logging.info(f"Accuracy: {accuracy:.2f}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    return accuracy, precision, recall, f1, confusion

# Function to log model and system metrics to MLflow
def log_to_mlflow(model, model_name, X_train, X_test, y_train, y_test):
    """Log model parameters, metrics, and system usage to MLflow."""
    with mlflow.start_run():
        # Log the model name and parameters
        mlflow.log_param("model_name", model_name)
        if isinstance(model, RandomForestClassifier):
            mlflow.log_param("max_depth", model.max_depth)
            mlflow.log_param("n_estimators", model.n_estimators)
        elif isinstance(model, LogisticRegression):
            mlflow.log_param("C", model.C)
            mlflow.log_param("solver", model.solver)

        # Evaluate and log metrics
        accuracy, precision, recall, f1, confusion = evaluate_model(model, X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log confusion matrix metrics
        confusion_dict = {
            "true_positive": confusion[1][1],
            "false_positive": confusion[0][1],
            "true_negative": confusion[0][0],
            "false_negative": confusion[1][0]
        }
        mlflow.log_metrics(confusion_dict)

        # Log system metrics (CPU, Memory)
        mlflow.log_metric("system_cpu_usage", psutil.cpu_percent(interval=1))
        mlflow.log_metric("system_memory_usage", psutil.virtual_memory().percent)

        # Log execution time for training the model
        start_time = time.time()
        model.fit(X_train, y_train)
        execution_time = time.time() - start_time
        mlflow.log_metric("training_time", execution_time)

        # Log the model to MLflow
        mlflow.sklearn.log_model(model, model_name)

# Main function to execute the workflow
def main():
    # Load the dataset
    data_path = '../dataset/cardiovascular_dataset.csv'  # Adjust path as needed
    data = pd.read_csv(data_path)

    # Preprocess the data
    X, y = preprocess_data(data)

    # Split the data into training (70%) and testing (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train and log the RandomForest model
    rf_model = train_rf_model(X_train, y_train)
    log_to_mlflow(rf_model, "RandomForest", X_train, X_test, y_train, y_test)

    # Train and log the LogisticRegression model
    lr_model = train_lr_model(X_train, y_train)
    log_to_mlflow(lr_model, "LogisticRegression", X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
