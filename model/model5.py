# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
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
    logging.info("Preprocessing the data by splitting features (X) and target (y).")
    X = data.drop('cardio', axis=1)
    y = data['cardio']
    return X, y

# Function to train a tuned RandomForest model
def train_rf_model(X_train, y_train):
    logging.info("Train a RandomForest model with manually tuned hyperparameters.")
    # Tuned hyperparameters
    clf = RandomForestClassifier(
        n_estimators=200,        # Increased number of trees for better learning
        max_depth=10,            # Deeper trees for capturing more patterns
        min_samples_split=5,     # Minimum samples required to split a node
        min_samples_leaf=4,      # Minimum samples required at a leaf node
        random_state=42,
        n_jobs=-1                # Use all processors for faster training
    )
    clf.fit(X_train, y_train)
    return clf

# Function to train a tuned LogisticRegression model
def train_lr_model(X_train, y_train):
    logging.info("Train a LogisticRegression model with manually tuned hyperparameters.")
    clf = LogisticRegression(
        C=1.0,                  # Tuned regularization parameter
        solver='liblinear',     # Suitable solver for small to medium datasets
        max_iter=2000,           # Increased iterations for convergence
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf

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
        mlflow.log_param("model_name", model_name)
        logging.info(f"Model: {model_name}")
        
        # Log hyperparameters for RandomForest
        if isinstance(model, RandomForestClassifier):
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth", model.max_depth)
            mlflow.log_param("min_samples_split", model.min_samples_split)
            mlflow.log_param("min_samples_leaf", model.min_samples_leaf)

        # Log hyperparameters for LogisticRegression
        if isinstance(model, LogisticRegression):
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
