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
    """Preprocess the data by splitting features (X) and target (y)."""
    X = data.drop('cardio', axis=1)
    y = data['cardio']
    return X, y

# Function to train a tuned RandomForest model
def train_rf_model(X_train, y_train):
    """Train a RandomForest model with more aggressively tuned hyperparameters."""
    clf = RandomForestClassifier(
        n_estimators=300,        # Further increased number of trees
        max_depth=20,            # Deeper trees for capturing more patterns
        min_samples_split=2,     # Minimum samples required to split a node
        min_samples_leaf=1,      # Minimum samples required at a leaf node
        max_features='sqrt',     # Use square root of features to reduce overfitting
        random_state=42,
        n_jobs=-1                # Use all processors for faster training
    )
    clf.fit(X_train, y_train)
    return clf

# Function to train a tuned LogisticRegression model
def train_lr_model(X_train, y_train):
    """Train a LogisticRegression model with further tuned hyperparameters."""
    clf = LogisticRegression(
        C=0.5,                  # More regularization to prevent overfitting
        solver='liblinear',     # Suitable solver for small to medium datasets
        max_iter=3000,           # Increased iterations for convergence
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return key metrics."""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    confusion = confusion_matrix(y_test, y_pred)

    logging.info(f"Accuracy: {accuracy:.2f}")
    logging.info(f"Precision: {precision:.2f}")
    logging.info(f"Recall: {recall:.2f}")
    logging.info(f"F1 Score: {f1:.2f}")
    logging.info(f"Confusion Matrix:\n{confusion}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    return accuracy, precision, recall, f1, confusion

# Function to log model and system metrics to MLflow
def log_to_mlflow(model, model_name, X_train, X_test, y_train, y_test):
    """Log model parameters, metrics, and system usage to MLflow."""
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        
        # Log hyperparameters for RandomForest
        if isinstance(model, RandomForestClassifier):
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth", model.max_depth)
            mlflow.log_param("min_samples_split", model.min_samples_split)
            mlflow.log_param("min_samples_leaf", model.min_samples_leaf)
            mlflow.log_param("max_features", model.max_features)

        # Log hyperparameters for LogisticRegression
        if isinstance(model, LogisticRegression):
            mlflow.log_param("C", model.C)
            mlflow.log_param("solver", model.solver)

        # Evaluate and log model metrics
        accuracy, precision, recall, f1, confusion = evaluate_model(model, X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log confusion matrix as individual metrics
        mlflow.log_metric("true_positive", confusion[1][1])
        mlflow.log_metric("false_positive", confusion[0][1])
        mlflow.log_metric("true_negative", confusion[0][0])
        mlflow.log_metric("false_negative", confusion[1][0])

        # Log system metrics (CPU and Memory)
        mlflow.log_metric("system_cpu_usage", psutil.cpu_percent(interval=1))
        mlflow.log_metric("system_memory_usage", psutil.virtual_memory().percent)

        # Log training time
        start_time = time.time()
        model.fit(X_train, y_train)
        mlflow.log_metric("training_time", time.time() - start_time)

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
