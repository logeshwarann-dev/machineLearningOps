# Installations (in your environment)
# 1. pip install mlflow
# 2. pip install psutil
# 3. pip install scikit-learn

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

# Set the MLflow tracking URI to 'http'
mlflow.set_tracking_uri("http://localhost:5000")

# Function for data preprocessing
def preprocess_data(data):
    # Convert categorical variables to one-hot encoding if needed
    #data = pd.get_dummies(data, columns=['Gender', 'CovidSeverity']) # Not used here
    
    # Split data into X (features) and y (target)
    X = data.drop('cardio', axis=1)
    y = data['cardio']
    
    return X, y

# Function to train a RandomForest model
def train_rf_model(X_train, y_train, max_depth=3, n_estimators=100):
    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Function to train a LogisticRegression model
def train_lr_model(X_train, y_train):
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    confusion = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy, precision, recall, f1, confusion

# Function to log model and system metrics to MLflow
def log_to_mlflow(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        # Log the model name (either RandomForest or LogisticRegression)
        mlflow.log_param("model_name", model_name)

        # Log model-specific parameters (RandomForest)
        if isinstance(model, RandomForestClassifier):
            mlflow.log_param("max_depth", model.max_depth)
            mlflow.log_param("n_estimators", model.n_estimators)

        # Log model metrics
        accuracy, precision, recall, f1, confusion = evaluate_model(model, X_test, y_test)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log confusion matrix
        confusion_dict = {
            "true_positive": confusion[1][1],
            "false_positive": confusion[0][1],
            "true_negative": confusion[0][0],
            "false_negative": confusion[1][0]
        }
        mlflow.log_metrics(confusion_dict)

        # Log system metrics (CPU and Memory)
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent

        mlflow.log_metric("system_cpu_usage", cpu_usage)
        mlflow.log_metric("system_memory_usage", memory_usage)

        # Log execution time for training the model
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        execution_time = end_time - start_time
        mlflow.log_metric("training_time", execution_time)

        # Log the model to MLflow
        mlflow.sklearn.log_model(model, model_name)

# Main function
def main():
    # Load the dataset
    data = pd.read_csv('../dataset/cardiovascular_dataset.csv')

    # Preprocess the data
    X, y = preprocess_data(data)

    # Split the data into training (70%) and testing (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the RandomForest model
    rf_model = train_rf_model(X_train, y_train)
    log_to_mlflow(rf_model, "RandomForest", X_train, X_test, y_train, y_test)

    # Train the LogisticRegression model
    lr_model = train_lr_model(X_train, y_train)
    log_to_mlflow(lr_model, "LogisticRegression", X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
