# Installations
# 1. pip install mlflow
# 2. pip install psutil
# 3. pip install scikit-learn pandas

# Steps to run
# 1. In terminal, run command -> mlflow ui --host 0.0.0.0 --port 5000
# 2. Run this script using Python: python main.py
# 3. Open http://localhost:5000 in browser and see the experimental results

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import psutil
import time

# Set the MLflow tracking URI to 'http'
mlflow.set_tracking_uri("http://localhost:5000")

# Function for data preprocessing
def preprocess_data(data):
    # Convert categorical variables to one-hot encoding if necessary
    # Example:
    # data = pd.get_dummies(data, columns=['Gender', 'CovidSeverity'])
    
    # Split data into X (features) and y (target)
    X = data.drop('cardio', axis=1)
    y = data['cardio']
    
    return X, y

# Define the classifiers
def get_classifiers():
    classifiers = {
        "RandomForest": RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', probability=True, random_state=42)
    }
    return classifiers

# Function to evaluate models
def evaluate_models(models, X_test, y_test):
    evaluation_results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        evaluation_results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }
        
        print(f"\n{name} Model Evaluation:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
    return evaluation_results

# Function to log metrics and models to MLflow
def log_to_mlflow(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        # Log model parameters
        if model_name == "RandomForest":
            mlflow.log_param("max_depth", model.max_depth)
            mlflow.log_param("n_estimators", model.n_estimators)
        elif model_name == "SVM":
            mlflow.log_param("kernel", model.kernel)
            mlflow.log_param("C", model.C)
        
        # Log system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        mlflow.log_metric("system_cpu_usage", cpu_usage)
        mlflow.log_metric("system_memory_usage", memory_usage)
        
        # Log execution time for training
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Log evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log confusion matrix as a dictionary
        confusion = confusion_matrix(y_test, y_pred)
        confusion_dict = {
            "true_positive": confusion[1][1],
            "false_positive": confusion[0][1],
            "true_negative": confusion[0][0],
            "false_negative": confusion[1][0]
        }
        mlflow.log_metrics(confusion_dict)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Logged {model_name} model to MLflow.")

# Main function
def main():
    # Load the dataset
    data = pd.read_csv("../dataset/cardiovascular_dataset.csv")  
    
    # Preprocess the data
    X, y = preprocess_data(data)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Get classifiers
    classifiers = get_classifiers()
    
    # Train classifiers
    trained_models = {}
    for name, clf in classifiers.items():
        print(f"Training {name} model...")
        clf.fit(X_train, y_train)
        trained_models[name] = clf
        print(f"{name} model trained.")
    
    # Evaluate the trained models
    evaluation_results = evaluate_models(trained_models, X_test, y_test)
    
    # Log all trained models to MLflow
    for name, model in trained_models.items():
        log_to_mlflow(model, name, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
