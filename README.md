# Machine Learning: Model Training

This project predicts the likelihood of cardiovascular disease using machine learning models such as **RandomForestClassifier** and **LogisticRegression**. The goal is to achieve high prediction accuracy (>0.70) using tuned models without complex preprocessing steps.

The project uses the **MLflow** framework to log hyperparameters, metrics, and model artifacts. Additionally, key performance metrics such as **precision**, **recall**, **F1 score**, and **confusion matrix** are tracked and logged for detailed evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [License](#license)

## Project Overview
This project builds and tunes two machine learning models:
- **RandomForestClassifier**
- **LogisticRegression**

Both models are trained on cardiovascular health data and are evaluated for accuracy, precision, recall, F1 score, and other important metrics. The aim is to achieve an accuracy above 70%.

## Dataset
The dataset used in this project contains various health indicators (age, cholesterol levels, blood pressure, etc.) to predict the presence of cardiovascular disease. Ensure you have the dataset available at the specified path in the project.

- **Input file**: `cardiovascular_dataset.csv`
  
- **Target variable**: `cardio` (indicating the presence of cardiovascular disease)

## Installation
To run this project locally, follow the steps below:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/logeshwarann-dev/machineLearningOps.git
   cd machineLearningOps
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start MLflow** (for logging and tracking):
   ```
   mlflow ui --host 0.0.0.0 --port 5000
   ```
   The MLflow UI will be accessible at `http://localhost:5000`.


## Usage
1. **Ensure the dataset** `cardiovascular_dataset.csv` is in the project directory or update the path in the `model5.py` script.

2. **Run the project**:
   ```
   python3 model5.py
   ```

3. After training, **model metrics** such as accuracy, precision, recall, F1 score, and confusion matrix will be logged in the console and MLflow.

## Evaluation Metrics
The following evaluation metrics are computed for both models:
- **Accuracy**: Overall correctness of the model.
- **Precision**: Correctness of positive predictions.
- **Recall**: Ability to find all positive instances.
- **F1 Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Table showing the distribution of predicted vs actual classes.


All metrics are also logged in **MLflow** for easier comparison across different runs.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Notes:
1. Make sure to replace the URL with your actual GitHub repository link.
2. Customize the project name, if necessary.
3. You might also want to add any specific instructions on tuning models further based on your dataset.