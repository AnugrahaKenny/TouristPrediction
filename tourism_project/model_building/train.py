# for data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# for model serialization
import joblib
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

api = HfApi()

Xtrain_path = "hf://datasets/AnugrahaKenny/TouristPrediction/Xtrain.csv"
Xtest_path = "hf://datasets/AnugrahaKenny/TouristPrediction/Xtest.csv"
ytrain_path = "hf://datasets/AnugrahaKenny/TouristPrediction/ytrain.csv"
ytest_path = "hf://datasets/AnugrahaKenny/TouristPrediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Drop 'CustomerID' as it's an identifier and not a feature
Xtrain = Xtrain.drop(columns=['CustomerID'])
Xtest = Xtest.drop(columns=['CustomerID'])

# Define features based on their original types for proper preprocessing in train.py
numerical_features = [
    'Age', 'CityTier', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
    'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting',
    'MonthlyIncome', 'PitchSatisfactionScore', 'NumberOfFollowups',
    'DurationOfPitch'
]
categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'MaritalStatus',
    'Designation', 'ProductPitched'
]

# Preprocessing pipeline: apply StandardScaler to numerical and OneHotEncoder to categorical
preprocessor = make_column_transformer(
    (StandardScaler(), numerical_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define XGBoost Classifier (since ProdTaken is binary classification)
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Define hyperparameter grid for classification
param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 200],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__subsample': [0.6, 0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.6, 0.8, 1.0]
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Grid search with cross-validation
grid_search = GridSearchCV(
    model_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
)
grid_search.fit(Xtrain, ytrain.values.ravel()) # .values.ravel() for ytrain

# Best model
best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

# Predictions
y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)
y_proba_train = best_model.predict_proba(Xtrain)[:, 1]
y_proba_test = best_model.predict_proba(Xtest)[:, 1]

# Evaluation for classification
print("\nTraining Performance:")
print("Accuracy:", accuracy_score(ytrain, y_pred_train))
print("Precision:", precision_score(ytrain, y_pred_train))
print("Recall:", recall_score(ytrain, y_pred_train))
print("F1 Score:", f1_score(ytrain, y_pred_train))
print("ROC AUC:", roc_auc_score(ytrain, y_proba_train))

print("\nTest Performance:")
print("Accuracy:", accuracy_score(ytest, y_pred_test))
print("Precision:", precision_score(ytest, y_pred_test))
print("Recall:", recall_score(ytest, y_pred_test))
print("F1 Score:", f1_score(ytest, y_pred_test))
print("ROC AUC:", roc_auc_score(ytest, y_proba_test))

# Save best model
joblib.dump(best_model, "best_tourism_prediction_model_v1.joblib")


# Upload to Hugging Face
repo_id = "AnugrahaKenny/TouristPrediction" 
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

api.upload_file(
    path_or_fileobj="best_tourism_prediction_model_v1.joblib",
    path_in_repo="best_tourism_prediction_model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
