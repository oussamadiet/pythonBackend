import os
import pandas as pd
import joblib
import time
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Paths to data in Colab
train_data_path = './predata/train_data.csv'
validation_data_path = './predata/validation_data.csv'
model_path = './model/diet_recommendation_modelyo.pkl'
label_encoder_path = './model/label_encoderyo.pkl'
preprocessor_path = './model/preprocessor.pkl'  # Path to save the preprocessor

# Load training and validation data
train_data = pd.read_csv(train_data_path)
validation_data = pd.read_csv(validation_data_path)

# Define features and target
features = ['Age', 'Weight (kg)', 'Height (cm)', 'Gender_encoded', 'Activity_Level_encoded', 'Dietary_Goal_encoded', 'caloric_needs', 'protein_needs']
target = 'Meal Plan Name'

X_train = train_data[features]
y_train = train_data[target]
X_val = validation_data[features]
y_val = validation_data[target]

# Encode target labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
joblib.dump(label_encoder, label_encoder_path)

# Preprocessing pipeline for numerical features
numeric_features = ['Age', 'Weight (kg)', 'Height (cm)', 'caloric_needs', 'protein_needs']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Column transformer to apply transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)],
    remainder='passthrough'
)

# Apply preprocessing to features
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# Save the fitted preprocessor
joblib.dump(preprocessor, preprocessor_path)

# Extended hyperparameter grid for XGBClassifier
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 7, 9, 11],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# Initialize XGBClassifier with GPU configuration
xgb = XGBClassifier(tree_method='hist', device='cuda', random_state=42)

# Timing
start_time = time.time()

# RandomizedSearchCV for hyperparameter tuning with extended grid
grid_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train_processed, y_train_encoded)

# Calculate training time
end_time = time.time()
print(f"Training Duration: {end_time - start_time:.2f} seconds")

# Best parameters and cross-validation score
print("Best Parameters:", grid_search.best_params_)
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.2f}")

# Train final model with best parameters
best_params = grid_search.best_params_
final_model = XGBClassifier(**best_params, tree_method='hist', device='cuda', random_state=42)
final_model.fit(X_train_processed, y_train_encoded)

# Save the trained model
joblib.dump(final_model, model_path)
print("Model and label encoder saved successfully.")

# Evaluate model performance on training data
y_train_pred = final_model.predict(X_train_processed)
print("Training Accuracy:", accuracy_score(y_train_encoded, y_train_pred))
print("\nTraining Classification Report:\n", classification_report(y_train_encoded, y_train_pred))

# Evaluate model performance on validation data
y_val_pred = final_model.predict(X_val_processed)
print("Validation Accuracy:", accuracy_score(y_val_encoded, y_val_pred))
print("\nValidation Classification Report:\n", classification_report(y_val_encoded, y_val_pred))
