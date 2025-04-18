import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

# Load dataset
df = pd.read_csv('churn_clean.csv')

# Select relevant columns based on D2 variable selection
selected_columns = [
    'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year', 'Outage_sec_perweek', 
    'Yearly_equip_failure', 'Age', 'Income', 'Email', 'Contacts',  # Continuous
    'Gender', 'Marital', 'Contract', 'PaymentMethod', 'InternetService', 
    'TechSupport', 'StreamingTV', 'PaperlessBilling', 'Techie', 'Area',  # Categorical
    'Item1', 'Item2', 'Item3', 'Item4', 'Item5', 'Item6', 'Item7', 'Item8',  # Ordinal (treated as numerical)
    'Churn'  # Target
]
df = df[selected_columns]

# Basic exploration
print("Dataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nFirst 5 Rows:")
print(df.head())

# Visualize target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.savefig('churn_distribution.png')
plt.show()

# Define numerical and categorical columns
numerical_cols = ['Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year', 'Outage_sec_perweek', 
                 'Yearly_equip_failure', 'Age', 'Income', 'Email', 'Contacts'] + \
                 ['Item' + str(i) for i in range(1, 9)]  # Treat survey items as numerical
categorical_cols = ['Gender', 'Marital', 'Contract', 'PaymentMethod', 'InternetService', 
                   'TechSupport', 'StreamingTV', 'PaperlessBilling', 'Techie', 'Area']

# Impute numerical missing values with median
num_imputer = SimpleImputer(strategy='median')
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

# Impute categorical missing values with mode
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# Encode categorical variables using one-hot encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Define features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Check class distribution
print("Class distribution before SMOTE:")
print(y.value_counts())

# Apply SMOTE if imbalance exists
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert back to DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['Churn'] = y_resampled

# Visualize new target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df_resampled)
plt.title('Churn Distribution After SMOTE')
plt.savefig('churn_distribution_smote.png')
plt.show()

print("Class distribution after SMOTE:")
print(df_resampled['Churn'].value_counts())

# Save cleaned dataset
df_resampled.to_csv('churn_cleaned.csv', index=False)
print("Cleaned dataset saved as 'churn_cleaned.csv'")

# Load the cleaned dataset
df_resampled = pd.read_csv('churn_cleaned.csv')

# Define features (X) and target (y)
X = df_resampled.drop('Churn', axis=1)
y = df_resampled['Churn']

# First split: 80% (train + validation), 20% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Second split: 75% of temp (60% of total) for train, 25% of temp (20% of total) for validation
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Save splits to files
pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
pd.DataFrame(X_val).to_csv('X_val.csv', index=False)
pd.DataFrame(y_val).to_csv('y_val.csv', index=False)
pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)

print("Data splits saved as CSV files")
print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

# Train initial Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict on training set
y_train_pred = rf_model.predict(X_train)

# Calculate metrics
accuracy = accuracy_score(y_train, y_train_pred)
precision = precision_score(y_train, y_train_pred)
recall = recall_score(y_train, y_train_pred)
f1 = f1_score(y_train, y_train_pred)
auc_roc = roc_auc_score(y_train, rf_model.predict_proba(X_train)[:, 1])

# Confusion matrix
cm = confusion_matrix(y_train, y_train_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix (Training Set)')
plt.savefig('confusion_matrix_train.png')
plt.show()

# Print metrics
print("Initial Model Metrics (Training Set):")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC-ROC: {auc_roc:.2f}")

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='f1',
                           n_jobs=-1)
grid_search.fit(X_val, y_val)

# Best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Save best hyperparameters to a file for screenshot
with open('best_hyperparameters.txt', 'w') as f:
    f.write(str(best_params))
    
# Combine training and validation sets for final training
X_train_val = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_val)])
y_train_val = pd.concat([pd.Series(y_train), pd.Series(y_val)])

# Train optimized model
optimized_rf = RandomForestClassifier(**best_params, random_state=42)
optimized_rf.fit(X_train_val, y_train_val)

# Predict on test set
y_test_pred = optimized_rf.predict(X_test)

# Calculate metrics
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
auc_roc_test = roc_auc_score(y_test, optimized_rf.predict_proba(X_test)[:, 1])

# Confusion matrix
cm_test = confusion_matrix(y_test, y_test_pred)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
disp_test.plot()
plt.title('Confusion Matrix (Test Set)')
plt.savefig('confusion_matrix_test.png')
plt.show()

# Print metrics
print("Optimized Model Metrics (Test Set):")
print(f"Accuracy: {accuracy_test:.2f}")
print(f"Precision: {precision_test:.2f}")
print(f"Recall: {recall_test:.2f}")
print(f"F1 Score: {f1_test:.2f}")
print(f"AUC-ROC: {auc_roc_test:.2f}")