import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
print("--- Advanced Insurance Fraud Detection System ---")

# --- 1. Data Loading and Basic Cleanup ---
print("\n1. Loading and Cleaning Data...")
try:
    df = pd.read_csv('insurance_claims.csv')
    # Replace '?' with NaN and drop columns not suitable for this model
    df.replace('?', np.nan, inplace=True)
    df.drop(columns=['policy_number', 'policy_bind_date', 'incident_date', 'incident_location', '_c39'], inplace=True)
    print("Data loaded. Shape:", df.shape)
except FileNotFoundError:
    print("\nError: 'insurance_claims.csv' not found.")
    print("Please download the dataset using the link provided and place it in the same directory as this script.")
    exit()

# --- 2. Advanced NLP with Sentence-BERT Embeddings ---
print("\n2. Generating Contextual NLP Embeddings (using Sentence-BERT)...")
# This might take a moment to download the model the first time
print("Loading pre-trained NLP model...")
nlp_model = SentenceTransformer('all-MiniLM-L6-v2')

# Ensure 'incident_details' is a string
df['incident_details'] = df['incident_details'].astype(str)

# Generate embeddings
# An embedding is a vector of numbers that represents the semantic meaning of the text
print("Encoding incident details into vectors...")
with torch.no_grad(): # Disable gradient calculations for efficiency
    text_embeddings = nlp_model.encode(df['incident_details'].tolist(), show_progress_bar=True)

# Create a DataFrame with the embeddings
embedding_features = pd.DataFrame(text_embeddings, columns=[f'embedding_{i}' for i in range(text_embeddings.shape[1])])
print("NLP Embeddings created. Shape:", embedding_features.shape)

# --- 3. Feature Engineering and Preprocessing Pipeline ---
print("\n3. Setting up Preprocessing Pipeline for Tabular Data...")

# Separate target variable
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported'].apply(lambda x: 1 if x == 'Y' else 0)

# Drop the original text column as it's now represented by embeddings
X = X.drop('incident_details', axis=1)

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns

# Create preprocessing pipelines for both feature types
# This is modern scikit-learn practice: clean, scalable, and prevents data leakage
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any)
)

# --- 4. Combine Features and Finalize Dataset ---
print("\n4. Combining Tabular and NLP Features...")

# Apply the preprocessing pipeline to the tabular data
X_tabular_processed = preprocessor.fit_transform(X)

# Get feature names after one-hot encoding
try:
    ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
except AttributeError: # For older scikit-learn versions
    ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names(categorical_features)

processed_tabular_features = np.concatenate([numerical_features, ohe_feature_names])

# Convert processed tabular data back to a DataFrame
X_tabular_df = pd.DataFrame(X_tabular_processed, columns=processed_tabular_features, index=X.index)

# Concatenate processed tabular data with NLP embedding features
X_final = pd.concat([X_tabular_df, embedding_features], axis=1)

# LightGBM can't handle special characters in feature names
X_final.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_final.columns]

print("Final combined feature set created. Shape:", X_final.shape)


# --- 5. Model Training with LightGBM ---
print("\n5. Training High-Performance LightGBM Model...")

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=42, stratify=y)

# Handle class imbalance by calculating scale_pos_weight
# This tells the model to pay more attention to the minority class (fraud)
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# Initialize and train the LightGBM model
lgbm = lgb.LGBMClassifier(objective='binary',
                          scale_pos_weight=scale_pos_weight,
                          random_state=42,
                          n_estimators=200,
                          learning_rate=0.05,
                          num_leaves=31)

lgbm.fit(X_train, y_train)
print("Model training complete.")

# --- 6. Performance Evaluation ---
print("\n6. Evaluating Model Performance...")
y_pred = lgbm.predict(X_test)
y_proba = lgbm.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Fraudulent', 'Fraudulent']))

print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# --- 7. Explainable AI (XAI) with SHAP ---
print("\n7. Generating Model Explanations with SHAP...")

# Create a SHAP explainer
explainer = shap.TreeExplainer(lgbm)
# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

print("SHAP values calculated. Generating plots...")

# Plot 1: SHAP Summary Plot (Beeswarm)
plt.figure()
shap.summary_plot(shap_values[1], X_test, plot_type="dot", show=False, max_display=15)
plt.title("SHAP Summary Plot: Feature Impact on Fraud Prediction")
plt.tight_layout()
plt.savefig("shap_summary_plot.png")
plt.show()
print("Saved 'shap_summary_plot.png'")


# Plot 2: Global Feature Importance (Bar Plot)
plt.figure()
shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False, max_display=15)
plt.title("Global Feature Importance (based on SHAP values)")
plt.tight_layout()
plt.savefig("shap_feature_importance.png")
plt.show()
print("Saved 'shap_feature_importance.png'")

print("\n--- Project Execution Finished ---")
