# c:\Users\HUAWEİ\Desktop\Sıla_LLM_Proje_etik\compas_ethical_analysis.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
print("Loading COMPAS dataset...")
df = pd.read_csv(r"c:\Users\HUAWEİ\Desktop\Sıla_LLM_Proje_etik\compas-analysis-master\compas-scores-raw.csv")

# Create name column for identification
df['name'] = df['FirstName'] + ' ' + df['LastName']

# Extract age from DOB
df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], format='%m/%d/%y', errors='coerce')
reference_date = pd.to_datetime('2013-01-01')  # Based on the screening dates in data
df['age'] = (reference_date - df['DateOfBirth']).dt.days // 365

# Get recidivism data
recid_data = df[df['DisplayText'] == 'Risk of Recidivism'].copy()

# Define recidivism as high risk scores (8-10)
recid_data['is_recid'] = (recid_data['DecileScore'] >= 8).astype(int)

# Use RawScore as proxy for priors_count (since we don't have direct priors in sample)
recid_data['priors_count'] = abs(recid_data['RawScore']).round().astype(int)

# Select required columns
final_df = recid_data[['Person_ID', 'name', 'age', 'Sex_Code_Text', 'Ethnic_Code_Text', 
                      'priors_count', 'is_recid']].copy()
final_df.rename(columns={'Sex_Code_Text': 'sex', 'Ethnic_Code_Text': 'race'}, inplace=True)

# Keep one record per person
final_df = final_df.drop_duplicates('Person_ID').dropna()

print(f"Processed data: {len(final_df)} unique individuals")
print(f"Recidivism rate in data: {final_df['is_recid'].mean():.2%}")

# Prepare for modeling
X = final_df[['age', 'sex', 'race', 'priors_count']]
y = final_df['is_recid']

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=['sex', 'race'], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)

# Train logistic regression model with class balancing
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(conf_matrix)

# Create prediction table
prediction_table = pd.DataFrame({
    'name': final_df.loc[y_test.index, 'name'].values,
    'actual': y_test.values,
    'predicted': y_pred,
    'probability': y_pred_prob,
    'correct': y_test.values == y_pred
})

print("\n===== PREDICTION EXAMPLES =====")
print(prediction_table.head(10))

# Feature importance from coefficients
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': np.abs(model.coef_[0]),
    'direction': model.coef_[0]
}).sort_values('coefficient', ascending=False)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='coefficient', y='feature', data=feature_importance, 
            palette=['red' if x < 0 else 'blue' for x in feature_importance['direction']])
plt.title('Feature Importance for Recidivism Prediction')
plt.xlabel('Absolute Coefficient Value')
plt.tight_layout()
plt.savefig(r"c:\Users\HUAWEİ\Desktop\Sıla_LLM_Proje_etik\feature_importance.png")
plt.close()

print("\n===== FEATURE IMPORTANCE RANKING =====")
for idx, row in feature_importance.iterrows():
    direction = "increases" if row['direction'] > 0 else "decreases"
    print(f"{idx+1}. {row['feature']}: {row['coefficient']:.4f} ({direction} risk)")

# Racial bias analysis
race_features = [col for col in X_test.columns if col.startswith('race_')]
gender_features = [col for col in X_test.columns if col.startswith('sex_')]

# Calculate fairness metrics
print("\n===== FAIRNESS ANALYSIS =====")
print("Model performance by race:")
for race_col in race_features:
    race_name = race_col.replace('race_', '')
    race_indices = X_test[race_col] == 1
    race_y_test = y_test[race_indices]
    race_y_pred = y_pred[race_indices]
    race_accuracy = accuracy_score(race_y_test, race_y_pred)
    race_positive_rate = race_y_pred.mean()
    print(f"  {race_name}: Accuracy={race_accuracy:.4f}, Positive prediction rate={race_positive_rate:.4f}")

print("\nModel performance by gender:")
for gender_col in gender_features:
    gender_name = gender_col.replace('sex_', '')
    gender_indices = X_test[gender_col] == 1
    gender_y_test = y_test[gender_indices]
    gender_y_pred = y_pred[gender_indices]
    gender_accuracy = accuracy_score(gender_y_test, gender_y_pred)
    gender_positive_rate = gender_y_pred.mean()
    print(f"  {gender_name}: Accuracy={gender_accuracy:.4f}, Positive prediction rate={gender_positive_rate:.4f}")

# Ethical analysis
sensitive_features = race_features + gender_features
most_important = feature_importance.iloc[0]['feature']

print("\n===== ETHICAL ANALYSIS =====")
if most_important in sensitive_features:
    print("⚠️ ETHICAL CONCERN DETECTED")
    print(f"The most influential feature is {most_important}, which is a sensitive attribute.")
    print("This suggests potential algorithmic bias in decision-making.")
else:
    print(f"The most influential feature is {most_important}, which is not a sensitive attribute.")

print("\nRanking of sensitive attributes:")
for feature in sensitive_features:
    rank = feature_importance[feature_importance['feature'] == feature].index[0] + 1
    importance = feature_importance[feature_importance['feature'] == feature]['coefficient'].values[0]
    direction = "increases" if feature_importance[feature_importance['feature'] == feature]['direction'].values[0] > 0 else "decreases"
    print(f"  - {feature}: Rank #{rank}, Importance: {importance:.4f} ({direction} risk)")

# Save results
prediction_table.to_csv(r"c:\Users\HUAWEİ\Desktop\Sıla_LLM_Proje_etik\prediction_results.csv", index=False)
feature_importance.to_csv(r"c:\Users\HUAWEİ\Desktop\Sıla_LLM_Proje_etik\feature_importance.csv", index=False)

print("\nAnalysis complete. Results and visualizations have been saved.")