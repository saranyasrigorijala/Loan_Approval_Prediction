import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# Load the dataset
data = pd.read_csv('loan_approval_dataset.csv')

# Strip spaces from column names
data.columns = data.columns.str.strip()

# Check for missing values and handle them
# Fill missing numerical columns with mean values
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

# Fill missing categorical columns with the most frequent value
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Convert categorical variables to numeric using LabelEncoder
label_encoder = LabelEncoder()

# Loop through all categorical columns and apply LabelEncoder
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Drop the 'loan_id' column from both training and prediction data
X = data.drop(['loan_status', 'loan_id'], axis=1)  # Features (without loan_id)
y = data['loan_status']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy of the model:")
print(accuracy_score(y_test, y_pred))

# Detailed classification report
print("Classification report:")
print(classification_report(y_test, y_pred))

# Visualizing the feature importance
feature_importance = model.feature_importances_
features = X.columns

# Creating a DataFrame for visualization
feature_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
})

# Sorting the features by importance
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Plotting the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title('Feature Importance')
plt.show()

# Predict and display the result for a new loan application (example)
# Example data for prediction (assuming it's already encoded)
new_data = pd.DataFrame({
    'no_of_dependents': [1],
    'education': [0],  # For example, 0 = Graduate, 1 = Not Graduate (you can map this based on your data)
    'self_employed': [1],  # 0 = No, 1 = Yes
    'income_annum': [500000],
    'loan_amount': [200000],
    'loan_term': [15],
    'cibil_score': [750],
    'residential_assets_value': [300000],
    'commercial_assets_value': [100000],
    'luxury_assets_value': [50000],
    'bank_asset_value': [200000]
})

# Drop the 'loan_id' column from the new data (to match the training features)
new_data = new_data.drop('loan_id', axis=1, errors='ignore')

# Make a prediction for the new data
prediction = model.predict(new_data)

# Output result: Approved or Rejected
if prediction == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")
