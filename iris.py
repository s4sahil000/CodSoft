import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


# Load the dataset
file_path = 'IRIS.csv'
iris_df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(iris_df.head())

# Check for missing values
print(iris_df.isnull().sum())

# Encode the target variable (species)
label_encoder = LabelEncoder()
iris_df['species'] = label_encoder.fit_transform(iris_df['species'])

# Display the first few rows of the dataset after encoding
print(iris_df.head())

# Define features and target variable
X = iris_df.drop(columns=['species'])
y = iris_df['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Example new data (replace with actual measurements)
new_data = pd.DataFrame({
    'sepal_length': [5.1, 7.0, 6.3],
    'sepal_width': [3.5, 3.2, 3.3],
    'petal_length': [1.4, 4.7, 6.0],
    'petal_width': [0.2, 1.4, 2.5]
})

# Make predictions
new_predictions = model.predict(new_data)

# Decode the predicted labels back to species names
predicted_species = label_encoder.inverse_transform(new_predictions)

print(f"Predicted species: {predicted_species}")


sns.pairplot(iris_df, hue='species')
plt.suptitle('Pair Plot of Iris Dataset', y=1.02)
plt.show()

# Predict on the test set
y_pred = model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

# Plot confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Get feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()
