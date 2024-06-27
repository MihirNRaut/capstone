"""import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset from a CSV file
data = pd.read_csv('Titanic-Dataset.csv')

# Displaying few row of dataset
print(data.head())

# Drop columns that won't be used for prediction
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Conversion of variables into numeric
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Define target and axis
X = data.drop('Survived', axis=1)
y = data['Survived']

# Splitting into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5 , random_state=45)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Use Logistic Regression 
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Plotting the Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load and preprocess data
df = pd.read_csv('Titanic-Dataset.csv')
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Split data
features = df.drop('Survived', axis=1)
target = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=45)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and predict
clf = LogisticRegression(random_state=42)
clf.fit(X_train_scaled, y_train)
predictions = clf.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
conf_mat = confusion_matrix(y_test, predictions)
class_rep = classification_report(y_test, predictions)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_mat)
print('Classification Report:')
print(class_rep)