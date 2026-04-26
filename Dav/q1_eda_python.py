# DAV_Practical/q1_eda_python.py
# Question 1: Getting introduced data analytics libraries (EDA) in Python. 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Data Import
# FIX: Added 'r' before the string to handle Windows backslashes properly
# Added skiprows=1 to ignore the comment line at the top of the CSV
df = pd.read_csv(r'C:\Users\LENOVO\Downloads\Datalab\Dav\DAV_dataset.csv', skiprows=1)
print("--- Original Data ---")
print(df.head())

# 2. Data Cleaning (Handling Missing Values)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())

# 3. Summary Statistics
print("\n--- Summary Statistics ---")
print(df.describe())

# 4. Visualization
plt.hist(df['Salary'], bins=5, color='skyblue', edgecolor='black')
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()

# 5. Splitting Data
X = df[['Age']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Data Splitting ---")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")
