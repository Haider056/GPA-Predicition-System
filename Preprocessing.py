import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")

# Read the Excel file
df = pd.read_excel("DataSet.xlsx", skiprows=1) 

# Clean up column names
df.columns = [col.strip() for col in df.columns]

# Display basic information about the dataset
print("Dataset Overview:")
print(df.info())

# Display the first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(df.head())

# Summary statistics for numerical columns
print("\nSummary statistics for numerical columns:")
print(df.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nNumber of duplicated rows:")
print(df.duplicated().sum())

# Explore categorical variables
categorical_columns = ['Program of Study', 'Gender', 'Nationality', 'Fathers Education', 'Mother\'s Education', 'Availing any scholarship']
for column in categorical_columns:
    print(f"\nCategories in '{column}' variable:")
    print(df[column].unique())

# Visualize some aspects of the data
# For example, you can create a bar plot for the 'Gender' column
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', data=df)
plt.title('Distribution of Gender')
plt.show()
