import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'DataSet.xlsx'
df = pd.read_excel(file_path, skiprows=1)  # Skip the first row as it contains headers

# Display basic information about the dataset
print("Dataset Information:")
print(df.info())

# Display basic statistics of numerical columns
print("\nDescriptive Statistics:")
print(df.describe())

# Display the first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(df.head())

# EDA: Count plot for categorical columns with color customization
categorical_columns = ['Gender', 'Nationality', 'Intermediate Stream']
for column in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df, palette='viridis')  # Adjust the color palette as needed
    plt.title(f'Count Plot for {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

# EDA: Histogram for 'Parental Income' with color customization
plt.figure(figsize=(12, 6))
sns.histplot(df['Parental Income'], bins=20, kde=True, color='skyblue')  # Adjust the color as needed
plt.title('Histogram for Parental Income')
plt.xlabel('Parental Income')
plt.ylabel('Frequency')
plt.show()

# EDA: Scatter plot for 'Matric percentage' vs 'Intermediate percentage'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Matric percentage', y='Intermediate percentage', data=df, color='orange')
plt.title('Scatter Plot for Matric percentage vs Intermediate percentage')
plt.xlabel('Matric percentage')
plt.ylabel('Intermediate percentage')
plt.show()

# EDA: Box plot for 'SGPA in BS First semester' with color customization
plt.figure(figsize=(10, 6))
sns.boxplot(x='SGPA in BS First semester', data=df, color='green')
plt.title('Box Plot for SGPA in BS First semester')
plt.xlabel('SGPA in BS First semester')
plt.show()

# Additional EDA for new fields
additional_columns = [
    'I understand the lecture more clearly if delivered in:',
    'I understand the lecture more clearly if delivered on:',
    'I understand the lecture more clearly if my sitting place is in:',
    'I understand the lecture more clearly if the duration is:',
    'I understand the lecture more clearly if delivered by:',
    'I understand the lecture more clearly if:',
    'Lecture attendance has a positive effect on my academic performance.',
    'TVFs are very difficult to access after class timings and hence I am unable to clarify my concepts.',
    'TVFs do not provide assessment results timely so I remain unaware of my standing in class.',
    'TVFs are not aware of the university\'s curriculum and other requirements.',
    'TVFs grade students according to their own beliefs and criteria and hence my GPA suffers.',
    'TVFs are not very well prepared for the lectures due to their busy schedule.',
    'What is your residence place?',
    'Due to class/lab cancellation or road blockage, lots of time gets wasted due to fixed official transport timings.',
    'Hostelites can be more focused towards their studies whereas a day scholar may have a lot of distractions due to family issues.',
    'Hostelites have more opportunities to engage with their peers and participate in social events, that improve their performance.',
    'Hostelites have better access to campus resources like libraries, instructors, labs and also get benefit of group study.',
    'Hostelites have a big social circle and connections with seniors who can guide them in studies and in other academic and administrative issues.',
    'Un-announced quizzes are quite stressful and cause unnecessary anxiety and pressure on students.',
    'Un-announced quizzes are a form of torture and have a negative impact on the learning environment.',
    'I am not usually prepared for surprise quizzes and hence these are not true measures of my knowledge and understandings.',
    'Un-announced quizzes are a waste of time and energy and are not effective and productive in the long term.',
    'I cannot miss a single class despite my health or any other emergency situations due to unannounced quizzes as it badly affects my grade.',
    'I usually do not perform very well in surprise quizzes, hence lose my interest in the subject.',
]
for column in additional_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df, palette='muted')  
    plt.title(f'Count Plot for {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()


plt.show()