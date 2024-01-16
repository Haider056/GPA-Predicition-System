This comprehensive project aims to predict Grade Point Averages (GPA) for the fifth semester of a university program based on various input features. The project is structured into three main components: Data Preprocessing, Exploratory Data Analysis (EDA), and a Graphical User Interface (GUI) for GPA prediction using machine learning models.

1. Data Preprocessing
File
PreProcessing.py
DataSet.xlsx
Description
The PreProcessing.py script employs data preprocessing techniques to ensure the dataset is suitable for machine learning analysis. Techniques include:

Reading the dataset from 'DataSet.xlsx'
Cleaning column names by removing leading and trailing whitespaces
Handling missing values using scikit-learn's SimpleImputer with the mean strategy
Selecting relevant features for training
Splitting the dataset into training and testing sets using train_test_split
Standardizing the features using scikit-learn's StandardScaler


2. Exploratory Data Analysis (EDA)
File
EDA.py
DataSet.xlsx
Description
The EDA.py script focuses on exploratory data analysis to understand the dataset and identify patterns. Techniques include:

Generating descriptive statistics for numerical columns
Creating count plots for categorical variables using seaborn
Visualizing the distribution of numerical variables through histograms and scatter plots
Assessing model performance with evaluation metrics such as mean squared error and R^2 score
Plotting comparison and evaluation results through scatter plots and residuals plots


3. GPA Prediction GUI
File
main.py
DataSet.xlsx
Description
The main.py script builds a graphical user interface (GUI) using Tkinter for predicting GPA based on user input. Key features include:

Entry boxes for users to input relevant features
Buttons for predicting GPA and displaying results
Utilizing trained machine learning models (not included in this snippet) for predictions
Visualizing comparison and evaluation results through scatter plots and residuals plots
Instructions for Running the Project
Install Required Libraries:

Ensure you have the necessary Python libraries installed: pandas, numpy, matplotlib, seaborn, scikit-learn, and Tkinter.
Download Dataset:

Download the DataSet.xlsx file and place it in the same directory as the Python scripts.
Run Data Preprocessing:

Execute PreProcessing.py to preprocess the dataset.
Explore Data:

Run EDA.py to perform exploratory data analysis and gain insights.
Train Machine Learning Models:

Train machine learning models using a separate script (not included in this snippet).
Run GPA Prediction GUI:

Execute main.py to use the GUI for GPA prediction.
