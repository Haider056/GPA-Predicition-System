from tkinter import Tk, Label, Entry, Button, StringVar
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = 'DataSet.xlsx'
df = pd.read_excel(file_path, skiprows=1)


relevant_features = ['Matric percentage', 'Intermediate percentage',
                     'SGPA in BS First semester', 'SGPA in BS Second semester',
                     'SGPA in BS Third semester', 'SGPA in BS Fourth semester']


X = df[relevant_features]
y_sgpa = df['SGPA in BS Fifth semester']
y_cgpa = df['CGPA in BS Fifth semester']


imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

y_sgpa_imputed = imputer.fit_transform(y_sgpa.values.reshape(-1, 1))
y_cgpa_imputed = imputer.fit_transform(y_cgpa.values.reshape(-1, 1))

y_sgpa = pd.Series(y_sgpa_imputed.flatten(), name='SGPA in BS Fifth semester')
y_cgpa = pd.Series(y_cgpa_imputed.flatten(), name='CGPA in BS Fifth semester')


X_train, X_test, y_sgpa_train, y_sgpa_test, y_cgpa_train, y_cgpa_test = train_test_split(
    X_imputed, y_sgpa, y_cgpa, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the Linear Regression model for SGPA
linear_model_sgpa = LinearRegression()
linear_model_sgpa.fit(X_train_scaled, y_sgpa_train)

# Training the Linear Regression model for CGPA
linear_model_cgpa = LinearRegression()
linear_model_cgpa.fit(X_train_scaled, y_cgpa_train)

# Training a Random Forest Regressor model for SGPA
rf_model_sgpa = RandomForestRegressor()
rf_model_sgpa.fit(X_train_scaled, y_sgpa_train)

# Training a Support Vector Regressor model for CGPA
svr_model_cgpa = SVR()
svr_model_cgpa.fit(X_train_scaled, y_cgpa_train)


def evaluate_model(model, X_test_scaled, y_test, name):
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return f"{name} Model:\nMean Squared Error: {mse:.2f}\nR^2 Score: {r2:.2f}", predictions



def get_performance_category(gpa):
    if 3.51 <= gpa <= 4.00:
        return "Extraordinary Performance"
    elif 3.00 <= gpa <= 3.50:
        return "Very Good Performance"
    elif 2.51 <= gpa <= 2.99:
        return "Good Performance"
    elif 2.00 <= gpa <= 2.50:
        return "Satisfactory Performance"
    elif 1.00 <= gpa <= 1.99:
        return "Poor Performance"
    elif 0.00 <= gpa <= 0.99:
        return "Very Poor Performance"
    else:
        return "Invalid GPA"

def predict_gpa(model_sgpa, model_cgpa):
    input_data = [float(entry.get()) for entry in entry_boxes]
    input_data = np.array(input_data).reshape(1, -1)


    input_data_imputed = []
    for i in range(input_data.shape[1]):
        imputed_value = imputer.transform(input_data[:, i:i+1])
        input_data_imputed.append(imputed_value[0, 0])

    input_data_imputed = np.array(input_data_imputed).reshape(1, -1)

    input_data_scaled = scaler.transform(input_data_imputed)

    # Predict SGPA
    predicted_sgpa = model_sgpa.predict(input_data_scaled)[0]

    # Predict CGPA
    predicted_cgpa = model_cgpa.predict(input_data_scaled)[0]

    result_var.set(f"Predicted SGPA of Linear Regression: {predicted_sgpa:.2f}\nPredicted CGPA of Linear Regression: {predicted_cgpa:.2f}")

    # Get performance category
    sgpa_performance = get_performance_category(predicted_sgpa)
    cgpa_performance = get_performance_category(predicted_cgpa)

    result_var.set(result_var.get() + f"\nSGPA Performance: {sgpa_performance}\nCGPA Performance: {cgpa_performance}")

    # Compare with dataset values
    actual_sgpa = y_sgpa_test.iloc[0]
    actual_cgpa = y_cgpa_test.iloc[0]

    comparison_result = f"\nComparison with dataset values:\nActual SGPA: {actual_sgpa:.2f}, Predicted SGPA: {predicted_sgpa:.2f}\nActual CGPA: {actual_cgpa:.2f}, Predicted CGPA: {predicted_cgpa:.2f}"

    linear_sgpa_evaluation, linear_sgpa_predictions = evaluate_model(linear_model_sgpa, X_test_scaled, y_sgpa_test, "Linear Regression SGPA")
    linear_cgpa_evaluation, linear_cgpa_predictions = evaluate_model(linear_model_cgpa, X_test_scaled, y_cgpa_test, "Linear Regression CGPA")
    rf_sgpa_evaluation, rf_sgpa_predictions = evaluate_model(rf_model_sgpa, X_test_scaled, y_sgpa_test, "Random Forest SGPA")
    svr_cgpa_evaluation, svr_cgpa_predictions = evaluate_model(svr_model_cgpa, X_test_scaled, y_cgpa_test, "SVR CGPA")

    print(comparison_result)
    print(linear_sgpa_evaluation)
    print(linear_cgpa_evaluation)
    print(rf_sgpa_evaluation)
    print(svr_cgpa_evaluation)

    # Plotting
    plt.figure(figsize=(12, 8))

    # Scatter plot for actual vs predicted SGPA
    plt.subplot(2, 2, 1)
    plt.scatter(y_sgpa_test, linear_sgpa_predictions, color='blue', label='Linear Regression')
    plt.scatter(y_sgpa_test, rf_sgpa_predictions, color='green', label='Random Forest')
    plt.title('SGPA Prediction Comparison')
    plt.xlabel('Actual SGPA')
    plt.ylabel('Predicted SGPA')
    plt.legend()

    # Scatter plot for actual vs predicted CGPA
    plt.subplot(2, 2, 2)
    plt.scatter(y_cgpa_test, linear_cgpa_predictions, color='blue', label='Linear Regression')
    plt.scatter(y_cgpa_test, svr_cgpa_predictions, color='red', label='SVR')
    plt.title('CGPA Prediction Comparison')
    plt.xlabel('Actual CGPA')
    plt.ylabel('Predicted CGPA')
    plt.legend()

    # Residuals plot for SGPA
    plt.subplot(2, 2, 3)
    plt.scatter(linear_sgpa_predictions, y_sgpa_test - linear_sgpa_predictions, color='blue', label='Linear Regression')
    plt.scatter(rf_sgpa_predictions, y_sgpa_test - rf_sgpa_predictions, color='green', label='Random Forest')
    plt.axhline(0, color='black', linestyle='--', linewidth=2)
    plt.title('Residuals Plot for SGPA Prediction')
    plt.xlabel('Predicted SGPA')
    plt.ylabel('Residuals')
    plt.legend()

    # Residuals plot for CGPA
    plt.subplot(2, 2, 4)
    plt.scatter(linear_cgpa_predictions, y_cgpa_test - linear_cgpa_predictions, color='blue', label='Linear Regression')
    plt.scatter(svr_cgpa_predictions, y_cgpa_test - svr_cgpa_predictions, color='red', label='SVR')
    plt.axhline(0, color='black', linestyle='--', linewidth=2)
    plt.title('Residuals Plot for CGPA Prediction')
    plt.xlabel('Predicted CGPA')
    plt.ylabel('Residuals')
    plt.legend()

    plt.tight_layout()
    plt.show()

# GUI Setup
root = Tk()
root.title("GPA Prediction")

# Entry boxes for relevant input variables
label_names = relevant_features
entry_boxes = [Entry(root) for _ in label_names]

for i, label in enumerate(label_names):
    Label(root, text=label).grid(row=i, column=0, padx=10, pady=5)
    entry_boxes[i].grid(row=i, column=1, padx=10, pady=5)

# Predict button
predict_button = Button(root, text="Predict GPA and SGPA", command=lambda: predict_gpa(linear_model_sgpa, linear_model_cgpa))
predict_button.grid(row=len(label_names) + 1, column=0, columnspan=2, pady=10)

# Result display
result_var = StringVar()
result_label = Label(root, textvariable=result_var)
result_label.grid(row=len(label_names) + 2, column=0, columnspan=2, pady=10)

root.mainloop()