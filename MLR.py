import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Load dataset
data_path = '/Users/qibuguojiuhuolala/Desktop/Project/Dataset/Unnormalized_data.xlsx'
df = pd.read_excel(data_path)

# Feature selection
# Column A (PI), Column C (PT), Column J (AA)
features = ['PI', 'PT', 'AA']
X = df[features].values

# Column W (SBP), Column X (DBP)
y_sys = df['systolic pressure'].values
y_dia = df['diastolic pressure'].values

# Multiple Linear Regression with Gradient Descent
def gradient_descent_lr(X_train, y_train, X_test, learning_rate=0.01, epochs=5000):
    m, n = X_train.shape
    
    # Add bias term to features
    X_train_b = np.c_[np.ones((m, 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    
    # Initialize weight parameters theta
    theta = np.zeros(n + 1)
    
    # Gradient descent iteration
    for epoch in range(epochs):
        predictions = X_train_b.dot(theta)
        errors = predictions - y_train
        # Calculate gradients
        gradients = (1 / m) * X_train_b.T.dot(errors)
        # Update parameters
        theta -= learning_rate * gradients
    
    # Predict on test set
    y_pred = X_test_b.dot(theta)
    return y_pred

# Bland-Altman Plotting
def plot_bland_altman(y_true, y_pred, title):
    mean_vals = (y_true + y_pred) / 2
    diff_vals = y_true - y_pred
    
    # Calculate mean difference and standard deviation
    md = np.mean(diff_vals)
    sd = np.std(diff_vals, ddof=1)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(mean_vals, diff_vals, alpha=0.6, color='teal')
    
    # Plot reference lines
    plt.axhline(md, color='red', linestyle='-', label=f'Mean Difference: {md:.2f} mmHg')
    plt.axhline(md + 1.96 * sd, color='blue', linestyle='--', label=f'+1.96 SD: {md + 1.96*sd:.2f} mmHg')
    plt.axhline(md - 1.96 * sd, color='blue', linestyle='--', label=f'-1.96 SD: {md - 1.96*sd:.2f} mmHg')
    
    plt.xlabel('Mean of True and Predicted Pressures (mmHg)')
    plt.ylabel('Difference (True - Predicted) (mmHg)')
    plt.title(f'Bland-Altman Plot: {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Evaluation Metrics Calculation Function
def calculate_metrics(y_true, y_pred, target_name):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # Calculate R-squared (coefficient of determination)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"--- {target_name} Prediction Results ---")
    print(f"MAE  : {mae:.4f} mmHg")
    print(f"RMSE : {rmse:.4f} mmHg")
    print(f"R²  : {r2:.4f}\n")

# Main Pipeline for 10-fold Cross Validation
kf = KFold(n_splits=10, shuffle=True)

# Store all true values and predictions
y_sys_true_all, y_sys_pred_all = [], []
y_dia_true_all, y_dia_pred_all = [], []

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    # Split training and test sets
    X_train, X_test = X[train_index], X[test_index]
    
    y_sys_train, y_sys_test = y_sys[train_index], y_sys[test_index]
    y_dia_train, y_dia_test = y_dia[train_index], y_dia[test_index]
    
    # Z-score normalization
    # Calculate mean and standard deviation from training set
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Normalize training and test sets separately
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    # Train model and predict systolic blood pressure
    # Learning rate set to 0.01
    y_sys_pred = gradient_descent_lr(X_train_norm, y_sys_train, X_test_norm, learning_rate=0.01, epochs=5000)
    y_sys_true_all.extend(y_sys_test)
    y_sys_pred_all.extend(y_sys_pred)
    
    # Train model and predict diastolic blood pressure
    y_dia_pred = gradient_descent_lr(X_train_norm, y_dia_train, X_test_norm, learning_rate=0.01, epochs=5000)
    y_dia_true_all.extend(y_dia_test)
    y_dia_pred_all.extend(y_dia_pred)

# Convert to numpy arrays for calculation
y_sys_true_all = np.array(y_sys_true_all)
y_sys_pred_all = np.array(y_sys_pred_all)
y_dia_true_all = np.array(y_dia_true_all)
y_dia_pred_all = np.array(y_dia_pred_all)

# Calculate evaluation metrics
calculate_metrics(y_sys_true_all, y_sys_pred_all, "Systolic Pressure")
calculate_metrics(y_dia_true_all, y_dia_pred_all, "Diastolic Pressure")

# Bland-Altman Plots
plot_bland_altman(y_sys_true_all, y_sys_pred_all, "Systolic Pressure Prediction")
plot_bland_altman(y_dia_true_all, y_dia_pred_all, "Diastolic Pressure Prediction")