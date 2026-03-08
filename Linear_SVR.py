import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Load data
data_path = '/Users/qibuguojiuhuolala/Desktop/Project/Dataset/Unnormalized_data.xlsx'
df = pd.read_excel(data_path)

# Feature selection
features = ['PI', 'PT', 'AA']
X = df[features].values
y_sys = df['systolic pressure'].values
y_dia = df['diastolic pressure'].values

# Implement Linear SVR with Sub-gradient Descent
def svr_gradient_descent(X_train, y_train, X_test, learning_rate=0.01, epochs=5000, C=100, epsilon=2.0):
    m, n = X_train.shape
    
    # Initialize weights w and bias b
    w = np.zeros(n)
    b = 0.0
    
    # Gradient descent iterations
    for epoch in range(epochs):
        # Calculate current predictions and errors
        predictions = np.dot(X_train, w) + b
        errors = y_train - predictions
        
        # Initialize gradients
        dw = w.copy()
        db = 0.0
        
        # Calculate subgradients for epsilon-insensitive loss
        idx_pos = errors > epsilon    # Predictions are much lower than true values
        idx_neg = errors < -epsilon   # Predictions are much higher than true values
        
        # Accumulate gradients (divide by m to keep gradients at a reasonable magnitude for convergence)
        dw -= (C / m) * np.sum(X_train[idx_pos], axis=0)
        dw += (C / m) * np.sum(X_train[idx_neg], axis=0)
        
        db -= (C / m) * np.sum(idx_pos)
        db += (C / m) * np.sum(idx_neg)
        
        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db
        
    # Make predictions on test set
    y_pred = np.dot(X_test, w) + b
    return y_pred

# Bland-Altman plotting 
def plot_bland_altman(y_true, y_pred, title):
    mean_vals = (y_true + y_pred) / 2
    diff_vals = y_true - y_pred
    
    md = np.mean(diff_vals)
    sd = np.std(diff_vals, ddof=1)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(mean_vals, diff_vals, alpha=0.6, color='teal')
    
    plt.axhline(md, color='red', linestyle='-', label=f'Mean Difference: {md:.2f} mmHg')
    plt.axhline(md + 1.96 * sd, color='blue', linestyle='--', label=f'+1.96 SD: {md + 1.96*sd:.2f} mmHg')
    plt.axhline(md - 1.96 * sd, color='blue', linestyle='--', label=f'-1.96 SD: {md - 1.96*sd:.2f} mmHg')
    
    plt.xlabel('Mean of True and Predicted Pressures (mmHg)')
    plt.ylabel('Difference (True - Predicted) (mmHg)')
    plt.title(f'Bland-Altman Plot: {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Evaluation metrics calculation function
def calculate_metrics(y_true, y_pred, target_name):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"--- {target_name} SVR Prediction Results ---")
    print(f"MAE  : {mae:.4f} mmHg")
    print(f"RMSE : {rmse:.4f} mmHg")
    print(f"R^2  : {r2:.4f}\n")

# 10-fold Cross Validation (Completely random split)
kf = KFold(n_splits=10, shuffle=True)

y_sys_true_all, y_sys_pred_all = [], []
y_dia_true_all, y_dia_pred_all = [], []

# Define SVR hyperparameters here
svr_C = 150.0      
svr_epsilon = 2.0  

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    
    y_sys_train, y_sys_test = y_sys[train_index], y_sys[test_index]
    y_dia_train, y_dia_test = y_dia[train_index], y_dia[test_index]
    
    # Z-score normalization 
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    # Train and predict SBP
    y_sys_pred = svr_gradient_descent(X_train_norm, y_sys_train, X_test_norm, 
                                      learning_rate=0.01, epochs=3000, C=svr_C, epsilon=svr_epsilon)
    y_sys_true_all.extend(y_sys_test)
    y_sys_pred_all.extend(y_sys_pred)
    
    # Train predict DBP
    y_dia_pred = svr_gradient_descent(X_train_norm, y_dia_train, X_test_norm, 
                                      learning_rate=0.01, epochs=3000, C=svr_C, epsilon=svr_epsilon)
    y_dia_true_all.extend(y_dia_test)
    y_dia_pred_all.extend(y_dia_pred)

# Convert to numpy arrays
y_sys_true_all = np.array(y_sys_true_all)
y_sys_pred_all = np.array(y_sys_pred_all)
y_dia_true_all = np.array(y_dia_true_all)
y_dia_pred_all = np.array(y_dia_pred_all)

# Calculate evaluation metrics
calculate_metrics(y_sys_true_all, y_sys_pred_all, "Systolic Pressure")
calculate_metrics(y_dia_true_all, y_dia_pred_all, "Diastolic Pressure")

# Bland-Altman Plots
plot_bland_altman(y_sys_true_all, y_sys_pred_all, "Systolic Pressure SVR Prediction")
plot_bland_altman(y_dia_true_all, y_dia_pred_all, "Diastolic Pressure SVR Prediction")