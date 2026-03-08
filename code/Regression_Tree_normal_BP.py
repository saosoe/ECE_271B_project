import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import random

# Load data
data_path = '/Users/qibuguojiuhuolala/Desktop/Project/Dataset/Unnormalized_data.xlsx'
df = pd.read_excel(data_path)

# Feature selection
features = ['PI', 'PT', 'AA']
X_all = df[features].values
y_sys_all = df['systolic pressure'].values
y_dia_all = df['diastolic pressure'].values


# Extract "normal blood pressure" data separately

normal_mask = (y_sys_all <= 130) & (y_sys_all >= 90) & (y_dia_all <= 85) & (y_dia_all >= 60)
X = X_all[normal_mask]
y_sys = y_sys_all[normal_mask]
y_dia = y_dia_all[normal_mask]

print(f"--- Data Filtering Complete ---")
print(f"Total number of full samples: {len(X_all)}")
print(f"Number of extracted normal blood pressure samples: {len(X)}")
print(f"Number of excluded abnormal blood pressure samples: {len(X_all) - len(X)}\n")


# implementation RT
class TreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx 
        self.threshold = threshold     
        self.left = left                
        self.right = right              
        self.value = value             

class RegressionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth                 
        self.min_samples_split = min_samples_split  
        self.root = None
        
    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        # Check if stopping criteria are met
        if n_samples >= self.min_samples_split and depth < self.max_depth:
            best_split = self._get_best_split(X, y, n_samples, n_features)
            if best_split is not None:
                # Recursively build left and right subtrees
                left_subtree = self._build_tree(best_split['X_left'], best_split['y_left'], depth + 1)
                right_subtree = self._build_tree(best_split['X_right'], best_split['y_right'], depth + 1)
                return TreeNode(best_split['feature_idx'], best_split['threshold'], left_subtree, right_subtree)
        
        # Stopping criteria met, calculate leaf node value
        leaf_value = np.mean(y)
        return TreeNode(value=leaf_value)
    
    def _get_best_split(self, X, y, n_samples, n_features):
        best_split = None
        min_error = float("inf")
        
        # iterate over all features
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            # Iterate over all unique values of the feature as candidate thresholds
            possible_thresholds = np.unique(feature_values)
            
            for threshold in possible_thresholds:
                # Split data based on threshold
                X_left, y_left, X_right, y_right = self._split(X, y, feature_idx, threshold)
                
                # ensure split subsets are non-empty
                if len(y_left) > 0 and len(y_right) > 0:
                    # Calculate error after splitting (weighted variance)
                    error = self._calculate_variance_reduction(y_left, y_right)
                    if error < min_error:
                        min_error = error
                        best_split = {
                            'feature_idx': feature_idx,
                            'threshold': threshold,
                            'X_left': X_left,
                            'y_left': y_left,
                            'X_right': X_right,
                            'y_right': y_right
                        }
        return best_split
    
    def _split(self, X, y, feature_idx, threshold):
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]
    
    def _calculate_variance_reduction(self, y_left, y_right):
        # Calculate weighted mean squared error (MSE) of split subsets
        n_left, n_right = len(y_left), len(y_right)
        total_n = n_left + n_right
        
        mse_left = np.var(y_left) * n_left
        mse_right = np.var(y_right) * n_right
        
        return (mse_left + mse_right) / total_n
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        # Reached leaf node, return value
        if node.value is not None:
            return node.value
        
        # Otherwise, traverse to left/right subtree based on feature threshold
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


# Bland-Altman Plotting
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

# Evaluation Metric Calculation
def calculate_metrics(y_true, y_pred, target_name):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"--- {target_name} Regression Tree Prediction Results ---")
    print(f"MAE  : {mae:.4f} mmHg")
    print(f"RMSE : {rmse:.4f} mmHg")
    print(f"R^2  : {r2:.4f}\n")

# 10-fold Cross Validation
kf = KFold(n_splits=10, shuffle=True)

y_sys_true_all, y_sys_pred_all = [], []
y_dia_true_all, y_dia_pred_all = [], []

# Define hyperparameters for Regression Tree here
tree_max_depth = 4 

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    
    y_sys_train, y_sys_test = y_sys[train_index], y_sys[test_index]
    y_dia_train, y_dia_test = y_dia[train_index], y_dia[test_index]
    
    #  Z-score Normalization 

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Prevent division by zero 
    std[std == 0] = 1e-8
    
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    # Initialize and train model for SBP
    tree_sys = RegressionTree(max_depth=tree_max_depth)
    tree_sys.fit(X_train_norm, y_sys_train)
    y_sys_pred = tree_sys.predict(X_test_norm)
    
    y_sys_true_all.extend(y_sys_test)
    y_sys_pred_all.extend(y_sys_pred)
    
    # Initialize and train model for DBP
    tree_dia = RegressionTree(max_depth=tree_max_depth)
    tree_dia.fit(X_train_norm, y_dia_train)
    y_dia_pred = tree_dia.predict(X_test_norm)
    
    y_dia_true_all.extend(y_dia_test)
    y_dia_pred_all.extend(y_dia_pred)

# Convert to numpy arrays
y_sys_true_all = np.array(y_sys_true_all)
y_sys_pred_all = np.array(y_sys_pred_all)
y_dia_true_all = np.array(y_dia_true_all)
y_dia_pred_all = np.array(y_dia_pred_all)

# calculate evaluation metrics
calculate_metrics(y_sys_true_all, y_sys_pred_all, "Systolic Pressure")
calculate_metrics(y_dia_true_all, y_dia_pred_all, "Diastolic Pressure")

# Generate Bland-Altman Plots
plot_bland_altman(y_sys_true_all, y_sys_pred_all, "Systolic Pressure Tree Prediction")
plot_bland_altman(y_dia_true_all, y_dia_pred_all, "Diastolic Pressure Tree Prediction")



# Ensure random selection of 30 cases
random.seed(42)  # Fix random seed for reproducibility
total_samples = len(y_sys_true_all)
sample_indices = random.sample(range(total_samples), 30)

# Extract corresponding Reference and Predicted Values
table_data = {
    'Sample ID': [i + 1 for i in range(30)],
    'Ref SBP (mmHg)': [y_sys_true_all[i] for i in sample_indices],
    'Pred SBP (mmHg)': [y_sys_pred_all[i] for i in sample_indices],
    'Ref DBP (mmHg)': [y_dia_true_all[i] for i in sample_indices],
    'Pred DBP (mmHg)': [y_dia_pred_all[i] for i in sample_indices]
}

# Create DataFrame and print table
df_samples = pd.DataFrame(table_data)
print("\n--- Quantification Comparison Table for 30 Randomly Selected Blood Pressure Cases (Similar to Figure 7) ---")
print(df_samples.to_string(index=False))

# Calculate Mean & SD
errors_sys = np.array(y_sys_pred_all) - np.array(y_sys_true_all)
errors_dia = np.array(y_dia_pred_all) - np.array(y_dia_true_all)

summary_stats = pd.DataFrame({
    'Metric': ['Mean Error (Bias)', 'Standard Deviation (SD)'],
    'SBP (mmHg)': [np.mean(errors_sys), np.std(errors_sys)],
    'DBP (mmHg)': [np.mean(errors_dia), np.std(errors_dia)]
})

print("\n--- Full Sample Performance Statistics ---")
print(summary_stats.to_string(index=False))


# Prepare data: 30 sample
sample_sys_ref = np.array([y_sys_true_all[i] for i in sample_indices])
sample_sys_pred = np.array([y_sys_pred_all[i] for i in sample_indices])

# Classify by Reference SBP 
hypertensive_mask = sample_sys_ref > 130
normotensive_mask = sample_sys_ref <= 130

# Extract data for both categories
ref_hyper = sample_sys_ref[hypertensive_mask]
pred_hyper = sample_sys_pred[hypertensive_mask]

ref_norm = sample_sys_ref[normotensive_mask]
pred_norm = sample_sys_pred[normotensive_mask]

# print classification table
print("\n" + "="*50)
print(f"{'Category':<15} | {'Ref SBP':<10} | {'Pred SBP':<10} | {'Error':<10}")
print("-" * 50)

def print_rows(refs, preds, label):
    for r, p in zip(refs, preds):
        print(f"{label:<15} | {r:<10.2f} | {p:<10.2f} | {p-r:<10.2f}")

print_rows(ref_hyper, pred_hyper, "Hypertensive")
print("-" * 50)
print_rows(ref_norm, pred_norm, "Normotensive")
print("="*50)

# Calculate Mean & SD
def calculate_stats(refs, preds):
    if len(refs) == 0: return 0, 0, 0
    errors = preds - refs
    return len(refs), np.mean(errors), np.std(errors)

count_h, mean_h, sd_h = calculate_stats(ref_hyper, pred_hyper)
count_n, mean_n, sd_n = calculate_stats(ref_norm, pred_norm)

# Generate statistical results table
summary_df = pd.DataFrame({
    'Category': ['Normotensive (≤130)'],
    'Sample Count': [count_n],
    'Mean Error (Bias)': [f"{mean_n:.2f} mmHg"],
    'Std Deviation (SD)': [f"{sd_n:.2f} mmHg"]
})

print("\n--- SBP Classification Performance Statistics ---")
print(summary_df.to_string(index=False))