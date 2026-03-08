import pandas as pd
import itertools
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Read data and clean missing values
df = pd.read_excel('/Users/qibuguojiuhuolala/Desktop/Project/Dataset/Unnormalized_data.xlsx')
df = df.dropna()

# Define scope: candidate independent variables (Column A to K, i.e., first 11 columns) and target dependent variable
candidate_cols = df.columns[0:11].tolist() 
target_col = 'diastolic pressure'  # If you want to predict diastolic pressure, keep this; for systolic, change to 'systolic pressure'
y = df[target_col]

# Set strict collinearity threshold
VIF_THRESHOLD = 5.0  

best_r2 = -1
best_combo = None
best_vif_details = {}

# Iterate through all combinations of 3 variables (total 165 combinations)
for combo in itertools.combinations(candidate_cols, 3):
    X_subset = df[list(combo)]
    X_with_const = sm.add_constant(X_subset)
    
    # Calculate VIF for current combination
    # range(1, 4) skips the 0th column (constant term)
    vifs = [variance_inflation_factor(X_with_const.values, i) for i in range(1, 4)]
    
    # If all VIF values in the combination are below threshold, evaluate prediction quality
    if max(vifs) < VIF_THRESHOLD:
        # Build OLS linear regression model and evaluate predictive power
        model = sm.OLS(y, X_with_const).fit()
        
        # Record as best combination if current R-squared is higher
        if model.rsquared > best_r2:
            best_r2 = model.rsquared
            best_combo = combo
            # Package variable names and corresponding VIF values into a dictionary
            best_vif_details = dict(zip(combo, vifs))

# Output final results
if best_combo:
    print(f"The best selected combination of 3 variables is: {best_combo}")
    print(f"Their combined explanatory power (R-squared) for {target_col} is: {best_r2:.4f}")
    print("-" * 30)
    print("VIF details for each variable (lower value = lower collinearity):")
    for var, vif_val in best_vif_details.items():
        print(f" - {var}: {vif_val:.2f}")
else:
    print(f"No valid 3-variable combination found under current threshold (VIF < {VIF_THRESHOLD}).")
    print("Suggestion: Try slightly increasing VIF_THRESHOLD (e.g., to 10), or check if extreme collinearity exists in the data.")
