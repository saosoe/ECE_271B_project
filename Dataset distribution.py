import pandas as pd
import matplotlib.pyplot as plt

# Read data 
file_path = '/Users/qibuguojiuhuolala/Desktop/Project/Dataset/Unnormalized_data.xlsx'

df = pd.read_excel(file_path)

# Define blood pressure classification function
def categorize_bp(row):
    sys = row['systolic pressure']
    dia = row['diastolic pressure']
    
    # Classification logic
    if sys < 90 or dia < 60:
        return 'Low Blood Pressure'
    elif sys >= 140 or dia >= 90:
        return 'High Blood Pressure'
    else:
        return 'Normal Blood Pressure'

# apply classification and count quantities
df['BP_Category'] = df.apply(categorize_bp, axis=1)
bp_counts = df['BP_Category'].value_counts()

# Set up canvas and colors
plt.figure(figsize=(8, 8))

# Assign colors to different categories
color_map = {
    'Normal Blood Pressure': '#a2d1a2', 
    'High Blood Pressure': '#ff9999',    
    'Low Blood Pressure': '#99c2ff'      
}
# Match colors based on the actual data categories present
pie_colors = [color_map.get(cat, '#cccccc') for cat in bp_counts.index]

# Draw pie chart

plt.pie(bp_counts, 
        labels=bp_counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=pie_colors, 
        shadow=True,
        textprops={'fontsize': 12}) # Adjust label font size

# Set title and display the chart
plt.tight_layout()
plt.show()
