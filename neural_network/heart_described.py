import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('Heart.csv')

# Remove the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)

# Separate numerical and categorical columns
numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()

# Create pairplots for numerical variables
plt.figure(figsize=(20, 15))
sns.pairplot(data, vars=numerical_columns, hue='AHD', diag_kind='kde', plot_kws={'alpha': 0.6})
plt.tight_layout()
plt.savefig('heart_disease_numerical_pairplot.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a correlation heatmap for numerical variables
plt.figure(figsize=(12, 10))
sns.heatmap(data[numerical_columns].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Variables')
plt.tight_layout()
plt.savefig('heart_disease_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Create box plots for numerical variables grouped by AHD
fig, axes = plt.subplots(3, 2, figsize=(20, 25))
axes = axes.flatten()
for i, col in enumerate(numerical_columns):
    if i < len(axes):
        sns.boxplot(x='AHD', y=col, data=data, ax=axes[i])
        axes[i].set_title(f'{col} by AHD')
plt.tight_layout()
plt.savefig('heart_disease_numerical_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

# Create bar plots for categorical variables
fig, axes = plt.subplots(3, 2, figsize=(20, 25))
axes = axes.flatten()
for i, col in enumerate(categorical_columns):
    if i < len(axes):
        sns.countplot(x=col, hue='AHD', data=data, ax=axes[i])
        axes[i].set_title(f'{col} Distribution by AHD')
        axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('heart_disease_categorical_barplots.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualizations have been saved as PNG files.")

# Display summary statistics for numerical variables
print("\nSummary Statistics for Numerical Variables:")
print(data[numerical_columns].describe())

# Display summary for categorical variables
print("\nSummary for Categorical Variables:")
for col in categorical_columns:
    print(f"\n{col}:")
    print(data[col].value_counts(normalize=True))

# Display correlation matrix for numerical variables
print("\nCorrelation Matrix for Numerical Variables:")
correlation_matrix = data[numerical_columns].corr()
print(correlation_matrix)