import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset into a pandas DataFrame from a CSV file
file_path = 'StudentPerformanceFactors.csv'
df = pd.read_csv(file_path)

# Display basic information about the dataset
print("Basic Information about the dataset:")
print(df.info())

# Display basic statistics of numerical columns
print("\nStatistical summary of numerical columns:")
print(df.describe())

# Check for missing data
print("\nMissing data in each column:")
print(df.isnull().sum())

# Handling missing values (if necessary)
df_cleaned = df.dropna()

# Check for duplicates
duplicates = df_cleaned.duplicated().sum()
print(f"\nNumber of duplicates: {duplicates}")

# Drop duplicates if any
df_cleaned = df_cleaned.drop_duplicates()

# Ensure that the data types of each column are appropriate for analysis
print("\nData types of each column:")
print(df_cleaned.dtypes)

# Data Exploration
print("\nExploring the dataset with descriptive statistics:")
print(df_cleaned.describe())

# Data Visualization: Correlation Heatmap (only for numeric columns)
plt.figure(figsize=(10, 8))

# Select only numeric columns for correlation
df_numeric = df_cleaned.select_dtypes(include=[np.number])

# Plot the heatmap
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Plotting relationships and distributions
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['Exam_Score'], kde=True, color='blue')
plt.title('Distribution of Exam Scores')
plt.xlabel('Exam Score')
plt.ylabel('Frequency')
plt.show()

# Scatter plot for Hours_Studied vs Exam_Score
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Hours_Studied', y='Exam_Score', data=df_cleaned, hue='Gender')
plt.title('Hours Studied vs Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.show()

# Data Manipulation: Grouping and Aggregation
# Example: Grouping by Gender and calculating average Exam Score
grouped_data = df_cleaned.groupby('Gender')['Exam_Score'].mean().reset_index()
print("\nAverage Exam Score by Gender:")
print(grouped_data)

# Data Analysis: Statistical Hypothesis Testing (Correlation)
correlation = df_cleaned['Hours_Studied'].corr(df_cleaned['Exam_Score'])
print(f"\nCorrelation between Hours Studied and Exam Score: {correlation:.2f}")

# Final report of insights
print("\nInsights:")
print("1. The correlation heatmap shows the relationships between different variables.")
print("2. Hours studied is positively correlated with exam scores.")
print("3. Data visualization helps uncover trends, like the impact of hours studied on exam performance.")
