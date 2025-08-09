import os
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math

#impoert breast cancer dataset
# Load the dataset
breast_cancer_df = pd.read_csv(r"C:\Users\Frenci De La Cruz\Desktop\breast_cancer\Breast_cancer_dataset.csv")
print(breast_cancer_df.head())


# Clean the dataset
# Check the shape of the DataFrame
print(breast_cancer_df.shape)

#drop Id column does not provide useful information
breast_cancer_df = breast_cancer_df.drop(columns=['id'])


# Drop the 'Unnamed: 32' column
breast_cancer_df = breast_cancer_df.drop(columns=['Unnamed: 32'])
print(breast_cancer_df.shape)


#rename the 'id' column to 'patient_id'
breast_cancer_df = breast_cancer_df.rename(columns={'id': 'patient_id'})
print(breast_cancer_df.head())

#check for missing values
print(breast_cancer_df.isnull().sum())
print(breast_cancer_df.head())


breast_cancer_df['diagnosis'] = breast_cancer_df['diagnosis'].map({'M': 1, 'B': 0})
print(breast_cancer_df['diagnosis'].value_counts())

#rename columns
breast_cancer_df.rename(columns={
    'radius_mean': 'mean_radius', 
    'texture_mean': 'mean_texture',}, inplace=True)


#check unique values in 'diagnosis' column
print(breast_cancer_df["diagnosis"].unique())

print(len(breast_cancer_df))


#count the number of malignant and benign cases
malignant_count = len(breast_cancer_df[breast_cancer_df['diagnosis'] == 1])
benign_count = len(breast_cancer_df[breast_cancer_df['diagnosis'] == 0])
print(f"Malignant count: {malignant_count}")    
print(f"Benign count: {benign_count}")

#chart for the number of malignant and benign cases

# Your counts
malignant_count = len(breast_cancer_df[breast_cancer_df['diagnosis'] == 1])
benign_count = len(breast_cancer_df[breast_cancer_df['diagnosis'] == 0])

# # Data for the chart
labels = ['Malignant', 'Benign']
counts = [malignant_count, benign_count]

# # Create bar chart
# plt.bar(labels, counts)
# plt.title('Comparison of Malignant and Benign Cases')
# plt.xlabel('Diagnosis')
# plt.ylabel('Number of Cases')

# # Add value labels on top of bars
# #for i, count in enumerate(counts):
#     #plt.text(i, count + 1, str(count), ha='center')

#lt.show()

#sorted the dataframe by 'mean_radius' in descending order
sorted_breast_cancer_df = breast_cancer_df.sort_values(by='mean_radius', ascending=False)
print(sorted_breast_cancer_df.head())

breast_cancer_df['radious_texture_avg'] = (breast_cancer_df['mean_radius'] + breast_cancer_df['mean_texture']) / 2     
print(breast_cancer_df.head())

breast_cancer_df.groupby('diagnosis').mean()
print(breast_cancer_df.groupby('diagnosis').mean())

#encore diadnosis as numeric for correlation analysis
breast_cancer_df['diagnosis'] = breast_cancer_df['diagnosis'].map({'M': 1, 'B': 0})

#correlation matrix
correlation_matrix = breast_cancer_df.corr(numeric_only=True)
print('n\ Top 5 features mos correlated with malignancy:')
print(correlation_matrix['diagnosis'].sort_values(ascending=False).head(5))

#heatmap of the correlation matrix
plt.figure(figsize=(12, 8)) 
plt.title('Correlation Matrix Heatmap')
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.tight_layout()  
plt.show()

#Get top features correlated with diagnosis
top_features = (
    correlation_matrix['diagnosis']
    .abs()
    .sort_values(ascending=False)
    .head(6)
    .index
)
top_features = [f for f in top_features if f != 'diagnosis']
print("\nTop correlated features:", top_features)


# 7. Create a new DataFrame with only the top features and diagnosis
top_features_df = breast_cancer_df[top_features + ['diagnosis']]        
print("\nTop features DataFrame:")
print(top_features_df.head())

# 8. Standardize the data
scaler = StandardScaler()
top_features_scaled = scaler.fit_transform(top_features_df.drop(columns=['diagnosis']))
top_features_scaled_df = pd.DataFrame(top_features_scaled, columns=top_features)
# 9. Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(top_features_scaled_df)
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
pca_df['diagnosis'] = top_features_df['diagnosis'].values
# 10. Plot the PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='diagnosis', palette='coolwarm', alpha=0.7)
plt.title('PCA of Top Features Correlated with Diagnosis')  
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Diagnosis', loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()  
