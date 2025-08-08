import os
import pandas as pd 
import matplotlib.pyplot as plt

#impoert breast cancer dataset
# Load the dataset
breast_cancer_df = pd.read_csv(r"C:\Users\Frenci De La Cruz\Desktop\breast_cancer\Breast_cancer_dataset.csv")
print(breast_cancer_df.head())


# Clean the dataset
# Check the shape of the DataFrame
print(breast_cancer_df.shape)


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
# labels = ['Malignant', 'Benign']
# counts = [malignant_count, benign_count]

# # Create bar chart
# plt.bar(labels, counts)
# plt.title('Comparison of Malignant and Benign Cases')
# plt.xlabel('Diagnosis')
# plt.ylabel('Number of Cases')

# # Add value labels on top of bars
# for i, count in enumerate(counts):
#     plt.text(i, count + 1, str(count), ha='center')

# plt.show()

#sorted the dataframe by 'mean_radius' in descending order
sorted_breast_cancer_df = breast_cancer_df.sort_values(by='mean_radius', ascending=False)
print(sorted_breast_cancer_df.head())

breast_cancer_df['radious_texture_avg'] = (breast_cancer_df['mean_radius'] + breast_cancer_df['mean_texture']) / 2     
print(breast_cancer_df.head())

breast_cancer_df.groupby('diagnosis').mean()
print(breast_cancer_df.groupby('diagnosis').mean())

#correlation matrix
correlation_matrix = breast_cancer_df.corr()
print(correlation_matrix)





