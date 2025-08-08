#imports
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
# Assuming the dataset is in CSV format and named 'breast_cancer_data.csv'


#reading in data
cancer_data_df = pd.read_csv('breast_cancer_data.csv')
# Display the first few rows of the dataset
print(cancer_data_df.head())

# Save the cleaned dataset to a new CSV file
cancer_data_df.to_csv('breast_cancer_data_cleaned.csv', index=False)# Display the first few rows of the dataset


# with open("breast_cancer_data.csv", newline='') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         print(row)

# # Display the first few rows of the dataset


