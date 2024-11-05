import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from IPython.display import FileLink
from IPython.core.display import HTML

# Get the file path from user input
dir_path = input("Enter the path to the file: ")
fname = input("Enter the file name: ")

os.chdir(dir_path)
df = pd.read_csv(fname, index_col=0)  # Read the CSV file into a DataFrame

print("DATASET INFO:")
df.info()

print("DATASET SHAPE:")
print(df.shape)
print("Raw Dataframe")
display(df.head())
# Check for missing values
missing_columns = df.columns[df.isnull().any()]
flag=missing_columns
print("Columns with missing values:", missing_columns)

# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number])

if flag.any():
    print("Null Values found!!")

    rm = input("Do you want to obtain the missing values? y/n ")
    if rm == 'y':
        mthd=input("How do you want to obtain the missing values :\n {min,max,mean,median,mode,upper_value,next_value} \n Else, Enter : rmv to drop \n>>>>")
    
        match mthd:
            case "mean":
                # Calculate the mean for each numeric column
                mean_values = numeric_cols.mean()
    
                # Fill missing values with the mean
                df[numeric_cols.columns] = numeric_cols.fillna(mean_values)
            case "upper_value":
                upper_value = numeric_cols.pad()
                df[numeric_cols.columns] = numeric_cols.fillna(upper_value)
            case "next_value":
                next_value = numeric_cols.bfill()
                df[numeric_cols.columns] = numeric_cols.fillna(next_value)
            case "median":
                median = numeric_cols.median()
                df[numeric_cols.columns] = numeric_cols.fillna(median)
            case "mode":
                mode = numeric_cols.mode()
                df[numeric_cols.columns] = numeric_cols.fillna(mode)
            case "max":
                maximum = numeric_cols.max()
                df[numeric_cols.columns] = numeric_cols.fillna(maximum)
            case "min":
                minimum = numeric_cols.min()
                df[numeric_cols.columns] = numeric_cols.fillna(minimum)
            case "rmv":
                user_code = input("Enter Python code to modify the DataFrame: ")
                try:
                    exec(user_code)
                    print("Updated DataFrame:")
                    display(df.head())
                except Exception as e:
                    print(f"Error executing code: {e}")
else:
    pass
print("\n>>>>>>>>>>>>>>>Missing values fixed>>>>>>>>>>>>>>>\n")
    
print("Updated DataFrame :")
display(df.head())

print("Searching for duplicate values............")
# Check for duplicate columns & rows
duplicate_rows = df.duplicated().sum()
print("Duplicate rows:", duplicate_rows)
duplicate_cols = df.columns.duplicated().sum()
print("Duplicate columns:", duplicate_cols)

print("Dropping duplicate values............")

# Drop duplicate rows & columns
df.drop_duplicates(inplace=True)
print("After dropping duplicate rows:")
df = df.loc[:, ~df.columns.duplicated()]
print("After dropping duplicate columns:")

print("\n>>>>>>>>>>>>>>>Duplicate values fixed>>>>>>>>>>>>>>>\n")
display(df.head())

# Lable Encoder
print("Encoding Lables............")
label_encoder = LabelEncoder()
n=int(input("Number of target variable or column: "))
while n != 0:
    trg = input("Enter the target variable or column: ").title()
    df[trg] = label_encoder.fit_transform(df[trg])
    n-=1

print("\n>>>>>>>>>>>>>>>Lable Encoded>>>>>>>>>>>>>>>\n")
print("Updated DataFrame:")
display(df.head())

#Normalization or standardization of numerical values
# Separate numerical and categorical features
print("Standardizing Numerical Values............")

numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

# Standardize numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("\n>>>>>>>>>>>>>>>Numerical Values Standardized>>>>>>>>>>>>>>>\n")
print("Dataframe Statistical description")
display(df.describe().round(3))
print("\n*********QUALITY ENHANCED DATASET OBTAINED!!!***********\n")
display(df.head())

# Export the processed DataFrame to a CSV file
processed_fname = "processed_" + fname
df.to_csv(processed_fname, index=False)

print("Processed DataFrame exported to", processed_fname)

# Create a file link to download the processed CSV file
file_link = HTML(f'<a href="data:application/octet-stream;base64,{processed_fname.encode().decode()}" download="{processed_fname}">Download: {processed_fname}</a>')
display(file_link)
