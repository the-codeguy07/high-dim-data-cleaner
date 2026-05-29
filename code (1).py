import pandas as pd
import numpy as np
import time
def clean_high_dimensional_data(input_path,output_path):
    start_time=time.time()
    print(f"Loading High-Dimensional Dataset from: {input_path}")
    df=pd.read_csv(input_path)
    initial_shape=df.shape
    print(f"Initial Shape: {initial_shape[0]} rows,{initial_shape[1]} columns")
    feature_cols=[col for col in df.columns if col.startswith('Feature_')]
    base_numeric_cols=['Age','Income','Credit_Score']
    identifier_col='CustomerID'
    target_col='Is_Active'
    print("\nHandling Base Demographics")
    df=df.dropna(subset=[identifier_col])
    for col in base_numeric_cols:
        if col in df.columns:
            df[col]=df[col].fillna(df[col].median())
    if target_col in df.columns:
        df[target_col]=df[target_col].ffill().astype(int)
    print("Native Pandas Imputation for High-Dimensional Features")
    df[feature_cols]=df[feature_cols].interpolate(method='linear',limit_direction='both')
    df[feature_cols]=df[feature_cols].fillna(df[feature_cols].median())
    print("Feature Scaling")
    for col in feature_cols:
        col_mean=df[col].mean()
        col_std=df[col].std()
        if col_std!=0:
            df[col]=(df[col]-col_mean)/col_std
        else:
            df[col]=0.0  
    df[feature_cols]=df[feature_cols].round(4)
    print("Outlier Capping")
    for col in feature_cols+base_numeric_cols:
        if col in df.columns:
            upper_limit=df[col].quantile(0.99)
            lower_limit=df[col].quantile(0.01)
            df[col]=np.where(df[col]>upper_limit,upper_limit,df[col])
            df[col]=np.where(df[col]<lower_limit,lower_limit,df[col])
    print("Memory Optimization")
    df[identifier_col]=df[identifier_col].astype('int32')
    df['Age']=df['Age'].astype('int16')
    df['Credit_Score']=df['Credit_Score'].astype('int16')
    if target_col in df.columns:
        df[target_col]=df[target_col].astype('int8')
    df[feature_cols]=df[feature_cols].astype('float32')
    df.to_csv(output_path,index=False)
    end_time=time.time()
    print("\nCleaning Complete")
    print(f"Final Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Time taken: {end_time-start_time:.2f} seconds")
    print(f"Cleaned high-dimensional data saved to '{output_path}'")
if __name__ =="__main__":
    input_file=r"C:\Users\91994\Desktop\project\uncleaned_dataset.csv"
    output_file=r"C:\Users\91994\Desktop\project\cleaned_dataset.csv"
    clean_high_dimensional_data(input_file,output_file)