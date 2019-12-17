import numpy as np
import pandas as pd
import os

def read_data():
    raw_data_path = os.path.join(os.path.pardir, 'data', 'raw')
    working_data_path = os.path.join(raw_data_path, 'WA_Fn-UseC_-HR-Employee-Attrition.csv')
    work_df = pd.read_csv(working_data_path, index_col='EmployeeNumber')
    return work_df


def process_data(df):
    return (df
            .drop(['EmployeeCount', 'MonthlyIncome', 'Over18', 'StandardHours'], axis=1)
            .pipe(categorical_binary)
            .pipe(categorical_onehot)
            .pipe(reorder_column)           
           )
            

def reorder_column(work_df):
    columns = [column for column in work_df.columns if column != 'Attrition']
    columns = ['Attrition'] + columns
    work_df = work_df[columns]
    return work_df

def categorical_onehot(work_df):
    categorical_col = []
    for column in work_df.columns:
        if work_df[column].dtype == object and len(work_df[column].unique())<=50:
            categorical_col.append(column)
                
    work_df = pd.get_dummies(work_df, columns = categorical_col)
    return work_df

def categorical_binary(work_df):
    work_df['Attrition'] = work_df['Attrition'].str.lower().replace({'yes': 1, 'no':0})
    work_df['Gender'] = work_df['Gender'].str.lower().replace({'male':1,'female':0})
    work_df['OverTime'] = work_df['OverTime'].str.lower().replace({'yes':1, 'no':0})
    return work_df

def write_data(df):
    processed_data_path = os.path.join(os.path.pardir, 'data', 'processed')
    write_data_path = os.path.join(processed_data_path, 'processed_Data_Employee-Attrition.csv')
    df.to_csv(write_data_path)


if __name__ == '__main__':
    df = read_data()
    df = process_data(df)
    write_data(df)
