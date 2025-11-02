import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def feature_engineer(df):
    """Applies feature engineering to the dataframe."""

    print("Performing feature engineering...")
    df_engineered = df.copy()
    
    # This is essential for date arithmetic and feature extraction.
    df_engineered['credit_history_length'] = (df_engineered['issue_d'] - df_engineered['earliest_cr_line']).dt.days

    # Extract Year and Month from Issue Date
    # These features can capture temporal patterns in loan issuance.
    df_engineered['issue_year'] = df_engineered['issue_d'].dt.year
    df_engineered['issue_month'] = df_engineered['issue_d'].dt.month

    # Loan Amount to Installment Ratio
    # This feature indicates how large the loan amount is relative to the monthly installment.
    # A higher ratio may indicate a riskier loan.
    df_engineered['loan_amnt_div_instlmnt'] = df_engineered['loan_amnt'] / (df_engineered['installment'] + 1e-6)

    # Loan Amount to Annual Income Ratio
    # This captures the borrower's debt burden relative to their income.
    # Add a small epsilon to the denominator to prevent division by zero
    df_engineered['loan_to_income_ratio'] = df_engineered['loan_amnt'] / (df_engineered['annual_inc'] + 1e-6)

    # DTI scaled by Income
    # A high DTI on a high income represents a different risk profile than a high DTI on a low income.
    # if 'dti' in df_engineered.columns and 'annual_inc' in df_engineered.columns:
    df_engineered['dti_x_income'] = df_engineered['dti'] * df_engineered['annual_inc']

    # Revolving Utilization x Recent Inquiries
    # Captures risk of a user with high existing debt actively seeking more credit.
    # FillNa for these columns is important before creating interaction term
    df_engineered['revol_util'] = df_engineered['revol_util'].fillna(df_engineered['revol_util'].median())
    df_engineered['inq_last_6mths'] = df_engineered['inq_last_6mths'].fillna(df_engineered['inq_last_6mths'].median())
    df_engineered['revol_util_x_inq'] = df_engineered['revol_util'] * df_engineered['inq_last_6mths']

    return df_engineered

def preprocess_dataset(df_engineered, train_end_date, val_end_date):
    """
    Performs feature engineering, data splitting, preprocessing, oversampling, and returns the final datasets.
    """
    df_engineered = df_engineered.sort_values(by='issue_d').reset_index(drop=True)

    X = df_engineered.drop('loan_status_binary', axis=1)
    y = df_engineered['loan_status_binary']

    # Define temporal split points
    train_end_date = pd.Timestamp(train_end_date)
    val_end_date = pd.Timestamp(val_end_date)

    # Get indices for splits
    train_indices = X[X['issue_d'] < train_end_date].index
    val_indices = X[(X['issue_d'] >= train_end_date) & (X['issue_d'] < val_end_date)].index
    test_indices = X[X['issue_d'] >= val_end_date].index

    # Create raw data splits
    X_train_raw, y_train = X.loc[train_indices], y.loc[train_indices]
    X_val_raw, y_val = X.loc[val_indices], y.loc[val_indices]
    X_test_raw, y_test = X.loc[test_indices], y.loc[test_indices]
    
    # Drop date columns after splitting
    date_cols_to_drop = ['issue_d', 'earliest_cr_line']
    X_train_raw = X_train_raw.drop(columns=date_cols_to_drop, errors='ignore')
    X_val_raw = X_val_raw.drop(columns=date_cols_to_drop, errors='ignore')
    X_test_raw = X_test_raw.drop(columns=date_cols_to_drop, errors='ignore')
    
    # Define preprocessing steps based on training data
    numerical_features = X_train_raw.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), 
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ], remainder='passthrough')
    
    print("Fitting preprocessor and transforming data...")
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    return X_train, X_val, X_test, y_train, y_val, y_test, X_test_raw, preprocessor