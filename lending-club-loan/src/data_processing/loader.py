import pandas as pd
import numpy as np

def load_and_clean_data(filepath="./data/accepted_2007_to_2018Q4.csv"):
    """
    Loads the Lending Club dataset and performs initial cleaning and target variable creation.
    """
    print("Loading data...")

    column_dtype_mapping = {
        'id': 'O', 'term': 'O', 'grade': 'O', 'sub_grade': 'O', 'emp_title': 'O', 
        'emp_length': 'O', 'home_ownership': 'O', 'verification_status': 'O', 'issue_d': 'O', 
        'loan_status': 'O', 'pymnt_plan': 'O', 'url': 'O', 'desc': 'O', 'purpose': 'O', 
        'title': 'O', 'zip_code': 'O', 'addr_state': 'O', 'earliest_cr_line': 'O', 
        'initial_list_status': 'O', 'last_pymnt_d': 'O', 'next_pymnt_d': 'O', 
        'last_credit_pull_d': 'O', 'application_type': 'O', 'verification_status_joint': 'O', 
        'sec_app_earliest_cr_line': 'O', 'hardship_flag': 'O', 'hardship_type': 'O', 
        'hardship_reason': 'O', 'hardship_status': 'O', 'hardship_start_date': 'O', 
        'hardship_end_date': 'O', 'payment_plan_start_date': 'O', 'hardship_loan_status': 'O', 
        'disbursement_method': 'O', 'debt_settlement_flag': 'O', 'debt_settlement_flag_date': 'O', 
        'settlement_status': 'O', 'settlement_date': 'O'
    }

    try:
        loan_data = pd.read_csv(filepath, dtype=column_dtype_mapping, low_memory=False)
        loan_data.dropna(subset=['loan_amnt'], inplace=True)
        print(f"Data loaded successfully. Initial shape: {loan_data.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: '{filepath}' not found. Please download the dataset.")

    fully_paid_statuses = [
        'Fully Paid',
        'Does not meet the credit policy. Status:Fully Paid'
    ]
    defaulted_statuses = [
        'Charged Off',
        'Default',
        'Does not meet the credit policy. Status:Charged Off',
    ]
    
    loan_data = loan_data[loan_data['loan_status'].isin(fully_paid_statuses + defaulted_statuses)].copy()
    # Use np.where for faster conditional assignment. Good Loan = 0, Bad Loan = 1
    loan_data['loan_status_binary'] = np.where(loan_data['loan_status'].isin(defaulted_statuses), 1, 0)

    for col in loan_data.select_dtypes(include=['object']).columns:
        loan_data[col] = loan_data[col].str.strip()

    def parse_emp_length(x):
        if pd.isnull(x) or x.strip() in ["", "n/a", "NA"]: return np.nan
        x = x.strip()
        if x == "< 1 year": return 0.5
        if "10+" in x: return 10
        try: return float(x.split()[0])
        except (ValueError, IndexError): return np.nan
    loan_data['emp_length'] = loan_data['emp_length'].apply(parse_emp_length)

    date_columns = ['issue_d', 'earliest_cr_line']
    for col in date_columns:
        if col in loan_data.columns:
            loan_data[col] = pd.to_datetime(loan_data[col], format='%b-%Y', errors='coerce')

    cols_to_drop_leakage = ['funded_amnt', 'funded_amnt_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 
                            'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 
                            'last_pymnt_amnt', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 
                            'debt_settlement_flag_date', 'settlement_date', 
                            'payment_plan_start_date', 'hardship_start_date', 'hardship_end_date', 
                            'last_fico_range_high', 'last_fico_range_low', 'out_prncp', 'out_prncp_inv']
    cols_to_drop_irrelevant = ['id', 'url', 'title', 'emp_title', 'zip_code', 'desc', 'member_id']
    cols_to_drop_redundant = ['loan_status', 'grade', 'policy_code']
    
    all_cols_to_drop = list(set(cols_to_drop_leakage + cols_to_drop_irrelevant + cols_to_drop_redundant))
    loan_data.drop(columns=all_cols_to_drop, errors='ignore', inplace=True)
    
    missing_summary = pd.DataFrame({'missing_percent': loan_data.isna().mean() * 100})
    cols_to_drop_missing = missing_summary[missing_summary['missing_percent'] > 38].index.tolist()
    loan_data.drop(columns=cols_to_drop_missing, errors='ignore', inplace=True)

    print(f"Shape after initial cleaning: {loan_data.shape}")
    return loan_data