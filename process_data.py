import pandas as pd

# load data and get selected columns
raw_data = pd.read_csv("lc_loan.csv", error_bad_lines=False)
print("Raw data shape: ", raw_data.shape)
raw_selected_features = raw_data[
    ['term', 'int_rate', 'loan_amnt', 'annual_inc', 'installment', 'dti', 'verification_status', 'loan_status']]
print("Raw selected features shape: ", raw_selected_features.shape)

# drop certain rows
raw_selected_features.dropna(inplace=True)
print("Data shape before drop: ", raw_selected_features.shape)

raw_selected_features = raw_selected_features.loc[((raw_selected_features['loan_status'] == "Fully Paid")
                                                   | (raw_selected_features['loan_status'] == "Charged Off")
                                                   | (raw_selected_features['loan_status'] == "Default")
                                                   | (raw_selected_features['loan_status'] == "Late (31-120 days)"))]
print("Data shape after drop: ", raw_selected_features.shape)
print(raw_selected_features['loan_status'].value_counts())

# pre process data
df_processed = pd.DataFrame(index=range(raw_selected_features.shape[0]),
                            columns=['term', 'int_rate', 'loan_amnt', 'verification_status', 'annual_inc',
                                     'installment', 'dti', 'loan_status'])


def preProcessData():
    for i in range(raw_selected_features.shape[0]):
        if i % 1000 == 0:
            print(str(round((i/raw_selected_features.shape[0])*100, 2)) + "%")

        # good to go features
        df_processed.iat[i, df_processed.columns.get_loc('int_rate')] = raw_selected_features['int_rate'].iloc[i]
        df_processed.iat[i, df_processed.columns.get_loc('loan_amnt')] = raw_selected_features['loan_amnt'].iloc[i]
        df_processed.iat[i, df_processed.columns.get_loc('annual_inc')] = raw_selected_features.iloc[i]['annual_inc']
        df_processed.iat[i, df_processed.columns.get_loc('installment')] = raw_selected_features.iloc[i]['installment']
        df_processed.iat[i, df_processed.columns.get_loc('dti')] = raw_selected_features.iloc[i]['dti']

        # features to convert
        df_processed.iat[i, df_processed.columns.get_loc('term')] = float(
            raw_selected_features.iloc[i]['term'].split()[0])
        df_processed.iat[i, df_processed.columns.get_loc('loan_status')] = raw_selected_features.iloc[i]['loan_status']
        if raw_selected_features['verification_status'].iloc[i] == "Source Verified":
            df_processed.iat[i, df_processed.columns.get_loc('verification_status')] = 1.0
        else:
            df_processed.iat[i, df_processed.columns.get_loc('verification_status')] = 0.0
        if raw_selected_features['loan_status'].iloc[i] == "Fully Paid":
            df_processed.iat[i, df_processed.columns.get_loc('loan_status')] = 1.0
        else:
            df_processed.iat[i, df_processed.columns.get_loc('loan_status')] = 0.0

    print(df_processed.shape)
    print(df_processed.head)
    df_processed.to_csv(r'processed_data.csv', index=False)


preProcessData()
