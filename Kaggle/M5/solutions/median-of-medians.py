# Credit: https://github.com/aerdem4/kaggle-wiki-traffic/blob/master/Model.ipynb 
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from datascroller import scroll


# Begin user input -----------------------------------------------------------
DATA_DIR = '/mnt/c/devl/m5'
YEAR_SHIFT = 364  # days in a year, multiple of 7 to capture week behavior
PERIOD = 49  #  Period size for determining if simpler backoff will be used
PREDICT_PERIOD = 56  # number of days which will be predicted
DAYS_IN_WEEK = 7  # So 7 doesn't seem like a magic number
WINDOWS = np.array([2, 3, 4, 7, 11, 18, 29, 47]) * DAYS_IN_WEEK

# Begin median of medians logic
cal_df = pd.read_csv(os.path.join(DATA_DIR, 'calendar.csv'),
                     parse_dates=['date'])
sales_df = pd.read_csv(os.path.join(DATA_DIR, 'sales_train_validation.csv'))
price_df = pd.read_csv(os.path.join(DATA_DIR, 'sell_prices.csv'))
sample_sub_df = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

# Utility functions ----------------------------------------------------------
def smape(x, y):
    """evaluation function"""
    if x == y:
        return 0
    else:
        return np.abs(x-y) / (x+y)

    
def safe_median(s):
    """median function ignoring nans"""
    return np.median([x for x in s if ~np.isnan(x)])


# Data Prep ------------------------------------------------------------------
d_df = pd.DataFrame()
d_df['d'] = list(sales_df.columns[6:])
d_df = d_df.merge(cal_df[['d', 'date']])

cols = ['id'] + list(sales_df.columns[6:])
sales_red = sales_df[cols]
new_cols = ['id'] + list(d_df['date'].dt.strftime('%Y-%m-%d'))
sales_red.columns = new_cols

sales = pd.melt(sales_red, id_vars='id', var_name='date',
                value_name='sales')

sales['date'] = sales['date'].astype('datetime64[ns]')

LAST_TRAIN_DAY = sales['date'].max()

train = sales.groupby(['id'])['sales'].apply(lambda x: list(x))
# train is a series, which maps id -> list of sales
# For instance,
train['FOODS_1_001_CA_1_validation']

# Method logic ---------------------------------------------------------------
count = 0
scount = 0
pred_dict = {}
for item_id, row in zip(train.index, train):
    # For debugging:
    #row = train['FOODS_3_090_CA_3_validation']

    preds = [0] * PREDICT_PERIOD  # Initialize preds to be zero
    medians = np.zeros((len(WINDOWS), DAYS_IN_WEEK))  # 7 columns 
    for day_i in range(DAYS_IN_WEEK):
        for window_j in range(len(WINDOWS)):
            window_size = WINDOWS[window_j]
            last_window_data = row[-window_size:] # window_size weeks of data
            weekly_window = (np.array(last_window_data)
                               .reshape(-1, DAYS_IN_WEEK))
            # Rows are weeks in weekly window
            # Use sliding 3-day windows centered around weekday day_i
            # e.g., Friday: [Thursday, Friday, Saturday]
            sliding_day_of_week_windows = (
                [
                    #weekly_window[:, (day_i - 1) % 7],
                    weekly_window[:, day_i] #,
                    #weekly_window[:, (day_i + 1) % DAYS_IN_WEEK]
                ]
            )
            flattened_windows = np.hstack(sliding_day_of_week_windows)
            medians[window_j, day_i] = safe_median(flattened_windows)
    for t in range(PREDICT_PERIOD):
        preds[t] = safe_median(medians[:, t % DAYS_IN_WEEK])
    
    pred_dict[item_id] = preds
    count += 1
    if count % 1000 == 0:
        print(f'Processed {count}')


predictions = []
for key in pred_dict.keys():
    item_data = pred_dict[key]
    item_root = '_'.join(key.split('_')[:-1])
    for period in ['validation', 'evaluation']:
        sub_id = '_'.join([item_root, period])
        if period == 'validation':
            sub_row = [sub_id] + list(pred_dict[key][:28])
        elif period == 'evaluation':
            sub_row = [sub_id] + list(pred_dict[key][28:])
        predictions.append(sub_row)


predictions_df = pd.DataFrame(predictions, columns=sample_sub_df.columns)

submission_name = ('median_of_medians_1_day_' + '_'.join([str(s) for s in WINDOWS])
                   + '.csv')
predictions_df.to_csv(os.path.join(DATA_DIR, 'submissions', submission_name),
                      index=False)
