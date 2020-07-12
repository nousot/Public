import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from datascroller import scroll

DATA_DIR = '/mnt/c/devl/m5'

cal_df = pd.read_csv(os.path.join(DATA_DIR, 'calendar.csv'),
                     parse_dates=['date'], index_col='date')
sales_df = pd.read_csv(os.path.join(DATA_DIR, 'sales_train_validation.csv'))
price_df = pd.read_csv(os.path.join(DATA_DIR, 'sell_prices.csv'))
sample_sub_df = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

predictions = []
for i in range(sales_df.shape[0]):
    print(i)
    item_row = sales_df.iloc[i, :]
    item_df = pd.DataFrame(item_row[6:])
    item_df = item_df.reset_index()
    item_df.columns = ['d', 'sales']

    prediction = np.median(item_df['sales'])
    for period in ['validation', 'evaluation']:
        sub_id = '_'.join([item_row['item_id'], item_row['store_id'], period])
        sub_row = [sub_id] + [prediction for d in range(1, 29)]
        predictions.append(sub_row)

predictions_df = pd.DataFrame(predictions, columns = sample_sub_df.columns)
predictions_df.to_csv(os.path.join(DATA_DIR, 'submissions/simple_median.csv'),
                      index=False)
