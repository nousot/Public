import os
import pickle

import pandas as pd
import numpy as np
# Begin user input -----------------------------------------------------------

DATA_DIR = '~/data'

cal_df = pd.read_csv(os.path.join(DATA_DIR, 'calendar.csv'),
                     parse_dates=['date'])
sales_df = pd.read_csv(os.path.join(DATA_DIR, 'sales_train_evaluation.csv'))
price_df = pd.read_csv(os.path.join(DATA_DIR, 'sell_prices.csv'))
sample_sub_df = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))


for d in range(1942, 1969 + 1):
   column_name = 'd_' + str(d)
   sales_df[column_name] = np.nan

sales_long = sales_df.melt(
    id_vars=['id'],
    value_vars=['d_' + str(i) for i in range(1, 1969 + 1)],
    var_name='d',
    value_name='sales'
)

sales_long = sales_long.merge(
    sales_df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']],
    on='id' 
)

sales_aug = sales_long.merge(
    cal_df[['d', 'date', 'wm_yr_wk', 'event_name_1',
            'snap_CA', 'snap_TX', 'snap_WI']], on='d')
  
sales_aug = sales_aug.merge(price_df, how='left',
                            on=['store_id', 'item_id', 'wm_yr_wk'])

sales_aug.sort_values(['id', 'date'], inplace=True)

df = sales_aug
df['t'] = df['d'].apply(lambda s: int(s[2:]))

df['snap'] = 0
df.loc[df['state_id'] == 'CA', 'snap'] = df.loc[df['state_id'] == 'CA', 'snap_CA']
df.loc[df['state_id'] == 'TX', 'snap'] = df.loc[df['state_id'] == 'TX', 'snap_TX']
df.loc[df['state_id'] == 'WI', 'snap'] = df.loc[df['state_id'] == 'WI', 'snap_WI']

df = df[['item_id', 'store_id', 'date', 't', 'sales', 'event_name_1',
         'snap', 'sell_price']]

df.to_csv('~/data/M5_lean_eval.csv.gz', index=False, compression='gzip')
