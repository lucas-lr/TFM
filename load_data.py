import os
import pandas as pd
from sklearn.externals import joblib
from TFM.settings import get_right_ts_name, get_target_name, get_user_id_name


def load_data(
    input_path, pref=None, right_ts=None, user_id=None,
    target=None, cols=None, sample_s=None, random_s=None
):
    '''
    Loads and concatenates dataframes stored as pickles.
    '''
    
    right_ts = get_right_ts_name(right_ts=right_ts)
    target = get_target_name(target=target)
    user_id = get_user_id_name(user_id=user_id)
    
    print('Loading data...')
    file_list = sorted(os.listdir(input_path))
    df_list = []
    
    for file in file_list:
        c1 = (file.startswith(pref)) or (pref is None)
        if c1 and file.endswith('.pkl'):
            temp = joblib.load(input_path + file)
            if cols:
                df_list.append(temp[cols].copy())
            else:
                df_list.append(temp.copy())
            print('> File "{}" loaded.'.format(file))
    del temp
    df = pd.concat(df_list)
    print()
    
    print('> Removing user-periods with null target...')
    df = df.loc[df[target].notnull()]
    
    if sample_s:
        print('> Selecting sample data...')
        sample_ids = df[user_id].drop_duplicates() \
            .sample(sample_s, random_state=random_s)
        df = df.loc[df[user_id].isin(sample_ids)]
    print()
    
    print('> Unique users ({}): {}'.format(
        user_id, df[user_id].unique().size))
    aux = df[right_ts].dt.date.unique()
    print('> Periods: {} ({} - {})'.format(
        aux.size, aux.min(), aux.max()))
    print('> User-periods: {}'.format(len(df)))
    
    return df
