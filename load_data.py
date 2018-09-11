import os
import pandas as pd
from sklearn.externals import joblib
from TFM.settings import get_right_ts_name, get_target_name, get_user_id_name, get_code_name


def load_data(input_path, pref=None, right_ts=None, user_id=None, target=None,
    cols=None, sample_s=None, random_s=None, verbose=True):
    '''
    Loads and concatenates dataframes stored as pickles.
    Parameters:
     > cols: columns to load keep.
     > random_s: size of sample data.
    '''
    
    right_ts = get_right_ts_name(right_ts=right_ts)
    target = get_target_name(target=target)
    user_id = get_user_id_name(user_id=user_id)

    if verbose:
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
            if verbose:
                print('> File "{}" loaded.'.format(file))
    del temp
    df = pd.concat(df_list).sort_values([user_id, right_ts])
    
    if verbose:
        print()
        print('> Removing user-periods with null target...')
    df = df.loc[df[target].notnull()]
    
    if sample_s:
        if verbose:
            print('> Selecting sample data...')
            print()
        sample_ids = df[user_id].drop_duplicates() \
            .sample(sample_s, random_state=random_s)
        df = df.loc[df[user_id].isin(sample_ids)]

    if verbose:
        print('> Unique users ({}): {}'.format(
            user_id, df[user_id].unique().size))
        aux = df[right_ts].dt.date.unique()
        print('> Periods: {} ({} - {})'.format(
            aux.size, aux.min(), aux.max()))
        print('> User-periods: {}'.format(len(df)))
    
    return df


def getting_events(events_type, c, user_id=None, code=None):

    user_id = get_user_id_name(user_id=user_id)
    code = get_code_name(code=code)

    m = events_type[user_id].isin(c)
    events_chosen = events_type.loc[m]

    cancer_mask = events_chosen[code].str[:3] == 'd_C'
    melanoma = events_chosen[code].str[:5] == 'd_C44'
    
    events_chosen = events_chosen.loc[~cancer_mask|melanoma]

    events_chosen.reset_index(drop=True, inplace=True)
    del events_type
    return events_chosen
