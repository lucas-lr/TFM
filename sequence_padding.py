import pandas as pd
from TFM.settings import get_right_ts_name, get_target_name, get_user_id_name


def presequence_padding(df0, val, right_ts=None, user_id=None, target=None,
                        p2p=None, fillna_cols=[], fillna_val=0.,
                        keep_first_cols=[]):
    '''
    Transforms data such that each sequence has a length equal to a multiple of
    a choosen value.
    Parameters:
     > p2p: periods to predict. Example: predict the sum of the expense of the
       next 4 weeks.
     > fillna_cols: fills null values with fillna_val.
     > keep_first_cols: fills null values with the first value for each user in
       case there is at least one not null value.
    '''


    right_ts = get_right_ts_name(right_ts=right_ts)
    target = get_target_name(target=target)
    user_id = get_user_id_name(user_id=user_id)
    
    df = df0.copy()
    
    print('Padding sequence lengths...')
    print('> Sequence length multiple: {}'.format(val))
    user_len = df[user_id].value_counts().to_frame(name='len0')
    user_len['len1'] = val*(((user_len['len0'] - 1)//val) + 1)
    fs = user_len['len0'].unique().size
    print('> Initial sequence lengths: {}'.format(fs))
    fs = user_len['len1'].unique().size
    print('> Final sequence lengths:   {}'.format(fs))
    print('> Creating "indexes" (user_id & right_ts)...')
    
    USER_TS = []
    
    for l in sorted(user_len['len1'].unique()):
        
        uids = user_len.loc[user_len['len1'] == l].index
        temp = df.loc[df[user_id].isin(uids)].copy()
        
        uids = user_len.loc[user_len['len0'] == l].index
        
        if len(uids) > 0:
            uid = uids[0]
        else:
            l = temp[user_id].value_counts().max()
            uid = user_len.loc[user_len['len0'] == l].index[0]

        ts = temp.loc[temp[user_id] == uid, right_ts].copy()
        
        for uid in temp[user_id].unique():
            user_ts = pd.DataFrame({right_ts: ts})
            user_ts[user_id] = uid
            USER_TS.append(user_ts)
    print('> Merging "indexes" with session data...')
    df = pd.concat(USER_TS).merge(df, on=[right_ts, user_id], how='left')
    del USER_TS
    print()
    
    print('Simple feature engineering')
    print('> Computing "weight"...')
    df['weight'] = df[target].notnull().astype(int)
    
    print('> Computing "month"...')
    aux = df[right_ts].dt.day/df[right_ts].dt.daysinmonth
    df['month'] = df[right_ts].dt.month + aux

    if (df[target].unique().size > 2) and p2p:
        print('> Computing "prior_target"...')
        df['prior_target'] = df.groupby(user_id)[target] \
            .transform(pd.Series.shift, p2p)
        df['prior_target'] = df['prior_target'].fillna(0.)

    print('> Filling nulls...')
    for col in fillna_cols + [target, 'prior_target']:
        df[col].fillna(fillna_val, inplace=True)

    for col in keep_first_cols:
        aux = df.loc[df[col].notnull(), [user_id, col]]
        aux.drop_duplicates(subset=user_id, inplace=True)
        aux.set_index(user_id, inplace=True)
        d = aux[col].to_dict()
        df.loc[df[col].isnull(), col] = df[user_id].map(d)
    
    return df