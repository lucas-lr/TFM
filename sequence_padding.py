import pandas as pd
from TFM.settings import get_right_ts_name, get_target_name, get_user_id_name


def presequence_padding(df0, val, right_ts=None, user_id=None, target=None,
                        verbose=True):
    '''
    Transforms data such that each sequence has a length equal to a multiple of
    a choosen value.
    '''
    
    right_ts = get_right_ts_name(right_ts=right_ts)
    target = get_target_name(target=target)
    user_id = get_user_id_name(user_id=user_id)
    
    df = df0.copy()

    unique_ts = sorted(df[right_ts].unique())
    d = dict(zip(unique_ts, range(len(unique_ts))))
    ts_list = sorted(list(d.keys()))
    df['time_i'] = df[right_ts].map(d)
    
    user_len = df[user_id].value_counts().to_frame(name='len0')
    u = user_len['len0'].max()
    user_len['len1'] = (val*(((user_len['len0'] - 1)//val) + 1)).clip_upper(u)

    if verbose:
        print('Padding sequence lengths...')
        print('> Sequence length multiple: {}'.format(val))
        fs = user_len['len0'].unique().size
        print('> Initial sequence lengths: {}'.format(fs))
        fs = user_len['len1'].unique().size
        print('> Final sequence lengths:   {}'.format(fs))
        print('> Creating "indexes" (user_id & time-stamp)...')
    
    USER_TS = []
    
    for length in sorted(user_len['len1'].unique()):
        uids = user_len.loc[user_len['len1'] == length].index
        temp = df.loc[df[user_id].isin(uids)].copy()
        
        for uid in uids:
            mti = temp.loc[temp[user_id] == uid, 'time_i'].max()
            ts = ts_list[(mti - length + 1):(mti + 1)]
            user_ts = pd.DataFrame({user_id: uid, right_ts: ts})
            USER_TS.append(user_ts)
    
    if verbose:
        print('> Merging "indexes" with session data...')
    df = pd.concat(USER_TS).merge(df, on=[right_ts, user_id], how='left')
    
    df.drop(columns=['time_i'], inplace=True)
    df['weight'] = df[target].notnull().astype(int)
    del USER_TS
    
    return df