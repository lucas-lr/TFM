import numpy as np
from TFM.settings import get_target_name, get_user_id_name

def build_datasets(df, con_feats, lstm_feats, cat_feats, train_p, p2p,
                   user_id=None, target=None, verbose=True):
    '''
    Returns validation and test sets for performing the out-of-sample and
    out-of-time validation methods.
    Parameters:
     > con_feats continuous features list
     > lstm_feats: LSTM / sequence features list
     > con_feats: continuous features list
     > train_p: proportion of data used for training
     > p2p: periods to predict
    '''

    target = get_target_name(target=target)
    user_id = get_user_id_name(user_id=user_id)
    
    data_valid = []
    data_test = []

    user_len = df[user_id].value_counts().sort_values(ascending=False)
    for length in user_len.unique():
        uids = user_len[user_len == length].index.values
        temp = df.loc[df[user_id].isin(uids)]
        temp_size = len(uids)
        train_size = int(temp_size*train_p)
        valid_size = temp_size - train_size

        temp_i = np.arange(temp_size)
        np.random.shuffle(temp_i)
        train_i = temp_i[:train_size]
        valid_i = temp_i[train_size:]

        if verbose:
            print('> Sequence length: {} | Train / Validation size: {} / {} ({:.1%})' \
                  .format(length, train_size, valid_size, train_size/temp_size))
        y = temp[target].values.reshape(temp_size, length, 1)
        w = (temp['weight']).values.reshape(temp_size, length, 1)
        X_con = temp[con_feats].values.reshape(temp_size, length, len(con_feats))
        X_lstm = temp[lstm_feats].values.reshape(temp_size, length, len(lstm_feats))
        X_cat = [temp[[i]].values.reshape(temp_size, length, 1) for i in cat_feats]

        # validation set (out-of-sample validation)
        data_valid.append({
            'y': y[valid_i, :-p2p, :],
            'w': w[valid_i, :-p2p, 0],
            'X_con': X_con[valid_i, :-p2p, :],
            'X_lstm': X_lstm[valid_i, :-p2p, :],
            'X_cat': [ar[valid_i, :-p2p, :] for ar in X_cat]
        })

        # test set (out-of-time validation)
        data_test.append({
            'y': y[:, :, :],
            'w': w[:, :, 0],
            'X_con': X_con[:, :, :],
            'X_lstm': X_lstm[:, :, :],
            'X_cat': [ar[:, :, :] for ar in X_cat],
            'train_i': train_i
        })
    
    return data_valid, data_test