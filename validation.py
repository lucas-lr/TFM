import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def test_score(model, data, only_last=False, return_y=False):
    y_pred = np.zeros((0))
    y = np.zeros((0))
    if only_last:
        for d in data:
            indexes = d['w'][:, -1:].ravel().nonzero()[0]
            y_pred = np.concatenate((y_pred, model.predict(d['X_cat'] + [
                d['X_con'], d['X_lstm']
            ])[:, -1:, :].ravel()[indexes]))
            y = np.concatenate((y, d['y'][:, -1:, :].ravel()[indexes]))        
    else:
        for d in data:
            indexes = d['w'][:, :].ravel().nonzero()[0]
            y_pred = np.concatenate((y_pred, model.predict(d['X_cat'] + [
                d['X_con'], d['X_lstm']
            ])[:, :, :].ravel()[indexes]))
            y = np.concatenate((y, d['y'][:, :, :].ravel()[indexes]))
    try:
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
    except:
        rmse = -1.
        mae = -1.

    if return_y:
        return rmse, mae, y, y_pred

    return rmse, mae


def get_shuffle_train(data_test, bs, p2p, noise_factor=None):
    data = []
    for d in data_test:
        train_size = len(d['train_i'])
        np.random.shuffle(d['train_i'])
        for i in range(0, train_size, bs):
            train_i = d['train_i'][i:min(train_size, i + bs)]
            
            if len(train_i) < (bs/2):
                continue

            y_train = d['y'][train_i, :-p2p, :]
            if noise_factor is not None:
                noise = y_train.copy()
                noise *= (noise_factor*((np.random.randint(0, high=201,
                    size=y_train.shape) - 100)/100))
                y_train += noise
            
            data.append({
                'y': y_train,
                'w': d['w'][train_i, :-p2p],
                'X_con': d['X_con'][train_i, :-p2p, :],
                'X_lstm': d['X_lstm'][train_i, :-p2p, :],
                'X_cat': [ar[train_i, :-p2p, :] for ar in d['X_cat']]})
    
    return data


def get_last_period_result(y, score, seq_len):
    '''
    Returns the result dataframe (y and score) for the last observation of
    each user. It only works if the length of all sequences is the same.
    '''

    def np_flatten(obj):
        if isinstance(obj, list):
            return np.concatenate(obj)
        else:
            return obj
    y = np_flatten(y)
    score = np_flatten(score)

    assert len(y) == len(score)
    result = pd.DataFrame({
        'y': y, 'score': score
    })
    idx = []
    for i in range(len(result)):
        if (i + 1)%seq_len == 0:
            idx.append(i)
    return result.loc[idx]

def run_test(i_batch_test, ps, num_model):
    '''
    i_batch_test is the list of the ids that compose the test
    num_model to store the model
    ps positive size, steps
    Returns test y and scores, getting the dataframe in batches.
    '''

    res_test =  []
    result_scores = []

    print(len(i_batch_test))
    positive_size = ps
    for i in range(0, len(i_batch_test), positive_size):
        start = dt.datetime.now()
        print('Iteration number: ', i)
        i_batch = i_batch_test[i:min(len(i_batch_test), i + positive_size)]

        df_test = get_df(i_batch, verbose = False, test = True)

        print('Transform features has started... ')
        temp, feat_dict = transform_features(df_test, con_cols, lstm_list, cat_cols, verbose=False)
        lstm_feats = feat_dict['lstm_feats']
        con_feats = feat_dict['con_feats']
        cat_feats = feat_dict['cat_feats']
        M = feat_dict['M']
        print('Transform features has finished... ')

        del df_test
        #temp = R[R['id'].isin(i_batch)]
        #break
        test_x_n = temp[lstm_feats].values.reshape(len(i_batch), 48, len(lstm_feats))
        test_y = temp['target'].values.reshape(len(i_batch), 48, 1)
        test_w = (temp['weight']).values.reshape(len(i_batch), 48)
        test_x_cc = [temp[[i]].values.reshape(len(i_batch), 48, 1) for i in cat_feats]
        test_x_con= temp[con_feats].values.reshape(len(i_batch), 48, len(con_feats))

        scores_test = model.predict(test_x_cc + [test_x_con, test_x_n]).ravel()

        y_test = test_y.ravel()
        w_test = test_w.ravel()
        y_test = y_test[w_test > 0]

        res_test.append(y_test)
        result_scores.append(scores_test)
        print('Batch computation time: ',dt.datetime.now() - start)

    joblib.dump(res_test, 'MODEL'+num_model+'_ALL/ALL_res_test_model'+num_model+'.pickle', compress = 3)
    joblib.dump(result_scores, 'MODEL'+num_model+'_ALL/ALL_scores_test'+num_model+'.pickle', compress = 3)
