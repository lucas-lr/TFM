import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def test_score(model, data, only_last=False):
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
        rmse = -1
        mae = -1
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