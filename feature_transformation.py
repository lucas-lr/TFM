from sklearn.preprocessing import LabelEncoder, Imputer, MinMaxScaler
from TFM.feature_engineering import clip_continuous_f


def has_nulls(s):
    return s.isnull().values.any()


def is_binary(s):
    return sorted(s.unique()) == [0, 1]


def transform_features(df0, con_cols=[], lstm_cols=[], cat_cols={},
    strategy='median', drop_cols=True, verbose=True):
    '''
    Returns:
     > dataframe with transformed features
     > feat_dict with lists of continuous, LSTM and categorical features and M
    Optionally, it applies 2 transformers to the continuous and LSTM features:
     > Imputer()
     > MinMaxScaler()
    Clips LSTM features before fitting the MinMaxScaler().
    Applies a label encoder to the categorical columns.
    Parameters:
     > con_cols: list of continuous columns
     > lstm_cols: list / dict of LSTM continuous columns (keys) and their lower and
       upper clip values (values).
     > cat_cols: dict of categorical columns (keys) and
       the number of features to be returned.
     > strategy: used in the Inputer() transformer.
     > drop_cols: if True, the original columns are excluded.
    '''
    
    df = df0.copy()

    # Preprocessing numerical features
    if verbose:
        print('> Preprocessing continuous features...')
    con_features = []
    for f in con_cols:
        if has_nulls(df[f]):
            t1 = Imputer(strategy=strategy)
            df['con__' + f] = t1.fit_transform(df[[f]])
        else:
            df['con__' + f] = df[f]
        if not is_binary(df[f]):
            t2 = MinMaxScaler()
            df['con__' + f] = t2.fit_transform(df[['con__' + f]])
        con_features.append('con__' + f)
        if verbose:
            print('  > {}'.format(f))

    if verbose:
        print()
        print('> Preprocessing LSTM continuous features...')
    lstm_features = []

    clip = isinstance(lstm_cols, dict)
    if clip:
        cols = lstm_cols.keys()
    else:
        cols = lstm_cols
    for f in cols:
        if has_nulls(df[f]):
            t1 = Imputer(strategy=strategy)
            df['con__' + f] = t1.fit_transform(df[[f]])
        else:
            df['con__' + f] = df[f]
        if is_binary(df[f]):
            if clip:
                s = df[['con__' + f]].clip(lstm_cols[f][0],
                                           lstm_cols[f][1])
            else: s = df[['con__' + f]]
            t2 = MinMaxScaler()
            t2.fit(s)
            df['con__' + f] = t2.transform(df[['con__' + f]])

        lstm_features.append('con__' + f)
        if verbose:
            print('  > {}'.format(f))

    # Preprocessing categorical features
    if verbose:
        print()
        print('> Preprocessing categorical features...')
    cat_features = []
    M = []

    for f in cat_cols.keys():
        le = LabelEncoder()
        df['cat__' + f] = le.fit_transform(df[f])
        m = df['cat__' + f].max() + 1
        cat_features.append('cat__' + f)
        M.append((m, cat_cols[f]))
        if verbose:
            print('  > {}'.format(f))
    
    if drop_cols:
        df.drop(columns=con_cols + list(lstm_cols.keys()) + list(cat_cols.keys()),
                inplace=True)
    
    feat_dict = {
        'con_feats': con_features,
        'lstm_feats': lstm_features,
        'cat_feats': cat_features,
        'M': M
    }

    return df, feat_dict