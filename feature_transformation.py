from sklearn.preprocessing import LabelEncoder, Imputer, MinMaxScaler
from TFM.feature_engineering import clip_continuous_f
from TFM.settings import get_right_ts_name, get_target_name, get_user_id_name, get_code_name
import pandas as pd


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
    If necessary, it applies 2 transformers to the continuous and LSTM features:
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

    # User profile continuous features
    if verbose:
        print('> Preprocessing continuous features...')
    con_features = []
    for f in con_cols:
        if has_nulls(df[f]):
            t1 = Imputer(strategy=strategy)
            df['con__' + f] = t1.fit_transform(df[[f]])
        else:
            df['con__' + f] = df[f]
        if not is_binary(df['con__' + f]):
            t2 = MinMaxScaler()
            df['con__' + f] = t2.fit_transform(df[['con__' + f]])
        con_features.append('con__' + f)
        if verbose:
            print('  > {}'.format(f))

    # LSTM / sequence continuous features
    if verbose:
        print()
        print('> Preprocessing LSTM continuous features...')
    lstm_features = []

    clip = isinstance(lstm_cols, dict)
    if clip:
        lstm_col_list = list(lstm_cols.keys())
    else:
        lstm_col_list = lstm_cols

    for f in lstm_col_list:

        if has_nulls(df[f]):
            t1 = Imputer(strategy=strategy)
            df['con__' + f] = t1.fit_transform(df[[f]])
        else:
            df['con__' + f] = df[f]

        if not is_binary(df['con__' + f]):
            if clip:
                s = df[['con__' + f]].clip(lstm_cols[f][0],
                                           lstm_cols[f][1])
            else:
                s = df[['con__' + f]]
            t2 = MinMaxScaler()
            t2.fit(s)
            df['con__' + f] = t2.transform(df[['con__' + f]])

        lstm_features.append('con__' + f)
        if verbose:
            print('  > {}'.format(f))

    # User profile categorical features
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
        df.drop(columns=con_cols + lstm_col_list + list(cat_cols.keys()),
                inplace=True)

    feat_dict = {
        'con_feats': con_features,
        'lstm_feats': lstm_features,
        'cat_feats': cat_features,
        'M': M
    }

    return df, feat_dict


def get_all_times(events, right_ts=None):

    right_ts = get_right_ts_name(right_ts=right_ts)
    time_index = pd.DataFrame(sorted(events[right_ts].unique()), columns=[right_ts])
    del events
    return time_index


def expand(df, time_index, user_id=None, right_ts=None):
    'Returns: a dataframe with all the missing dates fully with missing data'

    user_id = get_user_id_name(user_id=user_id)
    right_ts = get_right_ts_name(right_ts=right_ts)

    df.reset_index(inplace=True)
    df_expanded = []
    for i in df[user_id].unique():
        ti = time_index.copy()
        ti[user_id] = i
        df_expanded.append(ti)
    df_expanded = pd.concat(df_expanded, axis=0)

    # Merge index with event data
    df_expanded = df_expanded.merge(df, on=[user_id, right_ts], how='left', copy=False)

    del df
    return df_expanded


def adding_columns(chunk, list_type, verbose=True):
    a = chunk.columns.tolist()
    if verbose:
        print('Current number of columns of these type in this chunk: ', len(a))
    disjoint_list = set(list_type) - set(a)
    if verbose:
        print('Columns to add: ', len(disjoint_list))
    i = 0
    for column in disjoint_list:
        chunk[column] = 0.0
        i = i + 1
    del a
    del disjoint_list


def generating_catalegs(events_type, code=None):

    code = get_code_name(code=code)

    list_type = events_type[code].unique().tolist()

    #print('Number of columns of this type: ', len(list_type))
    cancer_mask = events_type[code].str[:3] != 'd_C'
    melanoma = events_type[code].str[:5] == 'd_C44'
    list_type = events_type[cancer_mask | melanoma].groupby([code])[code].size().to_frame('N')

    #print('Number of columns of this type wo cancers: ', len(list_type))

    list_type.reset_index(inplace=True)
    list_type = list_type[code].tolist()
    return list_type


def pivoting_events(events, agg_function, user_id=None, right_ts=None, code=None, verbose=True):

    user_id = get_user_id_name(user_id=user_id)
    right_ts = get_right_ts_name(right_ts=right_ts)
    code = get_code_name(code=code)

    input_data = pd.pivot_table(events, index=[user_id, right_ts], columns=code, aggfunc=agg_function)
    col_et = input_data.columns.tolist()
    input_data.reset_index(inplace=True)
    input_data.set_index([user_id, right_ts], inplace=True)
    input_data.columns = [str(s2) for (s1, s2) in input_data.columns.tolist()]
    if verbose:
        print(input_data.size)
        print(input_data.shape)
    del events
    return input_data
