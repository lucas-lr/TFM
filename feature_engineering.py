import pandas as pd
import matplotlib.pyplot as plt

from TFM.settings import get_right_ts_name, get_target_name, get_user_id_name


def get_decimal_month(dt_series):
    aux = dt_series.dt.day/dt_series.dt.daysinmonth
    return dt_series.dt.month + aux - 1


def add_prior_target(df, p2p, target=None, user_id=None):
    '''
    Adds the prior_target column to the input dataframe. Parameters:
     > p2p: periods to predict. Examples: predict the sum of the expense of
       each user in the next [4] weeks.
    '''

    target = get_target_name(target=target)
    user_id = get_user_id_name(user_id=user_id)

    df['prior_target'] = df.groupby(user_id)[target] \
        .transform(pd.Series.shift, p2p).fillna(0.)


def fillna_cols(df0, user_id=None, fillna_val={}, ffill_cols=[], bfill_cols=[],
                verbose=True):
    '''
    Fills null values in a dataframe by using the chosen method.
    Parameters:
     > fillna_cols: dict with col name (key) and fillna value (value).
     > ffill_cols: fills null values using the forward method per user.
     > bfill_cols: fills null values using the backward method per user.
    '''

    user_id = get_user_id_name(user_id=user_id)

    df = df0.copy()

    for col, val in fillna_val.items():
        if col in df.columns:
            df[col].fillna(value=val, inplace=True)

    for col in ffill_cols:
        if col in df.columns:
            df[col] = df.groupby(user_id, sort=False)[col] \
                .apply(lambda x: x.ffill())

    for col in bfill_cols:
        if col in df.columns:
            df[col] = df.groupby(user_id, sort=False)[col] \
                .apply(lambda x: x.bfill())

    return df


def clip_continuous_f(se0, q2cl=0, q2cu=1, neg_val=True, return_se1=False,
                      return_lu=False, show_hist=False, bins=20):
    '''
    Returns a clipped continuous feature.
    Alternatively, it returns the lower and upper clip values.
    Parameters:
     > se0: original pandas series.
     > q2cl: quantile value used to lower clip the feature.
     > q2cu: quantile value used to upper clip the feature.
       By default, no clipping takes place.
     > neg_val: if True, negative values are coherent for col.
     > return_se1: returns the non-zero clipped feature as a 2nd output.
     > return_lu: returns the lower and upper clip values.
     > show_hist: plots the feature histogram for its non-zero values.
     > bins: number of bins in the histogram.
    '''

    se1 = se0[se0 != 0].copy()
    l = se1.quantile(q=q2cl)
    u = se1.quantile(q=q2cu)
    if not neg_val:
        l = 0
    se0_clip = se0.clip(lower=l, upper=u)
    se1.clip(lower=l, upper=u, inplace=True)

    # histogram
    if show_hist:       
        fig, ax = plt.subplots(1, 1)
        ax0 = se1.hist(ax=ax, bins=bins)
        ax0.set_title('User-periods with feature != 0')
        ax0.set_xlabel('(Clipped) feature')
        ax0.set_ylabel('User-periods');
    
    if return_lu:
        return (l, u)
    
    if return_se1:
        return se0_clip, se1
    
    del se1
    return se0_clip


def trend_enrichment(df0, col, right_ts=None, q2cl=0, q2cu=1, neg_val=True,
                     drop_clip=True, show_plot=True, bins=20):
    '''
    Enriches dataset with global trend data for the chosen
    continuous feature.
    
    Parameters:
     > q2cl: quantile value used to lower clip the feature.
     > q2cu: quantile value used to upper clip the feature.
       By default, no clipping takes place.
     > neg_val: If True, negative values are coherent for col.
     > drop_clip: If True, the clipped col is excluded from the resulting
       dataset.
    
    The added columns are:
     > col_clip: clipped col (optional).
     > col_mean: global mean of col per period.
     > col_mean_diff: difference of the global mean between periods.
    '''

    right_ts = get_right_ts_name(right_ts=right_ts)

    df = df0.copy()

    # clip continuous feature
    df[col + '_clip'], se1 = clip_continuous_f(df[col], q2cl=q2cl, q2cu=q2cu,
                                               neg_val=neg_val,
                                               return_se1=True)

    # compute trend
    active_users = df.loc[df['weight'] == 1][right_ts].value_counts()
    global_col = df.loc[df['weight'] == 1] \
        .groupby(right_ts)[col + '_clip'].sum()
    trend = pd.concat([
        active_users, global_col
    ], axis=1).sort_index()
    trend.columns = ['users', col]
    trend[col + '_mean'] = trend[col]/trend['users']
    trend[col + '_mean_diff'] = trend[col + '_mean'].diff().fillna(0)

    for c in [col + '_mean', col + '_mean_diff']:
        d = trend[c].to_dict()
        df[c] = df[right_ts].map(d)

    if drop_clip:
        df.drop(columns=col + '_clip', inplace=True)

    if show_plot:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        ax0 = se1.hist(ax=axs[0], bins=bins)
        ax0.set_title('User-periods with {} != 0'.format(col))
        ax0.set_xlabel('(Clipped) {}'.format(col))
        ax0.set_ylabel('User-periods');

        ax1 = trend[col + '_mean'].plot(ax=axs[1])
        ax1.set_title('Periodic {} per user'.format(col))
        ax1.set_ylabel('(Clipped) {}'.format(col))

    del se1, active_users, global_col, trend, d
    return df