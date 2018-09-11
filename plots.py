import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_binary_roc_curve(y_test, y_score):
       
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_adjustment(result, target_name='y', score_name='score', groups=10,
                    lift=False, display_table=False):
    n = np.ceil(len(result)/groups)
    result.sort_values(score_name, inplace=True)
    result.reset_index(drop=True, inplace=True)
    result['group'] = np.floor((result.index)/n).astype(int) + 1
    groups = result.groupby('group')
    grouped_R = groups.mean()
    grouped_R['size'] = groups.size()
    
    ymean = grouped_R[target_name].mean()
    smean = grouped_R[score_name].mean()
    
    grouped_R['real_lift'] = grouped_R[target_name] / ymean
    grouped_R['pred_lift'] = grouped_R[score_name] / smean
    
    if lift:
        grouped_R[['pred_lift', 'real_lift']].plot(marker='.')
    else:
        grouped_R[[score_name, target_name]].plot(marker='.')
    
    if display_table:
        display(grouped_R)

    plt.xlabel('User groups')
    if lift:
        plt.ylabel('Lift')
    else:
        plt.ylabel('Target mean')
    plt.legend(['Prediction', 'Reality'])
    plt.grid()
    plt.show()