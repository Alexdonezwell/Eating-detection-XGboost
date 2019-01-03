import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from beyourself.core.util import maybe_create_folder
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from IPython.display import display, HTML
from sklearn import preprocessing
from beyourself.evaluation.ml import metrics_evaluate

from datetime import timedelta, date

import pandas as pd
import datetime


start_date_map = {  "P103": date(2017,6,19),
                    "P105": date(2017,6,23),
                    "P107": date(2017,7,12),
                    "P108": date(2017,8,3),
                    "P110": date(2017,8,4),
                    "P114": date(2017,8,9),
                    "P116": date(2017,8,11),
                    "P118": date(2017,8,18),
                    "P120": date(2017,8,23),
                    "P121": date(2017,8,24)}


def standardize_zscore(data, selected_columns):
    """normalize the whole dataset to make the system cleaner
    only normalize for columns belong to columns list
    Parameters
    ----------
        data:               dataFrame
        selected_columns:   list of columns which will be standardized
    Return
    ------
        dataZ           dataFrame
    """

    dataZ = data.copy()

    for col_header in selected_columns:
        dataZ[col_header] = zscore(dataZ[col_header])

    return dataZ

subj_list = ['P103','P105','P107','P108','P110','P114','P116','P118','P120','P121']

for subj in subj_list:

    print("************************")

    print("Testing on {}".format(subj))

    personalized_generalized_folder = '../data/wild/{}/person_general'.format(subj)
    maybe_create_folder(personalized_generalized_folder)


    df = pd.read_csv('../data/wild/{}/feature.csv'.format(subj))
    exclusion = ['label','date_exp','start','end']
    feature_names = [f for f in df.columns.values if not f in exclusion] 
    selected_columns_normalization = feature_names[:-5]

    concat_list = []
    for outside in subj_list:
        if outside == subj:
            print("Excluding {}".format(outside))
            continue
        df_outside = pd.read_csv('../data/wild/{}/feature.csv'.format(outside))
        concat_list.append(df_outside)

    df_generalized = pd.concat(concat_list)
    X_train_generalized = df_generalized[feature_names].values
    Y_train_generalized = df_generalized['label'].values

    # Leave one day out validation
    day = []
    day_index = []
    precision = []
    recall = []
    fscore = []

    precision_neg = []
    recall_neg = []
    fscore_neg = []


    start_date = start_date_map[subj]
    cm_total = np.zeros((2,2))

    for fold in range(14):
        print("=====================================================================================")
        print(fold)
        df_train = df[df['date_exp'] != fold]
        df_test = df[df['date_exp'] == fold]

        if df_test.shape[0] == 0:
            print("Skipping the fold {}".format(fold))
            continue
     
        X_train_subj = df_train[feature_names].values
        Y_train_subj = df_train['label'].values
        
        X_train = np.vstack((X_train_subj, X_train_generalized))
        Y_train = np.concatenate([Y_train_subj, Y_train_generalized])

        X_test = df_test[feature_names].values
        Y_test = df_test['label'].values

        std_scale = preprocessing.StandardScaler().fit(X_train)
        normalized_train = std_scale.transform(X_train)
        normalized_test = std_scale.transform(X_test)

        dtrain = xgb.DMatrix(normalized_train,label=Y_train)
        dtest = xgb.DMatrix(normalized_test,label=Y_test)
        param = {}
        param['objective'] = 'multi:softmax'
        param['lambda'] = 0.8
        param['eta'] = 0.05
        param['max_depth'] = 1
        param['subsample']= 0.8
        param['min_child_weight'] = 5
        param['silent'] = 0 
        param['num_class']=2
        param['nthread'] = 8
        
        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        bst = xgb.train(param, dtrain, evals=watchlist,num_boost_round=1000,early_stopping_rounds=200, maximize=False, verbose_eval=20)


        prediction = bst.predict(dtest)
        
        # save prediction along with features
        df_results = df_test.copy()
        df_results['prediction'] = prediction
        df_results.to_csv(os.path.join(personalized_generalized_folder, "prediction_day_{}.csv").format(fold))

        
        cm = confusion_matrix(Y_test, prediction, labels=[0,1])
        print(cm)
        
        cm_total = cm_total + cm

        np.savetxt(os.path.join(personalized_generalized_folder, "day_{}.csv".format(fold)), cm, fmt='%d')
        
        pr, rc, f1, support = precision_recall_fscore_support(Y_test, prediction, average=None, labels=[0, 1])
        
        print("Precision {} Recall {} F1 {}".format(pr[1], rc[1], f1[1]))

        day_index.append(fold)
        day.append(start_date + timedelta(days=fold))
        precision.append(pr[1])
        recall.append(rc[1])
        fscore.append(f1[1])
        precision_neg.append(pr[0])   
        recall_neg.append(rc[0]) 
        fscore_neg.append(f1[0])     


    df_result = pd.DataFrame({'day_index':day_index,'day':day,'precision':precision,'recall':recall,'fscore':fscore,'neg_precision':precision_neg,'neg_recall':recall_neg,"neg_fscore":fscore_neg},\
                        columns=['day_index','day','precision','recall','fscore','neg_precision','neg_recall','neg_fscore'])
    df_result.set_index('day_index', inplace=True)

    pr_total = cm_total[1,1]/(cm_total[1,1] + cm_total[0,1])
    rc_total = cm_total[1,1]/(cm_total[1,1] + cm_total[1,0])
    fscore_total = 2*pr_total*rc_total/(pr_total + rc_total)
    pr_neg_total = cm_total[0,0]/(cm_total[0,0] + cm_total[1,0])
    rc_neg_total = cm_total[0,0]/(cm_total[0,0] + cm_total[0,1])
    fscore_neg_total = 2*pr_neg_total*rc_neg_total/(pr_neg_total + rc_neg_total)

    np.savetxt(os.path.join(personalized_generalized_folder, "cm_total.csv".format(fold)), cm_total, fmt='%d')

    df_result.loc['Ave'] = ['Ave', pr_total, rc_total, fscore_total,\
                            pr_neg_total,rc_neg_total,fscore_neg_total]

    print ('Average F-score is:', df_result['fscore'].loc['Ave'])
    df_result.to_csv(os.path.join(personalized_generalized_folder, '{}.csv'.format(subj)))

    # convert to latex string
    latex = ""
    for i in range(df_result.shape[0]):
        for j in range(df_result.shape[1]):
            item = df_result.iloc[i][j]
            if isinstance(item, str):
                latex += item
            elif isinstance(item, datetime.date):
                latex += item.strftime("%m-%d")
            else:
                latex += "{:.1f}".format(100*item)
            if j != df_result.shape[1] - 1:
                latex += " & "
        latex += "\\\\"
        latex += "\n"
    print(latex)

    with open(os.path.join(personalized_generalized_folder, 'latex.txt'), 'w') as f:
        f.write(latex)
