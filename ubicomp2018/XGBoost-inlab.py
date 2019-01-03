
# coding: utf-8

# In[2]:


import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression

from sklearn.metrics import confusion_matrix
from IPython.display import display, HTML
from datetime import timedelta, date

import pandas as pd


# In[3]:


def calculate_confusion_matrix(m):
  #  accuracy = float(m[0, 0] + m[1, 1]) / m.sum()
    recall=m[1,1]/(m[1,0]+m[1,1]+10e-4)
    #true_positive = float(m[1, 1]) / (m[1, 0] + m[1, 1])
   # false_positive = float(m[0, 1]) / m[0, :].sum()
    precision = m[1, 1] / (m[0,1]+m[1,1]+10e-4)
    F1=2*precision*recall/(precision+recall)
    m1 = np.array([recall,precision,F1])
    return m1


# In[4]:


df = pd.read_csv('/Users/apple/Downloads/necklace_inlab/ubicomp2018/beyourself/CLEAN/Label_IS/inlab_feature.csv')

exclusion = ['label','subj','start','end']
feature_names = [f for f in df.columns.values if not f in exclusion] 
# feature_names = [f for f in df.columns.values if not f in exclusion and not 'fft' in f] 


# In[ ]:


print(len(feature_names))

display(feature_names)


# In[ ]:


# Leave one subject out validation
subj_index = []
precision = []
recall = []
fscore = []

precision_neg = []
recall_neg = []
fscore_neg = []

subj_list = ["P000", "P106", "P108", "P110", "P111", "P113", "P116", "P119", "P120", "P121"]
for subj in subj_list:
    print("=====================================================================================")
    print(subj)
    
    df_train = df[df['subj'] != subj]
    df_test = df[df['subj'] == subj]
    
    print(df_train.shape)
    print(df_test.shape)

    if df_test.shape[0] == 0:
        print("Skipping the subj {}".format(subj))
        continue
 
    X_train = df_train[feature_names].values
    Y_train = df_train['label'].values

    X_test = df_test[feature_names].values
    Y_test = df_test['label'].values

    dtrain=xgb.DMatrix(X_train,label=Y_train)
    dtest=xgb.DMatrix(X_test,label=Y_test)
    param = {}
    param['objective'] = 'multi:softmax'
    param['lambda'] = 0.8
    param['eta'] = 0.05
    param['max_depth'] = 3
    param['subsample']= 0.8
    param['min_child_weight'] = 5
    param['silent'] = 0 
    param['num_class']=2
    param['nthread'] = 8
    
    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    bst = xgb.train(param, dtrain, evals=watchlist,num_boost_round=2000,early_stopping_rounds=200, maximize=False, verbose_eval=20)


    prediction = bst.predict(dtest)
    
    df_results = df_test.copy()
    
    df_results['prediction'] = prediction
    
    df_results.to_csv("/Users/apple/Downloads/necklace_inlab/ubicomp2018/data/{}_prediction.csv".format(subj))
    
    cm = confusion_matrix(Y_test, prediction)
    cm1= confusion_matrix(1-Y_test,1-prediction)
    print(cm)
    if cm.shape[0] == 2:
        
        pr, rc, f1 = calculate_confusion_matrix(cm)
        pr_, rc_, f1_ = calculate_confusion_matrix(cm1)
        
        print("Precision {} Recall {} F1 {}".format(pr, rc, f1))

        subj_index.append(subj)
        precision.append(pr)
        recall.append(rc)
        fscore.append(f1)
        precision_neg.append(pr_)   
        recall_neg.append(rc_) 
        fscore_neg.append(f1_)
    


# In[ ]:


# display(df[['start','end']])


# In[ ]:


df_result = pd.DataFrame({'subj':subj_index,'precision':precision,'recall':recall,'fscore':fscore,                             'negprecision':precision_neg,'negrecall':recall_neg,'negfscore':fscore_neg},                        columns=['subj','precision','recall','fscore','negprecision','negrecall','negfscore'])
display(df_result)
ave_fscore=np.mean(np.asarray(fscore))
print ('Average F-score is:',ave_fscore)


# In[ ]:


latex = ""
for i in range(df_result.shape[0]):
    for j in range(df_result.shape[1]):
        if isinstance(df_result.ix[i][j], str):
            latex += df_result.ix[i][j]
        else:
            latex += "{:.1f}".format(100*df_result.ix[i][j])
        latex += " & "
    latex += "\\\\"
    latex += "\n"
print(latex)


# In[ ]:


df_result.to_csv('/Users/apple/Downloads/necklace_inlab/ubicomp2018/data/inlab_features/result_inlab.csv',index=None)


# In[ ]:


print(df_result.mean())


# In[ ]:


df_result[:-1].mean()

