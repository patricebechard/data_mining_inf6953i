import xgboost as xgb
from sklearn.utils import shuffle
from utils.data import prepare_data
from utils.gini import eval_gini, gini_xgb
import numpy as np
import pandas as pd
import xgboost as xgb
import time

data_path='./input'

train = pd.read_csv(data_path+'/train.csv')
test = pd.read_csv(data_path+'/test.csv')

prep = prepare_data(train,test)
train, targets, test = prep(True,False)

X, y = train.as_matrix()[:,1:], targets.as_matrix()
X, y = shuffle(X,y)
cutoff = int(len(X)*0.9)

train_X, train_y = X[:cutoff], y[:cutoff]
X_test, y_test = X[cutoff:], y[cutoff:]
X_sub = test.as_matrix()[:,1:]

del X, y, train, targets, test


param = {'max_depth':5, 'objective':'binary:logistic', 'subsample':0.8, 
         'colsample_bytree':0.8, 'eta':0.5, 'min_child_weight':1,
         'tree_method':'gpu_hist'}
num_round = 100

dtrain = xgb.DMatrix(train_X, train_y)
tic = time.time()
model = xgb.train(param, dtrain, num_round)
print('passed time with xgb (gpu): %.3fs'%(time.time()-tic))

xgb_param = {'max_depth':5, 'objective':'binary:logistic', 'subsample':0.8, 
         'colsample_bytree':0.8, 'learning_rate':0.5, 'min_child_weight':1,
         'tree_method':'gpu_hist'}
model = XGBClassifier(**xgb_param)
tic = time.time()
model.fit(train_X, train_y)
print('passed time with XGBClassifier (gpu): %.3fs'%(time.time()-tic))

param = {'max_depth':5, 'objective':'binary:logistic', 'subsample':0.8, 
         'colsample_bytree':0.8, 'eta':0.5, 'min_child_weight':1,
         'tree_method':'hist'}
num_round = 100

dtrain = xgb.DMatrix(train_X, train_y)
tic = time.time()
model = xgb.train(param, dtrain, num_round)
print('passed time with xgb (cpu): %.3fs'%(time.time()-tic))

xgb_param = {'max_depth':5, 'objective':'binary:logistic', 'subsample':0.8, 
         'colsample_bytree':0.8, 'learning_rate':0.5, 'min_child_weight':1,
         'tree_method':'hist'}
model = XGBClassifier(**xgb_param)
tic = time.time()
model.fit(train_X, train_y)
print('passed time with XGBClassifier (cpu): %.3fs'%(time.time()-tic))