# -*- coding: utf-8 -*-
"""
Created on Sun May 13 18:31:22 2018

@author: Alexandre Dos Santos

Data preparation, the code is greatly based on Bert Carremans Data Preparation 
& Exploration Kernel avaible at https://www.kaggle.com/bertcarremans/data-preparation-exploration
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', 100)

class prepare_data():
    def __init__(self, train, test):
        super(prepare_data, self).__init__()
        
        if type(train) == str:
            if os.path.exists(train):
                try:
                    self.train = pd.read_csv(train)
                except:
                    print('The train data is not readable')
                    raise
            else:
                print('The train path is wrong')
                
        elif not isinstance(train, pd.DataFrame):
            print('train is not a DataFrame')
            raise
        else:
            self.train = train

        if type(test) == str:
            if os.path.exists(test):
                try:
                    self.test = pd.read_csv(test)
                except:
                    print('The test data is not readable')
                    raise
            else:
                print('The test path is wrong')
                
        elif not isinstance(test, pd.DataFrame):
            print('test is not a DataFrame')
            raise
        else:
            self.test = test
        
        shape_before = self.train.shape
        self.train.drop_duplicates()
        print('%d duplicates dropped.' % (shape_before[0] - self.train.shape[0]))
        
        self.meta = self._create_meta(self.train)
        
    def show_data_types(self):
        print(pd.DataFrame({'count' : self.meta.groupby(['role', 'level'])['role'].size()}).reset_index())
    
    def show_stats(self):
        for data_type in np.unique(self.meta.level.values):
            print('Data of type %s :' % data_type)
            v = self.meta[(self.meta.level == data_type) & (self.meta.keep)].index
            print(self.train[v].describe())
            print('='*15)
    
    def show_corr(self, f):
        v = self.meta[(self.meta.level == f) & (self.meta.keep)].index
        print(v)
        v = v.append(pd.Index(['target']))
        print(v)
        data = self.train[v]
        self._corr_heatmap(data)
    
    def __call__(self, under_sample=False, use_feature_selection=False):
        train = self.train.copy()
        test = self.test.copy()
        meta = self._create_meta(train)
        
        if under_sample:
            print()
            print('Under-sampling...')
            desired_apriori=0.10

            # Get the indices per target value
            idx_0 = self.train[self.train.target == 0].index
            idx_1 = self.train[self.train.target == 1].index
            
            # Get original number of records per target value
            nb_0 = len(self.train.loc[idx_0])
            nb_1 = len(self.train.loc[idx_1])
            
            # Calculate the undersampling rate and resulting number of records with target=0
            undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
            undersampled_nb_0 = int(undersampling_rate*nb_0)
            print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
            print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))
            
            # Randomly select records with target=0 to get at the desired a priori
            undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)
            
            # Construct list with remaining indices
            idx_list = list(undersampled_idx) + list(idx_1)
            
            # Return undersample data frame
            del train
            train =self.train.loc[idx_list].reset_index(drop=True)
        
        # DATA QUALITY CHECK
        
        # Checking missing values
        print()
        print('Checking missing values...')
        vars_with_missing = []
        vars_to_drop = []

        for f in train.columns:
            missings = train[train[f] == -1][f].count()
            if missings > 0:
                vars_with_missing.append(f)
                missings_perc = missings/train.shape[0]
                if missings_perc > 0.4:
                    vars_to_drop.append(f)
                print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))
                
        print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))
        
        print()
        print('Dropping variables with more than 40% of missing values...')
        # Dropping the variables with too many missing values
        if len(vars_to_drop) > 0:
            print('Dropping %s' % (', '.join(vars_to_drop)))
            train.drop(vars_to_drop, inplace=True, axis=1)
            test.drop(vars_to_drop, inplace=True, axis=1)
            meta.loc[(vars_to_drop),'keep'] = False  # Updating the meta
        
        print()
        print('Replacing missing values...')
        # Imputing with the mean or mode
        if len(vars_with_missing) > 0:
            mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
            mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
            
            for f in vars_with_missing:
                if f in meta[(meta.level == 'interval')].index:
                    print('Replacing missing values of %s by mean' % f)
                    train[f] = mean_imp.fit_transform(train[[f]]).ravel()
                elif f in meta[(meta.level == 'ordinal')].index:
                    print('Replacing missing values of %s by mode' % f)
                    train[f] = mode_imp.fit_transform(train[[f]]).ravel()
        
        print()
        print('Checking the cardinality of the categorical variables...')
        # Checking the cardinality of the categorical variables
        v = meta[(meta.level == 'nominal') & (meta.keep)].index
        
        high_values_vars = []
        for f in v:
            dist_values = train[f].value_counts().shape[0]
            print('Variable {} has {} distinct values'.format(f, dist_values))
            if dist_values > 40:
                high_values_vars.append(f)
        
        for f in high_values_vars:
            train_encoded, test_encoded = self._target_encode(train[f],
                                                              test[f],
                                                              target=train.target,
                                                              min_samples_leaf=100,
                                                              smoothing=10,
                                                              noise_level=0.01)
    
            train[f] = train_encoded
            train.drop(f, axis=1, inplace=True)
            meta.loc[f,'keep'] = False  # Updating the meta
            test[f] = test_encoded
            test.drop(f, axis=1, inplace=True)
        
        # FEATURE ENGINEERING
        
        targets = train.target
        train.drop('target', inplace=True, axis=1)
        
        # Dummification
        print()
        print('Dummification...')
        v = meta[(meta.level == 'nominal') & (meta.keep)].index
        print('Before dummification we have {} variables in train'.format(train.shape[1]))
        train = pd.get_dummies(train, columns=v, drop_first=True)
        print('After dummification we have {} variables in train'.format(train.shape[1]))
        print('Before dummification we have {} variables in test'.format(test.shape[1]))
        test = pd.get_dummies(test, columns=v, drop_first=True)
        print('After dummification we have {} variables in test'.format(test.shape[1]))
        
        # Interaction variables
        print()
        print('Creating interaction variables...')
        v = meta[(meta.level == 'interval') & (meta.keep)].index
        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        interactions = pd.DataFrame(data=poly.fit_transform(train[v]), columns=poly.get_feature_names(v))
        interactions.drop(v, axis=1, inplace=True)  # Remove the original columns
        # Concat the interaction variables to the train data
        print('Before creating interactions we have {} variables in train'.format(train.shape[1]))
        train = pd.concat([train, interactions], axis=1)
        print('After creating interactions we have {} variables in train'.format(train.shape[1]))
        print('Before creating interactions we have {} variables in test'.format(test.shape[1]))
        test = pd.concat([test, interactions], axis=1)
        print('After creating interactions we have {} variables in test'.format(test.shape[1]))
        
        # Feature selection
        if use_feature_selection:
            print()
            print('Random Forest feature selection...')
            X_train = train.drop(['id', 'target'], axis=1)
            y_train = train['target']
            
            feat_labels = X_train.columns
            
            rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, verbose=10)
            
            rf.fit(X_train, y_train)
            importances = rf.feature_importances_
            
            indices = np.argsort(rf.feature_importances_)[::-1]
            
            for f in range(X_train.shape[1]):
                print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], importances[indices[f]]))
            
            sfm = SelectFromModel(rf, threshold='median', prefit=True)
            print('Number of features before selection: {}'.format(X_train.shape[1]))
            n_features = sfm.transform(X_train).shape[1]
            print('Number of features after selection: {}'.format(n_features))
            selected_vars = list(feat_labels[sfm.get_support()])
            
            train = train[selected_vars + ['target']]
        
        return train, targets, test
        
    def _create_meta(self, train):
        data = []
        for f in train.columns:
            # Defining the role
            if f == 'target':
                role = 'target'
            elif f == 'id':
                role = 'id'
            else:
                role = 'input'
                 
            # Defining the level
            if 'bin' in f or f == 'target':
                level = 'binary'
            elif 'cat' in f or f == 'id':
                level = 'nominal'
            elif train[f].dtype == 'float64':
                level = 'interval'
            elif train[f].dtype == 'int64':
                level = 'ordinal'
            else:
                print('Error')
            # Initialize keep to True for all variables except for id
            keep = True
            if f == 'id':
                keep = False
            
            # Defining the data type 
            dtype = train[f].dtype
            
            # Creating a Dict that contains all the metadata for the variable
            f_dict = {
                'varname': f,
                'role': role,
                'level': level,
                'keep': keep,
                'dtype': dtype
            }
            data.append(f_dict)
        
        meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
        meta.set_index('varname', inplace=True)
        
        return meta
    
    # Script by https://www.kaggle.com/ogrellier
    # Code: https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
    def _add_noise(self, series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))
    
    def _target_encode(self,
                       trn_series=None,
                       tst_series=None,
                       target=None,
                       min_samples_leaf=1,
                       smoothing=1,
                       noise_level=0):
        """
        Smoothing is computed like in the following paper by Daniele Micci-Barreca
        https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
        trn_series : training categorical feature as a pd.Series
        tst_series : test categorical feature as a pd.Series
        target : target data as a pd.Series
        min_samples_leaf (int) : minimum samples to take category average into account
        smoothing (int) : smoothing effect to balance categorical average vs prior  
        """ 
        assert len(trn_series) == len(target)
        assert trn_series.name == tst_series.name
        temp = pd.concat([trn_series, target], axis=1)
        # Compute target mean 
        averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
        # Compute smoothing
        smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
        # Apply average function to all target data
        prior = target.mean()
        # The bigger the count the less full_avg is taken into account
        averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(["mean", "count"], axis=1, inplace=True)
        # Apply averages to trn and tst series
        ft_trn_series = pd.merge(
            trn_series.to_frame(trn_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=trn_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_trn_series.index = trn_series.index 
        ft_tst_series = pd.merge(
            tst_series.to_frame(tst_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=tst_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_tst_series.index = tst_series.index
        return self._add_noise(ft_trn_series, noise_level), self._add_noise(ft_tst_series, noise_level)
    
    def _corr_heatmap(self, data):
        correlations = data.corr()

        # Create color map ranging between two colors
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',
                    square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})
        plt.show();
        
        
if __name__ == "__main__":
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    
    prep = prepare_data(train,test)
    print(prep.meta)
    prep.show_data_types()
    prep.show_stats()
    prep.show_corr('interval')
    prep.show_corr('ordinal')
    
    train, target, test = prep(True,False)