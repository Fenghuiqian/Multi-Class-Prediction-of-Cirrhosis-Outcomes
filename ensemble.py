#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import warnings 
warnings.filterwarnings('ignore') 
np.random.seed(2023) 
from sklearn.preprocessing import OrdinalEncoder, StandardScaler 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.linear_model import LinearRegression 
import lightgbm as lgb 
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV 
import optuna 
from optuna.integration import LightGBMPruningCallback 
from optuna.samplers import TPESampler 
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, KFold 
import seaborn as sns 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer 
from sklearn.decomposition import PCA 
from sklearn.metrics import log_loss, r2_score, roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, TargetEncoder, LabelBinarizer
from sklearn.metrics import mean_squared_error 
from sklearn.feature_selection import RFECV 
from datetime import datetime 
from tqdm import tqdm 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor,HistGradientBoostingClassifier, GradientBoostingClassifier, ExtraTreesClassifier,ExtraTreesRegressor
from itertools import combinations
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.calibration import CalibratedClassifierCV 
from xgboost import XGBClassifier, XGBRegressor
from fastai.tabular.all import TabularPandas, tabular_learner, cont_cat_split, Categorify, FillMissing, Normalize, F , CategoryBlock 
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
import fastai.metrics as metrics 
from fastai.tabular.all import TabularDataLoaders 
import torch 



def climb(oof_pred, y, test_pred, metric, direction):

    def climb_inner(oof_pred, y, test_pred, starter, metric):
        # initialize climb
        current_ens = oof_pred[starter].copy()
        current_test = test_pred[starter].copy()
        models = oof_pred.copy()
        models.pop(starter)

        weight_range = np.arange(-0.2, 2.01, 0.01)
        # INIT SCORE AS BEST
        best_score = metric(y, current_ens)
        if direction == 'minimize':
            best_score = best_score * (-1) 
        history = [best_score]
        
        climbing = True

        while climbing:

            new_best_found = False
            best_key = None
            best_weight = None

            # 找到最好的那个组合及权重
            for key, model in models.items():
                for weight in weight_range:

                    ens = (1 - weight) * current_ens + weight * model
                    score = metric(y, ens)
                    if direction == 'minimize':
                        score = score * (-1) 
                    if score > best_score:
                        best_score = score
                        best_key = key
                        best_weight = weight
                        new_best_found = True

            if new_best_found:
                current_ens = (1 - best_weight) * current_ens + best_weight * models[best_key]
                current_test = (1 - best_weight) * current_test + best_weight * test_pred[best_key]

                models.pop(best_key)

                if len(models) == 0:
                    climbing = False
                print('found better combination', best_key, 'weight', best_weight) 
                history.append(best_score)
    #             if len(history)>6:
    #                 climbing = False
            else:
                climbing = False
        print('history score', history) 
        diff_history = [j-i for i, j in zip(history[:-1], history[1:])]
        diff_history = [int(i*(1/min(diff_history))) for i in diff_history]
        print('diff', diff_history) 
        return current_ens, current_test, best_score 
    
    score_ens = -float('inf')
    for starter in oof_pred.keys():
        current_ens_, current_test_, best_score_ = climb_inner(oof_pred, y, test_pred, starter, metric)
        if score_ens < best_score_:
            oof_ens, test_ens, score_ens = current_ens_, current_test_, best_score_
    print('best score:', score_ens) 
    return oof_ens, test_ens, score_ens 

def lr_stack(X, Y, test):

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X) 
    if isinstance(Y, np.ndarray):
        Y = pd.DataFrame(Y) 
    params, score_cv = bayesGridSearchCVParams_LR(X, Y, cv=5, n_iter=50, objective='binary', 
#                                              scoring='average_precision',
                                                  scoring='roc_auc')
    print('LR Stacking srch cv score:', score_cv) 
    est = LogisticRegression(**params, n_jobs=-1, random_state=0)   
    est.fit(X, Y) 
    print(est.coef_) 
    oof_pred = est.predict_proba(X)[:, 1] 
    test_pred = est.predict_proba(test)[:, 1] 
#     score = average_precision_score(Y.values.reshape(-1), oof_pred) 
    score = roc_auc_score(Y.values.reshape(-1), oof_pred) 
    print('oof LR Stacking score', score) 
    return oof_pred, test_pred, score_cv, score 


def rf_stack(X, Y, test, objective='regression', scoring='mean_absolute_error', stack_srch_n_iter=100):
    
    def optunaSearchCVParams_RF_rf_pred(X, Y, cv=6, n_iter=30, sampler='tpe', study_name='new',
                                  objective_type='binary', scoring='average_precision', direction='maximize', 
                               n_jobs_est=10, n_jobs_optuna=5):

        """
        direction:  'minimize', 
                    'maximize'

        optuna.samplers.GridSampler(网格搜索采样)
        optuna.samplers.RandomSampler(随机搜索采样)
        optuna.samplers.TPESampler(贝叶斯优化采样)
        optuna.samplers.NSGAIISampler(遗传算法采样)
        optuna.samplers.CmaEsSampler(协方差矩阵自适应演化策略采样，非常先进的优化算法)

        贝叶斯优化（TPESampler）:
        基于过去的试验结果来选择新的参数组合，通常可以更快地找到好的解。适用于较大的参数空间
        当参数间的依赖关系比较复杂时，可能会更有优势。

        遗传算法（NSGAIISampler）:
        遗传算法是一种启发式的搜索方法，通过模拟自然选择过程来探索参数空间。
        对于非凸或非线性问题可能表现良好。

        协方差矩阵自适应演化策略（CmaEsSampler）:
        CMA-ES是一种先进的优化算法，适用于连续空间的非凸优化问题。
        通常需要较多的计算资源，但在一些困难问题上可能会表现得非常好。

        如果你的计算资源充足，网格搜索可能是一个可靠的选择，因为它可以穷举所有的参数组合来找到最优解。
        如果你希望在较短的时间内得到合理的解，随机搜索和贝叶斯优化可能是更好的选择。
        如果你面临的是一个非常复杂或非线性的问题，遗传算法和CMA-ES可能值得尝试。
        """
        optuna.logging.set_verbosity(optuna.logging.ERROR) 

#             tpe_params = TPESampler.hyperopt_parameters()
#             tpe_params['n_startup_trials'] = 30  
        samplers = { 
    #                 'grid': optuna.samplers.GridSampler(), 
                    'random': optuna.samplers.RandomSampler(), 
                    'anneal': optuna.samplers, 
#                         'tpe': optuna.samplers.TPESampler(**tpe_params), 
                    'tpe': optuna.samplers.TPESampler(), 
                    'cma': optuna.samplers.CmaEsSampler(), 
                    'nsgaii': optuna.samplers.NSGAIISampler()} 
        optuna.logging.set_verbosity(optuna.logging.ERROR) 

        if isinstance(Y, pd.DataFrame):
            Y = Y.values.reshape(-1) 
        if objective_type == 'regression':
            objective_list = ['squared_error', 'absolute_error']  # “squared_error”, “absolute_error”, “friedman_mse”, “poisson”
        elif objective_type == 'binary':
            objective_list = ['gini', 'entropy', 'log_loss']   # “gini”, “entropy”, “log_loss”
        elif objective_type == 'multiclass': 
            objective_list = ['gini', 'entropy', 'log_loss'] 

        def objective(trial):
            #params
            param_grid = { 'criterion': trial.suggest_categorical('criterion', objective_list), 
                            'max_depth': trial.suggest_int('max_depth', 3, 63),
                            'max_samples': trial.suggest_float('max_samples', 0.05, 1.0),
                            'max_features': trial.suggest_float('max_features', 0.05, 1.0),
                            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                            'min_samples_split': trial.suggest_int('min_samples_split', 2, 500),
                            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 300),
                            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 5, 200), 
                            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 1e-8, 2, log=True), 
                            'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-8, 2, log=True),
                            'n_jobs': trial.suggest_int('n_jobs', n_jobs_est, n_jobs_est)
                         } 

            if objective_type in ('binary', 'multiclass'):
                clf_params = {'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample'])}
                param_grid.update(clf_params) 
            # 交叉验证
            if objective_type in ('binary', 'multiclass'):
                kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0) 
                kf_split = kf.split(X, Y)
            else:
                kf = KFold(n_splits=cv, shuffle=True, random_state=0) 
                kf_split = kf.split(X)
            # oof  
            if objective_type == 'multiclass':
                if scoring in ('multi_logloss', 'log_loss'): 
                    oof_pred = np.zeros((X.shape[0], len(set(Y))))  
                elif scoring in ('accuracy'):
                    raise ValueError('scoring TODO') 
                else: 
                    raise ValueError('scoring TODO') 
            elif objective_type =='regression':
                oof_pred = np.zeros(X.shape[0])  

            for idx, (train_idx, test_idx) in enumerate(kf_split): 
                X_train, X_val = X.iloc[train_idx], X.iloc[test_idx] 
                y_train, y_val = Y[train_idx], Y[test_idx] 
                # RF建模
                preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X_train.select_dtypes(include=['int', 'float']).columns.tolist()),
#                                                             ('cat', OneHotEncoder(), X_train.select_dtypes(
#                                                                 include=['category', 'object']).columns.tolist())
                                                              ])
                model = Pipeline([('preprocessor', preprocessor),
                                    ('est', RandomForestRegressor(**param_grid, verbose=0) if objective_type =='regression' else RandomForestClassifier(**param_grid) )]) 
                model.fit(X_train, y_train)  
                # 模型预测 
                if objective_type =='multiclass':
                    if scoring in ('log_loss', 'multi_logloss'):
                        preds = model.predict_proba(X_val) 
                        oof_pred[test_idx] = preds 
                    elif scoring == 'accuracy':
                        preds = model.predict(X_val) 
                        oof_pred.iloc[test_idx] = preds 
                    else:
                        raise ValueError('scoring 未设置') 
                elif objective_type =='binary':
                    preds = model.predict_proba(X_val)[:, 1]
                    oof_pred[test_idx] = preds 
                elif objective_type =='regression':
                    preds = model.predict(X_val) 
                    oof_pred[test_idx] = preds
                else:
                    raise ValueError('objective_type Error')

                # 优化METRIC 
            if scoring == 'average_precision':
                score = average_precision_score(Y, oof_pred) 
            elif scoring == 'roc_auc':
                score = roc_auc_score(Y, oof_pred)
            elif scoring == 'accuracy_score':
                score = accuracy_score(Y, oof_pred) 
            elif scoring == 'log_loss':
                score = log_loss(Y, oof_pred) 
            elif scoring == 'multi_logloss':
                lb = LabelBinarizer() 
                Y_lb = lb.fit_transform(Y) 
                score = log_loss(Y_lb, oof_pred) 
            elif scoring == 'mean_absolute_error':
                score = mean_absolute_error(Y, oof_pred)
            elif scoring == 'mean_squared_error':
                score = mean_squared_error(Y, oof_pred)
            elif scoring == 'median_absolute_error':
                score = median_absolute_error(Y, oof_pred)
            else:
                raise ValueError('scoring TODO')
            return score 

        study = optuna.create_study(direction=direction, 
                                    sampler=samplers[sampler], 
    #                                 pruner=optuna.pruners.HyperbandPruner(max_resource='auto'), 
    #                                 pruner=optuna.pruners.MedianPruner(interval_steps=20, n_min_trials=8),
                                    storage=optuna.storages.RDBStorage(url="sqlite:///db.sqlite3", 
                                                                       engine_kwargs={"connect_args": {"timeout": 500}}),  # 指定database URL 
                                    study_name= '%s_%s_%s'%(study_name, sampler, X.shape[1]), load_if_exists=False) 
        study.optimize(objective, n_trials=n_iter, show_progress_bar=True, n_jobs=n_jobs_optuna) 
        print('Best lgbm params:', study.best_trial.params) 
        print('Best CV score:', study.best_value) 

        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_param_importances(study).show()
#         params_df = study.trials_dataframe()
#         params_df = params_df.sort_values(by=['value']) 
        return study.best_trial.params, study.best_value 
    
    
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X) 
    if isinstance(Y, np.ndarray):
        Y = pd.DataFrame(Y) 
    direction = 'minimize' if scoring in ('mean_absolute_error', 'mean_squared_error', 'median_absolute_error',
                                         'log_loss', 'multi_logloss') else 'maximize' 
    params, score_cv = optunaSearchCVParams_RF_rf_pred(X, Y, cv=5, n_iter=stack_srch_n_iter, sampler='tpe', 
                                                       study_name='new%s'%(np.random.randint(0, 1e7)),
                                                       objective_type=objective, scoring=scoring, 
                                                       direction=direction, 
                                                       n_jobs_est=10, n_jobs_optuna=1) 

    print('RF Stacking srch cv score:', score_cv) 
    if objective in ('binary', 'multiclass'):
        
        preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X.select_dtypes(include=['int', 'float']).columns.tolist()),
#                                                             ('cat', OneHotEncoder(), X.select_dtypes(
#                                                                 include=['category', 'object']).columns.tolist())
                                                      ]) 
        est = Pipeline([('preprocessor', preprocessor),
                            ('est', RandomForestClassifier(**params, verbose=0))]) 
        est.fit(X, Y) 
        if scoring in ('log_loss'):
            oof_pred = est.predict_proba(X)[:, 1] 
            test_pred = est.predict_proba(test)[:, 1] 
        elif scoring in ('multi_logloss'):
            oof_pred = est.predict_proba(X)
            test_pred = est.predict_proba(test)
    #   
        if scoring == 'roc_auc':
            score = roc_auc_score(Y.values.reshape(-1), oof_pred) 
        elif scoring == 'average_precision':
            score = average_precision_score(Y.values.reshape(-1), oof_pred) 
        elif scoring in ('log_loss', 'multi_logloss'):
            lb = LabelBinarizer() 
            Y_lb = lb.fit_transform(Y) 
            score = log_loss(Y_lb, oof_pred) 
        else: 
            raise ValueError('TOBD')
        print('oof RF Stacking score', score) 

    else:
        preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X.select_dtypes(include=['int', 'float']).columns.tolist()),
#                                                             ('cat', OneHotEncoder(), X.select_dtypes(
#                                                                 include=['category', 'object']).columns.tolist())
                                                      ]) 
        est = Pipeline([('preprocessor', preprocessor),
                            ('est', RandomForestRegressor(**params,verbose=0))]) 
        est.fit(X, Y) 
        oof_pred = est.predict(X) 
        test_pred = est.predict(test) 
        if scoring == 'mean_absolute_error':
            score = mean_absolute_error(Y.values.reshape(-1), oof_pred)
        elif scoring == 'mean_squared_error':
            score = mean_squared_error(Y.values.reshape(-1), oof_pred) 
        elif scoring == 'median_absolute_error':
            score = median_absolute_error(Y.values.reshape(-1), oof_pred)
        else:
            raise ValueError('TODO')
    return oof_pred, test_pred, score_cv, score 


def kFoldStackingTest(X=[], Y=None, 
                      objective='multiclass', scoring='mean_absolute_error', 
                      n_splits=7, search_n_iters=50, search_cv=7, stack_srch_n_iter=100):

    """
    输入格式
    X = [['LGBM', 'MARK1', train, test, params], 
         ['RF', 'MARK2', train, test, params], 
         ['LR', 'MARK3', train, test, params]],
         ['CAT', 
         ['XGB'
    """
    
#     if isinstance(Y, pd.DataFrame):
#         Y = Y.values.reshape(-1) 
    # 5-fold CV 
    if objective in ('binary', 'multiclass'):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2024)
        kf_split = kf.split(X[0][2].values, Y) 
    else: 
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        kf_split = kf.split(X[0][2].values)
    
    len_x, len_test = X[0][2].shape[0], X[0][3].shape[0]  
    
    if objective in ('binary', 'multiclass'):
        if scoring in ('log_loss', 'multi_logloss'):
            oof_preds = {'climb': np.zeros((len_x, Y[Y.columns[0]].nunique() )), 'lr': np.zeros((len_x, Y[Y.columns[0]].nunique() )), 'rf': np.zeros((len_x, Y[Y.columns[0]].nunique() ))}  
            test_preds = {'climb': np.zeros((len_test, Y[Y.columns[0]].nunique() )), 'lr': np.zeros((len_test, Y[Y.columns[0]].nunique() )), 'rf': np.zeros((len_test, Y[Y.columns[0]].nunique()))}  
    elif objective == 'regression': 
        oof_preds = {'climb': np.zeros(len_x), 'lr': np.zeros(len_x), 'rf': np.zeros(len_x)}
        test_preds = {'climb': np.zeros(len_test), 'lr': np.zeros(len_test), 'rf': np.zeros(len_test)}
    # 保存每个model的oof_pred与test_pred 
    
    oof_preds_est, test_preds_est = {}, {}
    if objective in ('regression'):
        for est_list in X: 
            oof_preds_est[est_list[1]] = np.zeros(len_x)
            test_preds_est[est_list[1]] = np.zeros(len_test)
    elif objective in ('binary', 'multiclass'):
        for est_list in X: 
            oof_preds_est[est_list[1]] = np.zeros((len_x, Y[Y.columns[0]].nunique() )) 
            test_preds_est[est_list[1]] = np.zeros((len_test, Y[Y.columns[0]].nunique() ))  
    
    def lgbm_pred(X_tr, y_tr, X_val, y_val, t, p):             
        if len(p) >0:
            params = p
        else: 
            raise ValueError('TODO') 
        if objective in ('binary', 'multiclass'):
            
            preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X_tr.select_dtypes(
                                                                                        include=['int', 'float']).columns.tolist()),
#                                                             ('cat', OneHotEncoder(), X_tr.select_dtypes(include=['category', 'object']).columns.tolist())
                                                          ]) 
            model = Pipeline([('preprocessor', preprocessor),
                                ('est', lgb.LGBMClassifier(**params, verbosity=-2))]) 
            model.fit(X_tr, y_tr, est__eval_set=[(X_val, y_val)]) 
            if objective == 'binary':
                val_pred = model.predict_proba(X_val, num_iteration=model.named_steps['est'].best_iteration_)[:, 1]
                test_pred = model.predict_proba(t, num_iteration=model.named_steps['est'].best_iteration_)[:, 1] 
            elif objective == 'multiclass':
                val_pred = model.predict_proba(X_val, num_iteration=model.named_steps['est'].best_iteration_)
                test_pred = model.predict_proba(t, num_iteration=model.named_steps['est'].best_iteration_)
        else: 
            preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X_tr.select_dtypes(include=['int', 'float']).columns.tolist()),
#                                                             ('cat', OneHotEncoder(), X_tr.select_dtypes(include=['category', 'object']).columns.tolist())
                                                          ]) 
            model = Pipeline([('preprocessor', preprocessor),
                                ('est', lgb.LGBMRegressor(**params, verbosity=-2))]) 
            model.fit(X_tr, y_tr, est__eval_set=[(X_val, y_val)]) 
            val_pred = model.predict(X_val, num_iteration=model.named_steps['est'].best_iteration_) 
            test_pred = model.predict(t, num_iteration=model.named_steps['est'].best_iteration_) 
        return val_pred, test_pred, params 
    
    def rf_pred(X_tr, y_tr, X_val, y_val, t, p): 
        # # RF
        def optunaSearchCVParams_RF_rf_pred(X, Y, cv=6, n_iter=30, sampler='tpe', study_name='new',
                                      objective_type='binary', scoring='average_precision', direction='maximize', 
                                   n_jobs_est=10, n_jobs_optuna=5):

            """
            direction:  'minimize', 
                        'maximize'

            optuna.samplers.GridSampler(网格搜索采样)
            optuna.samplers.RandomSampler(随机搜索采样)
            optuna.samplers.TPESampler(贝叶斯优化采样)
            optuna.samplers.NSGAIISampler(遗传算法采样)
            optuna.samplers.CmaEsSampler(协方差矩阵自适应演化策略采样，非常先进的优化算法)

            贝叶斯优化（TPESampler）:
            基于过去的试验结果来选择新的参数组合，通常可以更快地找到好的解。适用于较大的参数空间
            当参数间的依赖关系比较复杂时，可能会更有优势。

            遗传算法（NSGAIISampler）:
            遗传算法是一种启发式的搜索方法，通过模拟自然选择过程来探索参数空间。
            对于非凸或非线性问题可能表现良好。

            协方差矩阵自适应演化策略（CmaEsSampler）:
            CMA-ES是一种先进的优化算法，适用于连续空间的非凸优化问题。
            通常需要较多的计算资源，但在一些困难问题上可能会表现得非常好。

            如果你的计算资源充足，网格搜索可能是一个可靠的选择，因为它可以穷举所有的参数组合来找到最优解。
            如果你希望在较短的时间内得到合理的解，随机搜索和贝叶斯优化可能是更好的选择。
            如果你面临的是一个非常复杂或非线性的问题，遗传算法和CMA-ES可能值得尝试。
            """
            optuna.logging.set_verbosity(optuna.logging.ERROR) 

#             tpe_params = TPESampler.hyperopt_parameters()
#             tpe_params['n_startup_trials'] = 100 
            samplers = { 
        #                 'grid': optuna.samplers.GridSampler(), 
                        'random': optuna.samplers.RandomSampler(), 
                        'anneal': optuna.samplers, 
#                         'tpe': optuna.samplers.TPESampler(**tpe_params), 
                        'tpe': optuna.samplers.TPESampler(), 
                        'cma': optuna.samplers.CmaEsSampler(), 
                        'nsgaii': optuna.samplers.NSGAIISampler()} 
            optuna.logging.set_verbosity(optuna.logging.ERROR) 

            if isinstance(Y, pd.DataFrame):
                Y = Y.values.reshape(-1) 
            if objective_type == 'regression':
                objective_list = ['squared_error', 'absolute_error']  # “squared_error”, “absolute_error”, “friedman_mse”, “poisson”
            elif objective_type == 'binary':
                objective_list = ['gini', 'entropy', 'log_loss']   # “gini”, “entropy”, “log_loss”
            elif objective_type == 'multiclass': 
                objective_list = ['gini', 'entropy', 'log_loss'] 

            def objective(trial):
                #params
                param_grid = { 'criterion': trial.suggest_categorical('criterion', objective_list), 
                                'max_depth': trial.suggest_int('max_depth', 3, 63),
                                'max_samples': trial.suggest_float('max_samples', 0.05, 1.0),
                                'max_features': trial.suggest_float('max_features', 0.05, 1.0),
                                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                                'min_samples_split': trial.suggest_int('min_samples_split', 2, 500),
                                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 300),
                                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 5, 200), 
                                'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 1e-8, 2, log=True), 
                                'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-8, 2, log=True),
                                'n_jobs': trial.suggest_int('n_jobs', n_jobs_est, n_jobs_est)
                             } 

                if objective_type in ('binary', 'multiclass'):
                    clf_params = {'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample'])}
                    param_grid.update(clf_params) 
                # 交叉验证
                if objective_type in ('binary', 'multiclass'):
                    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0) 
                    kf_split = kf.split(X, Y)
                else:
                    kf = KFold(n_splits=cv, shuffle=True, random_state=0) 
                    kf_split = kf.split(X)
                # oof  
                oof_pred = pd.Series([None]*X.shape[0]) if objective_type =='multiclass' else np.zeros(X.shape[0]) 

                for idx, (train_idx, test_idx) in enumerate(kf_split): 
                    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx] 
                    y_train, y_val = Y[train_idx], Y[test_idx] 

                    # RF建模
                    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X_train.select_dtypes(include=['int', 'float']).columns.tolist()),
        #                                                             ('cat', OneHotEncoder(), X_train.select_dtypes(include=['category', 'object']).columns.tolist())
                                                                  ]) 
                    model = Pipeline([('preprocessor', preprocessor),
                                        ('est', RandomForestRegressor(**param_grid, verbose=0) if objective_type =='regression' else RandomForestClassifier(**param_grid))]) 
                    model.fit(X_train, y_train) 
                    # 模型预测 
                    if objective_type =='multiclass':
                        if scoring == 'accuracy':
                            preds = model.predict(X_val)
                            oof_pred.iloc[test_idx] = preds
                        elif scoring in ('log_loss', 'multi_logloss'):
                            raise ValueError('TODO')

                    elif objective_type =='binary':
                        preds = model.predict_proba(X_val)[:, 1]
                        oof_pred[test_idx] = preds
                    elif objective_type =='regression':
                        preds = model.predict(X_val) 
                        oof_pred[test_idx] = preds
                    else:
                        raise ValueError('objective_type Error')

                    # 优化average_precision
                if scoring == 'average_precision':
                    score = average_precision_score(Y, oof_pred) 
                elif scoring == 'roc_auc':
                    score = roc_auc_score(Y, oof_pred)
                elif scoring == 'accuracy_score':
                    score = accuracy_score(Y, oof_pred) 
                elif scoring == 'log_loss':
                    score = log_loss(Y, oof_pred) 
                elif scoring == 'mean_absolute_error':
                    score = mean_absolute_error(Y, oof_pred)
                elif scoring == 'mean_squared_error':
                    score = mean_squared_error(Y, oof_pred)
                elif scoring == 'median_absolute_error':
                    score = median_absolute_error(Y, oof_pred)
                else:
                    raise ValueError('scoring Error')
                return score 

            study = optuna.create_study(direction=direction, 
                                        sampler=samplers[sampler], 
        #                                 pruner=optuna.pruners.HyperbandPruner(max_resource='auto'), 
        #                                 pruner=optuna.pruners.MedianPruner(interval_steps=20, n_min_trials=8),
                                        storage=optuna.storages.RDBStorage(url="sqlite:///db.sqlite3", 
                                                                           engine_kwargs={"connect_args": {"timeout": 500}}),  # 指定database URL 
                                        study_name= '%s_%s_%s_%s'%(study_name, sampler, X.shape[1], 
                                                                  np.random.randint(0, 1e7)
                                                                  ), load_if_exists=False) 
            study.optimize(objective, n_trials=n_iter, show_progress_bar=True, n_jobs=n_jobs_optuna) 
            print('Best lgbm params:', study.best_trial.params) 
            print('Best CV score:', study.best_value) 

            optuna.visualization.plot_optimization_history(study).show()
#             optuna.visualization.plot_param_importances(study).show()
#             params_df = study.trials_dataframe()
#             params_df = params_df.sort_values(by=['value'])
            return study.best_trial.params, study.best_value

        
        
        if len(p) >0:
            params = p
        else:  
            raise ValueError('TODO') 
        if objective in ('binary', 'multiclass'): 
            preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X_tr.select_dtypes(include=['int', 'float']).columns.tolist()),
#                                                             ('cat', OneHotEncoder(), X_tr.select_dtypes(include=['category', 'object']).columns.tolist())
                                                          ]) 
            model = Pipeline([('preprocessor', preprocessor),
                                ('est', RandomForestClassifier(**params, verbose=0))]) 
            model.fit(X_tr, y_tr) 
            if scoring in ('log_loss', 'multi_logloss'): 
                val_pred = model.predict_proba(X_val) 
                test_pred = model.predict_proba(t) 
            else:
                raise ValueError('TODO')
        else:
            preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X_tr.select_dtypes(include=['int', 'float']).columns.tolist()),
#                                                             ('cat', OneHotEncoder(), X_tr.select_dtypes(include=['category', 'object']).columns.tolist())
                                                          ]) 
            model = Pipeline([('preprocessor', preprocessor),
                                ('est', RandomForestRegressor(**dict(params, verbose=0)))]) 
            model.fit(X_tr, y_tr) 
            val_pred = model.predict(X_val)
            test_pred = model.predict(t)
        return val_pred, test_pred, params 

    def cat_pred(X_tr, y_tr, X_val, y_val, t, p): 
        # # RF2
        if len(p) >0:
            params = p
        else:  
            raise ValueError('TODO') 
        if objective in ('binary', 'multiclass'):
            preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X_tr.select_dtypes(include=['int', 'float']).columns.tolist()),
#                                                             ('cat', OneHotEncoder(), X_tr.select_dtypes(include=['category', 'object']).columns.tolist())
                                                          ]) 
            model = Pipeline([('preprocessor', preprocessor),
                                ('est', CatBoostClassifier(**dict(params, verbosity=0)))]) 
            model.fit(X_tr, y_tr, est__eval_set=[(X_val, y_val)]) 
            val_pred = model.predict_proba(X_val)[:, 1]
            test_pred = model.predict_proba(t)[:, 1]
        else:
            preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X_tr.select_dtypes(include=['int', 'float']).columns.tolist()),
#                                                             ('cat', OneHotEncoder(), X_tr.select_dtypes(include=['category', 'object']).columns.tolist())
                                                          ]) 
            model = Pipeline([('preprocessor', preprocessor),
                                ('est', CatBoostRegressor(**dict(params, verbosity=0)))]) 
            model.fit(X_tr, y_tr, est__eval_set=[(X_val, y_val)]) 
            val_pred = model.predict(X_val)
            test_pred = model.predict(t)
        return val_pred, test_pred, params 

    def xgb_pred(X_tr, y_tr, X_val, y_val, t, p): 
        if len(p) >0:
            params = p
        else:
            raise ValueError('TODO') 
        if objective in ('binary', 'multiclass'):
#             preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X_tr.select_dtypes(include=['int', 'float']).columns.tolist()),
# #                                                             ('cat', OneHotEncoder(), X_tr.select_dtypes(include=['category', 'object']).columns.tolist())
#                                                           ]) 
#             model = Pipeline([('preprocessor', preprocessor),
#                                 ('est', XGBClassifier(**dict(params), verbosity=0 ))]) 
            model = XGBClassifier(**dict(params), verbosity=0)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)]) 
            if scoring in ('log_loss', 'multi_logloss'): 
                val_pred = model.predict_proba(X_val)
                test_pred = model.predict_proba(t)
            else:
                raise ValueError('TODO')
        else:
#             preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X_tr.select_dtypes(include=['int', 'float']).columns.tolist()),
# #                                                             ('cat', OneHotEncoder(), X_tr.select_dtypes(include=['category', 'object']).columns.tolist())
#                                                           ]) 
#             model = Pipeline([('preprocessor', preprocessor),
#                                 ('est', XGBRegressor(**dict(params), verbosity=0))]) 
            model = XGBRegressor(**dict(params), verbosity=0) 
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)]) 
            val_pred = model.predict(X_val) 
            test_pred = model.predict(t)
        return val_pred, test_pred, params 

    def lr_pred(X_tr, y_tr, X_val, y_val, t, p): 
        # LR 
        if len(p) >0:
            params = p
        else: 
            raise ValueError('TODO') 
        preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X_tr.select_dtypes(include=['int', 'float']).columns.tolist()),
#                                                             ('cat', OneHotEncoder(), X_tr.select_dtypes(include=['category', 'object']).columns.tolist())
                                                          ]) 
        model = Pipeline([('preprocessor', preprocessor),
                                ('est', LogisticRegression(**dict(params)))]) 
        model.fit(X_tr, y_tr) 
        val_pred = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict_proba(t)[:, 1]
        return val_pred, test_pred, params 
    
    def dnn_pred(X_tr, y_tr, X_val, y_val, t, p): 
        # # RF2 
        if len(p) >0:
            params = p 
        else:  # 搜索参数
            raise ValueError('DNN params needed')
        
        tr_df = pd.concat((X_tr, y_tr), axis=1)
        val_df = pd.concat((X_val, y_val), axis=1)
        df = pd.concat((tr_df, val_df), axis=0).reset_index(drop=True) 
        valid_idx = df.index.tolist()[-len(val_df):] 
        procs = [Categorify, FillMissing, Normalize] 
        dep_var = y_tr.columns.tolist()[0] 
        cont, cat = cont_cat_split(tr_df, 1, dep_var=dep_var) 
        
        valid_df = val_df.copy().reset_index(drop=True) 
        dls = TabularDataLoaders.from_df(df, procs=procs, cat_names=cat, cont_names=cont, 
                                         y_names=dep_var, valid_idx=valid_idx, 
                                         y_block = CategoryBlock() if objective in ('binary', 'multiclass') else None,  
                                         bs=1024*2) 
        
        # n_out 
        if objective == 'regression': 
            loss_func = F.median_absolute_error()
            n_out = 1 
        else:
            loss_func = torch.nn.CrossEntropyLoss()
            n_out = y[y.columns[0]].nunique()

        #params 
        def layers_n_units(params):
            layers = []
            for i in range(params['n_layers']):
                layers.append(params['n_units_l%s'%i])
            return layers 
        def emb_szs_dict(params):
            r = {} 
            for k in params.keys():
                if 'emb_sz_' in k:
                    r[k[7:]] = params[k]
            return r 

        layers = layers_n_units(params) 
        config = {'ps': params['dropout_rate'], 
                  'embed_p': params['embed_p']} 
        emb_szs = emb_szs_dict(params)  

        if objective == 'regression': 
            y_range = trial.suggest_categorical('y_range', [(float(y.iloc[:, 0].min()), float(y.iloc[:, 0].max()))]) 
        else: 
            y_range = None 


        learn = tabular_learner(dls, 
                                y_range=y_range, 
                                layers=layers, 
#                                 metrics=metrics.mae, 
                                wd=params['wd'], 
                                train_bn=params['train_bn'],
                                config=config, 
                                emb_szs=emb_szs, 
                                n_out=n_out, 
#                                 loss_func=loss_func, 
                                cbs=[EarlyStoppingCallback(monitor='valid_loss', 
                                patience=params['early_stopping_rounds']), 
                                SaveModelCallback(monitor='valid_loss', fname='best_model', every_epoch=False)] 
                               )  
        
        
        # 训练模型 
        with learn.no_logging():
            learn.fit_one_cycle(100000, params['lr']) 
        # 载入最佳模型 
        learn.load('best_model')
        # val/test 
        dl_val = learn.dls.test_dl(valid_df.drop(columns=[dep_var])) 
        val_pred, _ = learn.get_preds(dl=dl_val, reorder=True) 
        dl_test = learn.dls.test_dl(t) 
        test_pred, _ = learn.get_preds(dl=dl_test, reorder=True) 
        val_pred = np.array(val_pred) 
        test_pred = np.array(test_pred) 
        return val_pred, test_pred, params 

    
    for train_index, valid_index in kf_split:
        y_tr, y_val = Y.iloc[train_index], Y.iloc[valid_index]
        # datas for stack 
        oof_models, test_models = {}, {} 
        for idx, (name, mark, x, t, p) in enumerate(X): 
#             if isinstance(x, pd.DataFrame):
#                 x = x.values 
            X_tr, X_val = x.iloc[train_index], x.iloc[valid_index]
            if name == 'LGBM': 
                val_pred, test_pred, p_srch = lgbm_pred(X_tr, y_tr, X_val, y_val, t, p)
            if name =='RF':
                val_pred, test_pred, p_srch = rf_pred(X_tr, y_tr, X_val, y_val, t, p)
            if name == 'LR':
                val_pred, test_pred, p_srch = lr_pred(X_tr, y_tr, X_val, y_val, t, p)
            if name == 'XGB':
                val_pred, test_pred, p_srch = xgb_pred(X_tr, y_tr, X_val, y_val, t, p)
            if name == 'CAT':
                val_pred, test_pred, p_srch = cat_pred(X_tr, y_tr, X_val, y_val, t, p)
            if name == 'DNN':
                val_pred, test_pred, p_srch = dnn_pred(X_tr, y_tr, X_val, y_val, t, p)

            oof_models[mark] = val_pred 
            test_models[mark] = test_pred 
            if len(p) == 0:
                X[idx][4] = p_srch 
            
            # 保存每个模型结果
            oof_preds_est[mark][valid_index] = val_pred
            test_preds_est[mark] += test_pred / n_splits

        # STACK 
        # climb stacking
        if scoring == 'mean_absolute_error':
            scoring_func = mean_absolute_error
            direction = 'minimize'
        elif scoring == 'multi_logloss':
            scoring_func = log_loss
            direction = 'minimize'
        else: 
            raise ValueError('TODO') 
        climb_stack_oof, climb_stack_test, climb_stack_score = climb(oof_models, y_val.values.reshape(-1), 
                                                                     test_models, scoring_func, direction) 
        oof_preds['climb'][valid_index] = climb_stack_oof 
        test_preds['climb'] += climb_stack_test/ kf.n_splits  
        # LR / RF stacking 
        if objective in ('regression'):
            oof_df = pd.DataFrame(oof_models) 
            test_df = pd.DataFrame(test_models)  
        elif objective in ('multiclass'): 
            oof_df = pd.DataFrame() 
            for k in oof_models.keys():
                oof_prob = pd.DataFrame(oof_models[k], columns=['%s_%s'%(k, i) for i in range(oof_models[k].shape[1])]) 
                oof_df = pd.concat((oof_df, oof_prob.copy()), axis=1) 
        
            test_df = pd.DataFrame() 
            for k in test_models.keys():
                test_prob = pd.DataFrame(test_models[k], columns=['%s_%s'%(k, i) for i in range(test_models[k].shape[1])]) 
                test_df = pd.concat((test_df, test_prob.copy()), axis=1) 
        else: 
            raise ValueError('TODO')
#         # lr 
#         if objective in ('binary', 'multiclass'):
#             lr_stack_oof, lr_stack_test, lr_srch_cv_score, lr_stack_score = lr_stack(oof_df, y_val, test_df)
#             oof_preds['lr'][valid_index] = lr_stack_oof 
#             test_preds['lr'] +=   lr_stack_test/ kf.n_splits  
#         else:
#             lr_stack_oof, lr_stack_test, lr_srch_cv_score, lr_stack_score = linear_Reg_stack(oof_df, y_val, test_df)
#             oof_preds['lr'][valid_index] = lr_stack_oof 
#             test_preds['lr'] +=   lr_stack_test/ kf.n_splits  
#       # rf 
#         rf_stack_oof, rf_stack_test, rf_srch_cv_score, rf_stack_score = rf_stack(oof_df, y_val, test_df, 
#                                                                                  objective=objective, scoring=scoring, 
#                                                                                 stack_srch_n_iter=stack_srch_n_iter)
#         oof_preds['rf'][valid_index] = rf_stack_oof 
#         test_preds['rf'] +=   rf_stack_test / kf.n_splits  

#         print('search params cv score: LR %s, RF %s'%(round(lr_srch_cv_score, 4), round(rf_srch_cv_score, 4)))
#         print('stacked oof score: climb %s,  LR %s, RF %s'%(round(climb_stack_score, 4), round(lr_stack_score, 4), round(rf_stack_score, 4))) 
#         print('*************************') 
#     np.save('./data/stack/climb_4models_oof_{}.npy'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), oof_preds)
#     np.save('./data/stack/climb_4models_test_{}.npy'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), test_preds)
    
    
    oof_preds_est.update(oof_preds)
    test_preds_est.update(test_preds)
    
    return test_preds, oof_preds_est, test_preds_est



x = pd.read_csv('./data/x.csv') 
y = pd.read_csv('./data/y.csv')
test_data = pd.read_csv('./data/test_processed.csv')  





# ##### 参数


# lgbm 
lgbm_prms1 =  {'boosting_type': 'gbdt', 'objective': 'softmax', 'n_estimators': 5000, 'learning_rate': 0.03412509185437405, 'num_leaves': 31, 'max_depth': 16, 'min_data_in_leaf': 10, 'min_sum_hessian_in_leaf': 1.4811192344196536e-07, 'lambda_l1': 1.0793441072446075e-05, 'lambda_l2': 6.476927186856316e-05, 'min_gain_to_split': 4.267613946159837e-06, 'bagging_fraction': 0.85, 'bagging_freq': 6, 'feature_fraction': 0.16999999999999998, 'feature_fraction_bynode': 0.5700000000000001, 'max_bin': 245, 'min_data_in_bin': 8, 'early_stopping_rounds': 40, 'seed': 0, 'n_jobs': -1, 'class_weight': None, 'num_class': 3, 'metric': 'multi_logloss'}
# Best CV score: 0.4156276502230447  
lgbm_prms1['n_estimators'] = 10000 
lgbm_prms1['early_stopping_rounds'] = 200  


lgbm_prms2 =  {'boosting_type': 'gbdt', 'objective': 'softmax', 'n_estimators': 3000, 'learning_rate': 0.006859582595622571, 'num_leaves': 21, 'max_depth': 7, 'min_data_in_leaf': 57, 'min_sum_hessian_in_leaf': 0.04962917283157315, 'lambda_l1': 0.014677317557436457, 'lambda_l2': 2.6897717967274723e-08, 'min_gain_to_split': 0.00018998741421903567, 'bagging_fraction': 0.9500000000000001, 'bagging_freq': 2, 'feature_fraction': 0.07, 'feature_fraction_bynode': 0.99, 'max_bin': 500, 'min_data_in_bin': 9, 'early_stopping_rounds': 50, 'seed': 0, 'n_jobs': -1, 'class_weight': None, 'num_class': 3, 'metric': 'multi_logloss'}
# Best CV score: 0.4178293978108016 
lgbm_prms2['n_estimators'] = 10000 
lgbm_prms2['early_stopping_rounds'] = 200  


# In[6]:


lgbm_prms3 = {'objective': 'multiclass',    'metric': 'multi_logloss',     'max_depth': 15,     'min_child_samples': 13, 'learning_rate': 0.05285597081335651, 'n_estimators': 284,     'min_child_weight': 5,     'subsample': 0.7717873512945741,    'colsample_bytree': 0.10012816493265511,     'reg_alpha': 0.8767668608061822,     'reg_lambda': 0.8705834466355764,    'random_state': 42} 
lgbm_prms3['n_estimators'] = 10000 
lgbm_prms3['early_stopping_rounds'] = 200  


lgbm_prms4 = { 'objective': 'multiclass',    'metric': 'multi_logloss',      'max_depth': 9,     'min_child_samples': 14, 'learning_rate': 0.034,     'n_estimators': 274,     'min_child_weight': 9,     'colsample_bytree': 0.17029,    
              'reg_alpha': 0.107,     'reg_lambda': 0.723, 'random_state': 42} 
lgbm_prms4['n_estimators'] = 10000 
lgbm_prms4['early_stopping_rounds'] = 200  


rf_prms1 = {'criterion': 'log_loss', 'max_depth': 35, 'max_samples': 0.9623108597184384, 'max_features': 0.6164900254500953, 'n_estimators': 155, 'min_samples_split': 12, 'min_samples_leaf': 1, 'max_leaf_nodes': 197, 'min_impurity_decrease': 1.4528232630054616e-08, 'ccp_alpha': 2.7239340219801595e-08, 'n_jobs': 10, 'class_weight': None}
# Best CV score: 0.455768066671348 
rf_prms1['n_estimators'] = 500 


xgb_prms1 = {'booster': 'gbtree', 'objective': 'multi:softprob', 'tree_method': 'hist', 'grow_policy': 'depthwise', 'n_estimators': 7000, 'eta': 0.009703297257526708,   'max_depth': 23, 'min_child_weight': 7.752094698172611e-07, 'alpha': 6.868500227256957e-05, 'lambda': 2.7999591104350574,  'gamma': 0.00014971126326939103, 'subsample': 0.6, 'colsample_bytree': 0.11,              'colsample_bynode': 0.93, 'max_bin': 471, 'eval_metric': 'mlogloss',              'early_stopping_rounds': 100, 'seed': 0, 'n_jobs': 10, 'num_class': 3}
# Best CV score: 0.4127667202010639 
xgb_prms1['n_estimators'] = 10000 
xgb_prms1['early_stopping_rounds'] = 200 


xgb_prms2 ={ 'objective': 'multiclass',    'metric': 'multi_logloss',    'n_estimators': 397,
    'max_depth': 44,    'min_child_weight': 4.8419409783368215,    'learning_rate': 0.049792103525168455,
    'subsample': 0.7847543051746505,    'gamma': 0.4377096783729759,    'colsample_bytree': 0.22414960640035653,
    'colsample_bylevel': 0.8173336142032213,    'colsample_bynode': 0.9468109886478254,
    'random_state': 42 } 
xgb_prms2['n_estimators'] = 100000  
xgb_prms2['early_stopping_rounds'] = 150 



# ##### kaggle

kaggle_x = pd.read_csv('./data/kagglebook/train_data.csv')

kaggle_test = pd.read_csv('./data/kagglebook/test_data.csv')  

SELECTED = ['Platelets', 'Copper','Alk_Phos','Diagnosis_Date','SGOT','Age_Years','N_Days','Cholesterol',          'Tryglicerides','Albumin','Bilirubin','Prothrombin','Symptom_Score','Stage','Drug', 
         'Hepatomegaly', 'Spiders', 'Sex', 'Edema_N', 'Edema_S', 'Edema_Y', 'Bilirubin_deviation',       'Cholesterol_deviation', 'Albumin_deviation', 'Copper_deviation',
       'Alk_Phos_deviation', 'SGOT_deviation', 'Tryglicerides_deviation', 'Platelets_deviation',       'Prothrombin_deviation'] 


# In[ ]:





# In[12]:


kaggle_xgb_params = {'lambda': 1.1369029459700144e-06, 'alpha': 0.012063715109367643, 'max_depth': 6, 'eta': 0.0016842485569386354, 
              'gamma': 1.8110005586084708e-08, 'colsample_bytree': 0.14198953405080517, 'subsample': 0.7387879239640978, 
              'min_child_weight': 1, 'n_estimators': 931, 'learning_rate': 0.07244128492444549, 'reg_alpha': 0.7206512712096103, 
              'reg_lambda': 0.33555254247327354} 

kaggle_xgb_params['n_estimators'] = 100000  
kaggle_xgb_params['early_stopping_rounds'] = 200 


# In[13]:


kaggle_xgb_prms2 = {'booster': 'gbtree', 'objective': 'multi:softprob', 'tree_method': 'hist', 'grow_policy': 'lossguide', 'n_estimators': 7000, 'eta': 0.030137562471404363, 'max_depth': 28, 'min_child_weight': 0.0019868862772378978, 'alpha': 1.479324715372322, 'lambda': 0.1835943863949339, 'gamma': 2.080825449803173e-06, 'subsample': 0.8, 'colsample_bytree': 0.07, 'colsample_bynode': 0.41, 'max_bin': 400, 'eval_metric': 'mlogloss', 'early_stopping_rounds': 100, 'seed': 0, 'n_jobs': 10, 'num_class': 3}
# Best CV score: 0.41632443773296496
kaggle_xgb_prms2['n_estimators'] = 100000  
kaggle_xgb_prms2['early_stopping_rounds'] = 200 


# ##### dnn

# In[14]:


x_pred_prob_xgb = pd.read_csv('./data/pred_prob.csv') 
test_pred_prob_xgb = pd.read_csv('./data/xgb_test_pred.csv')
x_pred_prob_xgb.columns = test_pred_prob_xgb.columns 
dnn_x = pd.concat((x, x_pred_prob_xgb), axis=1) 
dnn_test = pd.concat((test_data, test_pred_prob_xgb), axis=1) 


# In[15]:


dnn_params_0423 = {'n_layers': 2, 'n_units_l0': 14, 'n_units_l1': 25, 'lr': 0.060908082737224915,  'wd': 1.1555079081263367e-05, 'dropout_rate': 0.45991713909042586,   'embed_p': 0.2419301357571613, 'train_bn': False}
dnn_params_0423['early_stopping_rounds'] = 500 






result, oof_perEST, test_perEST = kFoldStackingTest(X=[ 
                           ['LGBM', 'lgbm1', x, test_data, lgbm_prms1], 
                           ['LGBM', 'lgbm2', x, test_data, lgbm_prms2], 
                           ['LGBM', 'lgbm3', x, test_data, lgbm_prms3], 
                           ['LGBM', 'lgbm4', x, test_data, lgbm_prms4], 
                           ['RF', 'rf1', x, test_data, rf_prms1], 
                           ['XGB', 'xgb1', x, test_data, xgb_prms1], 
                           ['XGB', 'xgb2', x, test_data, xgb_prms2], 
                           ['XGB', 'xgb3', kaggle_x[SELECTED], kaggle_test[SELECTED], kaggle_xgb_params], 
                           ['XGB', 'xgb4', kaggle_x[SELECTED], kaggle_test[SELECTED], kaggle_xgb_prms2], 
                           ['DNN', 'dnn_xgb1', dnn_x, dnn_test, dnn_params_0423], 
                #             ['CAT', 'cat1', X[rawcols], test[rawcols], prms_cat1], 
                        ], 
                           Y=y, 
                           objective='multiclass', scoring='multi_logloss', 
                           n_splits=6, search_n_iters=50, search_cv=5, 
                           stack_srch_n_iter=100) 



# In[88]:


# result['climb'][result['climb']<0] = 0 
# submit = pd.read_csv('./data/sample_submission.csv') 
# submit.iloc[:, 1:] = result['climb']
# submit.columns=['id', 'Status_D', 'Status_C', 'Status_CL']   
# submit.to_csv('./data/submit/climb_submit6.csv', index=False) 


# ##### lgbm stack 


oof_combined = np.hstack([oof_perEST['lgbm1'], oof_perEST['lgbm2'],oof_perEST['lgbm3'],oof_perEST['lgbm4'], 
                         oof_perEST['rf1'], 
                          oof_perEST['xgb1'], oof_perEST['xgb2'], 
                          oof_perEST['xgb3'], oof_perEST['xgb4'], 
                          oof_perEST['dnn_xgb1']]) 
test_combined = np.hstack([
                           test_perEST['lgbm1'], test_perEST['lgbm2'], test_perEST['lgbm3'],test_perEST['lgbm4'], 
                           test_perEST['rf1'], 
                           test_perEST['xgb1'],test_perEST['xgb2'], 
                           test_perEST['xgb3'],test_perEST['xgb4'],
                           test_perEST['dnn_xgb1']])  

oof_combined = pd.DataFrame(oof_combined, columns=['col'+ str(i) for i in range(oof_combined.shape[1])])  
test_combined = pd.DataFrame(test_combined, columns=['col'+ str(i) for i in range(test_combined.shape[1])])  


# In[ ]:


# oof_combined.to_csv('./data/oof_combined_cols29.csv', index=False) 
# test_combined.to_csv('./data/test_combined_cols29.csv', index=False)   



oof_combined_vars29 = pd.read_csv('./data/oof_combined_cols29.csv')
test_combined_vars29 = pd.read_csv('./data/test_combined_cols29.csv') 







def optunaSearchCVParams_LGBM(X, Y, cv=6, n_iter=30, sampler='tpe',  study_name = 'new',
                              objective_type='binary',  scoring='average_precision', direction='maximize',
                             n_jobs_est=10, n_jobs_optuna=3, use_gpu=False):
    
    """
    direction:  'minimize', 
                'maximize'
    objective_type: binary, multiclass, 
    
    scoring: binary: 'average_precision','roc_auc'
             multiclass: 'log_loss', 'accuracy_score'
             
    model: lgbm
    X, Y dtype: int, float, category
    objective:  'regression': 传统的均方误差回归。
                'regression_l1': 使用L1损失的回归，也称为 Mean Absolute Error (MAE)。
                'huber': 使用Huber损失的回归，这是均方误差和绝对误差的结合，特别适用于有异常值的情况。
                'fair': 使用Fair损失的回归，这也是另一种对异常值鲁棒的损失函数。
                'binary', 
                'multiclass'
    scoring:
    'neg_root_mean_squared_error', 'precision_micro', 'jaccard_micro', 'f1_macro', 
    'recall_weighted', 'neg_mean_absolute_percentage_error', 'f1_weighted', 
    'completeness_score', 'neg_brier_score', 'neg_mean_gamma_deviance', 'precision', 
    'adjusted_mutual_info_score', 'f1_samples', 'jaccard', 'neg_mean_poisson_deviance', 
    'precision_samples', 'recall', 'recall_samples', 'top_k_accuracy', 'roc_auc_ovr', 
    'mutual_info_score', 'jaccard_samples', 'positive_likelihood_ratio', 'f1_micro', 
    'adjusted_rand_score', 'accuracy', 'matthews_corrcoef', 'neg_mean_squared_log_error', 
    'precision_macro', 'rand_score', 'neg_log_loss', 'recall_macro', 'roc_auc_ovo', 
    'average_precision', 'jaccard_weighted', 'max_error', 'neg_median_absolute_error', 
    'jaccard_macro', 'roc_auc_ovo_weighted', 'fowlkes_mallows_score', 'precision_weighted', 
    'balanced_accuracy', 'v_measure_score', 'recall_micro', 'normalized_mutual_info_score', 
    'neg_mean_squared_error', 'roc_auc', 'roc_auc_ovr_weighted', 'f1', 'homogeneity_score', 
    'explained_variance', 'r2', 'neg_mean_absolute_error', 'neg_negative_likelihood_ratio'

             
    optuna.samplers.GridSampler(网格搜索采样)
    optuna.samplers.RandomSampler(随机搜索采样)
    optuna.samplers.TPESampler(贝叶斯优化采样)
    optuna.samplers.NSGAIISampler(遗传算法采样)
    optuna.samplers.CmaEsSampler(协方差矩阵自适应演化策略采样，非常先进的优化算法)

    贝叶斯优化（TPESampler）:
    基于过去的试验结果来选择新的参数组合，通常可以更快地找到好的解。
    当参数间的依赖关系比较复杂时，可能会更有优势。
    
    遗传算法（NSGAIISampler）:
    遗传算法是一种启发式的搜索方法，通过模拟自然选择过程来探索参数空间。
    对于非凸或非线性问题可能表现良好。
    
    协方差矩阵自适应演化策略（CmaEsSampler）:
    CMA-ES是一种先进的优化算法，适用于连续空间的非凸优化问题。
    通常需要较多的计算资源，但在一些困难问题上可能会表现得非常好。
    
    如果你的计算资源充足，网格搜索可能是一个可靠的选择，因为它可以穷举所有的参数组合来找到最优解。
    如果你希望在较短的时间内得到合理的解，随机搜索和贝叶斯优化可能是更好的选择。
    如果你面临的是一个非常复杂或非线性的问题，遗传算法和CMA-ES可能值得尝试。
    """
    
    optuna.logging.set_verbosity(optuna.logging.ERROR) 
#     def lgb_median_absolute_error(y_true, y_pred):
#         return 'median_absolute_error', median_absolute_error(y_true, y_pred), False 
    
#     lb = LabelBinarizer()
#     Y_lb = lb.fit_transform(Y) 
#     def custom_eval_metric(y_true, y_pred):
#         predicted_categories = np.argmax(y_pred, axis=1).reshape(-1, 1)
#         y_pred_cate = lb.inverse_transform(predicted_categories) 
        
#         y_pred_ = y_pred_cate.copy().astype(np.float64) 
#         y_true_ = y_true.copy().astype(np.float64)
#         return 'median_absolute_error', median_absolute_error(y_true_, y_pred_), False 

#     def custom_eval_metric2(y_true, y_pred):
#         y_pred_ = y_pred.copy().astype(np.float64) 
#         y_true_ = y_true.copy().astype(np.float64)
#         return  mean_absolute_error(y_true_, y_pred_)
    
    
    # tpe params
#     tpe_params = TPESampler.hyperopt_parameters()
#     tpe_params['n_startup_trials'] = 30
    samplers = {
#                 'grid': optuna.samplers.GridSampler(), 
                'random': optuna.samplers.RandomSampler(), 
#                 'anneal': optuna.samplers, 
#                 'tpe': optuna.samplers.TPESampler(**tpe_params), 
                'tpe': optuna.samplers.TPESampler(), 
                'cma': optuna.samplers.CmaEsSampler(), 
                'nsgaii': optuna.samplers.NSGAIISampler()} 
#     optuna.logging.set_verbosity(optuna.logging.ERROR) 
    
    if isinstance(Y, pd.DataFrame):
        Y = Y.values.reshape(-1) 
    if objective_type == 'regression':
        objective_list = ['regression', 'regression_l1', 'quantile']  # ['regression', 'regression_l1', 'quantile','huber', 'mape']
    elif objective_type == 'binary':
        objective_list = ['binary'] 
    elif objective_type == 'multiclass':
        objective_list = ['softmax', 'multiclassova'] 
        
    def objective(trial):
        #params
        param_grid = { 
                        'boosting_type': trial.suggest_categorical("boosting_type", ['gbdt', 'dart', 'rf']), # 'gbdt', 'dart', 'rf'
                        'objective': trial.suggest_categorical("objective", objective_list), 
                        "n_estimators": trial.suggest_int("n_estimators", 100, 3000),
                        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
                        "num_leaves": trial.suggest_int("num_leaves", 2, 63),
                        "max_depth": trial.suggest_int("max_depth", 2, 31),
                        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 300),
                        'min_sum_hessian_in_leaf': trial.suggest_float("min_sum_hessian_in_leaf", 1e-8, 5, log=True), 
                        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10, log=True),
                        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10, log=True),
                        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-8, 10, log=True),
                        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1, step=0.05),
                        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7), 
                        "feature_fraction": trial.suggest_float("feature_fraction", 0.05, 1, step=0.02),
                        'feature_fraction_bynode':  trial.suggest_float("feature_fraction_bynode", 0.05, 1, step=0.02),
                        'max_bin': trial.suggest_int("max_bin", 63, 511), # 默认255 
                        'min_data_in_bin': trial.suggest_int("min_data_in_bin", 1, 20),
                        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 50, 50), 
                        'seed': trial.suggest_int('seed', 0, 0), 
                        'n_jobs': trial.suggest_int('n_jobs', n_jobs_est, n_jobs_est) 
                     } 
        if objective_type == 'regression':
            param_grid.update({"alpha": trial.suggest_float("alpha", 0.5, 0.5), 
                              'metric': trial.suggest_categorical("metric", ['', 'mae', 'mse', 
                                                                             'quantile', 'huber']), 
                              }) 
            cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
            if len(cat_cols)>0: 
                param_grid.update({"categorical_features": trial.suggest_categorical("categorical_features", [cat_cols]),
                                  'cat_smooth': trial.suggest_float("cat_smooth", 1e-5, 200),
                                   'cat_l2': trial.suggest_float("cat_l2", 1e-5, 200)
                                  }) 
        elif objective_type == 'binary': 
            param_grid.update({'class_weight': trial.suggest_categorical("class_weight", ['balanced', None]), 
                              'metric': trial.suggest_categorical("metric", ['', 'binary_logloss', 'average_precision', 
                                                                             'auc'])
                              }) 
        elif objective_type == 'multiclass':
            param_grid.update({'class_weight': trial.suggest_categorical("class_weight", ['balanced', None]), 
                              'num_class': trial.suggest_int('num_class', np.unique(Y).shape[0], 
                                                            np.unique(Y).shape[0]),
                              'metric': trial.suggest_categorical("metric", [ 'multi_logloss'])  #  '',  'auc_mu', 'multi_error'
                              }) 
        else:
            raise ValueError('objective_type error')
            
        if use_gpu == True:
            gpu_params = {'device_type': trial.suggest_categorical("device_type", ['gpu']),   # cpu, gpu, cuda 
                          'gpu_platform_id': trial.suggest_categorical("gpu_platform_id", [0]),  
                          'gpu_device_id': trial.suggest_categorical("gpu_device_id", [0])} 
            param_grid.update(gpu_params)
        
        # est : 
#         est_lgb = lgb.LGBMClassifier(**param_grid)
#         score = cross_val_score(est_lgb, X, Y, 
#                                 cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=0), 
#                                 scoring=scoring).mean() 
#         return score 

    
        # 交叉验证
        if objective_type in ('binary', 'multiclass'):
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=100) 
            kf_split = kf.split(X, Y)
        else:
            kf = KFold(n_splits=cv, shuffle=True, random_state=0) 
            kf_split = kf.split(X)
        
        if objective_type in ('binary', 'multiclass'):
            if scoring in ('accuracy_score', 'balanced_accuracy_score'): 
                oof_pred = pd.Series([None]*X.shape[0]) 
            elif scoring in ('average_precision', 'roc_auc', 'log_loss', 'multi_logloss'):
                oof_pred = np.zeros([Y.shape[0], np.unique(Y).shape[0]])
                oof_pred_balAccu = pd.Series([None]*X.shape[0])
            else:
                raise ValueError('check scoring')
        else:
            oof_pred = np.zeros(X.shape[0]) 

        for idx, (train_idx, test_idx) in enumerate(kf_split): 
            X_train, X_val = X.iloc[train_idx], X.iloc[test_idx] 
            y_train, y_val = Y[train_idx], Y[test_idx] 
            if objective_type in ('binary', 'multiclass'):
                # LGBM建模
                
                preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X_train.select_dtypes(include=['int', 'float']).columns.tolist()),
        #                                                             ('cat', OneHotEncoder(), X_train.select_dtypes(include=['category', 'object']).columns.tolist())
                                                                  ]) 
                model = Pipeline([('preprocessor', preprocessor),
                                        ('est', lgb.LGBMClassifier(**param_grid, verbose=-2))]) 
                model.fit(X_train, y_train, est__eval_set=[(X_val, y_val)],
    #                 callbacks=[LightGBMPruningCallback(trial, "average_precision")]
                        ) 
            else:
                preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X_train.select_dtypes(include=['int', 'float']).columns.tolist()),
        #                                                             ('cat', OneHotEncoder(), X_train.select_dtypes(include=['category', 'object']).columns.tolist())
                                                                  ]) 
                model = Pipeline([('preprocessor', preprocessor),
                                        ('est', lgb.LGBMRegressor(**param_grid, verbose=-2) )]) 
                model.fit(X_train, y_train, est__eval_set=[(X_val, y_val)],
    #                 callbacks=[LightGBMPruningCallback(trial, "average_precision")] 
                        )
            # 预测
            if scoring in ('average_precision', 'roc_auc', 'log_loss', 'multi_logloss'):
                y_pred_prob = model.predict_proba(X_val, num_iteration=model.named_steps['est'].best_iteration_) 
                oof_pred[test_idx] = y_pred_prob 
            elif scoring in ('accuracy_score', 'balanced_accuracy_score'): 
                preds = model.predict(X_val, num_iteration=model.named_steps['est'].best_iteration_)
                oof_pred.iloc[test_idx] = preds 
                
            elif scoring in ('mean_absolute_error', 'mean_squared_error', 'median_absolute_error'): 
                preds = model.predict(X_val, num_iteration=model.named_steps['est'].best_iteration_)
                oof_pred[test_idx] = preds 
            else:
                raise ValueError('check scoring')
        
        if scoring == 'average_precision':
            lb = LabelBinarizer()
            # 字母序encoding
            Y_lb = lb.fit_transform(Y) 
            score = average_precision_score(Y_lb, oof_pred) 
        elif scoring == 'roc_auc':
            lb = LabelBinarizer()
            Y_lb = lb.fit_transform(Y) 
            score = roc_auc_score(Y_lb, oof_pred) 
        elif scoring in ('log_loss', 'multi_logloss'):
            lb = LabelBinarizer() 
            Y_lb = lb.fit_transform(Y) 
            score = log_loss(Y_lb, oof_pred) 
        elif scoring in ('accuracy_score'):
            score = accuracy_score(Y, oof_pred) 
        elif scoring in ( 'balanced_accuracy_score'):
#             score = balanced_accuracy_score(Y, oof_pred) 
#             print(oof_pred) 
            score = custom_eval_metric2(Y, oof_pred)
            
        elif scoring == 'mean_absolute_error':
            score = mean_absolute_error(Y, oof_pred)
        elif scoring == 'mean_squared_error':
            score = mean_squared_error(Y, oof_pred)
        elif scoring == 'median_absolute_error':
            score = median_absolute_error(Y, oof_pred) 
            score2 = mean_absolute_error(Y, oof_pred)
        else:
            raise ValueError('check scoring')  
        return score 


    study = optuna.create_study(direction=direction, 
                                sampler=samplers[sampler], 
#                                 pruner=optuna.pruners.HyperbandPruner(max_resource='auto'), 
#                                 pruner=optuna.pruners.MedianPruner(interval_steps=20, n_min_trials=8),
                                storage=optuna.storages.RDBStorage(url="sqlite:///db.sqlite3", 
                                                                   engine_kwargs={"connect_args": {"timeout": 500}}),  # 指定database URL 
                                study_name= '%s_%s_%s'%(study_name, sampler, X.shape[1]), load_if_exists=True) 
    study.optimize(objective, n_trials=n_iter, show_progress_bar=True, n_jobs=n_jobs_optuna) 
    print('Best lgbm params:', study.best_trial.params) 
    print('Best CV score:', study.best_value) 
    
    optuna.visualization.plot_optimization_history(study).show() 
#     optuna.visualization.plot_intermediate_values(study).show() 
#     optuna.visualization.plot_parallel_coordinate(study).show() 
#     optuna.visualization.plot_parallel_coordinate(study, params=["max_depth", "min_samples_leaf"]).show() 
#     optuna.visualization.plot_contour(study).show() 
#     optuna.visualization.plot_contour(study, params=["max_depth", "min_samples_leaf"]).show() 
#     optuna.visualization.plot_slice(study).show() 
    optuna.visualization.plot_param_importances(study).show() 
    params_df = study.trials_dataframe() 
    params_df = params_df.sort_values(by=['value']) 
    return params_df, study 



def kF_oof_score_And_predictTest_LGB(X, Y, test, params, n_split=6, 
                                     objective_type='regression', metric='average_precision'):
        # 交叉验证
        # # params参数中应该有early_stopping_rounds  
    lb = LabelBinarizer()
    Y_lb = lb.fit_transform(Y) 
    def custom_eval_metric(y_true, y_pred):
        predicted_categories = np.argmax(y_pred, axis=1).reshape(-1, 1)
        y_pred_cate = lb.inverse_transform(predicted_categories) 
        
        y_pred_ = y_pred_cate.copy().astype(np.float64) 
        y_true_ = y_true.copy().astype(np.float64)
        return 'median_absolute_error', median_absolute_error(y_true_, y_pred_), False 

        
        
    if objective_type in ('binary', 'multiclass'):
        kf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=0) 
        kf_split = kf.split(X, Y)
    else:
        kf = KFold(n_splits=n_split, shuffle=True, random_state=0) 
        kf_split = kf.split(X)
        
    if objective_type =='regression':
        oof_pred = np.zeros(X.shape[0]) 
        test_pred = np.zeros(test.shape[0])
    if objective_type == 'binary':
        oof_pred = np.zeros(X.shape[0]) 
        oof_pred_accu = pd.Series([None] * X.shape[0]) 
        test_pred = np.zeros(test.shape[0])
    if objective_type == 'multiclass':
        oof_pred = np.zeros((X.shape[0], Y[Y.columns[0]].nunique()))
        oof_pred_accu = pd.Series([None] * X.shape[0]) 
        test_pred_soft = np.zeros((test.shape[0], Y[Y.columns[0]].nunique()))
        test_pred_hard = pd.DataFrame()

    
    if isinstance(Y, pd.DataFrame):
        Y = Y.values.reshape(-1) 
        
    for idx, (train_idx, val_idx) in enumerate(kf_split): 
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx] 
        y_train, y_val = Y[train_idx], Y[val_idx] 
        if objective_type in ('binary', 'multiclass'):
            # LGBM建模            
            preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X_train.select_dtypes(include=['int', 'float']).columns.tolist()),
        #                                                             ('cat', OneHotEncoder(), X_train.select_dtypes(include=['category', 'object']).columns.tolist())
                                                                  ]) 
            model = Pipeline([('preprocessor', preprocessor),
                              ('est', lgb.LGBMClassifier(**params, verbose=-2))]) 
            model.fit(X_train, y_train, est__eval_set=[(X_val, y_val)]) 
            # 模型预测val
            if objective_type == 'binary':
                preds = model.predict_proba(X_val, num_iteration=model.named_steps['est'].best_iteration_)[:, 1]
            if objective_type == 'multiclass':
                preds = model.predict_proba(X_val, num_iteration=model.named_steps['est'].best_iteration_)
            oof_pred[val_idx] = preds 
            preds_accu = model.predict(X_val, num_iteration=model.named_steps['est'].best_iteration_)
            oof_pred_accu.iloc[val_idx] = preds_accu
            # soft vote 
            preds_test_soft = model.predict_proba(test, num_iteration=model.named_steps['est'].best_iteration_)
            test_pred_soft += preds_test_soft / n_split 
            # hard vote 
            preds_test_hard = model.predict(test, num_iteration=model.named_steps['est'].best_iteration_)
            test_pred_hard[str(test_pred_hard.shape[1])] = preds_test_hard
            
        if objective_type in ('regression'):
            # LGBM建模
            def lgb_median_absolute_error(y_true, y_pred):
                return 'median_absolute_error', median_absolute_error(y_true, y_pred), False 
            preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X_train.select_dtypes(include=['int', 'float']).columns.tolist()),
        #                                                             ('cat', OneHotEncoder(), X_train.select_dtypes(include=['category', 'object']).columns.tolist())
                                                                  ]) 
            model = Pipeline([('preprocessor', preprocessor),
                              ('est', lgb.LGBMRegressor(**params, verbose=-2))]) 
            model.fit(X_train, y_train,
                        eval_set=[(X_val, y_val)]) 
            # 模型预测val 
            preds = model.predict(X_val, num_iteration=model.best_iteration_) 
            oof_pred[val_idx] = preds 
            # predict test 
            preds_test = model.predict(test, num_iteration=model.best_iteration_) 
            test_pred += preds_test / n_split

    if metric == 'average_precision':
        score = average_precision_score(Y, oof_pred) 
        score2 = roc_auc_score(Y, oof_pred) 
    elif metric in ('log_loss', 'multi_logloss'):
        score = log_loss(Y, oof_pred) 
    elif metric in ('balanced_accuracy_score', 'accuracy_score'):
        score = balanced_accuracy_score(Y, oof_pred_accu)  
    elif metric == 'mean_absolute_error':
        score = mean_absolute_error(Y, oof_pred)
    elif metric == 'median_absolute_error':
        score = median_absolute_error(Y, oof_pred)
    else: 
        raise ValueError('scoring TODO')
    return score, oof_pred, test_pred_hard, test_pred_soft





params_stack_1, study_lgb_stack_1 = optunaSearchCVParams_LGBM(oof_combined_vars29, y, 
                                                                cv=6, 
                                                                n_iter=100, 
                                                                sampler='tpe',  
                                                                study_name = 'lgb_stack_oof_01',
                                                                objective_type='multiclass',  
                                                                scoring='multi_logloss', 
                                                                direction='minimize', 
                                                                n_jobs_est= -1, n_jobs_optuna=1, use_gpu=False)  






stack_lgb_prms = {'boosting_type': 'gbdt', 'objective': 'multiclassova', 'n_estimators': 993, 'learning_rate': 0.09933452659495313, 'num_leaves': 43, 'max_depth': 3, 'min_data_in_leaf': 128, 'min_sum_hessian_in_leaf': 0.00030481205859573526, 'lambda_l1': 0.001209060777348702, 'lambda_l2': 6.7064433568734e-07, 'min_gain_to_split': 0.015252027808765105, 'bagging_fraction': 0.6, 'bagging_freq': 3, 'feature_fraction': 0.8300000000000001, 'feature_fraction_bynode': 0.33, 'max_bin': 206, 'min_data_in_bin': 6, 'early_stopping_rounds': 50, 'seed': 0, 'n_jobs': -1, 'class_weight': None, 'num_class': 3, 'metric': 'multi_logloss'}
stack_lgb_prms['n_estimators'] = 10000 
stack_lgb_prms['early_stopping_rounds'] = 200 

score, oof_pred, test_pred_hard, test_pred_soft = kF_oof_score_And_predictTest_LGB(oof_combined_vars29, 
                                                                                   y, 
                                                                                   test_combined_vars29, 
                                                                                   stack_lgb_prms, 
                                                                                   n_split=6, 
                                                                                   objective_type='multiclass', 
                                                                                   metric='multi_logloss') 





submit = pd.read_csv('./data/sample_submission.csv') 
submit.iloc[:, 1:] = test_pred_soft
submit.columns=['id', 'Status_D', 'Status_C', 'Status_CL']   
submit.to_csv('./data/submit/stack_lgb_submit005.csv', index=False) 




# In[ ]:





# In[ ]:




