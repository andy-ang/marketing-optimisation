# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:23:23 2025

@author: andya
"""

import os
import pandas as pd
import numpy as np

os.chdir("C:\\Users\\andya\\Desktop\\marketing optimisation\\")
os.getcwd()

# =============================================================================
# import data
# =============================================================================
soc_dem = pd.read_excel('data input\\'+ 'DataScientist_CaseStudy_Dataset.xlsx', sheet_name='Soc_Dem', index_col = 'Client')
pdt_acct_bal = pd.read_excel('data input\\'+ 'DataScientist_CaseStudy_Dataset.xlsx', sheet_name='Products_ActBalance', index_col = 'Client')
inflow_outflow = pd.read_excel('data input\\'+ 'DataScientist_CaseStudy_Dataset.xlsx', sheet_name='Inflow_Outflow', index_col = 'Client')
sales_rev = pd.read_excel('data input\\'+ 'DataScientist_CaseStudy_Dataset.xlsx', sheet_name='Sales_Revenues', index_col = 'Client')


# =============================================================================
# pre-processing data
# =============================================================================
# assuming soc_dem contains all customers, sales_rev Sale columns are the targets
# left joining all information to soc_dem to create an input data set for train / validate / test
input_data = pd.merge(soc_dem, pdt_acct_bal, how = 'left', left_index = True, right_index = True)
input_data = pd.merge(input_data, inflow_outflow, how = 'left', left_index = True, right_index = True)
input_data = pd.merge(input_data, sales_rev[['Sale_MF','Sale_CC','Sale_CL']], how = 'left', left_index = True, right_index = True)

# one hot for categorical columns
# in addition, since all data are transactions, balances and target, i assume missing/nan = 0
cate_cols = ['Sex']
cate_data = pd.get_dummies(input_data[cate_cols])
input_data = pd.concat([input_data[input_data.columns[~input_data.columns.isin(cate_cols)]], cate_data], axis=1)
input_data = input_data.fillna(0)


# =============================================================================
# split data into train/test
# finding the best hyperparameter for the model
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import scipy.stats as stats
import joblib

# split data into train/test
target_cols = ['Sale_MF','Sale_CC','Sale_CL']
input_cols = input_data.columns[~input_data.columns.isin(target_cols)]

d_train, d_test, lab_train, lab_test = train_test_split(input_data[input_cols], input_data[target_cols], test_size=0.30, random_state=42, stratify=input_data[target_cols])

# set the hyperparameter search space
param_dist = {
    'max_depth': stats.randint(3, 10),
    'learning_rate': stats.uniform(0.01, 0.1),
    'subsample': stats.uniform(0.5, 0.5),
    'n_estimators':stats.randint(10,200),
    'min_child_weight': stats.randint(3, 10)
}

# get best hyperparameter for each model and save them
for target in target_cols:
    xgb_model = xgb.XGBClassifier()
    
    # RandomizedSearchCV to find best hyperparameters, lots of other ways to do this like bayesian optimisation, gridsearchcv
    # scoring = f1 as data is imbalanced
    random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=30, cv=3, scoring='f1', random_state=42)
    random_search.fit(d_train, lab_train[target])
    
    # saving best params for model building later
    joblib.dump(random_search.best_params_, 'model and params\\' + target + '_params.pkl')
    

# =============================================================================
# build models using saved params
# calibrate probabilities
# =============================================================================
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

for target in target_cols:
    # load best params
    xgb_params = joblib.load('model and params\\' + target + '_params.pkl')
    xgb_model = xgb.XGBClassifier(learning_rate = xgb_params['learning_rate'],
                                  max_depth = xgb_params['max_depth'],
                                  min_child_weight = xgb_params['min_child_weight'],
                                  n_estimators = xgb_params['n_estimators'],
                                  subsample = xgb_params['subsample']
                                  )
    
    xgb_model = xgb_model.fit(d_train, lab_train[target])
    uncali_pred = xgb_model.predict_proba(d_test)[:, 1]
    
    # calibrate the probabilities to have a better reflection of actual likelihood, i do this because we want to compare across models
    # usually not advisable to compare different model scores as they dont mean the same thing i.e. 0.9 for model A is not better than 0.6 of model B
    cali_xgb_model = CalibratedClassifierCV(xgb_model, method='sigmoid', cv='prefit')
    cali_xgb_model.fit(d_train, lab_train[target])
    cali_pred = cali_xgb_model.predict_proba(d_test)[:, 1]
    
    # save model
    joblib.dump(xgb_model, 'model and params\\' + target + '_uncali_model.pkl')
    joblib.dump(cali_xgb_model, 'model and params\\' + target + '_cali_model.pkl')
    
    # calibration plots, to check if calibration has made the probability distribution reflect better of actual likely hood
    prob_true_uncal, prob_pred_uncal = calibration_curve(lab_test[target], uncali_pred, n_bins=10, strategy='uniform')
    prob_true_cal, prob_pred_cal = calibration_curve(lab_test[target], cali_pred, n_bins=10, strategy='uniform')
    
    plt.figure(figsize=(10, 7))
    plt.plot(prob_pred_uncal, prob_true_uncal, marker='o', label='Uncalibrated', color='red')
    plt.plot(prob_pred_cal, prob_true_cal, marker='o', label='Calibrated', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.title('Calibration Curve')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.grid()
    plt.savefig('plots//' + target + "_cali_curve.png", bbox_inches="tight")


# =============================================================================
# predict on test set
# target optimisation based on highest expected rev per customer
# =============================================================================
pred_prob = pd.DataFrame(index=d_test.index)

# predict on test, get probabilties and expected revenue
for target in target_cols:
    rev_col = target.replace("Sale", "Revenue")
    xgb_model = joblib.load('model and params\\' + target + '_cali_model.pkl')
    pred_prob[target.replace("Sale", "Prob")] = xgb_model.predict_proba(d_test)[:, 1]
    # get expected revenue based on probabilities * average revenue of the product
    pred_prob[target.replace("Sale", "Exp_Rev")] = pred_prob[target.replace("Sale", "Prob")] * np.mean(sales_rev[sales_rev[rev_col]>0][rev_col])
    # for checking performance for top 15%, i want to compare with random baseline
    pred_prob[target.replace("Sale", "Ind")] = pred_prob[target.replace("Sale", "Prob")].apply(lambda x: 1 if x <= pred_prob[target.replace("Sale", "Prob")].quantile(0.15) else 0)

# target optimsation, each customer only have 1 recommendation based on highest expected rev
rec_prod = list()
for each_pred in range(0, len(pred_prob)):
    if pred_prob['Exp_Rev_MF'].iloc[each_pred] == pred_prob[['Exp_Rev_MF','Exp_Rev_CC','Exp_Rev_CL']].max(axis=1).iloc[each_pred]:
        rec_prod.append('MF')
    elif pred_prob['Exp_Rev_CC'].iloc[each_pred] == pred_prob[['Exp_Rev_MF','Exp_Rev_CC','Exp_Rev_CL']].max(axis=1).iloc[each_pred]:
        rec_prod.append('CC')
    else: rec_prod.append('CL')

pred_prob['Exp_Rev_Max'] = pred_prob[['Exp_Rev_MF','Exp_Rev_CC','Exp_Rev_CL']].max(axis=1)
pred_prob['rec_prod'] = rec_prod

# calculate average expected revenue per targeted customer for top 15% customers
avg_exp_rev_per_cx = sum(pred_prob[pred_prob['Exp_Rev_Max'] > pred_prob['Exp_Rev_Max'].quantile(0.85)]['Exp_Rev_Max'])/len(pred_prob[pred_prob['Exp_Rev_Max'] > pred_prob['Exp_Rev_Max'].quantile(0.85)]['Exp_Rev_Max'])

# save prediction output
pred_prob.to_csv('data output\\' + 'test_recommendation.csv')

# x = pd.merge(pred_prob, d_test, how = 'left', left_index = True, right_index = True)
# x = pd.merge(x, lab_test, how = 'left', left_index = True, right_index = True)
# x.to_csv('final.csv')


# =============================================================================
# SHAP explainer to understand how each feature affects model's prediction
# =============================================================================
import shap

for target in target_cols:
    
    # load uncali model, cali model doesn't work with shap package
    xgb_model = joblib.load('model and params\\' + target + '_uncali_model.pkl')
    explainer = shap.TreeExplainer(xgb_model)
    
    # calculate shap values
    shapValues = explainer.shap_values(d_test)
    
    # get summary plot
    # this gives us a summary of how how each feature affects model's prediction, but it cant tell us which value range
    shap.summary_plot(shapValues, d_test)

    plt.savefig('plots//' + target + '_shap_summary_plot.png', bbox_inches='tight')
    plt.clf()

# get dependancy plot for each feature
# plotting individually will provide us with the feature range.

# target = 'Sale_MF'
# xgb_model = joblib.load(target+'_uncali_model.pkl')
# explainer = shap.TreeExplainer(xgb_model)
# shapValues = explainer.shap_values(d_test)

# d_test.columns

# feature = 'TransactionsDeb'
# shap.dependence_plot(feature, shapValues, d_test, interaction_index=None)
# plt.savefig('plots//' + target + '_' + feature + '_shap_dep_plot.png', bbox_inches='tight')










