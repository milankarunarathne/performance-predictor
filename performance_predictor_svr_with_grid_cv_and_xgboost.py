import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVR
from xgboost import XGBRegressor
import sklearn.metrics as mx
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import joblib
import xgboost as xgb
from sklearn.metrics import make_scorer, mean_absolute_error
import time


summary_data = 'resources/train/old/wso2apimanagerperformanceresults.csv'
summary_data_test = 'resources/test/old/wso2apimanagerperformanceresults.csv'
t_splitter = ","

csv_select_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 26, 27, 28, 29, 52, 53, 54]
# 4 = M * C, 5 =  M * S, 6 = S * C, 7 = M * S * C, 8 =	M / C,
# 9 = M / S, 10 = C / M, 11 = C / S, 12 = S / M, 13 = S / C, 14 = M / S * C, 15 = S / M * C,
# 16 =	C / M * S, 17 = 1 / M * C * S, 18 = N * C
x_select_cols = [0, 1, 2, 3, 4, 6]  # select columns to x (features)
y_select_col_latency = 20
y_select_col_90th_percentile = 21
y_select_col_95th_percentile = 22
y_select_col_99th_percentile = 23
y_select_col_throughput = 24
y_select_col_load_average_1_minute = 25
y_select_col_load_average_5_minute = 26
y_select_col_load_average_15_minute = 27
t_size = 0.0  # percentage for testing (test size)
n_rows = 117   # total rows
row_start = 25  # testing rows at start
r_seed = 42  # seed of random (random seed)
kernel_type_array = ['rbf', 'poly', 'linear']
c_array = [1E-3, 1E-2, 1E-1, 1E0, 1E1, 1E2, 1E3, 1E4, 1E5, 1E6]
epsilion_array = [0.001, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
gamma_default = 'auto'


def data_reader(csv_file, total_row, thousands_splitter, csv_select_columns, x_column_numbers, y_column_number):
    # time5 = time.time()
    # read the file
    data_file = pd.read_csv(csv_file, thousands=thousands_splitter, usecols=csv_select_columns,)
    #  replace Echo API and Mediation API with 1 and 2
    datapd = pd.DataFrame.replace(data_file, to_replace=['Echo API', 'Mediation API'], value=[1, 2])
    data = np.array(datapd, copy=True, )
    dataset_row_n = data[0:total_row, :]  # select specific number of rows
    x = preprocessing.scale(dataset_row_n[:, x_column_numbers])  # machine learning to be in a range of -1 to 1.
    # x = dataset_row_n[:, x_column_numbers]
    # This may do nothing, but it usually speeds up processing and can also help with accuracy.
    # Because this range is so popularly used
    y = data[:, y_column_number]
    # print (y)
    # print ("time 5", time.time()-time5)  # 0.0036
    return x, y


def mean_absolute_percentage_error(y_true, y_pred):
    """Mean absolute error regression loss

    Read more in the :ref:`User Guide <mean_absolute_error>`.

    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : float or ndarray of floats
        If multioutput is 'raw_values', then mean absolute error is returned
        for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.

        MAE output is non-negative floating point. The best value is 0.0.
    """
    output_errors = 100*np.average(np.abs(y_true - y_pred)/y_true)
    return np.average(output_errors)


def evaluator(y_real_ev, y_pred_ev):
    confidence_rmse = np.sqrt(mx.mean_squared_error(y_true=y_real_ev, y_pred=y_pred_ev))
    confidence_mape = mean_absolute_percentage_error(y_true=y_real_ev, y_pred=y_pred_ev)
    return confidence_rmse, confidence_mape


def array_print(array_get):
    for i in range(0, len(array_get)):
        print array_get[i]
    print "okay"


time1 = time.time()
time2 = time.time()
# ###################################################################################
# throughput
print "\n\n\nThroughput "

data_split_throughput = np.array([], dtype='float64')
data_split_throughput = data_reader(csv_file=summary_data, total_row=n_rows,
                                    thousands_splitter=t_splitter, csv_select_columns=csv_select_cols,
                                    x_column_numbers=x_select_cols, y_column_number=y_select_col_throughput)

data_split_throughput_test = np.array([], dtype='float64')
data_split_throughput_test = data_reader(csv_file=summary_data_test, total_row=n_rows, thousands_splitter=t_splitter,
                                         csv_select_columns=csv_select_cols, x_column_numbers=x_select_cols,
                                         y_column_number=y_select_col_throughput)

#  ##################################################################################
# print "\n\n\nSVR Grid Search CV "
# parameters = {'kernel': kernel_type_array, 'C': c_array, 'epsilon': epsilion_array}
# svr = SVR()
# svr_model = GridSearchCV(svr, parameters, cv=10, n_jobs=6, return_train_score=False, scoring='neg_median_absolute_error')
# svr_model.fit(data_split_throughput[0], data_split_throughput[1])
# y_prd = svr_model.predict(data_split_throughput[0])
# print 'train data'
# print array_print(data_split_throughput[1])
# print 'train pred'
# print array_print(y_prd)
# y_prd_test = svr_model.predict(data_split_throughput_test[0])  # predict y values by train x_train
# print 'test data'
# print array_print(data_split_throughput_test[1])
# print 'test pred data'
# print array_print(y_prd_test)
# print svr_model
# y_prd = svr_model.predict(data_split_throughput[0])
# y_prd_test = svr_model.predict(data_split_throughput_test[0])  # predict y values by train x_train
# current_result_array = evaluator(y_train_ev=data_split_throughput[1], y_pred_ev=y_prd)
# arr_rmse_in = current_result_array[0]
# arr_mape_in = current_result_array[1]
# current_result_array_test = evaluator(y_train_ev=data_split_throughput_test[1], y_pred_ev=y_prd_test)
# arr_rmse_in_test = current_result_array_test[0]
# arr_mape_in_test = current_result_array_test[1]
# # #############################################################################
# print "\n\n\nresults "
# print "rmse"
# print arr_rmse_in
# print "mape"
# print arr_mape_in
# print "test rmse"
# print arr_rmse_in_test
# print "test mape"
# print arr_mape_in_test
#
# print svr_model.best_params_

print 'time', time.time()-time2
# #############################################################################
print "\n\n\nXGBoost "




#train=pd.read_csv('')
# TrainSet=np.loadtxt('Trainset.csv',delimiter=',',skiprows=1)
# print(TrainSet.shape)
# features = TrainSet[:,1:99]
# labels = TrainSet[:,99]

# tr_features, ts_features, tr_labels, ts_labels = train_test_split(features,labels, test_size=0.30, random_state=42)

# imputer=Imputer(missing_values=-191, strategy='median')
# tr_features=imputer.fit_transform(tr_features)
# ts_features=imputer.transform(ts_features)

# data=xgb.DMatrix(data=tr_features,label=tr_labels)
# test=xgb.DMatrix(data=ts_features)
#
# eval_set=xgb.DMatrix(ts_features,ts_labels)
#
# params={'booster':'gbtree','eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:linear','max_depth':3,'min_child_weight':2,'silent':1,'eval_metric':'mae','verbose':True,'gamma':0}
# bst=xgb.train(params, data, num_boost_round=7000,evals=[(eval_set,'eval')],early_stopping_rounds=10)
#
# y_pred=bst.predict(test)
#
# res=xgb.cv(params = params, dtrain = data, num_boost_round = 7000, nfold = 10, metrics = ['mae'],early_stopping_rounds = 10)
# my_score = make_scorer(mean_absolute_error)
parameters_throughput = {'max_depth' : [3], 'learning_rate': [0.1], 'n_estimators': [100], 'min_child_weight': [1], 'max_delta_step': [0], 'objective': ['reg:linear']}
xgboost_throughput = XGBRegressor(max_depth=3, learning_rate=0.2, n_estimators=100, silent=True,
                                  objective='reg:linear', gamma=0, min_child_weight=1, max_delta_step=0,
                                  subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0,
                                  reg_lambda=1, scale_pos_weight=1, base_score=0.5)
# best = GridSearchCV(xgboost_throughput,parameters_throughput , n_jobs=5,  cv=10, refit=True, return_train_score=True)
# bst = best(data_split_throughput[0], data_split_throughput[1])
# print best.cv_results_

xgboost_throughput.fit(data_split_throughput[0], data_split_throughput[1])

print array_print(data_split_throughput_test[1])
predict_throughput = xgboost_throughput.predict(data_split_throughput_test[0])
print array_print(predict_throughput)

print array_print(evaluator(y_real_ev=data_split_throughput[1], y_pred_ev=xgboost_throughput.predict(data_split_throughput[0])))
print array_print(evaluator(y_real_ev=data_split_throughput_test[1], y_pred_ev=predict_throughput))
# print predict_final_semi
# res = xgb.cv(params=params, dtrain=data, num_boost_round=200, nfold=10, metrics= ['mae'], early_stopping_rounds=10)
#
#
# predict_final = bst.predict(test)
# print predict_final

# xgb_model = XGBRegressor(max_depth=4,learning_rate=0.06, n_estimators=100,silent=True, nthread=6,gamma=0,max_delta_step=0,subsample=1,scale_pos_weight=1,base_score=0.5,seed=37)  # not parameters for tune
# xg_model = xgb.train()
# xgb_model.fit(data_split_throughput[0], data_split_throughput[1])
# y_prd = xgb_model.predict(data_split_throughput[0])
# print 'train data'
# print array_print(data_split_throughput[1])
# print 'train pred'
# print array_print("wait")
# y_prd_test = bst.predict(test)  # predict y values by train x_train
# print 'test data'
# print array_print(data_split_throughput_test[1])
# print 'test pred data'
# print array_print(y_prd_test)
# current_result_array = evaluator(y_train_ev=data_split_throughput[1], y_pred_ev=y_pred)
# arr_rmse_in_xg = current_result_array[0]
# arr_mape_in_xg = current_result_array[1]
# current_result_array_test = evaluator(y_train_ev=data_split_throughput_test[1], y_pred_ev=y_prd_test)
# arr_rmse_in_test_xg = current_result_array_test[0]
# arr_mape_in_test_xg = current_result_array_test[1]
# # #################################################################################
#
# print "rmse xg"
# print arr_rmse_in_xg
# print "mape xg"
# print arr_mape_in_xg
# print "test rmse xg"
# print arr_rmse_in_test_xg
# print "test mape xg"
# print arr_mape_in_test_xg

print (time.time()-time2)

# #
# time2 = time.time()
# # ###################################################################################
# latency
print "\n\n\nLatency "

data_split_latency = np.array([], dtype='float64')
data_split_latency = data_reader(csv_file=summary_data, total_row=n_rows,
                                    thousands_splitter=t_splitter, csv_select_columns=csv_select_cols,
                                    x_column_numbers=x_select_cols, y_column_number=y_select_col_latency)

data_split_latency_test = np.array([], dtype='float64')
data_split_latency_test = data_reader(csv_file=summary_data_test, total_row=n_rows, thousands_splitter=t_splitter,
                                         csv_select_columns=csv_select_cols, x_column_numbers=x_select_cols,
                                         y_column_number=y_select_col_latency)

# #  ##################################################################################
# # SVR Grid Search to latency
# print "\n\n\nSVR Grid Search CV "
# parameters = {'kernel': kernel_type_array , 'C': c_array, 'epsilon': epsilion_array}
# svr = SVR()
# svr_model = GridSearchCV(svr, parameters)
# svr_model.fit(data_split_latency[0], data_split_latency[1])
# y_prd = svr_model.predict(data_split_latency[0])
# print 'train data'
# print array_print(data_split_latency[1])
# print 'train pred'
# print array_print(y_prd)
# y_prd_test = svr_model.predict(data_split_latency_test[0])  # predict y values by train x_train
# print 'test data'
# print array_print(data_split_latency_test[1])
# print 'test pred data'
# print array_print(y_prd_test)
# print svr_model
# y_prd = svr_model.predict(data_split_latency[0])
# y_prd_test = svr_model.predict(data_split_latency_test[0])  # predict y values by train x_train
# current_result_array = evaluator(y_train_ev=data_split_latency[1], y_pred_ev=y_prd)
# arr_rmse_in = current_result_array[0]
# arr_mape_in = current_result_array[1]
# current_result_array_test = evaluator(y_train_ev=data_split_latency_test[1], y_pred_ev=y_prd_test)
# arr_rmse_in_test = current_result_array_test[0]
# arr_mape_in_test = current_result_array_test[1]
# # #############################################################################
# print "\n\n\nresults "
# print "rmse"
# print arr_rmse_in
# print "mape"
# print arr_mape_in
# print "test rmse"
# print arr_rmse_in_test
# print "test mape"
# print arr_mape_in_test
#
# print svr_model.best_params_
#
# print 'time', time.time()-time2
#
# # ######################################################################################
print "\n\n\nXGBoost Latency"


parameters_latency = {'max_depth' : [3], 'learning_rate': [0.1], 'n_estimators': [100], 'min_child_weight': [1], 'max_delta_step': [0], 'objective': ['reg:linear']}
xgboost_latency = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, min_child_weight= 1, max_delta_step= 0, objective= 'reg:linear', silent=True, gamma=0)
# best = GridSearchCV(xgboost_latency,parameters_latency , n_jobs=5,  cv=10, refit=True, return_train_score=True)
# bst = best(data_split_latency[0], data_split_latency[1])
# print best.cv_results_

xgboost_latency.fit(data_split_latency[0], data_split_latency[1])

print array_print(data_split_latency_test[1])
predict_latency = xgboost_latency.predict(data_split_latency_test[0])
print array_print(predict_latency)

print array_print(evaluator(y_real_ev=data_split_latency[1], y_pred_ev=xgboost_latency.predict(data_split_latency[0])))
print array_print(evaluator(y_real_ev=data_split_latency_test[1], y_pred_ev=predict_latency))
# print predict_final_semi
# res = xgb.cv(params=params, dtrain=data, num_boost_round=200, nfold=10, metrics= ['mae'], early_stopping_rounds=10)
#
#
# predict_final = bst.predict(test)
# print predict_final

# xgb_model = XGBRegressor()  # not parameters for tune
# xgb_model.fit(data_split_latency[0], data_split_latency[1])
# y_prd = xgb_model.predict(data_split_latency[0])
# print 'train data'
# print array_print(data_split_latency[1])
# print 'train pred'
# print array_print(y_prd)
# y_prd_test = xgb_model.predict(data_split_latency_test[0])  # predict y values by train x_train
# print 'test data'
# print array_print(data_split_latency_test[1])
# print 'test pred data'
# print array_print(y_prd_test)
# current_result_array = evaluator(y_train_ev=data_split_latency[1], y_pred_ev=y_prd)
# arr_rmse_in_xg = current_result_array[0]
# arr_mape_in_xg = current_result_array[1]
# current_result_array_test = evaluator(y_train_ev=data_split_latency_test[1], y_pred_ev=y_prd_test)
# arr_rmse_in_test_xg = current_result_array_test[0]
# arr_mape_in_test_xg = current_result_array_test[1]
#
# # ########################################################################
# print "\n\n\nresults "
#
# print "rmse xg"
# print arr_rmse_in_xg
# print "mape xg"
# print arr_mape_in_xg
# print "test rmse xg"
# print arr_rmse_in_test_xg
# print "test mape xg"
# print arr_mape_in_test_xg
#
# print (time.time()-time2)
#
#
# time2 = time.time()
# # ###################################################################################
# 90th percentile
print "\n\n\n90th percentile "

data_split_90th_percentile = np.array([], dtype='float64')
data_split_90th_percentile = data_reader(csv_file=summary_data, total_row=n_rows,
                                    thousands_splitter=t_splitter, csv_select_columns=csv_select_cols,
                                    x_column_numbers=x_select_cols, y_column_number=y_select_col_90th_percentile)

data_split_90th_percentile_test = np.array([], dtype='float64')
data_split_90th_percentile_test = data_reader(csv_file=summary_data_test, total_row=n_rows, thousands_splitter=t_splitter,
                                         csv_select_columns=csv_select_cols, x_column_numbers=x_select_cols,
                                         y_column_number=y_select_col_90th_percentile)

# #  ##################################################################################
# # SVR Grid Search to 90th percentile
# print "\n\n\nSVR Grid Search CV "
# parameters = {'kernel': kernel_type_array , 'C': c_array, 'epsilon': epsilion_array}
# svr = SVR()
# svr_model = GridSearchCV(svr, parameters)
# svr_model.fit(data_split_90th_percentile[0], data_split_90th_percentile[1])
# y_prd = svr_model.predict(data_split_90th_percentile[0])
# print 'train data'
# print array_print(data_split_90th_percentile[1])
# print 'train pred'
# print array_print(y_prd)
# y_prd_test = svr_model.predict(data_split_90th_percentile_test[0])  # predict y values by train x_train
# print 'test data'
# print array_print(data_split_90th_percentile_test[1])
# print 'test pred data'
# print array_print(y_prd_test)
# print svr_model
# y_prd = svr_model.predict(data_split_90th_percentile[0])
# y_prd_test = svr_model.predict(data_split_90th_percentile_test[0])  # predict y values by train x_train
# current_result_array = evaluator(y_train_ev=data_split_90th_percentile[1], y_pred_ev=y_prd)
# arr_rmse_in = current_result_array[0]
# arr_mape_in = current_result_array[1]
# current_result_array_test = evaluator(y_train_ev=data_split_90th_percentile_test[1], y_pred_ev=y_prd_test)
# arr_rmse_in_test = current_result_array_test[0]
# arr_mape_in_test = current_result_array_test[1]
# # #############################################################################
# print "\n\n\nresults "
#
# print "rmse"
# print arr_rmse_in
# print "mape"
# print arr_mape_in
# print "test rmse"
# print arr_rmse_in_test
# print "test mape"
# print arr_mape_in_test
#
# print svr_model.best_params_
#
# print 'time', time.time()-time2
#
# ####################################################################################
print "\n\n\nXGBoost 90th percentile"

parameters_90th_percentile = {'max_depth' : [3], 'learning_rate': [0.1], 'n_estimators': [100], 'min_child_weight': [1], 'max_delta_step': [0], 'objective': ['reg:linear']}
xgboost_90th_percentile = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, min_child_weight= 1, max_delta_step= 0, objective= 'reg:linear', silent=True, gamma=0)
# best = GridSearchCV(xgboost_90th_percentile,parameters_90th_percentile , n_jobs=5,  cv=10, refit=True, return_train_score=True)
# bst = best(data_split_90th_percentile[0], data_split_90th_percentile[1])
# print best.cv_results_

xgboost_90th_percentile.fit(data_split_90th_percentile[0], data_split_90th_percentile[1])

print array_print(data_split_90th_percentile_test[1])
predict_90th_percentile = xgboost_90th_percentile.predict(data_split_90th_percentile_test[0])
print array_print(predict_90th_percentile)

print array_print(evaluator(y_real_ev=data_split_90th_percentile[1], y_pred_ev=xgboost_90th_percentile.predict(data_split_90th_percentile[0])))
print array_print(evaluator(y_real_ev=data_split_90th_percentile_test[1], y_pred_ev=predict_90th_percentile))
# print predict_final_semi
# res = xgb.cv(params=params, dtrain=data, num_boost_round=200, nfold=10, metrics= ['mae'], early_stopping_rounds=10)
#
#
# predict_final = bst.predict(test)
# print predict_final



# xgb_model = XGBRegressor()  # not parameters for tune
# xgb_model.fit(data_split_90th_percentile[0], data_split_90th_percentile[1])
# y_prd = xgb_model.predict(data_split_90th_percentile[0])
# print 'train data'
# print array_print(data_split_90th_percentile[1])
# print 'train pred'
# print array_print(y_prd)
# y_prd_test = xgb_model.predict(data_split_90th_percentile_test[0])  # predict y values by train x_train
# print 'test data'
# print array_print(data_split_90th_percentile_test[1])
# print 'test pred data'
# print array_print(y_prd_test)
# current_result_array = evaluator(y_train_ev=data_split_90th_percentile[1], y_pred_ev=y_prd)
# arr_rmse_in_xg = current_result_array[0]
# arr_mape_in_xg = current_result_array[1]
# current_result_array_test = evaluator(y_train_ev=data_split_90th_percentile_test[1], y_pred_ev=y_prd_test)
# arr_rmse_in_test_xg = current_result_array_test[0]
# arr_mape_in_test_xg = current_result_array_test[1]
# # ################################################################################
# print "\n\n\nresults "
#
# print "rmse xg"
# print arr_rmse_in_xg
# print "mape xg"
# print arr_mape_in_xg
# print "test rmse xg"
# print arr_rmse_in_test_xg
# print "test mape xg"
# print arr_mape_in_test_xg
#
# print (time.time()-time2)
#
#
# time2 = time.time()
# # ###################################################################################
# 95th percentile
print "\n\n\n95th percentile "

data_split_95th_percentile = np.array([], dtype='float64')
data_split_95th_percentile = data_reader(csv_file=summary_data, total_row=n_rows,
                                    thousands_splitter=t_splitter, csv_select_columns=csv_select_cols,
                                    x_column_numbers=x_select_cols, y_column_number=y_select_col_95th_percentile)

data_split_95th_percentile_test = np.array([], dtype='float64')
data_split_95th_percentile_test = data_reader(csv_file=summary_data_test, total_row=n_rows, thousands_splitter=t_splitter,
                                         csv_select_columns=csv_select_cols, x_column_numbers=x_select_cols,
                                         y_column_number=y_select_col_95th_percentile)

# #  ##################################################################################
# # SVR Grid Search to 95th percentile
# print "\n\n\nSVR Grid Search CV "
# parameters = {'kernel': kernel_type_array , 'C': c_array, 'epsilon': epsilion_array}
# svr = SVR()
# svr_model = GridSearchCV(svr, parameters)
# svr_model.fit(data_split_95th_percentile[0], data_split_95th_percentile[1])
# y_prd = svr_model.predict(data_split_95th_percentile[0])
# print 'train data'
# print array_print(data_split_95th_percentile[1])
# print 'train pred'
# print array_print(y_prd)
# y_prd_test = svr_model.predict(data_split_95th_percentile_test[0])  # predict y values by train x_train
# print 'test data'
# print array_print(data_split_95th_percentile_test[1])
# print 'test pred data'
# print array_print(y_prd_test)
# print svr_model
# y_prd = svr_model.predict(data_split_95th_percentile[0])
# y_prd_test = svr_model.predict(data_split_95th_percentile_test[0])  # predict y values by train x_train
# current_result_array = evaluator(y_train_ev=data_split_95th_percentile[1], y_pred_ev=y_prd)
# arr_rmse_in = current_result_array[0]
# arr_mape_in = current_result_array[1]
# current_result_array_test = evaluator(y_train_ev=data_split_95th_percentile_test[1], y_pred_ev=y_prd_test)
# arr_rmse_in_test = current_result_array_test[0]
# arr_mape_in_test = current_result_array_test[1]
# # #############################################################################
# print "\n\n\nresults "
# print "rmse"
# print arr_rmse_in
# print "mape"
# print arr_mape_in
# print "test rmse"
# print arr_rmse_in_test
# print "test mape"
# print arr_mape_in_test
#
# print svr_model.best_params_
#
# print 'time', time.time()-time2
# #############################################################################################
print "\n\n\nXGBoost 95th percentile"

parameters_95th_percentile = {'max_depth' : [3], 'learning_rate': [0.1], 'n_estimators': [100], 'min_child_weight': [1], 'max_delta_step': [0], 'objective': ['reg:linear']}
xgboost_95th_percentile = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, min_child_weight= 1, max_delta_step= 0, objective= 'reg:linear', silent=True, gamma=0)
# best = GridSearchCV(xgboost_95th_percentile,parameters_95th_percentile , n_jobs=5,  cv=10, refit=True, return_train_score=True)
# bst = best(data_split_95th_percentile[0], data_split_95th_percentile[1])
# print best.cv_results_

xgboost_95th_percentile.fit(data_split_95th_percentile[0], data_split_95th_percentile[1])

print array_print(data_split_95th_percentile_test[1])
predict_95th_percentile = xgboost_95th_percentile.predict(data_split_95th_percentile_test[0])
print array_print(predict_95th_percentile)

print array_print(evaluator(y_real_ev=data_split_95th_percentile[1], y_pred_ev=xgboost_95th_percentile.predict(data_split_95th_percentile[0])))
print array_print(evaluator(y_real_ev=data_split_95th_percentile_test[1], y_pred_ev=predict_95th_percentile))
# print predict_final_semi
# res = xgb.cv(params=params, dtrain=data, num_boost_round=200, nfold=10, metrics= ['mae'], early_stopping_rounds=10)
#
#
# predict_final = bst.predict(test)
# print predict_final


# xgb_model = XGBRegressor()  # not parameters for tune
# xgb_model.fit(data_split_95th_percentile[0], data_split_95th_percentile[1])
# y_prd = xgb_model.predict(data_split_95th_percentile[0])
# print 'train data'
# print array_print(data_split_95th_percentile[1])
# print 'train pred'
# print array_print(y_prd)
# y_prd_test = xgb_model.predict(data_split_95th_percentile_test[0])  # predict y values by train x_train
# print 'test data'
# print array_print(data_split_95th_percentile_test[1])
# print 'test pred data'
# print array_print(y_prd_test)
# current_result_array = evaluator(y_train_ev=data_split_95th_percentile[1], y_pred_ev=y_prd)
# arr_rmse_in_xg = current_result_array[0]
# arr_mape_in_xg = current_result_array[1]
# current_result_array_test = evaluator(y_train_ev=data_split_95th_percentile_test[1], y_pred_ev=y_prd_test)
# arr_rmse_in_test_xg = current_result_array_test[0]
# arr_mape_in_test_xg = current_result_array_test[1]
# # ####################################################################################
# print "\n\n\nresults "
# print "rmse xg"
# print arr_rmse_in_xg
# print "mape xg"
# print arr_mape_in_xg
# print "test rmse xg"
# print arr_rmse_in_test_xg
# print "test mape xg"
# print arr_mape_in_test_xg
#
# print (time.time()-time2)
#
#
# time2 = time.time()
# # ###################################################################################
# 99th_percentile
print "\n\n\n99th_percentile "

data_split_99th_percentile = np.array([], dtype='float64')
data_split_99th_percentile = data_reader(csv_file=summary_data, total_row=n_rows,
                                    thousands_splitter=t_splitter, csv_select_columns=csv_select_cols,
                                    x_column_numbers=x_select_cols, y_column_number=y_select_col_99th_percentile)

data_split_99th_percentile_test = np.array([], dtype='float64')
data_split_99th_percentile_test = data_reader(csv_file=summary_data_test, total_row=n_rows, thousands_splitter=t_splitter,
                                         csv_select_columns=csv_select_cols, x_column_numbers=x_select_cols,
                                         y_column_number=y_select_col_99th_percentile)
#
# #  ##################################################################################
# # SVR Grid Search to 99th_percentile
# print "\n\n\nSVR Grid Search CV "
# parameters = {'kernel': kernel_type_array , 'C': c_array, 'epsilon': epsilion_array}
# svr = SVR()
# svr_model = GridSearchCV(svr, parameters)
# svr_model.fit(data_split_99th_percentile[0], data_split_99th_percentile[1])
# y_prd = svr_model.predict(data_split_99th_percentile[0])
# print 'train data'
# print array_print(data_split_99th_percentile[1])
# print 'train pred'
# print array_print(y_prd)
# y_prd_test = svr_model.predict(data_split_99th_percentile_test[0])  # predict y values by train x_train
# print 'test data'
# print array_print(data_split_99th_percentile_test[1])
# print 'test pred data'
# print array_print(y_prd_test)
# print svr_model
# y_prd = svr_model.predict(data_split_99th_percentile[0])
# y_prd_test = svr_model.predict(data_split_99th_percentile_test[0])  # predict y values by train x_train
# current_result_array = evaluator(y_train_ev=data_split_99th_percentile[1], y_pred_ev=y_prd)
# arr_rmse_in = current_result_array[0]
# arr_mape_in = current_result_array[1]
# current_result_array_test = evaluator(y_train_ev=data_split_99th_percentile_test[1], y_pred_ev=y_prd_test)
# arr_rmse_in_test = current_result_array_test[0]
# arr_mape_in_test = current_result_array_test[1]
# # #######################################################################################
# print "\n\n\nresults "
# print "rmse"
# print arr_rmse_in
# print "mape"
# print arr_mape_in
# print "test rmse"
# print arr_rmse_in_test
# print "test mape"
# print arr_mape_in_test
#
# print svr_model.best_params_
#
# print 'time', time.time()-time2
# #######################################################################################
print "\n\n\nXGBoost "

parameters_99th_percentile = {'max_depth' : [3], 'learning_rate': [0.1], 'n_estimators': [100], 'min_child_weight': [1], 'max_delta_step': [0], 'objective': ['reg:linear']}
xgboost_99th_percentile = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, min_child_weight= 1, max_delta_step= 0, objective= 'reg:linear', silent=True, gamma=0)
# best = GridSearchCV(xgboost_99th_percentile,parameters_99th_percentile , n_jobs=5,  cv=10, refit=True, return_train_score=True)
# bst = best(data_split_99th_percentile[0], data_split_99th_percentile[1])
# print best.cv_results_

xgboost_99th_percentile.fit(data_split_99th_percentile[0], data_split_99th_percentile[1])

print array_print(data_split_99th_percentile_test[1])
predict_99th_percentile = xgboost_99th_percentile.predict(data_split_99th_percentile_test[0])
print array_print(predict_99th_percentile)

print array_print(evaluator(y_real_ev=data_split_99th_percentile[1], y_pred_ev=xgboost_99th_percentile.predict(data_split_99th_percentile[0])))
print array_print(evaluator(y_real_ev=data_split_99th_percentile_test[1], y_pred_ev=predict_99th_percentile))
# print predict_final_semi
# res = xgb.cv(params=params, dtrain=data, num_boost_round=200, nfold=10, metrics= ['mae'], early_stopping_rounds=10)
#
#
# predict_final = bst.predict(test)
# print predict_final


# xgb_model = XGBRegressor()  # not parameters for tune
# xgb_model.fit(data_split_99th_percentile[0], data_split_99th_percentile[1])
# y_prd = xgb_model.predict(data_split_99th_percentile[0])
# print 'train data'
# print array_print(data_split_99th_percentile[1])
# print 'train pred'
# print array_print(y_prd)
# y_prd_test = xgb_model.predict(data_split_99th_percentile_test[0])  # predict y values by train x_train
# print 'test data'
# print array_print(data_split_99th_percentile_test[1])
# print 'test pred data'
# print array_print(y_prd_test)
# current_result_array = evaluator(y_train_ev=data_split_99th_percentile[1], y_pred_ev=y_prd)
# arr_rmse_in_xg = current_result_array[0]
# arr_mape_in_xg = current_result_array[1]
# current_result_array_test = evaluator(y_train_ev=data_split_99th_percentile_test[1], y_pred_ev=y_prd_test)
# arr_rmse_in_test_xg = current_result_array_test[0]
# arr_mape_in_test_xg = current_result_array_test[1]
# # ##################################################################################
# print "\n\n\nresults "
# print "rmse xg"
# print arr_rmse_in_xg
# print "mape xg"
# print arr_mape_in_xg
# print "test rmse xg"
# print arr_rmse_in_test_xg
# print "test mape xg"
# print arr_mape_in_test_xg
#
# print (time.time()-time2)
#
#
# time2 = time.time()
# # ###################################################################################
# load_average_1_minute
print "\n\n\nload_average_1_minute "

data_split_load_average_1_minute = np.array([], dtype='float64')
data_split_load_average_1_minute = data_reader(csv_file=summary_data, total_row=n_rows,
                                    thousands_splitter=t_splitter, csv_select_columns=csv_select_cols,
                                    x_column_numbers=x_select_cols, y_column_number=y_select_col_load_average_1_minute)

data_split_load_average_1_minute_test = np.array([], dtype='float64')
data_split_load_average_1_minute_test = data_reader(csv_file=summary_data_test, total_row=n_rows, thousands_splitter=t_splitter,
                                         csv_select_columns=csv_select_cols, x_column_numbers=x_select_cols,
                                         y_column_number=y_select_col_load_average_1_minute)

# #  ##################################################################################
# # SVR Grid Search to load_average_1_minute
# print "\n\n\nSVR Grid Search CV "
# parameters = {'kernel': kernel_type_array , 'C': c_array, 'epsilon': epsilion_array}
# svr = SVR()
# svr_model = GridSearchCV(svr, parameters)
# svr_model.fit(data_split_load_average_1_minute[0], data_split_load_average_1_minute[1])
# y_prd = svr_model.predict(data_split_load_average_1_minute[0])
# print 'train data'
# print array_print(data_split_load_average_1_minute[1])
# print 'train pred'
# print array_print(y_prd)
# y_prd_test = svr_model.predict(data_split_load_average_1_minute_test[0])  # predict y values by train x_train
# print 'test data'
# print array_print(data_split_load_average_1_minute_test[1])
# print 'test pred data'
# print array_print(y_prd_test)
# print svr_model
# y_prd = svr_model.predict(data_split_load_average_1_minute[0])
# y_prd_test = svr_model.predict(data_split_load_average_1_minute_test[0])  # predict y values by train x_train
# current_result_array = evaluator(y_train_ev=data_split_load_average_1_minute[1], y_pred_ev=y_prd)
# arr_rmse_in = current_result_array[0]
# arr_mape_in = current_result_array[1]
# current_result_array_test = evaluator(y_train_ev=data_split_load_average_1_minute_test[1], y_pred_ev=y_prd_test)
# arr_rmse_in_test = current_result_array_test[0]
# arr_mape_in_test = current_result_array_test[1]
# # #############################################################################
# print "\n\n\nresults "
# print "rmse"
# print arr_rmse_in
# print "mape"
# print arr_mape_in
# print "test rmse"
# print arr_rmse_in_test
# print "test mape"
# print arr_mape_in_test
#
# print svr_model.best_params_
#
# print 'time', time.time()-time2
# # ##################################################################################
print "\n\n\nXGBoost "

parameters_load_average_1_minute = {'max_depth' : [3], 'learning_rate': [0.1], 'n_estimators': [100], 'min_child_weight': [1], 'max_delta_step': [0], 'objective': ['reg:linear']}
xgboost_load_average_1_minute = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, min_child_weight= 1, max_delta_step= 0, objective= 'reg:linear', silent=True, gamma=0)
# best = GridSearchCV(xgboost_load_average_1_minute,parameters_load_average_1_minute , n_jobs=5,  cv=10, refit=True, return_train_score=True)
# bst = best(data_split_load_average_1_minute[0], data_split_load_average_1_minute[1])
# print best.cv_results_

xgboost_load_average_1_minute.fit(data_split_load_average_1_minute[0], data_split_load_average_1_minute[1])

print array_print(data_split_load_average_1_minute_test[1])
predict_load_average_1_minute = xgboost_load_average_1_minute.predict(data_split_load_average_1_minute_test[0])
print array_print(predict_load_average_1_minute)

print array_print(evaluator(y_real_ev=data_split_load_average_1_minute[1], y_pred_ev=xgboost_load_average_1_minute.predict(data_split_load_average_1_minute[0])))
print array_print(evaluator(y_real_ev=data_split_load_average_1_minute_test[1], y_pred_ev=predict_load_average_1_minute))
# print predict_final_semi
# res = xgb.cv(params=params, dtrain=data, num_boost_round=200, nfold=10, metrics= ['mae'], early_stopping_rounds=10)
#
#
# predict_final = bst.predict(test)
# print predict_final


# xgb_model = XGBRegressor()  # not parameters for tune
# xgb_model.fit(data_split_load_average_1_minute[0], data_split_load_average_1_minute[1])
# y_prd = xgb_model.predict(data_split_load_average_1_minute[0])
# print 'train data'
# print array_print(data_split_load_average_1_minute[1])
# print 'train pred'
# print array_print(y_prd)
# y_prd_test = xgb_model.predict(data_split_load_average_1_minute_test[0])  # predict y values by train x_train
# print 'test data'
# print array_print(data_split_load_average_1_minute_test[1])
# print 'test pred data'
# print array_print(y_prd_test)
# current_result_array = evaluator(y_train_ev=data_split_load_average_1_minute[1], y_pred_ev=y_prd)
# arr_rmse_in_xg = current_result_array[0]
# arr_mape_in_xg = current_result_array[1]
# current_result_array_test = evaluator(y_train_ev=data_split_load_average_1_minute_test[1], y_pred_ev=y_prd_test)
# arr_rmse_in_test_xg = current_result_array_test[0]
# arr_mape_in_test_xg = current_result_array_test[1]
# # ####################################################################################
# print "\n\n\nresults "
# print "rmse xg"
# print arr_rmse_in_xg
# print "mape xg"
# print arr_mape_in_xg
# print "test rmse xg"
# print arr_rmse_in_test_xg
# print "test mape xg"
# print arr_mape_in_test_xg
#
# print (time.time()-time2)
#
#
# time2 = time.time()
# ###################################################################################
# load_average_5_minutes
print "\n\n\nload_average_5_minutes "

data_split_load_average_5_minute = np.array([], dtype='float64')
data_split_load_average_5_minute = data_reader(csv_file=summary_data, total_row=n_rows,
                                    thousands_splitter=t_splitter, csv_select_columns=csv_select_cols,
                                    x_column_numbers=x_select_cols, y_column_number=y_select_col_load_average_5_minute)

data_split_load_average_5_minute_test = np.array([], dtype='float64')
data_split_load_average_5_minute_test = data_reader(csv_file=summary_data_test, total_row=n_rows, thousands_splitter=t_splitter,
                                         csv_select_columns=csv_select_cols, x_column_numbers=x_select_cols,
                                         y_column_number=y_select_col_load_average_5_minute)
#
# #  ##################################################################################
# # SVR Grid Search to load_average_5_minutes
# print "\n\n\nSVR Grid Search CV "
# parameters = {'kernel': kernel_type_array , 'C': c_array, 'epsilon': epsilion_array}
# svr = SVR()
# svr_model = GridSearchCV(svr, parameters)
# svr_model.fit(data_split_load_average_5_minute[0], data_split_load_average_5_minute[1])
# y_prd = svr_model.predict(data_split_load_average_5_minute[0])
# print 'train data'
# print array_print(data_split_load_average_5_minute[1])
# print 'train pred'
# print array_print(y_prd)
# y_prd_test = svr_model.predict(data_split_load_average_5_minute_test[0])  # predict y values by train x_train
# print 'test data'
# print array_print(data_split_load_average_5_minute_test[1])
# print 'test pred data'
# print array_print(y_prd_test)
# print svr_model
# y_prd = svr_model.predict(data_split_load_average_5_minute[0])
# y_prd_test = svr_model.predict(data_split_load_average_5_minute_test[0])  # predict y values by train x_train
# current_result_array = evaluator(y_train_ev=data_split_load_average_5_minute[1], y_pred_ev=y_prd)
# arr_rmse_in = current_result_array[0]
# arr_mape_in = current_result_array[1]
# current_result_array_test = evaluator(y_train_ev=data_split_load_average_5_minute_test[1], y_pred_ev=y_prd_test)
# arr_rmse_in_test = current_result_array_test[0]
# arr_mape_in_test = current_result_array_test[1]
# # #############################################################################
# print "\n\n\nresults "
# print "rmse"
# print arr_rmse_in
# print "mape"
# print arr_mape_in
# print "test rmse"
# print arr_rmse_in_test
# print "test mape"
# print arr_mape_in_test
#
# print svr_model.best_params_
#
# print 'time', time.time()-time2
# #################################################################################
print "\n\n\nXGBoost "

parameters_load_average_5_minute = {'max_depth' : [3], 'learning_rate': [0.1], 'n_estimators': [100], 'min_child_weight': [1], 'max_delta_step': [0], 'objective': ['reg:linear']}
xgboost_load_average_5_minute = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, min_child_weight= 1, max_delta_step= 0, objective= 'reg:linear', silent=True, gamma=0)
# best = GridSearchCV(xgboost_load_average_5_minute,parameters_load_average_5_minute , n_jobs=5,  cv=10, refit=True, return_train_score=True)
# bst = best(data_split_load_average_5_minute[0], data_split_load_average_5_minute[1])
# print best.cv_results_

xgboost_load_average_5_minute.fit(data_split_load_average_5_minute[0], data_split_load_average_5_minute[1])

print array_print(data_split_load_average_5_minute_test[1])
predict_load_average_5_minute = xgboost_load_average_5_minute.predict(data_split_load_average_5_minute_test[0])
print array_print(predict_load_average_5_minute)

print array_print(evaluator(y_real_ev=data_split_load_average_5_minute[1], y_pred_ev=xgboost_load_average_5_minute.predict(data_split_load_average_5_minute[0])))
print array_print(evaluator(y_real_ev=data_split_load_average_5_minute_test[1], y_pred_ev=predict_load_average_5_minute))
# print predict_final_semi
# res = xgb.cv(params=params, dtrain=data, num_boost_round=200, nfold=10, metrics= ['mae'], early_stopping_rounds=10)
#
#
# predict_final = bst.predict(test)
# print predict_final


# xgb_model = XGBRegressor()  # not parameters for tune
# xgb_model.fit(data_split_load_average_5_minute[0], data_split_load_average_5_minute[1])
# y_prd = xgb_model.predict(data_split_load_average_5_minute[0])
# print 'train data'
# print array_print(data_split_load_average_5_minute[1])
# print 'train pred'
# print array_print(y_prd)
# y_prd_test = xgb_model.predict(data_split_load_average_5_minute_test[0])  # predict y values by train x_train
# print 'test data'
# print array_print(data_split_load_average_5_minute_test[1])
# print 'test pred data'
# print array_print(y_prd_test)
# current_result_array = evaluator(y_train_ev=data_split_load_average_5_minute[1], y_pred_ev=y_prd)
# arr_rmse_in_xg = current_result_array[0]
# arr_mape_in_xg = current_result_array[1]
# current_result_array_test = evaluator(y_train_ev=data_split_load_average_5_minute_test[1], y_pred_ev=y_prd_test)
# arr_rmse_in_test_xg = current_result_array_test[0]
# arr_mape_in_test_xg = current_result_array_test[1]
# # ###################################################################################
# print "\n\n\nresults "
# print "rmse xg"
# print arr_rmse_in_xg
# print "mape xg"
# print arr_mape_in_xg
# print "test rmse xg"
# print arr_rmse_in_test_xg
# print "test mape xg"
# print arr_mape_in_test_xg
#
# print (time.time()-time2)
#
#
# time2 = time.time()
# # ###################################################################################
# load_average_15_minutes
print "\n\n\nload_average_15_minutes "

data_split_load_average_15_minute = np.array([], dtype='float64')
data_split_load_average_15_minute = data_reader(csv_file=summary_data, total_row=n_rows,
                                    thousands_splitter=t_splitter, csv_select_columns=csv_select_cols,
                                    x_column_numbers=x_select_cols, y_column_number=y_select_col_load_average_15_minute)

data_split_load_average_15_minute_test = np.array([], dtype='float64')
data_split_load_average_15_minute_test = data_reader(csv_file=summary_data_test, total_row=n_rows, thousands_splitter=t_splitter,
                                         csv_select_columns=csv_select_cols, x_column_numbers=x_select_cols,
                                         y_column_number=y_select_col_load_average_15_minute)
<<<<<<< HEAD
#
# #  ##################################################################################
# # SVR Grid Search to load_average_15_minutes
# print "\n\n\nSVR Grid Search CV "
# parameters = {'kernel': kernel_type_array , 'C': c_array, 'epsilon': epsilion_array}
# svr = SVR()
# svr_model = GridSearchCV(svr, parameters)
# svr_model.fit(data_split_load_average_15_minute[0], data_split_load_average_15_minute[1])
# y_prd = svr_model.predict(data_split_load_average_15_minute[0])
# print 'train data'
# print array_print(data_split_load_average_15_minute[1])
# print 'train pred'
# print array_print(y_prd)
# y_prd_test = svr_model.predict(data_split_load_average_15_minute_test[0])  # predict y values by train x_train
# print 'test data'
# print array_print(data_split_load_average_15_minute_test[1])
# print 'test pred data'
# print array_print(y_prd_test)
# print svr_model
# y_prd = svr_model.predict(data_split_load_average_15_minute[0])
# y_prd_test = svr_model.predict(data_split_load_average_15_minute_test[0])  # predict y values by train x_train
# current_result_array = evaluator(y_train_ev=data_split_load_average_15_minute[1], y_pred_ev=y_prd)
# arr_rmse_in = current_result_array[0]
# arr_mape_in = current_result_array[1]
# current_result_array_test = evaluator(y_train_ev=data_split_load_average_15_minute_test[1], y_pred_ev=y_prd_test)
# arr_rmse_in_test = current_result_array_test[0]
# arr_mape_in_test = current_result_array_test[1]
# # #############################################################################
# print "\n\n\nresults "
# print "rmse"
# print arr_rmse_in
# print "mape"
# print arr_mape_in
# print "test rmse"
# print arr_rmse_in_test
# print "test mape"
# print arr_mape_in_test
#
# print svr_model.best_params_
#
# print 'time', time.time()-time2
# ################################################################################
print "\n\n\nXGBoost 15 minutes load average"
=======
>>>>>>> 6bb2a43c62e01957ee9f341883b705108f1d6c53

parameters_load_average_15_minute = {'max_depth' : [3], 'learning_rate': [0.1], 'n_estimators': [100], 'min_child_weight': [1], 'max_delta_step': [0], 'objective': ['reg:linear']}
xgboost_load_average_15_minute = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=50, silent=True, objective= 'reg:linear')
# best = GridSearchCV(xgboost_load_average_15_minute,parameters_load_average_15_minute , n_jobs=5,  cv=10, refit=True, return_train_score=True)
# bst = best(data_split_load_average_15_minute[0], data_split_load_average_15_minute[1])
# print best.cv_results_

xgboost_load_average_15_minute.fit(data_split_load_average_15_minute[0], data_split_load_average_15_minute[1])

print array_print(data_split_load_average_15_minute_test[1])
predict_load_average_15_minute = xgboost_load_average_15_minute.predict(data_split_load_average_15_minute_test[0])
print array_print(predict_load_average_15_minute)

print array_print(evaluator(y_real_ev=data_split_load_average_15_minute[1], y_pred_ev=xgboost_load_average_15_minute.predict(data_split_load_average_15_minute[0])))
print array_print(evaluator(y_real_ev=data_split_load_average_15_minute_test[1], y_pred_ev=predict_load_average_15_minute))
# print predict_final_semi
# res = xgb.cv(params=params, dtrain=data, num_boost_round=200, nfold=10, metrics= ['mae'], early_stopping_rounds=10)
#
#
# predict_final = bst.predict(test)
# print predict_final


# xgb_model = XGBRegressor()  # not parameters for tune
# xgb_model.fit(data_split_load_average_15_minute[0], data_split_load_average_15_minute[1])
# y_prd = xgb_model.predict(data_split_load_average_15_minute[0])
# print 'train data'
# print array_print(data_split_load_average_15_minute[1])
# print 'train pred'
# print array_print(y_prd)
# y_prd_test = xgb_model.predict(data_split_load_average_15_minute_test[0])  # predict y values by train x_train
# print 'test data'
# print array_print(data_split_load_average_15_minute_test[1])
# print 'test pred data'
# print array_print(y_prd_test)
# current_result_array = evaluator(y_train_ev=data_split_load_average_15_minute[1], y_pred_ev=y_prd)
# arr_rmse_in_xg = current_result_array[0]
# arr_mape_in_xg = current_result_array[1]
# current_result_array_test = evaluator(y_train_ev=data_split_load_average_15_minute_test[1], y_pred_ev=y_prd_test)
# arr_rmse_in_test_xg = current_result_array_test[0]
# arr_mape_in_test_xg = current_result_array_test[1]
# # ####################################################################################
# print "\n\n\nresults "
# print "rmse xg"
# print arr_rmse_in_xg
# print "mape xg"
# print arr_mape_in_xg
# print "test rmse xg"
# print arr_rmse_in_test_xg
# print "test mape xg"
# print arr_mape_in_test_xg
#
# print 'time', time.time()-time2
# ################################################################################
print 'total time = ', time.time()-time1
