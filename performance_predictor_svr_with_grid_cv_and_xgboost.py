import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from xgboost import XGBRegressor
import sklearn.metrics as mx
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
# import joblib
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
# x_select_cols = [0, 1, 2, 3, 10]  # select columns to x (features)
x_select_cols_throughput_svr = [0, 1, 2, 3, 4]
x_select_cols_throughput_xgboost = [0, 1, 2, 3, 14]
x_select_cols_latency_svr = [0, 1, 2, 3]
x_select_cols_latency_xgboost = [0, 1, 2, 3]
x_select_cols_90th_percentile_svr = [0, 1, 2, 3]
x_select_cols_90th_percentile_xgboost = [0, 1, 2, 3]
x_select_cols_95th_percentile_svr = [0, 1, 2, 3]
x_select_cols_95th_percentile_xgboost = [0, 1, 2, 3]
x_select_cols_99th_percentile_svr = [0, 1, 2, 3]
x_select_cols_99th_percentile_xgboost = [0, 1, 2, 3]
x_select_cols_load_average_1_minute_svr = [0, 1, 2, 3]
x_select_cols_load_average_1_minute_xgboost = [0, 1, 2, 3]
x_select_cols_load_average_5_minute_svr = [0, 1, 2, 3]
x_select_cols_load_average_5_minute_xgboost = [0, 1, 2, 3]
x_select_cols_load_average_15_minute_svr = [0, 1, 2, 3]
x_select_cols_load_average_15_minute_xgboost = [0, 1, 2, 3]
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
    evaluate_rmse = np.sqrt(mx.mean_squared_error(y_true=y_real_ev, y_pred=y_pred_ev))
    evaluate_mape = mean_absolute_percentage_error(y_true=y_real_ev, y_pred=y_pred_ev)
    return evaluate_rmse, evaluate_mape


def array_print(array_get):
    for i in range(0, len(array_get)):
        print array_get[i]
    print "okay"


time1 = time.time()
time2 = time.time()
# ###################################################################################
# throughput
print "\n\n\nThroughput "

data_split_throughput_svr = np.array([], dtype='float64')
data_split_throughput_svr = data_reader(csv_file=summary_data, total_row=n_rows, thousands_splitter=t_splitter,
                                        csv_select_columns=csv_select_cols,
                                        x_column_numbers=x_select_cols_throughput_svr,
                                        y_column_number=y_select_col_throughput)

data_split_throughput_svr_test = np.array([], dtype='float64')
data_split_throughput_svr_test = data_reader(csv_file=summary_data_test, total_row=n_rows,
                                             thousands_splitter=t_splitter, csv_select_columns=csv_select_cols,
                                             x_column_numbers=x_select_cols_throughput_svr,
                                             y_column_number=y_select_col_throughput)
# additional feature 10 = Concurrency / Message size


#  ##################################################################################
print "\n\n\nSVR Grid Search CV Throughput"
parameters_svr_throughput = {'kernel': ['rbf', 'poly', 'linear'], 'C': [1E2, 1E3],
                             'epsilon': [0.0001, 0.0005, 0.001, 0.005,  0.01, 0.05, 0.1, 0.5, 10]}

svr_throughput = SVR(coef0=0.1, tol=0.001, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

svr_best_model_throughput = GridSearchCV(svr_throughput, parameters_svr_throughput, cv=10, n_jobs=6,
                                         return_train_score=True, refit=True, scoring='neg_mean_squared_error')

svr_best_throughput = svr_best_model_throughput.fit(data_split_throughput_svr[0], data_split_throughput_svr[1])

print array_print(data_split_throughput_svr_test[1])
print array_print(svr_best_model_throughput.predict(data_split_throughput_svr_test[0]))
print array_print(evaluator(data_split_throughput_svr[1],
                            svr_best_throughput.predict(data_split_throughput_svr[0])))
print array_print(evaluator(data_split_throughput_svr_test[1],
                            svr_best_throughput.predict(data_split_throughput_svr_test[0])))

print 'time', time.time()-time2
# #############################################################################


print "\n\n\nXGBoost Grid Search CV Throughput "

data_split_throughput_xgboost = np.array([], dtype='float64')
data_split_throughput_xgboost = data_reader(csv_file=summary_data, total_row=n_rows, thousands_splitter=t_splitter,
                                            csv_select_columns=csv_select_cols,
                                            x_column_numbers=x_select_cols_throughput_xgboost,
                                            y_column_number=y_select_col_throughput)

data_split_throughput_xgboost_test = np.array([], dtype='float64')
data_split_throughput_xgboost_test = data_reader(csv_file=summary_data_test, total_row=n_rows,
                                                 thousands_splitter=t_splitter, csv_select_columns=csv_select_cols,
                                                 x_column_numbers=x_select_cols_throughput_xgboost,
                                                 y_column_number=y_select_col_throughput)
# additional feature 10 = Concurrency / Message size

parameters_xgboost_throughput = {'max_depth': [3], 'learning_rate': [0.1], 'n_estimators': [1000],
                                 'min_child_weight': [1], 'max_delta_step': [0], 'objective': ['reg:linear']}

xgboost_throughput = xgb.XGBRegressor(silent=True, objective='reg:linear', gamma=0,
                                      subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0,
                                      reg_lambda=1, scale_pos_weight=1, base_score=0.5, missing=None)


xgboost_best_model_throughput = GridSearchCV(xgboost_throughput, parameters_xgboost_throughput, n_jobs=5,
                                             cv=10, refit=True, return_train_score=True)

xgboost_best_throughput = xgboost_best_model_throughput.fit(X=data_split_throughput_xgboost[0],
                                                            y=data_split_throughput_xgboost[1],
                                                            eval_set=[(data_split_throughput_xgboost[0],
                                                                       data_split_throughput_xgboost[1]),
                                                                      (data_split_throughput_xgboost_test[0],
                                                                       data_split_throughput_xgboost_test[1])],
                                                            eval_metric='rmse', early_stopping_rounds=10)


print array_print(data_split_throughput_xgboost_test[1])
print array_print( xgboost_best_throughput.predict(data_split_throughput_xgboost_test[0]))
print array_print(evaluator(data_split_throughput_xgboost[1],
                            xgboost_best_throughput.predict(data_split_throughput_xgboost[0])))
print array_print(evaluator(data_split_throughput_xgboost_test[1],
                            xgboost_best_throughput.predict(data_split_throughput_xgboost_test[0])))

print (time.time()-time2)
# ###########################################################################

# ###################################################################################
# latency
print "\n\n\nlatency "

data_split_latency_svr = np.array([], dtype='float64')
data_split_latency_svr = data_reader(csv_file=summary_data, total_row=n_rows, thousands_splitter=t_splitter,
                                        csv_select_columns=csv_select_cols,
                                        x_column_numbers=x_select_cols_latency_svr,
                                        y_column_number=y_select_col_latency)

data_split_latency_svr_test = np.array([], dtype='float64')
data_split_latency_svr_test = data_reader(csv_file=summary_data_test, total_row=n_rows,
                                             thousands_splitter=t_splitter, csv_select_columns=csv_select_cols,
                                             x_column_numbers=x_select_cols_latency_svr,
                                             y_column_number=y_select_col_latency)
# additional feature 10 = Concurrency / Message size


#  ##################################################################################
print "\n\n\nSVR Grid Search CV latency"
parameters_svr_latency = {'kernel': ['rbf', 'poly', 'linear'], 'C': [1E2, 1E3],
                          'epsilon': [0.0001, 0.0005, 0.001, 0.005,  0.01, 0.05, 0.1, 0.5, 10]}

svr_latency = SVR(coef0=0.1, tol=0.001, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

svr_best_model_latency = GridSearchCV(svr_latency, parameters_svr_latency, cv=10, n_jobs=6,
                                      return_train_score=True, refit=True, scoring='neg_mean_squared_error')

svr_best_latency = svr_best_model_latency.fit(data_split_latency_svr[0], data_split_latency_svr[1])

print array_print(data_split_latency_svr_test[1])
print array_print(svr_best_model_latency.predict(data_split_latency_svr_test[0]))
print array_print(evaluator(data_split_latency_svr[1],
                            svr_best_latency.predict(data_split_latency_svr[0])))
print array_print(evaluator(data_split_latency_svr_test[1],
                            svr_best_latency.predict(data_split_latency_svr_test[0])))

print 'time', time.time()-time2
# #############################################################################


print "\n\n\nXGBoost Grid Search CV latency "

data_split_latency_xgboost = np.array([], dtype='float64')
data_split_latency_xgboost = data_reader(csv_file=summary_data, total_row=n_rows, thousands_splitter=t_splitter,
                                            csv_select_columns=csv_select_cols,
                                            x_column_numbers=x_select_cols_latency_xgboost,
                                            y_column_number=y_select_col_latency)

data_split_latency_xgboost_test = np.array([], dtype='float64')
data_split_latency_xgboost_test = data_reader(csv_file=summary_data_test, total_row=n_rows,
                                                 thousands_splitter=t_splitter, csv_select_columns=csv_select_cols,
                                                 x_column_numbers=x_select_cols_latency_xgboost,
                                                 y_column_number=y_select_col_latency)
# additional feature 10 = Concurrency / Message size

parameters_xgboost_latency = {'max_depth': [3], 'learning_rate': [0.1], 'n_estimators': [100],
                              'min_child_weight': [1], 'max_delta_step': [0], 'objective': ['reg:linear']}

xgboost_latency = xgb.XGBRegressor(silent=True, objective='reg:linear', gamma=0,
                                      subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0,
                                      reg_lambda=1, scale_pos_weight=1, base_score=0.5, missing=None)


xgboost_best_model_latency = GridSearchCV(xgboost_latency, parameters_xgboost_latency, n_jobs=5,
                                          cv=10, refit=True, return_train_score=True)

xgboost_best_latency = xgboost_best_model_latency.fit(X=data_split_latency_xgboost[0],
                                                            y=data_split_latency_xgboost[1],
                                                            eval_set=[(data_split_latency_xgboost[0],
                                                                       data_split_latency_xgboost[1]),
                                                                      (data_split_latency_xgboost_test[0],
                                                                       data_split_latency_xgboost_test[1])],
                                                            eval_metric='rmse', early_stopping_rounds=10)


print array_print(data_split_latency_xgboost_test[1])
print array_print( xgboost_best_latency.predict(data_split_latency_xgboost_test[0]))
print array_print(evaluator(data_split_latency_xgboost[1],
                            xgboost_best_latency.predict(data_split_latency_xgboost[0])))
print array_print(evaluator(data_split_latency_xgboost_test[1],
                            xgboost_best_latency.predict(data_split_latency_xgboost_test[0])))

print (time.time()-time2)
# ###########################################################################
