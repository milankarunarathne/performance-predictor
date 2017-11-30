import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVR
from xgboost import XGBRegressor
import sklearn.metrics as mx
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
import time


summary_data = 'resources/train/old/wso2apimanagerperformanceresults.csv'
summary_data_test = 'resources/test/old/wso2apimanagerperformanceresults.csv'
t_splitter = ","

csv_select_cols = [0, 1, 2, 3, 7, 10, 11, 12, 13, 36, 37, 38]
x_select_cols = [0, 1, 2, 3]  # select columns to x (features)
y_select_col_latency = 4
y_select_col_90th_percentile = 5
y_select_col_95th_percentile = 6
y_select_col_99th_percentile = 7
y_select_col_throughput = 8
y_select_col_load_average_1_minute = 9
y_select_col_load_average_5_minute = 10
y_select_col_load_average_15_minute = 11
target = y_select_col_99th_percentile
t_size = 0.0  # percentage for testing (test size)
n_rows = 117   # total rows
row_start = 25  # testing rows at start
r_seed = 42  # seed of random (random seed)
kernel_type_array = ['rbf', 'poly', 'linear']
c_array = [1E-3, 1E-2, 1E-1,1E0, 1E1, 1E2, 1E3, 1E4, 1E5, 1E6, 1E7, 1E8 ]
epsilion_array = [0.001, 0.5, 0.999, 1.498, 1.997, 2.496, 2.995, 3.494, 3.993, 4.492, 4.991, 5.49, 5.989, 6.488, 6.987, 7.486, 7.985, 8.484, 8.983, 9.482, 9.981]
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


def evaluator(y_train_ev, y_pred_ev):
    confidence_rmse = np.sqrt(mx.mean_squared_error(y_true=y_train_ev, y_pred=y_pred_ev))
    confidence_mape = mean_absolute_percentage_error(y_true=y_train_ev, y_pred=y_pred_ev)
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
print "\n\n\nSVR Grid Search CV "
parameters = {'kernel': kernel_type_array, 'C': c_array, 'epsilon': epsilion_array}
svr = SVR()
svr_model = GridSearchCV(svr, parameters)
svr_model.fit(data_split_throughput[0], data_split_throughput[1])
y_prd = svr_model.predict(data_split_throughput[0])
print 'train data'
print array_print(data_split_throughput[1])
print 'train pred'
print array_print(y_prd)
y_prd_test = svr_model.predict(data_split_throughput_test[0])  # predict y values by train x_train
print 'test data'
print array_print(data_split_throughput_test[1])
print 'test pred data'
print array_print(y_prd_test)
print svr_model
y_prd = svr_model.predict(data_split_throughput[0])
y_prd_test = svr_model.predict(data_split_throughput_test[0])  # predict y values by train x_train
current_result_array = evaluator(y_train_ev=data_split_throughput[1], y_pred_ev=y_prd)
arr_rmse_in = current_result_array[0]
arr_mape_in = current_result_array[1]
current_result_array_test = evaluator(y_train_ev=data_split_throughput_test[1], y_pred_ev=y_prd_test)
arr_rmse_in_test = current_result_array_test[0]
arr_mape_in_test = current_result_array_test[1]
# #############################################################################
print "\n\n\nresults "
print "rmse"
print arr_rmse_in
print "mape"
print arr_mape_in
print "test rmse"
print arr_rmse_in_test
print "test mape"
print arr_mape_in_test

print svr_model.best_params_

print 'time', time.time()-time2
# #############################################################################
print "\n\n\nXGBoost "
xgb_model = XGBRegressor()  # not parameters for tune
xgb_model.fit(data_split_throughput[0], data_split_throughput[1])
y_prd = xgb_model.predict(data_split_throughput[0])
print 'train data'
print array_print(data_split_throughput[1])
print 'train pred'
print array_print(y_prd)
y_prd_test = xgb_model.predict(data_split_throughput_test[0])  # predict y values by train x_train
print 'test data'
print array_print(data_split_throughput_test[1])
print 'test pred data'
print array_print(y_prd_test)
current_result_array = evaluator(y_train_ev=data_split_throughput[1], y_pred_ev=y_prd)
arr_rmse_in_xg = current_result_array[0]
arr_mape_in_xg = current_result_array[1]
current_result_array_test = evaluator(y_train_ev=data_split_throughput_test[1], y_pred_ev=y_prd_test)
arr_rmse_in_test_xg = current_result_array_test[0]
arr_mape_in_test_xg = current_result_array_test[1]
# #################################################################################

print "rmse xg"
print arr_rmse_in_xg
print "mape xg"
print arr_mape_in_xg
print "test rmse xg"
print arr_rmse_in_test_xg
print "test mape xg"
print arr_mape_in_test_xg

print (time.time()-time2)


time2 = time.time()
# ###################################################################################
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

#  ##################################################################################
# SVR Grid Search to latency
print "\n\n\nSVR Grid Search CV "
parameters = {'kernel': kernel_type_array , 'C': c_array, 'epsilon': epsilion_array}
svr = SVR()
svr_model = GridSearchCV(svr, parameters)
svr_model.fit(data_split_latency[0], data_split_latency[1])
y_prd = svr_model.predict(data_split_latency[0])
print 'train data'
print array_print(data_split_latency[1])
print 'train pred'
print array_print(y_prd)
y_prd_test = svr_model.predict(data_split_latency_test[0])  # predict y values by train x_train
print 'test data'
print array_print(data_split_latency_test[1])
print 'test pred data'
print array_print(y_prd_test)
print svr_model
y_prd = svr_model.predict(data_split_latency[0])
y_prd_test = svr_model.predict(data_split_latency_test[0])  # predict y values by train x_train
current_result_array = evaluator(y_train_ev=data_split_latency[1], y_pred_ev=y_prd)
arr_rmse_in = current_result_array[0]
arr_mape_in = current_result_array[1]
current_result_array_test = evaluator(y_train_ev=data_split_latency_test[1], y_pred_ev=y_prd_test)
arr_rmse_in_test = current_result_array_test[0]
arr_mape_in_test = current_result_array_test[1]
# #############################################################################
print "\n\n\nresults "
print "rmse"
print arr_rmse_in
print "mape"
print arr_mape_in
print "test rmse"
print arr_rmse_in_test
print "test mape"
print arr_mape_in_test

print svr_model.best_params_

print 'time', time.time()-time2

# ######################################################################################
print "\n\n\nXGBoost "
xgb_model = XGBRegressor()  # not parameters for tune
xgb_model.fit(data_split_latency[0], data_split_latency[1])
y_prd = xgb_model.predict(data_split_latency[0])
print 'train data'
print array_print(data_split_latency[1])
print 'train pred'
print array_print(y_prd)
y_prd_test = xgb_model.predict(data_split_latency_test[0])  # predict y values by train x_train
print 'test data'
print array_print(data_split_latency_test[1])
print 'test pred data'
print array_print(y_prd_test)
current_result_array = evaluator(y_train_ev=data_split_latency[1], y_pred_ev=y_prd)
arr_rmse_in_xg = current_result_array[0]
arr_mape_in_xg = current_result_array[1]
current_result_array_test = evaluator(y_train_ev=data_split_latency_test[1], y_pred_ev=y_prd_test)
arr_rmse_in_test_xg = current_result_array_test[0]
arr_mape_in_test_xg = current_result_array_test[1]

# ########################################################################
print "\n\n\nresults "

print "rmse xg"
print arr_rmse_in_xg
print "mape xg"
print arr_mape_in_xg
print "test rmse xg"
print arr_rmse_in_test_xg
print "test mape xg"
print arr_mape_in_test_xg

print (time.time()-time2)


time2 = time.time()
# ###################################################################################
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

#  ##################################################################################
# SVR Grid Search to 90th percentile
print "\n\n\nSVR Grid Search CV "
parameters = {'kernel': kernel_type_array , 'C': c_array, 'epsilon': epsilion_array}
svr = SVR()
svr_model = GridSearchCV(svr, parameters)
svr_model.fit(data_split_90th_percentile[0], data_split_90th_percentile[1])
y_prd = svr_model.predict(data_split_90th_percentile[0])
print 'train data'
print array_print(data_split_90th_percentile[1])
print 'train pred'
print array_print(y_prd)
y_prd_test = svr_model.predict(data_split_90th_percentile_test[0])  # predict y values by train x_train
print 'test data'
print array_print(data_split_90th_percentile_test[1])
print 'test pred data'
print array_print(y_prd_test)
print svr_model
y_prd = svr_model.predict(data_split_90th_percentile[0])
y_prd_test = svr_model.predict(data_split_90th_percentile_test[0])  # predict y values by train x_train
current_result_array = evaluator(y_train_ev=data_split_90th_percentile[1], y_pred_ev=y_prd)
arr_rmse_in = current_result_array[0]
arr_mape_in = current_result_array[1]
current_result_array_test = evaluator(y_train_ev=data_split_90th_percentile_test[1], y_pred_ev=y_prd_test)
arr_rmse_in_test = current_result_array_test[0]
arr_mape_in_test = current_result_array_test[1]
# #############################################################################
print "\n\n\nresults "

print "rmse"
print arr_rmse_in
print "mape"
print arr_mape_in
print "test rmse"
print arr_rmse_in_test
print "test mape"
print arr_mape_in_test

print svr_model.best_params_

print 'time', time.time()-time2

# ####################################################################################
print "\n\n\nXGBoost "
xgb_model = XGBRegressor()  # not parameters for tune
xgb_model.fit(data_split_90th_percentile[0], data_split_90th_percentile[1])
y_prd = xgb_model.predict(data_split_90th_percentile[0])
print 'train data'
print array_print(data_split_90th_percentile[1])
print 'train pred'
print array_print(y_prd)
y_prd_test = xgb_model.predict(data_split_90th_percentile_test[0])  # predict y values by train x_train
print 'test data'
print array_print(data_split_90th_percentile_test[1])
print 'test pred data'
print array_print(y_prd_test)
current_result_array = evaluator(y_train_ev=data_split_90th_percentile[1], y_pred_ev=y_prd)
arr_rmse_in_xg = current_result_array[0]
arr_mape_in_xg = current_result_array[1]
current_result_array_test = evaluator(y_train_ev=data_split_90th_percentile_test[1], y_pred_ev=y_prd_test)
arr_rmse_in_test_xg = current_result_array_test[0]
arr_mape_in_test_xg = current_result_array_test[1]
# ################################################################################
print "\n\n\nresults "

print "rmse xg"
print arr_rmse_in_xg
print "mape xg"
print arr_mape_in_xg
print "test rmse xg"
print arr_rmse_in_test_xg
print "test mape xg"
print arr_mape_in_test_xg

print (time.time()-time2)


time2 = time.time()
# ###################################################################################
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

#  ##################################################################################
# SVR Grid Search to 95th percentile
print "\n\n\nSVR Grid Search CV "
parameters = {'kernel': kernel_type_array , 'C': c_array, 'epsilon': epsilion_array}
svr = SVR()
svr_model = GridSearchCV(svr, parameters)
svr_model.fit(data_split_95th_percentile[0], data_split_95th_percentile[1])
y_prd = svr_model.predict(data_split_95th_percentile[0])
print 'train data'
print array_print(data_split_95th_percentile[1])
print 'train pred'
print array_print(y_prd)
y_prd_test = svr_model.predict(data_split_95th_percentile_test[0])  # predict y values by train x_train
print 'test data'
print array_print(data_split_95th_percentile_test[1])
print 'test pred data'
print array_print(y_prd_test)
print svr_model
y_prd = svr_model.predict(data_split_95th_percentile[0])
y_prd_test = svr_model.predict(data_split_95th_percentile_test[0])  # predict y values by train x_train
current_result_array = evaluator(y_train_ev=data_split_95th_percentile[1], y_pred_ev=y_prd)
arr_rmse_in = current_result_array[0]
arr_mape_in = current_result_array[1]
current_result_array_test = evaluator(y_train_ev=data_split_95th_percentile_test[1], y_pred_ev=y_prd_test)
arr_rmse_in_test = current_result_array_test[0]
arr_mape_in_test = current_result_array_test[1]
# #############################################################################
print "\n\n\nresults "
print "rmse"
print arr_rmse_in
print "mape"
print arr_mape_in
print "test rmse"
print arr_rmse_in_test
print "test mape"
print arr_mape_in_test

print svr_model.best_params_

print 'time', time.time()-time2
# #############################################################################################
print "\n\n\nXGBoost "
xgb_model = XGBRegressor()  # not parameters for tune
xgb_model.fit(data_split_95th_percentile[0], data_split_95th_percentile[1])
y_prd = xgb_model.predict(data_split_95th_percentile[0])
print 'train data'
print array_print(data_split_95th_percentile[1])
print 'train pred'
print array_print(y_prd)
y_prd_test = xgb_model.predict(data_split_95th_percentile_test[0])  # predict y values by train x_train
print 'test data'
print array_print(data_split_95th_percentile_test[1])
print 'test pred data'
print array_print(y_prd_test)
current_result_array = evaluator(y_train_ev=data_split_95th_percentile[1], y_pred_ev=y_prd)
arr_rmse_in_xg = current_result_array[0]
arr_mape_in_xg = current_result_array[1]
current_result_array_test = evaluator(y_train_ev=data_split_95th_percentile_test[1], y_pred_ev=y_prd_test)
arr_rmse_in_test_xg = current_result_array_test[0]
arr_mape_in_test_xg = current_result_array_test[1]
# ####################################################################################
print "\n\n\nresults "
print "rmse xg"
print arr_rmse_in_xg
print "mape xg"
print arr_mape_in_xg
print "test rmse xg"
print arr_rmse_in_test_xg
print "test mape xg"
print arr_mape_in_test_xg

print (time.time()-time2)


time2 = time.time()
# ###################################################################################
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

#  ##################################################################################
# SVR Grid Search to 99th_percentile
print "\n\n\nSVR Grid Search CV "
parameters = {'kernel': kernel_type_array , 'C': c_array, 'epsilon': epsilion_array}
svr = SVR()
svr_model = GridSearchCV(svr, parameters)
svr_model.fit(data_split_99th_percentile[0], data_split_99th_percentile[1])
y_prd = svr_model.predict(data_split_99th_percentile[0])
print 'train data'
print array_print(data_split_99th_percentile[1])
print 'train pred'
print array_print(y_prd)
y_prd_test = svr_model.predict(data_split_99th_percentile_test[0])  # predict y values by train x_train
print 'test data'
print array_print(data_split_99th_percentile_test[1])
print 'test pred data'
print array_print(y_prd_test)
print svr_model
y_prd = svr_model.predict(data_split_99th_percentile[0])
y_prd_test = svr_model.predict(data_split_99th_percentile_test[0])  # predict y values by train x_train
current_result_array = evaluator(y_train_ev=data_split_99th_percentile[1], y_pred_ev=y_prd)
arr_rmse_in = current_result_array[0]
arr_mape_in = current_result_array[1]
current_result_array_test = evaluator(y_train_ev=data_split_99th_percentile_test[1], y_pred_ev=y_prd_test)
arr_rmse_in_test = current_result_array_test[0]
arr_mape_in_test = current_result_array_test[1]
# #######################################################################################
print "\n\n\nresults "
print "rmse"
print arr_rmse_in
print "mape"
print arr_mape_in
print "test rmse"
print arr_rmse_in_test
print "test mape"
print arr_mape_in_test

print svr_model.best_params_

print 'time', time.time()-time2
# #######################################################################################
print "\n\n\nXGBoost "
xgb_model = XGBRegressor()  # not parameters for tune
xgb_model.fit(data_split_99th_percentile[0], data_split_99th_percentile[1])
y_prd = xgb_model.predict(data_split_99th_percentile[0])
print 'train data'
print array_print(data_split_99th_percentile[1])
print 'train pred'
print array_print(y_prd)
y_prd_test = xgb_model.predict(data_split_99th_percentile_test[0])  # predict y values by train x_train
print 'test data'
print array_print(data_split_99th_percentile_test[1])
print 'test pred data'
print array_print(y_prd_test)
current_result_array = evaluator(y_train_ev=data_split_99th_percentile[1], y_pred_ev=y_prd)
arr_rmse_in_xg = current_result_array[0]
arr_mape_in_xg = current_result_array[1]
current_result_array_test = evaluator(y_train_ev=data_split_99th_percentile_test[1], y_pred_ev=y_prd_test)
arr_rmse_in_test_xg = current_result_array_test[0]
arr_mape_in_test_xg = current_result_array_test[1]
# ##################################################################################
print "\n\n\nresults "
print "rmse xg"
print arr_rmse_in_xg
print "mape xg"
print arr_mape_in_xg
print "test rmse xg"
print arr_rmse_in_test_xg
print "test mape xg"
print arr_mape_in_test_xg

print (time.time()-time2)


time2 = time.time()
# ###################################################################################
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

#  ##################################################################################
# SVR Grid Search to load_average_1_minute
print "\n\n\nSVR Grid Search CV "
parameters = {'kernel': kernel_type_array , 'C': c_array, 'epsilon': epsilion_array}
svr = SVR()
svr_model = GridSearchCV(svr, parameters)
svr_model.fit(data_split_load_average_1_minute[0], data_split_load_average_1_minute[1])
y_prd = svr_model.predict(data_split_load_average_1_minute[0])
print 'train data'
print array_print(data_split_load_average_1_minute[1])
print 'train pred'
print array_print(y_prd)
y_prd_test = svr_model.predict(data_split_load_average_1_minute_test[0])  # predict y values by train x_train
print 'test data'
print array_print(data_split_load_average_1_minute_test[1])
print 'test pred data'
print array_print(y_prd_test)
print svr_model
y_prd = svr_model.predict(data_split_load_average_1_minute[0])
y_prd_test = svr_model.predict(data_split_load_average_1_minute_test[0])  # predict y values by train x_train
current_result_array = evaluator(y_train_ev=data_split_load_average_1_minute[1], y_pred_ev=y_prd)
arr_rmse_in = current_result_array[0]
arr_mape_in = current_result_array[1]
current_result_array_test = evaluator(y_train_ev=data_split_load_average_1_minute_test[1], y_pred_ev=y_prd_test)
arr_rmse_in_test = current_result_array_test[0]
arr_mape_in_test = current_result_array_test[1]
# #############################################################################
print "\n\n\nresults "
print "rmse"
print arr_rmse_in
print "mape"
print arr_mape_in
print "test rmse"
print arr_rmse_in_test
print "test mape"
print arr_mape_in_test

print svr_model.best_params_

print 'time', time.time()-time2
# ##################################################################################
print "\n\n\nXGBoost "
xgb_model = XGBRegressor()  # not parameters for tune
xgb_model.fit(data_split_load_average_1_minute[0], data_split_load_average_1_minute[1])
y_prd = xgb_model.predict(data_split_load_average_1_minute[0])
print 'train data'
print array_print(data_split_load_average_1_minute[1])
print 'train pred'
print array_print(y_prd)
y_prd_test = xgb_model.predict(data_split_load_average_1_minute_test[0])  # predict y values by train x_train
print 'test data'
print array_print(data_split_load_average_1_minute_test[1])
print 'test pred data'
print array_print(y_prd_test)
current_result_array = evaluator(y_train_ev=data_split_load_average_1_minute[1], y_pred_ev=y_prd)
arr_rmse_in_xg = current_result_array[0]
arr_mape_in_xg = current_result_array[1]
current_result_array_test = evaluator(y_train_ev=data_split_load_average_1_minute_test[1], y_pred_ev=y_prd_test)
arr_rmse_in_test_xg = current_result_array_test[0]
arr_mape_in_test_xg = current_result_array_test[1]
# ####################################################################################
print "\n\n\nresults "
print "rmse xg"
print arr_rmse_in_xg
print "mape xg"
print arr_mape_in_xg
print "test rmse xg"
print arr_rmse_in_test_xg
print "test mape xg"
print arr_mape_in_test_xg

print (time.time()-time2)


time2 = time.time()
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

#  ##################################################################################
# SVR Grid Search to load_average_5_minutes
print "\n\n\nSVR Grid Search CV "
parameters = {'kernel': kernel_type_array , 'C': c_array, 'epsilon': epsilion_array}
svr = SVR()
svr_model = GridSearchCV(svr, parameters)
svr_model.fit(data_split_load_average_5_minute[0], data_split_load_average_5_minute[1])
y_prd = svr_model.predict(data_split_load_average_5_minute[0])
print 'train data'
print array_print(data_split_load_average_5_minute[1])
print 'train pred'
print array_print(y_prd)
y_prd_test = svr_model.predict(data_split_load_average_5_minute_test[0])  # predict y values by train x_train
print 'test data'
print array_print(data_split_load_average_5_minute_test[1])
print 'test pred data'
print array_print(y_prd_test)
print svr_model
y_prd = svr_model.predict(data_split_load_average_5_minute[0])
y_prd_test = svr_model.predict(data_split_load_average_5_minute_test[0])  # predict y values by train x_train
current_result_array = evaluator(y_train_ev=data_split_load_average_5_minute[1], y_pred_ev=y_prd)
arr_rmse_in = current_result_array[0]
arr_mape_in = current_result_array[1]
current_result_array_test = evaluator(y_train_ev=data_split_load_average_5_minute_test[1], y_pred_ev=y_prd_test)
arr_rmse_in_test = current_result_array_test[0]
arr_mape_in_test = current_result_array_test[1]
# #############################################################################
print "\n\n\nresults "
print "rmse"
print arr_rmse_in
print "mape"
print arr_mape_in
print "test rmse"
print arr_rmse_in_test
print "test mape"
print arr_mape_in_test

print svr_model.best_params_

print 'time', time.time()-time2
# #################################################################################
print "\n\n\nXGBoost "
xgb_model = XGBRegressor()  # not parameters for tune
xgb_model.fit(data_split_load_average_5_minute[0], data_split_load_average_5_minute[1])
y_prd = xgb_model.predict(data_split_load_average_5_minute[0])
print 'train data'
print array_print(data_split_load_average_5_minute[1])
print 'train pred'
print array_print(y_prd)
y_prd_test = xgb_model.predict(data_split_load_average_5_minute_test[0])  # predict y values by train x_train
print 'test data'
print array_print(data_split_load_average_5_minute_test[1])
print 'test pred data'
print array_print(y_prd_test)
current_result_array = evaluator(y_train_ev=data_split_load_average_5_minute[1], y_pred_ev=y_prd)
arr_rmse_in_xg = current_result_array[0]
arr_mape_in_xg = current_result_array[1]
current_result_array_test = evaluator(y_train_ev=data_split_load_average_5_minute_test[1], y_pred_ev=y_prd_test)
arr_rmse_in_test_xg = current_result_array_test[0]
arr_mape_in_test_xg = current_result_array_test[1]
# ###################################################################################
print "\n\n\nresults "
print "rmse xg"
print arr_rmse_in_xg
print "mape xg"
print arr_mape_in_xg
print "test rmse xg"
print arr_rmse_in_test_xg
print "test mape xg"
print arr_mape_in_test_xg

print (time.time()-time2)


time2 = time.time()
# ###################################################################################
# load_average_15_minutes
print "\n\n\nload_average_15_minutes "

data_split_load_average_15_minute = np.array([], dtype='float64')
data_split_load_average_15_minute = data_reader(csv_file=summary_data, total_row=n_rows,
                                    thousands_splitter=t_splitter, csv_select_columns=csv_select_cols,
                                    x_column_numbers=x_select_cols, y_column_number=y_select_col_load_average_5_minute)

data_split_load_average_15_minute_test = np.array([], dtype='float64')
data_split_load_average_15_minute_test = data_reader(csv_file=summary_data_test, total_row=n_rows, thousands_splitter=t_splitter,
                                         csv_select_columns=csv_select_cols, x_column_numbers=x_select_cols,
                                         y_column_number=y_select_col_load_average_5_minute)

#  ##################################################################################
# SVR Grid Search to load_average_15_minutes
print "\n\n\nSVR Grid Search CV "
parameters = {'kernel': kernel_type_array , 'C': c_array, 'epsilon': epsilion_array}
svr = SVR()
svr_model = GridSearchCV(svr, parameters)
svr_model.fit(data_split_load_average_15_minute[0], data_split_load_average_15_minute[1])
y_prd = svr_model.predict(data_split_load_average_15_minute[0])
print 'train data'
print array_print(data_split_load_average_15_minute[1])
print 'train pred'
print array_print(y_prd)
y_prd_test = svr_model.predict(data_split_load_average_15_minute_test[0])  # predict y values by train x_train
print 'test data'
print array_print(data_split_load_average_15_minute_test[1])
print 'test pred data'
print array_print(y_prd_test)
print svr_model
y_prd = svr_model.predict(data_split_load_average_15_minute[0])
y_prd_test = svr_model.predict(data_split_load_average_15_minute_test[0])  # predict y values by train x_train
current_result_array = evaluator(y_train_ev=data_split_load_average_15_minute[1], y_pred_ev=y_prd)
arr_rmse_in = current_result_array[0]
arr_mape_in = current_result_array[1]
current_result_array_test = evaluator(y_train_ev=data_split_load_average_15_minute_test[1], y_pred_ev=y_prd_test)
arr_rmse_in_test = current_result_array_test[0]
arr_mape_in_test = current_result_array_test[1]
# #############################################################################
print "\n\n\nresults "
print "rmse"
print arr_rmse_in
print "mape"
print arr_mape_in
print "test rmse"
print arr_rmse_in_test
print "test mape"
print arr_mape_in_test

print svr_model.best_params_

print 'time', time.time()-time2
# ################################################################################
print "\n\n\nXGBoost "
xgb_model = XGBRegressor()  # not parameters for tune
xgb_model.fit(data_split_load_average_15_minute[0], data_split_load_average_15_minute[1])
y_prd = xgb_model.predict(data_split_load_average_15_minute[0])
print 'train data'
print array_print(data_split_load_average_15_minute[1])
print 'train pred'
print array_print(y_prd)
y_prd_test = xgb_model.predict(data_split_load_average_15_minute_test[0])  # predict y values by train x_train
print 'test data'
print array_print(data_split_load_average_15_minute_test[1])
print 'test pred data'
print array_print(y_prd_test)
current_result_array = evaluator(y_train_ev=data_split_load_average_15_minute[1], y_pred_ev=y_prd)
arr_rmse_in_xg = current_result_array[0]
arr_mape_in_xg = current_result_array[1]
current_result_array_test = evaluator(y_train_ev=data_split_load_average_15_minute_test[1], y_pred_ev=y_prd_test)
arr_rmse_in_test_xg = current_result_array_test[0]
arr_mape_in_test_xg = current_result_array_test[1]
# ####################################################################################
print "\n\n\nresults "
print "rmse xg"
print arr_rmse_in_xg
print "mape xg"
print arr_mape_in_xg
print "test rmse xg"
print arr_rmse_in_test_xg
print "test mape xg"
print arr_mape_in_test_xg

print 'time', time.time()-time2
# ################################################################################
print 'total time = ', time.time()-time1