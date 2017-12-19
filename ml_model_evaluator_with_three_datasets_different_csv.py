import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import sklearn.metrics as mx
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt
import thread
import time

summary_data = 'resources/train/old/wso2apimanagerperformanceresults.csv'
summary_data_test = 'resources/test/old/wso2apimanagerperformanceresults.csv'
t_splitter = ","
csv_select_cols = [0, 1, 2, 3, 7, 10, 11, 12, 13, 36, 37, 38]
x_select_cols = [0, 1, 2, 3]  # select columns to x (features)
y_select_col_latency = 4
y_select_col_throughput = 8
y_select_col_load_average_last_1minute = 5
y_select_col_load_average_last_5minute = 6
y_select_col_load_average_last_15minute = 7
y_select_col_90th_percentile = 9
y_select_col_95th_percentile = 10
y_select_col_99th_percentile = 11
target = y_select_col_90th_percentile
tst_size = 0.3  # percentage for testing (test size)
n_rows = 117   # total rows
row_start = 25  # testing rows at start
r_seed = 42  # seed of random (random seed)
# epsilion_default = 0.01
gamma_default = 'auto'
# c_defalut = 1e5
c_s = -3
c_e = 8
e_s = 0.001
e_e = 10


arr_rmse = np.array([], dtype='float64')

arr_mape = np.array([], dtype='float64')
arr_rmse_test = np.array([], dtype='float64')
arr_mape_test = np.array([], dtype='float64')

arr_c_values = np.array([], dtype='float64')
arr_epsilion_values = np.array([], dtype='float64')


def data_reader(csv_file, total_row, thousands_splitter, csv_select_columns, x_column_numbers, y_column_number):
    # time5 = time.time()
    # read the file
    data_file = pd.read_csv(csv_file, thousands=thousands_splitter, usecols=csv_select_columns,)
    #  replace Echo API and Mediation API with 1 and 2
    datapd = pd.DataFrame.replace(data_file, to_replace=['Echo API', 'Mediation API'], value=[1, 2])
    data = np.array(datapd, copy=True, )
    dataset_row_n = data[0:total_row, :]  # select specific number of rows
    x = preprocessing.scale(dataset_row_n[:, x_column_numbers])  # machine learning to be in a range of -1 to 1.
    # This may do nothing, but it usually speeds up processing and can also help with accuracy.
    # Because this range is so popularly used
    y = data[:, y_column_number]
    return x, y


def svr_regression_fit(d_array, test_size, random_seed, c, gamma, epsilion, kernel_type):    # linear and rbf
    x_train, x_test, y_train, y_test = train_test_split(d_array[0], d_array[1], test_size=test_size, random_state=random_seed)  # 30% testing
    svr_kernel = SVR(kernel=kernel_type, C=c, gamma=gamma, epsilon=epsilion)  # kernel='rbf', C=1e6, gamma=0.1, epsilon=0.01
    svr_kernel.fit(x_train, y_train)
    return svr_kernel


def svr_regression_fit_poly(d_array, test_size, random_seed, c, gamma, epsilion, kernel_type, deg_svr):
    x_train, x_test, y_train, y_test = train_test_split(d_array[0], d_array[1], test_size=test_size, random_state=random_seed)  # 30% testing
    svr_kernel_poly = SVR(kernel=kernel_type, degree=deg_svr, C=c, gamma=gamma, epsilon=epsilion)  # kernel='rbf', C=1e6, gamma=0.1, epsilon=0.01
    svr_kernel_poly.fit(x_train, y_train)
    return svr_kernel_poly


def xgboost_regression_fit(d_array, test_size, random_seed):
    x_train, x_test, y_train, y_test = train_test_split(d_array[0], d_array[1], test_size=test_size, random_state=random_seed)  # 30% testing
    xgb_kernel = XGBRegressor()  # not parameters for tune
    xgb_kernel.fit(x_train, y_train)
    return xgb_kernel


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


def hy_para_finder(data_split, data_split_test, kernel_ty, degree, arr_c_values_in, arr_epsilion_values_in, arr_rmse_in, arr_rmse_in_test, arr_mape_in,arr_mape_in_test, t_size, rnd_state, c_start, c_end, epsilion_start, epsilion_end ):
    print kernel_ty, degree
    time3 = time.time()
    l_i = c_start
    while l_i <= c_end:  # C , best 1e5 , 1e-3 >> 1e8
        l_e = epsilion_start
        while l_e <= epsilion_end:  # epsilion , example 0.01 , 0.001 >> 10    10*
            "SVR - epsilion"
            if kernel_ty == 'poly':
                model = svr_regression_fit_poly(data_split, test_size=t_size, random_seed=rnd_state, c=10**l_i,
                                                gamma=gamma_default, epsilion=l_e, kernel_type=kernel_ty,
                                                deg_svr=degree)
            else:
                model = svr_regression_fit(data_split, test_size=t_size, random_seed=rnd_state, c=10**l_i,
                                           gamma=gamma_default, epsilion=l_e, kernel_type=kernel_ty)
            print model
            x_tra, x_tes, y_tra, y_tes = train_test_split(data_split[0], data_split[1], test_size=t_size, random_state=rnd_state)  # 30% testing
            y_prd = model.predict(x_tes)  # predict y values by x_test (part of the train set but it's not use to train)
            y_prd_test = model.predict(data_split_test[0])  # predict y values by train x_train
            current_result_array = evaluator(y_train_ev=y_tes, y_pred_ev=y_prd)
            arr_rmse_in = np.append(arr_rmse_in, current_result_array[0])
            arr_mape_in = np.append(arr_mape_in, current_result_array[1])
            arr_epsilion_values_in = np.append(arr_epsilion_values_in, l_e)
            arr_c_values_in = np.append(arr_c_values_in, 10**l_i)
            current_result_array_test = evaluator(y_train_ev=data_split_test[1], y_pred_ev=y_prd_test)
            arr_rmse_in_test = np.append(arr_rmse_in_test, current_result_array_test[0])
            arr_mape_in_test = np.append(arr_mape_in_test, current_result_array_test[1])
            l_e += 0.499
        l_i += 1
    # #############################################################################

    print (time.time() - time3)
    array_print(arr_c_values_in)
    array_print(arr_epsilion_values_in)
    print "rmse"
    array_print(arr_rmse_in)
    print "mape"
    array_print(arr_mape_in)
    print "test rmse"
    array_print(arr_rmse_in_test)
    print "test mape"
    array_print(arr_mape_in_test)

    return arr_rmse_in, arr_mape_in, arr_c_values_in, arr_epsilion_values_in


# ###################################################################################
# throughput
data_split_throughput = np.array([], dtype='float64')
data_split_throughput = data_reader(csv_file=summary_data, total_row=n_rows,
                                    thousands_splitter=t_splitter, csv_select_columns=csv_select_cols,
                                    x_column_numbers=x_select_cols, y_column_number=target)

data_split_throughput_test = np.array([], dtype='float64')
data_split_throughput_test = data_reader(csv_file=summary_data_test, total_row=n_rows, thousands_splitter=t_splitter,
                                         csv_select_columns=csv_select_cols, x_column_numbers=x_select_cols,
                                         y_column_number=target)


# # linear
arrays_linear_throughput = hy_para_finder(data_split=data_split_throughput, data_split_test=data_split_throughput_test,
                                          arr_rmse_in_test=arr_rmse_test, arr_mape_in_test=arr_mape_test,
                                          arr_rmse_in=arr_rmse, arr_mape_in=arr_mape, arr_c_values_in=arr_c_values,
                                          arr_epsilion_values_in=arr_epsilion_values, kernel_ty='rbf',
                                          degree='auto', t_size=tst_size, rnd_state=r_seed, c_start=c_s, c_end=c_e,
                                          epsilion_start=e_s, epsilion_end=e_e)
