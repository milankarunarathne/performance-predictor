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
x_select_cols_throughput_svr = [0, 1, 2, 3, 4]  # additional feature is 4 = Message size * Concurrency users
x_select_cols_throughput_xgboost = [0, 1, 2, 3, 7]  # additional feature is 7 = Message size * Sleep time
x_select_cols_latency_svr = [0, 1, 2, 3, 4]  # additional feature
x_select_cols_latency_xgboost = [0, 1, 2, 3, 4]
x_select_cols_90th_percentile_svr = [0, 1, 2, 3, 4]
x_select_cols_90th_percentile_xgboost = [0, 1, 2, 3, 4]
x_select_cols_95th_percentile_svr = [0, 1, 2, 3, 4]
x_select_cols_95th_percentile_xgboost = [0, 1, 2, 3, 4]
x_select_cols_99th_percentile_svr = [0, 1, 2, 3, 4]
x_select_cols_99th_percentile_xgboost = [0, 1, 2, 3, 4]
x_select_cols_load_average_1_minute_svr = [0, 1, 2, 3, 10]
x_select_cols_load_average_1_minute_xgboost = [0, 1, 2, 3, 11]
x_select_cols_load_average_5_minute_svr = [0, 1, 2, 3]
x_select_cols_load_average_5_minute_xgboost = [0, 1, 2, 3, 13]
x_select_cols_load_average_15_minute_svr = [0, 1, 2, 3]
x_select_cols_load_average_15_minute_xgboost = [0, 1, 2, 3, 8]
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
# kernel_type_array = ['rbf', 'poly', 'linear']
# c_array = [1E-3, 1E-2, 1E-1, 1E0, 1E1, 1E2, 1E3, 1E4, 1E5, 1E6]
# epsilion_array = [0.001, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
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
