import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import sklearn.metrics as mx
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV

summary_data = 'resources/wso2apimanagerperformanceresults.csv'
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
t_size = 0.30  # percentage for testing (test size)
n_rows = 117   # total rows
r_seed = 98 # seed of random (random seed)
kernel_grid = ['rbf', 'poly', 'linear']
c_grid = [1E2, 1E3, 1E4, 1E5, 1E6]
epsilion_grid = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
target = y_select_col_90th_percentile  # change the target as y_select_col_throughput,
# y_select_col_latency,y_select_col_90th_percentile,
# y_select_col_95th_percentile,y_select_col_99th_percentile

arr_rmse = np.array([], dtype='float64')
arr_rmse_test = np.array([], dtype='float64')

arr_mape = np.array([], dtype='float64')
arr_mape_test = np.array([], dtype='float64')

arr_c_values = np.array([], dtype='float64')
arr_epsilion_values = np.array([], dtype='float64')


def data_reader(csv_file, total_row, thousands_splitter, csv_select_columns, x_column_numbers, y_column_number):
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
    return x, y


def mean_absolute_percentage_error(y_true, y_pred):
    output_errors = 100*np.average(np.abs(y_true - y_pred)/y_true)
    return np.average(output_errors)


def evaluator(y_train_ev, y_pred_ev):
    confidence_rmse = np.sqrt(mx.mean_squared_error(y_true=y_train_ev, y_pred=y_pred_ev))
    confidence_mape = mean_absolute_percentage_error(y_true=y_train_ev, y_pred=y_pred_ev)
    return confidence_rmse, confidence_mape


def array_print(array_get):
    for i in range(0, len(array_get)-1):
        print array_get[i]


# ###################################################################################
# data splits
data_splits = np.array([], dtype='float64')
data_splits = data_reader(csv_file=summary_data, total_row=n_rows, thousands_splitter=t_splitter,
                          csv_select_columns=csv_select_cols, x_column_numbers=x_select_cols,
                          y_column_number=target)

x_tra, x_tes, y_tra, y_tes = train_test_split(data_splits[0], data_splits[1], test_size=t_size, random_state=r_seed)

# ##################################################################################
# SVR Grid Search with Cross- Validation
parameters = {'kernel': kernel_grid, 'C': c_grid, 'epsilon': epsilion_grid}
svr = SVR()
svr_model = GridSearchCV(svr, parameters)
svr_model.fit(x_tra, y_tra)
y_prd_test = svr_model.predict(x_tes)  # predict y values by train x_train

print '\ntest data\n', array_print(y_tes)
print '\ntest pred data - svr gird cv\n', array_print(y_prd_test)

current_result_array_test = evaluator(y_train_ev=y_tes, y_pred_ev=y_prd_test)
arr_rmse_in_test = current_result_array_test[0]
arr_mape_in_test = current_result_array_test[1]

print "\ntest rmse = ", arr_rmse_in_test
print "test mape = ", arr_mape_in_test
print "best hyper parameters = ", svr_model.best_params_, "\n\n"

# #################################################################################
# xgboost regression
print 'XGBoost - Regression'
xgb_model = XGBRegressor()  # haven't parameters
xgb_model.fit(x_tra, y_tra)
y_prd_test = xgb_model.predict(x_tes)  # predict y values by train x_train
print '\ntest data\n', array_print(y_tes)
print '\ntest pred data - xgboost\n', array_print(y_prd_test)

current_result_array_test = evaluator(y_train_ev=y_tes, y_pred_ev=y_prd_test)
arr_rmse_in_test_xg = current_result_array_test[0]
arr_mape_in_test_xg = current_result_array_test[1]

print "\nrmse on test data and xg boost = ", arr_rmse_in_test_xg
print "mape on test data and xg boost = ", arr_mape_in_test_xg
