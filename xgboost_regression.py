import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn import preprocessing
# from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


summary_data = 'resources/wso2apimanagerperformanceresults.csv'
x_select_columns = [0, 1, 2, 3]  # select columns to x (features)
y_select_column_throughput = 5
y_select_column_latency = 4
test_size = 0.33  # percentage for testing
n_rows = 117   # total rows
row_start = 25  # testing rows at start

# read the file
datasetno = pd.read_csv(summary_data, thousands=",", usecols=[0, 1, 2, 3, 7, 13],)
#  replace Echo API and Mediation API with 1 and 2
datapd = pd.DataFrame.replace(datasetno, to_replace=['Echo API', 'Mediation API'], value=[1, 2])
data = np.array(datapd, copy=True, )


def xgboost_regression_throughput(dataset, r):
    dataset_row_n = dataset[0:r, :]  # select specific number of rows
    x = preprocessing.scale(dataset_row_n[:, x_select_columns])  # machine learning to be in a range of -1 to 1.
    # This may do nothing, but it usually speeds up processing and can also help with accuracy.
    # Because this range is so popularly used
    y = dataset_row_n[:, y_select_column_throughput]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    model = XGBRegressor()
    model.fit(x_train, y_train)
    # svr_rbf.fit(x_train, y_train)
    # confidence_throughput = svr_rbf.score(x_test, y_test)
    confidence_throughput = model.score(x_test, y_test)
    return confidence_throughput
# #############################################################################


def xgboost_regression_latency(dataset, r):
    dataset_row_n = dataset[0:r, :]  # select specific number of rows
    x = preprocessing.scale(dataset_row_n[:, x_select_columns])  # machine learning to be in a range of -1 to 1.
    # This may do nothing, but it usually speeds up processing and can also help with accuracy.
    # Because this range is so popularly used
    y = dataset_row_n[:, y_select_column_latency]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    model = XGBRegressor()
    model.fit(x_train, y_train)
    # svr_rbf.fit(x_train, y_train)
    # confidence_latency = svr_rbf.score(x_test, y_test)
    confidence_latency = model.score(x_test, y_test)
    return confidence_latency
# ################################################################################


confidence_results_throughput = np.array([], dtype='float64')

for i in range(row_start, n_rows):
    confidence_results_throughput = np.append(confidence_results_throughput, xgboost_regression_throughput(data, i))


confidence_results_latency = np.array([], dtype='float64')

for i in range(row_start, n_rows):
    confidence_results_latency = np.append(confidence_results_latency, xgboost_regression_latency(data, i))
###########################################################################


lw = 2
plt.plot(confidence_results_throughput, color='navy', lw=lw, label='Thr')
plt.xlim(row_start, n_rows)
# plt.xlabel('total rows')
plt.ylabel('success score (1 is best)')
plt.title('XGBoost')
# plt.legend()
#  plt.show()

lw = 2
plt.plot(confidence_results_latency, color='red', lw=lw, label='Lat')
plt.xlim(row_start, n_rows)
plt.xlabel('number of rows (use ML by increasing volume of data)')
plt.legend()
plt.show()
