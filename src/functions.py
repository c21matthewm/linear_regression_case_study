import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as spicystats
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from pandas.plotting import scatter_matrix
import statsmodels.api as sm



def rmse(true, predicted):
    mse = mean_squared_error(true, predicted)
    return np.sqrt(mse)



def cross_val(X_train, y_train, k):    
    kf = KFold(k)
    kf.get_n_splits(X_train)

    # print(kf.split(X_train))

    train_dict = {}
    test_dict = {}

    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        
        X_train_kfold = X_train.iloc[train_index]
        y_train_kfold = y_train.iloc[train_index]
        X_test_kfold = X_train.iloc[test_index]
        y_test_kfold = y_train.iloc[test_index]

        reg = LinearRegression()
        reg.fit(X_train_kfold, y_train_kfold)

        # Call predict to get the predicted values for training and test set
        train_predicted = reg.predict(X_train_kfold)
        test_predicted = reg.predict(X_test_kfold)

        # Calculate RMSE for training and test set
        train_dict[i] = rmse(y_train_kfold, train_predicted)
        test_dict[i] = rmse(y_test_kfold, test_predicted)

    return np.mean(list(train_dict.values())), np.mean(list(test_dict.values()))


if __name__ == '__main__':
    pass