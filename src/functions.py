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



def cross_val_linear(X_train, y_train, k):    
    kf = KFold(k)
    kf.get_n_splits(X_train)

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



def residual_plot(ax, x, y, y_hat, n_bins=100):
    residuals = y - y_hat
    ax.axhline(0, color="black", linestyle="--")
    ax.scatter(x, residuals, color="grey", alpha=0.5)
    ax.set_ylabel("Residuals ($y - \hat y$)")



def linear_regression_model(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    return y_hat




def model_summary(y_train, X_train):
    model_info = sm.OLS(np.array(y_train, dtype=float), np.array(X_train, dtype=float))
    return model_info.fit().summary()




def cross_val_ridge(X_train, y_train, k, alpha=0.5, model=Ridge):    

    kf = KFold(k)
    kf.get_n_splits(X_train)

    train_dict = {}
    test_dict = {}

    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        
        X_train_kfold = X_train.iloc[train_index]
        y_train_kfold = y_train.iloc[train_index]
        X_test_kfold = X_train.iloc[test_index]
        y_test_kfold = y_train.iloc[test_index]

        reg = model(alpha=alpha)
        reg.fit(X_train_kfold, y_train_kfold)

        # Call predict to get the predicted values for training and test set
        train_predicted = reg.predict(X_train_kfold)
        test_predicted = reg.predict(X_test_kfold)

        # Calculate RMSE for training and test set
        train_dict[i] = rmse(y_train_kfold, train_predicted)
        test_dict[i] = rmse(y_test_kfold, test_predicted)

    return np.mean(list(train_dict.values())), np.mean(list(test_dict.values()))




def normalize_data(data, target):

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    X_train_norm = pd.DataFrame([(lambda x: (x-X_train[i].min())/(X_train[i].max()-X_train[i].min()))(X_train[i]) for i in X_train.columns]).T
    y_train_norm = pd.DataFrame((lambda x: (x-y_train.min())/(y_train.max()-y_train.min()))(y_train))
    X_test_norm = pd.DataFrame([(lambda x: (x-X_test[i].min())/(X_test[i].max()-X_test[i].min()))(X_test[i]) for i in X_test.columns]).T
    y_test_norm = pd.DataFrame((lambda x: (x-y_test.min())/(y_test.max()-y_test.min()))(y_test))

    return X_train_norm, y_train_norm, X_test_norm, y_test_norm




def train_at_various_alphas(X, y, model, alphas, n_folds=10, **kwargs):
    cv_errors_train = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                     columns=alphas)
    cv_errors_test = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                        columns=alphas)
    for j in range(n_folds):   
        for i in alphas:
            w, v = cross_val_ridge(X, y, k=n_folds, alpha=i, model=model)
            cv_errors_train.at[j, i] = w
            cv_errors_test.at[j, i] = v
    return cv_errors_train.iloc[0], cv_errors_test.iloc[0]




if __name__ == '__main__':
    pass