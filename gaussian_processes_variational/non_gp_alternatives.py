from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import linear_model


def fit_svm(X, y, plot=True):
    """Fit an SVM."""
    clf = SVR(C=1.0, epsilon=0.2)
    clf.fit(X, y.flatten())
    z = clf.predict(X)
    if plot:
        plt.plot(X, z)
        plt.scatter(X, y)
        plt.title("SVM")
        plt.show()
    return clf


def linear_regression(X, y, plot=True):
    """Fit a linear regression model."""
    clf = LinearRegression().fit(X, y)
    z = clf.predict(X)
    if plot:
        plt.plot(X, z)
        plt.scatter(X, y)
        plt.title("Linear Regression")
        plt.show()


def bayesian_ridge_regression(X_train, y_train, X_test):
    clf = linear_model.BayesianRidge()
    clf.fit(X_train, y_train)
    z, std = clf.predict(X_test, return_std=True)
    return z, std