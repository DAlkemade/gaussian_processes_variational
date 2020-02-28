from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


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
