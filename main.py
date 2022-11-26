# This is a sample Python script.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# Importer les libraries:
from tools import *

if __name__ == '__main__':

    pca_tune(pipe, param_dict, X_train, y_train, X_test, y_test)
    knn_tune(pipe3, param_dict1, X_train, y_train, X_test, y_test)
    pipeline_logreg(X_train, y_train, X_test, y_test)
    pipeline_knn(X_train, y_train, X_test, y_test)
    evaluation_model(pipeline_logreg(X_train, y_train, X_test, y_test), X_train, y_train)
    evaluation_model(pipeline_knn(X_train, y_train, X_test, y_test), X_train, y_train)
