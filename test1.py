import unittest
from tools import *


class MyTestCase(unittest.TestCase):

    def test_values_defaults(self):
        self.assertEqual(oversampling_undersampling(X_train, y_train, over=False)[3],
                         oversampling_undersampling(X_train, y_train, over=False)[4])

    def test_pca_tune(self):
        self.assertEqual(pca_tune(pipe, param_dict, X_train, y_train, X_test, y_test), {'reducer__n_components': 12})

    def test_knn_tune(self):
        self.assertEqual(knn_tune(pipe3, param_dict1, X_train, y_train, X_test, y_test), {'knn__n_neighbors': 11})


if __name__ == '__main__':
    unittest.main()
