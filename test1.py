import unittest
from tools2 import *


class MyTestCase(unittest.TestCase):
    def test_preprocess(self):
        self.assertNotEqual(preprocessing(loan)[["UseOfLoan", "MaritalStatus", "EmploymentStatus",
                                                "OccupationArea", "HomeOwnershipType"]].values.any(), -1)
    def test_values_defaults(self):
        self.assertEqual(oversampling_undersampling(X_train, y_train, over=False)[3],
                         oversampling_undersampling(X_train, y_train, over=False)[4])

    def test_pca_tune(self):
        self.assertEqual(pca_tune(pipe, param_dict, X_train, y_train, X_test, y_test), {'reducer__n_components': 16})

    def test_knn_tune(self):
        self.assertEqual(knn_tune(pipe3, param_dict1, X_train, y_train, X_test, y_test), {'knn__n_neighbors': 19})


if __name__ == '__main__':
    unittest.main()