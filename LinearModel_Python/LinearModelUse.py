import ctypes
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":

    path_to_dll = "../LinearModelCppLib/cmake-build-debug" \
                  "/LinearModelCppLib.dll "

    my_lib = ctypes.CDLL(path_to_dll)

    my_lib.linear_create_model.argtypes = [ctypes.c_int]
    my_lib.linear_create_model.restype = ctypes.c_void_p

    my_lib.linear_dispose_model.argtypes = [ctypes.c_void_p]
    my_lib.linear_dispose_model.restype = None

    my_lib.linear_train_model_classification.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int
    ]
    my_lib.linear_train_model_classification.restype = None

    my_lib.linear_train_model_regression.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]
    my_lib.linear_train_model_regression.restype = None

    my_lib.linear_predict_model_classification.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int
    ]
    my_lib.linear_predict_model_classification.restype = ctypes.c_double

    my_lib.linear_predict_model_regression.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int
    ]
    my_lib.linear_predict_model_regression.restype = ctypes.c_double

    model = my_lib.linear_create_model(ctypes.c_int(2))
 #   model = my_lib.linear_create_model(ctypes.c_int(7164160))

    #CLASSIFICATION
    X = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ], dtype='float64')

    Y = np.array([
        1,
        -1,
        -1
    ], dtype='float64')

#REGRESSION
    K = np.array([
            [1, 1],
            [2, 2],
            [3, 1]
    ], dtype='float64')

    L = np.array([
        2,
        3,
        2.5
    ], dtype='float64')

    flattened_X = K.flatten()

    print("Before Training !!!")
    for inputs_k in K:
        print(my_lib.linear_predict_model_regression(
            model,
            inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(inputs_k)
        ))

    my_lib.linear_train_model_regression(
        model,
        flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        L.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        K.shape[0],
        K.shape[1],
        L.shape[0]
        #0.01,
        #1000
    )

    print("After Regression Training")
    for inputs_k in K:
        print(my_lib.linear_predict_model_regression(
            model,
            inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(inputs_k)
        ))

    my_lib.linear_dispose_model(model)

















    """
#DESSIN DE LA COURBE
    plt.scatter(X[0, 0], X[0, 1], color='blue')
    plt.scatter(X[1:3, 0], X[1:3, 1], color='red')
    plt.show()
    plt.clf()

    test_points = np.array([[i, j] for i in range(50) for j in range(50)], dtype='float64') / 50.0 * 2.0 + 1.0

    test_points_predicted = np.zeros(len(test_points))
    red_points = []
    blue_points = []
    for k, test_input_k in enumerate(test_points):
        predicted_value = my_lib.linear_predict_model_classification(
                model,
                test_input_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                len(test_input_k))
        if predicted_value == 1.0:
            blue_points.append(test_input_k)
        else:
            red_points.append(test_input_k)

    red_points = np.array(red_points)
    blue_points = np.array(blue_points)

    if len(red_points) > 0:
        plt.scatter(red_points[:, 0], red_points[:, 1], color='red', alpha=0.5, s=2)
    if len(blue_points) > 0:
        plt.scatter(blue_points[:, 0], blue_points[:, 1], color='blue', alpha=0.5, s=2)
    plt.scatter(X[0, 0], X[0, 1], color='blue', s=10)
    plt.scatter(X[1:3, 0], X[1:3, 1], color='red', s=10)
    plt.show()
    plt.clf()

    my_lib.linear_dispose_model(model)

"""
