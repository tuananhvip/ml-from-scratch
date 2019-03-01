class KNN:
    """
    Characteristics of KNN algorithm:
        - Non-parametric ML algorithm. It means KNN doesn't have training phase to find optimal parameter.
        - Therefore, training phase and predicting phase in 1 phase make the algorithm very slow as the data larger
            and larger, also the dimension of vector is big.
        - ...

    Idea of KNN algorithm:
        - Given a specific dataset, for each row is a n-D dimension vector and the label.
        - Pre-processing dataset (normalization, standardization) into same scale (optional).
        - For any new point in predicting phase, the algorithm finds the distance between that point and all other
            points in training set (L_1, L_2, L_inf).
        - Base on K hyper-parameter, the algorithm will find K nearest neighbor and classify that point into
            which class.

    """
    def __init__(self, K, input_data):
        self.K = K

