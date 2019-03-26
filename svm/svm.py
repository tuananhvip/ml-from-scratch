from utils import load_mat_file, Graph, load_vocabulary, process_email, email_feature
from sklearn.svm import SVC


def linear_kernel():
    X, y = load_mat_file('ex6data1.mat')
    plot = Graph(X, y, 'on')
    plot.plot_data()

    C = 1.0
    linear_svm = SVC(C=C, kernel='linear')

    linear_svm.fit(X, y)

    W = linear_svm.coef_
    b = linear_svm.intercept_
    plot.visualize_boundary_linear(W, b)


def rbf_kernel():
    X, y = load_mat_file('ex6data2.mat')
    plot = Graph(X, y, 'on')
    plot.plot_data()

    C = 1.0
    rbf_svm = SVC(C=C, kernel='rbf')

    rbf_svm.fit(X, y)


def spam_classification():
    vocabs = load_vocabulary('vocab.txt')
    X, y = load_mat_file('spamTrain.mat')
    y = y.reshape((-1, ))
    C = 0.1
    svm = SVC(C=C, kernel='linear')
    svm.fit(X, y)

    pred_train = svm.predict(X)
    print("Training accuracy:", len(pred_train[pred_train == y])/len(pred_train))

    X_test, y_test = load_mat_file('spamTest.mat')
    y_test = y_test.reshape((-1, ))

    pred_test = svm.predict(X_test)
    print("Testing accuracy:", len(pred_test[pred_test == y_test]) / len(pred_test))

    # Try with emailSample1.txt
    with open('emailSample1.txt') as f:
        sample_1 = f.read()
    word_indices = process_email(sample_1, vocabs)
    x = email_feature(word_indices, vocabs)
    x = x.reshape((-1, 1)).T
    is_spam = svm.predict(x)
    print("Spam" if is_spam[0] == 1 else "No spam")

spam_classification()