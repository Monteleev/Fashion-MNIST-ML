import collections
import time
from sklearn import metrics

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print("Shape of x_train: {}".format(x_train.shape))
print("Shape of y_train: {}".format(y_train.shape))
print()
print("Shape of x_test: {}".format(x_test.shape))
print("Shape of y_test: {}".format(y_test.shape))

labelNames = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print(x_train.shape)
print(x_test.shape)

# SVM Model
start1 = time.time()

svc = SVC(C=1, kernel='sigmoid', gamma="auto")
svc.fit(x_train, y_train)

end1 = time.time()
svm_time = end1-start1

# KNN Model
start2 = time.time()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)

end2 = time.time()
knn_time = end2-start2

# NN Model
start3 = time.time()

nn = MLPClassifier(hidden_layer_sizes=(500,), max_iter=10, alpha=1e-4,solver='sgd', verbose=10, random_state=1,learning_rate_init=.1)
nn.fit(x_train, y_train)
y_pred_nn = nn.predict(x_test)

end3 = time.time()
nn_time = end3-start3

print("SVM Time: {:0.2f} minute".format(svm_time/60.0))
print("KNN Time: {:0.2f} minute".format(knn_time/60.0))
print("NN Time: {:0.2f} minute".format(nn_time/60.0))



# SVM report and analysis
y_pred_svc = svc.predict(x_test)
svc_f1 = metrics.f1_score(y_test, y_pred_svc, average= "weighted")
svc_accuracy = metrics.accuracy_score(y_test, y_pred_svc)
print("-----------------SVM Report---------------")
print("F1 score: {}".format(svc_f1))
print("Accuracy score: {}".format(svc_accuracy))

# KNN report and analysis
knn_f1 = metrics.f1_score(y_test, y_pred_knn, average= "weighted")
knn_accuracy = metrics.accuracy_score(y_test, y_pred_knn)
print("-----------------K-nearest neighbors Report---------------")
print("F1 score: {}".format(knn_f1))
print("Accuracy score: {}".format(knn_accuracy))

# NN report and analysis
nn_f1 = metrics.f1_score(y_test, y_pred_knn, average= "weighted")
nn_accuracy = metrics.accuracy_score(y_test, y_pred_nn)
print("-----------------Neural Networks Report---------------")
print("F1 score: {}".format(nn_f1))
print("Accuracy score: {}".format(nn_accuracy))

