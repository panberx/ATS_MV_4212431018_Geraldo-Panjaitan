import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn import datasets
from mlxtend.data import loadlocal_mnist
from sklearn.model_selection import LeaveOneOut as loocv
from sklearn.preprocessing import LabelBinarizer as lb
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, classification_report


#Load EMNIST Dataset
train_images, train_labels = loadlocal_mnist(images_path='./images/emnist-mnist-train-images-idx3-ubyte',
                                             labels_path='./images/emnist-mnist-train-labels-idx1-ubyte')
test_images, test_labels = loadlocal_mnist(images_path='./images/emnist-mnist-test-images-idx3-ubyte',
                                             labels_path='./images/emnist-mnist-test-labels-idx1-ubyte')

#Test HOG Extraction
train_feature, hog_img = hog(train_images[0].reshape(28,28), orientations = 9, pixels_per_cell=(8,8), visualize = True, cells_per_block=(2,2), block_norm='L2')
test_feature, hog_img = hog(test_images[0].reshape(28,28), orientations = 9, pixels_per_cell=(8,8), visualize = True, cells_per_block=(2,2), block_norm='L2')

#Preprocessing
train_n_dims = train_feature.shape[0]
train_n_samples = train_images.shape[0]
test_n_dims = test_feature.shape[0]
test_n_samples = test_images.shape[0]
x_train, y_train = datasets.make_classification(n_samples=train_n_samples, n_features=train_n_dims)
x_test, y_test = datasets.make_classification(n_samples=test_n_samples, n_features=test_n_dims)

for i in range (train_n_samples):
    x_train[i],_ = hog(train_images[0].reshape(28,28), orientations = 9, pixels_per_cell=(8,8), visualize = True, cells_per_block=(2,2), block_norm='L2')
    y_train[i] = train_labels[i]

for i in range (test_n_samples):
    x_test[i],_ = hog(test_images[0].reshape(28,28), orientations = 9, pixels_per_cell=(8,8), visualize = True, cells_per_block=(2,2), block_norm='L2')
    y_test[i] = test_labels[i]


#Classification with SVM
svm_model = SVC(kernel = 'rbf', C=1, gamma='scale', random_state = 42)
svm_model.fit(x_train, y_train)

#Dataset Prediction
y_true, y_pred = [], []
for train_index, test_index in loocv.split(x_train):
    X_train_fold, X_test_fold = x_train[train_index], x_train[test_index]
    y_train_fold, y_test_fold = train_labels[train_index], train_labels[test_index]
    svm_model.fit(X_train_fold, y_train_fold)
    y_pred_test = svm_model.predict(X_test_fold)
    y_true.append(y_test_fold[0])
    y_pred.append(y_pred_test[0])

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
conf_matrix = confusion_matrix(y_true, y_pred)

print("LOOCV Accuracy (Train):", accuracy)
print("LOOCV Precision (Train):", precision)
print("LOOCV F1-Score (Train):", f1)
print("LOOCV Confusion Matrix (Train):\n", conf_matrix)

svm_model.fit(x_train, train_labels)
y_test_pred = svm_model.predict(x_test)
test_accuracy = accuracy_score(test_labels, y_test_pred)
test_precision = precision_score(test_labels, y_test_pred, average = 'macro')
test_f1 = f1_score(test_labels, y_test_pred, average = 'macro')
test_conf_matrix = confusion_matrix(test_labels, y_test_pred)

print("\nTest Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test F1-Score:", test_f1)
print("Test Confusion Matrix:\n", test_conf_matrix)