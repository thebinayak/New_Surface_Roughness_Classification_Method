from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
import tensorflow
import matplotlib as plt
import numpy as np

x_test = np.load('predp_data_YCbCr_Histogram.npy')
x_test_1 = np.load('predicp_data_stnd_dev.npy')
x_test_2 = np.load('predicp_data_mean.npy')
y_test = np.load('predicp_label.npy')

x_test = x_test.astype('float32')
x_test /= 255

fpr = [None] * 3
tpr = [None] * 3
thr = [None] * 3

target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))

model = load_model('./saved_models/Back_Up/test_3(select).h5', custom_objects={'leaky_relu': tensorflow.nn.leaky_relu})

for i in range(3):
    result = model.predict([x_test_1, x_test_2, x_test])
    GaussianNB().fit(X, y[:, i])
    fpr[i], tpr[i], thr[i] = roc_curve(y[:, i], model.predict_proba(X)[:, 1])
    plt.plot(fpr[i], tpr[i])

plt.xlabel('위양성률(Fall-Out)')
plt.ylabel('재현률(Recall)')
plt.show()