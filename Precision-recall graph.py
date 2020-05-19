from sklearn.metrics import precision_recall_curve, average_precision_score
from tensorflow.keras.models import load_model
import tensorflow
import matplotlib.pyplot as plt
import numpy as np

num_classes = 3

x_test = np.load('predp_data_YCbCr_Histogram.npy')
x_test_1 = np.load('predicp_data_stnd_dev.npy')
x_test_2 = np.load('predicp_data_mean.npy')
y_test = np.load('predicp_label.npy')

x_test = x_test.astype('float32')
x_test /= 255

model = load_model('./saved_models/Back_Up/test_3(select).h5', custom_objects={'leaky_relu': tensorflow.nn.leaky_relu})
result = model.predict([x_test_1, x_test_2, x_test])

precision_ = dict()
recall_ = dict()
ap_ = dict()
for i in range(num_classes):
    precision_[i], recall_[i], _ = precision_recall_curve(y_test[:, i], result[:, i])
    ap_[i] = average_precision_score(y_test[:, i], result[:, i])

w, h = plt.figaspect(0.618)
plt.figure(figsize=(w, h))
plt.grid(True)
plt.xlabel('Recall')
plt.ylabel('Precision')

linestyles = ['-', '--', '-.']
for i in range(num_classes):
    plt.plot(precision_[i], recall_[i], linestyle=linestyles[i], label='Class '+str(i+1)+' (AP = %0.2F)' % ap_[i])

plt.legend(loc='lower left')
plt.show()