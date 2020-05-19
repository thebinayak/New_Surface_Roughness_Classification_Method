from __future__ import print_function
from tensorflow import keras
import tensorflow
import warnings
import os
import numpy as np
from tensorflow.keras.layers import Input, Dropout, Activation, Flatten
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Embedding, LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch

# Define training condition and flags
warnings.filterwarnings('ignore')
tensorflow.keras.backend.set_image_data_format('channels_first')
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    for k in range(len(gpus)):
        tensorflow.config.experimental.set_memory_growth(gpus[k], True)
        print('memory growth:', tensorflow.config.experimental.get_memory_growth(gpus[k]))
else:
    print("Not enough GPU hardware devices available")

# Currently, memory growth needs to be the same across GPUs
for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)

batch_size = 4
epochs = 20
num_classes = 3
max_features = 489
maxlen = 2
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'test_3.h5'
leaky_relu = tensorflow.nn.leaky_relu

# The data, split between train and test sets:
x_train, x_test, x_train_1, x_test_1, x_train_2, x_test_2, \
y_train, y_test = train_test_split(np.load('train_data_YCbCr_Histogram.npy'),
                                   np.load('train_data_stnd_dev.npy'),
                                   np.load('train_data_mean.npy'),
                                   np.load('train_label_YCbCr_Histogram.npy'),
                                   test_size=0.2, random_state=123)

print('y_test shape:', y_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

<<<<<<< .merge_file_a12928
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

def build_model(hp):
    num_layers = hp.Int('num_layers', 2, 8, default=6)-1

    # Define training condition and flags
    stnd_dev = Input(shape=(1,))
    mean = Input(shape=(1,))

    # Configuring Convolution Neural Network by functional API
    model1_input = Input(shape=x_train.shape[1:])
    filters = hp.Int('filters_0', 32, 256, step=32, default=64)
=======
# Define input shape
stnd_dev = Input(shape=(1,))
mean = Input(shape=(1,))

# Define training condition on LSTM layers
LSTM_layer = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
End_LSTM_Layer = LSTM(128, dropout=0.5, recurrent_dropout=0.5)

# Configuring Convolution Neural Network by functional API
model1_input = Input(shape=x_train.shape[1:])
>>>>>>> .merge_file_a22972

    model1 = Conv2D(filters, (3, 3), padding='same')(model1_input)
    model1 = Activation(leaky_relu)(model1)
    model1 = MaxPooling2D(pool_size=(2, 2))(model1)
    model1 = Dropout(0.25)(model1)

    for idx in range(1, num_layers):
        idx = str(idx+1)
        filters = hp.Int('filters_' + idx, 32, 256, step=32, default=64)

        model1 = Conv2D(filters, (3, 3), padding='same')(model1)
        model1 = Activation(leaky_relu)(model1)
        model1 = MaxPooling2D(pool_size=(2, 2))(model1)
        model1 = Dropout(0.25)(model1)

    filters = hp.Int('filters_' + str(num_layers), 32, 256, step=32, default=64)

    model1 = Conv2D(filters, (3, 3), padding='same')(model1)
    model1 = Activation(leaky_relu)(model1)
    model1 = MaxPooling2D(pool_size=(2, 2))(model1)
    model1 = Dropout(0.5)(model1)

<<<<<<< .merge_file_a12928
    model1 = Flatten()(model1)

    model2_input = keras.layers.concatenate([stnd_dev, mean])

    emb_len = hp.Int('Output_dim', 32, 512, step=32, default=128)
=======
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Configuring Long-Short Term Memory by functional API
embedded_s = Embedding(max_features, 128)(stnd_dev)
embedded_m = Embedding(max_features, 128)(mean)
encoded_s_1 = LSTM_layer(embedded_s)
encoded_m_1 = LSTM_layer(embedded_m)
encoded_s_2 = LSTM_layer(encoded_s_1)
encoded_m_2 = LSTM_layer(encoded_m_1)
encoded_s_3 = LSTM_layer(encoded_s_2)
encoded_m_3 = LSTM_layer(encoded_m_2)
encoded_s_4 = LSTM_layer(encoded_s_3)
encoded_m_4 = LSTM_layer(encoded_m_3)
encoded_s_5 = LSTM_layer(encoded_s_4)
encoded_m_5 = LSTM_layer(encoded_m_4)
encoded_s_6 = End_LSTM_Layer(encoded_s_5)
encoded_m_6 = End_LSTM_Layer(encoded_m_5)
>>>>>>> .merge_file_a22972

    model2 = Embedding(max_features, emb_len, input_length=maxlen)(model2_input)

<<<<<<< .merge_file_a12928
    num_layers_L = hp.Int('num_layers_L', 2, 8, default=6)
    for idx in range(num_layers_L):
        idx = str(idx)
        units = hp.Int('units_' + idx, 32, 512, step=32, default=64)
=======
# Concatenate CNN and LSTM
merged_vector = keras.layers.concatenate([encoded_s_6, encoded_m_6, model1])
>>>>>>> .merge_file_a22972

        # Configuring Feed-Forward Neural Network by functional API
        model2 = Bidirectional(LSTM(units, return_sequences=True))(model2)
        model2 = Dropout(0.2)(model2)

    units = hp.Int('units_' + str(num_layers_L), 32, 512, step=32, default=64)

    model2 = Bidirectional(LSTM(units))(model2)
    Dropout(0.5)

<<<<<<< .merge_file_a12928
    print('Build model...')

    # Concatenate CNN and FFNN
    merged_vector = keras.layers.concatenate([model2, model1])

    output = Dense(num_classes, activation='softmax')(merged_vector)

    model = Model(inputs=[stnd_dev, mean, model1_input], outputs=output)

    optimizer_name = hp.Choice(
        'optimizer', ['adam', 'rmsprop', 'sgd'], default='adam')
    optimizer = keras.optimizers.get(optimizer_name)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=3,
    directory='hyper_Model',
    project_name='sre_Model')

tuner.search_space_summary()

tuner.search([x_train_1, x_train_2, x_train], y_train, batch_size=batch_size, epochs=epochs,
             validation_data=([x_test_1, x_test_2, x_test], y_test), callbacks=[TensorBoard('./logs')])

models = tuner.get_best_models(num_models=1)[0]
=======
history = model.fit([x_train_1, x_train_2, x_train], y_train, batch_size, epochs,
                    validation_data=([x_test_1, x_test_2, x_test], y_test))
>>>>>>> .merge_file_a22972

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
models.save(model_path)
print('Saved trained model at %s ' % model_path)

<<<<<<< .merge_file_a12928
score, acc = models.evaluate([x_test_1, x_test_2, x_test], y_test,
=======
# Model evaluation
score, acc = model.evaluate([x_test_1, x_test_2, x_test], y_test,
>>>>>>> .merge_file_a22972
                            batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)

<<<<<<< .merge_file_a12928
tuner.results_summary()
=======
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
>>>>>>> .merge_file_a22972
