from autoformer.model import Autoformer
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam               
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,  PowerTransformer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def create_time_windows(data, target, window_size, horizon=1):
    X_windows = []
    y_windows = []
    for i in range(len(data) - window_size - horizon + 1):
        X_windows.append(data[i:i+window_size])  # این باید یک sequence باشد
        y_windows.append(target[i+window_size:i+window_size+horizon])
    return np.array(X_windows), np.array(y_windows)

df= pd.read_csv('data/Tetuan City power consumption.csv')
# split data
X= df.drop(columns= ['Zone 3  Power Consumption'])
y= df[['Zone 3  Power Consumption']]

# create timestamp
X['DateTime'] = pd.to_datetime(X['DateTime'])
X['timestamp'] = X['DateTime'].view('int')
X = X.set_index(['DateTime'])

# normalizing data
pt1 = PowerTransformer(method='yeo-johnson')
pt2 = PowerTransformer(method='yeo-johnson')
X['normed_general diffuse flows'] = pt1.fit_transform(X[['general diffuse flows']])
X['normed_diffuse flows'] = pt2.fit_transform(X[['diffuse flows']])
X['normed_humidity'] = np.clip(X['Humidity'], a_min=40, a_max=90)
X = X.drop(['Humidity', 'general diffuse flows', 'diffuse flows'], axis=1)

scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

window_size = 144  # last 24 hour
horizon = 144  # forecast next hour

X_windows, y_windows = create_time_windows(X_scaled, y_scaled, window_size, horizon)

if len(y_windows.shape) == 1:
    y_windows = y_windows.reshape(-1, 1)
    
x_train, x_test, y_train, y_test= train_test_split(
    X_windows, y_windows, train_size=0.8, shuffle=False)

x_valid, x_test, y_valid, y_test= train_test_split(
    x_test, y_test, train_size=0.5, shuffle=False)

input_shape = (window_size, X.shape[1])
inputs= layers.Input(shape=input_shape)
x= Autoformer(d_out= 1,
            d_model= 32,
            n_heads= 2,
            conv_filter= 32,
            num_decoder=1,
            num_encoder=1)(inputs)

'''x= layers.Dense(horizon)(x)
x= x[:, -1, :]
outputs= layers.Reshape((horizon, 1))(x)'''

model= Model(inputs= inputs, outputs= x)      
model.compile(optimizer=Adam(learning_rate= 0.001, clipnorm= 0.1), 
              loss='mse',
              metrics= ['RootMeanSquaredError'])    
model.summary()  

callback= EarlyStopping(monitor='val_loss',
                        patience=40,
                        restore_best_weights= True)
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_valid, y_valid),
    shuffle=False,
    epochs=1000,
    batch_size=32,
    callbacks= callback
)

score = model.evaluate(x_test, y_test)

y_pred= model.predict(x_test)

y_pred_rescaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
y_true_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

plt.figure(figsize=(12,6))
plt.plot(y_true_rescaled[0], label="Actual", marker='o')
plt.plot(y_pred_rescaled[0], label="Predicted", marker='x')
plt.title("Test Sequence Prediction vs Actual")
plt.xlabel("Time step")
plt.ylabel("Target value")
plt.legend()
plt.show()

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10,6))
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()






























































































