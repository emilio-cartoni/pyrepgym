import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.loadtxt("data_filtered")
X = data[:, [7, 8, 9]]
Y = data[:, [0, 1, 3, 5]]/np.pi

# %%
model = keras.Sequential()
opt = keras.optimizers.Adam(learning_rate=0.0005)

model.add(keras.layers.Input(shape=(3,)))
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dense(1000, activation="relu"))
model.add(keras.layers.Dense(4, activation="tanh"))

# %%
model.compile(optimizer=opt, loss="mse", metrics=['accuracy'])
model.summary()

# %%
hist = model.fit(X, Y, batch_size=40, verbose=2, epochs=50)
np.save("weights", model.get_weights())
for w in model.get_weights():
    print(w.shape)
plt.plot(hist.history["loss"])
plt.savefig("/tmp/loss.png")
