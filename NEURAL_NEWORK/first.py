# Simple Neural Network using TensorFlow/Keras

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# -----------------------------------
# Load Dataset
# -----------------------------------

data = load_iris()

X = data.data
y = data.target

# Convert output into categorical form
y = to_categorical(y)

# -----------------------------------
# Split Dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



model = Sequential()

model.add(Dense(16, activation='relu', input_shape=(4,)))

model.add(Dense(8, activation='relu'))

model.add(Dense(3, activation='softmax'))


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------
# Train Model
# -----------------------------------

model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8
)



loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)



predictions = model.predict(X_test)

print("\nPredicted Probabilities:")
print(predictions)