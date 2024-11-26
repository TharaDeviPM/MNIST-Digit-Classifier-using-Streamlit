import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import to_categorical


# Title of the app
st.title("MNIST Digit Classifier")
st.write("This app trains a neural network to classify handwritten digits from the MNIST dataset.")

# Load and preprocess the MNIST dataset
@st.cache_resource
def load_and_preprocess_data():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train / 255.0  # Normalize the training data
    X_test = X_test / 255.0    # Normalize the test data
    Y_train = to_categorical(Y_train, num_classes=10)
    Y_test = to_categorical(Y_test, num_classes=10)
    return (X_train, Y_train), (X_test, Y_test)

(X_train, Y_train), (X_test, Y_test) = load_and_preprocess_data()

# Build the model
@st.cache_resource
def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()
st.success("Model built successfully!")

# Train the model
if st.button("Train Model"):
    with st.spinner("Training the model... Please wait."):
        history = model.fit(X_train, Y_train, epochs=5, batch_size=32, verbose=1)
        st.success("Model training completed!")
        st.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.2f}")

# Test the model with a sample image
st.subheader("Test the Model")
index = st.slider("Select an image index from the test set:", 0, len(X_test) - 1, 0)
test_image = X_test[index]
true_label = np.argmax(Y_test[index])

# Display the test image
st.write("Selected Test Image:")
fig, ax = plt.subplots()
ax.imshow(test_image, cmap="gray")
ax.axis("off")
st.pyplot(fig)

# Predict the digit
if st.button("Predict"):
    prediction = model.predict(test_image.reshape(1, 28, 28))
    predicted_label = np.argmax(prediction)
    st.write(f"True Label: {true_label}")
    st.write(f"Predicted Label: {predicted_label}")
    st.write(f"Confidence Scores: {prediction.flatten()}")



