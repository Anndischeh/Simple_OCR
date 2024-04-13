import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from preprocessing import preprocessing
from model import create_models

# Read the training and testing datasets
ds_train = pd.read_csv("/content/emnist/emnist-balanced-train.csv", header=None)
ds_test = pd.read_csv("/content/emnist/emnist-balanced-test.csv", header=None)

# Check if 'label' column exists before dropping it
if 0 in ds_train.columns:
    ds_train.columns = ['label'] + list(range(1, len(ds_train.columns)))
if 0 in ds_test.columns:
    ds_test.columns = ['label'] + list(range(1, len(ds_test.columns)))

# Reset index
ds_train = ds_train.reset_index(drop=True)
ds_test = ds_test.reset_index(drop=True)

# Extract features and labels
x_train = ds_train.drop(['label'], axis=1)
y_train = ds_train['label']
x_test = ds_test.drop(['label'], axis=1)
y_test = ds_test['label']

alpha_num_to_char = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j',
    20: 'k', 21: 'l', 22: 'm', 23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't',
    30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z',
    36: 'A', 37: 'B', 38: 'C', 39: 'D', 40: 'E', 41: 'F', 42: 'G', 43: 'H', 44: 'I', 45: 'J',
    46: 'K', 47: 'L', 48: 'M', 49: 'N', 50: 'O', 51: 'P', 52: 'Q', 53: 'R', 54: 'S', 55: 'T',
    56: 'U', 57: 'V', 58: 'W', 59: 'X', 60: 'Y', 61: 'Z'
}

def show_example(image, true_val, predicted_val):
    plt.title(f"Actual: {true_val}, Predicted: {predicted_val}")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

# Function to generate a 2x3 table of examples with predicted and real labels
def display_examples(model, x_data, y_data):
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))

    for i, ax in enumerate(axes.flat):
        # Predict
        example_index = np.random.randint(len(x_data))
        example_image = x_data[example_index].reshape(28, 28)
        true_val = alpha_num_to_char[y_data.iloc[example_index]]
        predicted_val = alpha_num_to_char[np.argmax(model.predict(x_data[example_index].reshape(1, 28, 28, 1)))]

        # Display example
        ax.imshow(example_image, cmap='gray')
        ax.set_title(f"Actual: {true_val}, Predicted: {predicted_val}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Define input shape
input_shape = (28, 28, 1)

# Create models
models = create_models(input_shape)

# Compile and train models
x_train_processed, y_train_processed, x_test_processed, y_test_processed = preprocessing(ds_train, ds_test, x_train, y_train, x_test, y_test)

for i, model in enumerate(models):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_processed, y_train_processed, epochs=5, batch_size=32)

# Evaluate models
accuracies = []
for i, model in enumerate(models):
    loss, accuracy = model.evaluate(x_test_processed, y_test_processed)
    accuracies.append(accuracy)
    print(f"Model {i+1} Accuracy: {accuracy}")

# Select best model
best_model_index = accuracies.index(max(accuracies))
best_model = models[best_model_index]
print(f"Best Model: Model {best_model_index+1}")

# Display examples
display_examples(best_model, x_train_processed, y_train)