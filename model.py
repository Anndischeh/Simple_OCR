from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_models(input_shape):
    models = []

    # Model 1
    model1 = Sequential()
    model1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model1.add(MaxPooling2D((2, 2)))
    model1.add(Flatten())
    model1.add(Dense(128, activation='relu'))
    model1.add(Dense(62, activation='softmax'))
    models.append(model1)

    # Model 2
    model2 = Sequential()
    model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model2.add(Dropout(0.25)) 
    model2.add(MaxPooling2D((2, 2)))
    model2.add(Flatten())
    model2.add(Dense(128, activation='relu'))
    model2.add(Dropout(0.4))
    model2.add(Dense(62, activation='softmax'))
    models.append(model2)

    # Model 3
    model3 = Sequential()
    model3.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model3.add(Dropout(0.25)) 
    model3.add(MaxPooling2D((2, 2)))
    model3.add(Flatten())
    model3.add(Dense(256, activation='relu'))
    model3.add(Dropout(0.4))
    model3.add(Dense(128, activation='relu'))
    model3.add(Dense(62, activation='softmax'))
    models.append(model3)

    return models
