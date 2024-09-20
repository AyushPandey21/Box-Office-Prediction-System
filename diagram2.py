from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16

# Example model setup
model = Sequential()
model.add(VGG16(include_top=False, input_shape=(224, 224, 3)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(28, activation='softmax'))

# Display the model architecture
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
