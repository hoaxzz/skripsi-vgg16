import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'C:/Users/akip/Desktop/vgg16/Citrus Leaf Disease Image/Black spot',
    target_size=(240, 240),
    batch_size=3,
    class_mode='binary'  # Sesuaikan dengan tipe masalah (binary, categorical, atau lainnya)
)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(240, 240, 3)))
model.add(MaxPooling2D((2, 2)))
# Tambahkan layer-layer lain sesuai kebutuhan

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator, epochs=50)

validation_generator = train_datagen.flow_from_directory(
    'C:/Users/akip/Desktop/vgg16/valid',
    target_size=(240, 240),
    batch_size=3,
    class_mode='binary'
)

model.evaluate(validation_generator)
