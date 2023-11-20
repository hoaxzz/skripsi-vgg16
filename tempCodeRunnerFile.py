validation_generator = train_datagen.flow_from_directory(
    'C:/Users/akip/Desktop/vgg16/valid',
    target_size=(240, 240),
    batch_size=32,
    class_mode='binary'
)

model.evaluate(validation_generator)
