from fastapi import FastAPI
app = FastAPI()
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Define hyperparameters
batch_size = 32
num_epochs = 50
learning_rate = 0.001

# Define data generator
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                   rotation_range=20, width_shift_range=0.1,
                                   height_shift_range=0.1, zoom_range=0.1,
                                   horizontal_flip=True)

# Load data
train_data = train_datagen.flow_from_directory(directory='basedata/training/detected',
                                               target_size=(224, 224),
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               subset='training')

val_data = train_datagen.flow_from_directory(directory='basedata/training/original',
                                             target_size=(224, 224),
                                             batch_size=batch_size,
                                             class_mode='categorical',
                                             subset='validation')

# Define model architecture
model = tf.keras.models.Sequential([
    tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_data, epochs=num_epochs, validation_data=val_data)

# Evaluate model
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(directory='/path/to/test/folder',
                                             target_size=(224, 224),
                                             batch_size=batch_size,
                                             class_mode='categorical')

test_loss, test_acc = model.evaluate(test_data)

# Save model
model.save('bmi_detection_model.h5')
@app.get("/")
async def root():
    return {"message": "yes"}

