from jinja2 import Template
from keras.models import load_model
import base64
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.python import keras
from PIL import Image, ImageChops, ImageEnhance
import os
from keras.models import Sequential
import itertools
import PIL.Image
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Input
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing import image
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
app = FastAPI()


# %%

import tensorflow as tf
from tensorflow.python import keras


def convert_to_ela_image(input_path,  quality):
    temp_filename = 'a.jpg'
    ela_filename = 'temp_ela.png'

    image = PIL.Image.open(input_path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = PIL.Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    ela_image.save('b.jpg')
    return ela_image

# %%


def build_model():

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(
        128, 128, 3), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

    model.add(Flatten())

    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))

    model.add(Dense(2, activation="softmax"))

    return model


# %%
model = build_model()
model.summary()


def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path,90).resize(image_size)).flatten() / 255.0


X=[]
Y=[]

import random
path='basedata/training/original'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png'):
            full_path = os.path.join(dirname,filename)
            X.append(prepare_image(full_path))
            Y.append(1)
            if len(Y) % 500 ==0:
                print(f'processing {len(Y)} images')

                
random.shuffle(X)
X = X[:2100]
Y = Y[:2100]
print(len(X), len(Y))   

path='basedata/training/detected'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png'):
            full_path = os.path.join(dirname,filename)
            X.append(prepare_image(full_path))
            Y.append(0)
            #if len(Y) % 500 ==0:
              # print(f'processing {len(Y)} images'
                
print(len(X), len(Y))  


X=np.array(X)
Y=to_categorical(Y,2)
X=X.reshape(-1,128,128,3)


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size =0.2, random_state=5)
X=X.reshape(-1,1,1,1)
print(len(X_train), len(Y_train))
print(len(X_val), len(Y_val))


# %%
epochs = 20
batch_size = 22

init_lr = 1e-4
optimizer = Adam(learning_rate = init_lr, decay = init_lr/epochs)


model.compile(optimizer = optimizer,
              loss= 'binary_crossentropy',
              metrics=['accuracy'])


early_stopping=EarlyStopping(monitor='val_accuracy',
                            min_delta =0,
                            patience = 2,
                            verbose=0,
                            mode='auto')


hist = model.fit(X_train,
                Y_train,
                batch_size= batch_size,
                epochs = epochs,
                validation_data = (X_val,Y_val),
                callbacks= [early_stopping])



model.save('aa.h5')# %%
# model = load_model('model_casia_run1.h5')


# %%
image_size = (128, 128)

# %%


def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0


# %%
class_names = ['detected', 'original']

# %%
real_image_path = 'basedata/training/detected/images.jpeg'

image = prepare_image(real_image_path)

image = image.reshape(-1, 128, 128, 3)

y_pred = model.predict(image)

y_pred_class = np.argmax(y_pred, axis=1)[0]

print(
    f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) *100:0.2f}')

# %%6
real_image_path = 'basedata/training/detected/images.jpeg'

image = prepare_image(real_image_path)

image = image.reshape(-1, 128, 128, 3)

y_pred = model.predict(image)
print(y_pred)

y_pred_class = np.argmax(y_pred, axis=1)[0]

confidence = np.amax(y_pred) * 100

print(
    f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) *100:0.2f}')


@app.get("/")
async def root():
    return {"message": "Hello World"}


class Item(BaseModel):
    name: str
    price: float


@app.get("/home")
async def read_index():
    return FileResponse("index.html")


@app.post("/items/")
async def create_item(item: Item):
    """
    Create an item with the provided name and price.
    """
    return {"item": item, "Confidence": confidence}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...),):
    """
    Upload a file and return the file path.
    """
    # Save the file to disk
    file_path = f"{file.filename}"
    base_path = os.getcwd()
    localised_path = f"{base_path}/{file_path}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    app.mount("/", StaticFiles(directory="/"), name="images")
    imagea = prepare_image(localised_path)

    image1 = imagea.reshape(-1, 128, 128, 3)

    y_pred1 = model.predict(image1)
    y_pred_class1 =np.argmax(y_pred1, axis= 1)[0]
    y_pred_class2 = class_names[y_pred_class1]
    print(base_path)
    confidence = f'{np.amax(y_pred1) *100:0.2f}'
    print(confidence)
    # forged = 100 - confidence
    localised = convert_to_ela_image(localised_path, 90)


    template = Template(open("./uploded.html").read())
    rendered_template = template.render(
        confidence=confidence, file_name=localised_path, localised=f'{base_path}/b.jpg', classmame=y_pred_class2)
    # Return the file path and caption
    # return HTMLResponse(rendered_template, media_type="text/html")
    return {"item": True , "Confidence": confidence}
