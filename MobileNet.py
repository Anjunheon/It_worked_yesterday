import os
from PIL import Image
import glob
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import numpy as np
import cv2


poketmon = ["Squirtle", "Digda", "Dragonite", "Modafi", "Yoongella", "Leesang", "Snorlax", "Charmander", "Purin", "Pikachu"]
img_path = "D:\\Gongmo\\It_worked\\img\\"

label_list = os.listdir(img_path)
label = [i for i in range(10)]

x_train = []
x_label = []

y_verfi = []
y_label = []

z_test = []
z_label = []
for b in label:

    directory = img_path + str(b) + "\\" + poketmon[int(b)] + "_pp\\"
    files = glob.glob(directory + "*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        data = np.asarray(img) / 255
        if i <= 599:
            x_train.append(np.array(data).flatten())
            x_label.append(b)
        elif 599 < i and i <= 799:
            y_verfi.append(np.array(data).flatten())
            y_label.append(b)
        elif i < 900 and i > 799:
            z_test.append(np.array(data).flatten())
            z_label.append(b)
        else:
            break

# print(len(x_train))
x_train = np.array(x_train)
# print(len(x_label))
x_label = to_categorical(x_label)
# print(len(y_verfi))
y_verfi = np.array(y_verfi)
# print(len(y_label))
y_label = to_categorical(y_label)
# print(len(z_test))
z_test = np.array(z_test)
# print(len(z_label))
z_label = to_categorical(z_label)

model = Sequential()
model.add(Dense(256, activation='relu', input_dim = 400 * 400))
model.add(Dense(256, activation='relu'))
model.add(Dense(256))
model.add(Dense(10, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
model.fit(x_train, x_label, epochs=5, batch_size=32, validation_data=(y_verfi, y_label))

# 5. 모델 평가하기
loss_and_metrics = model.evaluate(z_test, z_label, batch_size=32)
print('')
print('loss_and_metrics : ' + str(loss_and_metrics))

# 6. 모델 사용하기
xhat_idx = np.random.choice(z_test.shape[0], 5)
xhat = z_test[xhat_idx]
yhat = model.predict_classes(xhat)

for i in range(5):
    print('True : ' + str(np.argmax(z_label[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))
