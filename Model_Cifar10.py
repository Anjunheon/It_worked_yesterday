from os import times_result

import keras
from PIL import Image
import glob
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.utils import to_categorical
import numpy as np

poketmon = ["Squirtle", "Digda", "Dragonite", "Modafi", "Yoongella", "Leesang", "Snorlax", "Charmander", "Purin", "Pikachu"]
img_path = "D:\\Projects\\It_worked_yesterday\\img\\"

label = [i for i in range(10)]

train_data = []
train_label = []

verfi_data = []
verfi_label = []

test_data = []
test_label = []

width = 100
height = 100

# 학습, 검증, 테스트 데이터 개수 조절
train_num = 1800
verfi_num = 900
test_num = 300
# --------------------------------- #

verfi_num += train_num
test_num += verfi_num

for b in label:
    directory = img_path + str(b) + "\\" + poketmon[int(b)] + "_pp\\"
    files = glob.glob(directory + "*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        data = np.asarray(img) / 255
        # rgb_data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        if i < train_num:
            train_data.append(np.array(data))
            train_label.append(b)
        elif train_num <= i and i < verfi_num:
            verfi_data.append(np.array(data))
            verfi_label.append(b)
        elif i >= verfi_num and i < test_num :
            test_data.append(np.array(data))
            test_label.append(b)
        else:
            break
files = 0 # files에 trash 값 남아있음

# data -> numpy array, label -> one hot encoding
train_num = 1800; verfi_num = 900; test_num = 300

train_data = np.array(train_data)
train_data = train_data.reshape((train_num * 10, width, height, 1))
train_label = to_categorical(train_label)

verfi_data = np.array(verfi_data)
verfi_data = verfi_data.reshape((verfi_num * 10, width, height, 1))
verfi_label = to_categorical(verfi_label)

test_data = np.array(test_data)
test_data = test_data.reshape((test_num * 10, width, height, 1))
test_label = to_categorical(test_label)
# ----------------------------------------- #

# ------------------------------------------#

# fit() 메서드의 callbacks 매개변수를 사용하여 원하는 개수만큼 콜백을 모델로 전달
callback_list = [
  keras.callbacks.EarlyStopping(
    monitor='val_acc', # 모델의 검증 정확도 모니터링
    patience=25, # 1 에포크보다 더 길게 향상되지 않으면 중단
  ),
  keras.callbacks.ModelCheckpoint(
    filepath='my_model.h5', # 저장
    monitor='val_loss',
    save_best_only=True, # 가장 좋은 모델
  )
]

# 모델 구성
model = Sequential()

#-- layer 1
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(width, height, 1)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))

#-- layer 2
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))

#-- layer 3
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))

#-- layer 4
model.add(Flatten())
model.add(Dense(512, activation='relu'))

# -- layer 5
model.add(Dense(512, activation='relu'))

# -- layer 6
model.add(Dense(10, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# 4. 모델 학습시키기
model.fit(train_data, train_label, epochs=500, batch_size=32, validation_data=(verfi_data, verfi_label), verbose=1, shuffle=True,
          callbacks=callback_list)

# 5. 모델 평가하기
loss_and_metrics = model.evaluate(test_data, test_label, verbose=0)
print('')
print('loss_and_metrics : ' + str(loss_and_metrics))

# 6. 모델 사용하기
xhat_idx = np.random.choice(test_data.shape[0], 5)
xhat = test_data[xhat_idx]
yhat = model.predict_classes(xhat)

for i in range(5):
    print('True : ' + str(np.argmax(test_label[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))