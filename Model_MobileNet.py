from os import times_result

import keras
from PIL import Image
import glob
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, Dropout, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.applications import MobileNet
import numpy as np

pokemon = ["Squirtle", "Digda", "Dragonite", "Modafi", "Yoongella", "Leesang", "Snorlax", "Charmander", "Purin", "Pikachu"]
img_path = "D:\\It_worked_yesterday\\img_100x100\\"

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
    directory = img_path + str(b) + "\\" + pokemon[int(b)] + "_pp\\"
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
  # keras.callbacks.EarlyStopping(
  #   monitor='val_accuracy', # 모델의 검증 정확도 모니터링
  #   patience=25, # 25 에포크보다 더 길게 향상되지 않으면 중단
  # ),
  keras.callbacks.ModelCheckpoint(
    filepath='D:\\It_worked_yesterday\\venv\\models\\check\\', # 저장 경로
    monitor='val_loss',
    save_best_only=True, # 가장 좋은 모델
  )
]

# weights='imagenet' : imagenet으로 pretrain된 가중치를 사용
# include_top=False : 네트워크 최상단에 Fully-Connected Layer가 들어가지 않음
# input_shape : MobileNet은 224x224 이미지에서 작동되도록 설계됨
MobileNet = MobileNet(weights='imagenet', include_top=False, input_shape=(width, height, 3))

# 학습을 진행하면서 가중치 조정이 가능하도록 설정(기존의 물체, 생물이 아닌 포켓몬을 분류해야하기 때문)
for layer in MobileNet.layers:
    layer.trainable = True

for (i, layer) in enumerate(MobileNet.layers):
    print(str(i), layer.__class__.__name__, layer.trainable)

# 마지막 분류층(Classification head) 정의
def addTopModelMobileNet(bottom_model, model_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)

    top_model = Dense(1024,activation='relu')(top_model)

    top_model = Dense(512, activation='relu')(top_model)

    top_model = Dense(len(pokemon), activation='softmax')(top_model)

    return top_model

# 모바일넷과 분류층 결합하여 모델 객체 생성
FC_Head = addTopModelMobileNet(MobileNet, len(pokemon))

model = Model(inputs = MobileNet.input, outputs = FC_Head)

# 모델 구조 출력
print(model.summary())

# 모델 학습과정 설정
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

# 모델 학습
model.fit(train_data, train_label, epochs=25, batch_size=32, validation_data=(verfi_data, verfi_label), verbose=1, shuffle=True,
          callbacks=callback_list)

# 모델 평가
loss_and_metrics = model.evaluate(test_data, test_label, verbose=0)
print('')
print('loss_and_metrics : ' + str(loss_and_metrics))

# 모델 테스트
xhat_idx = np.random.choice(test_data.shape[0], 100)
xhat = test_data[xhat_idx]
yhat = model.predict_classes(xhat)

cnt = 0
for i in range(100):
    if np.argmax(test_label[xhat_idx[i]]) == yhat[i]:
        cnt += 1
    print('True : ' + str(np.argmax(test_label[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))

print(str(cnt) + ' / ' + str(100))