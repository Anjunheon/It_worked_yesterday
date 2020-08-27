from PIL import Image
import glob
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import numpy as np

poketmon = ["Squirtle", "Digda", "Dragonite", "Modafi", "Yoongella", "Leesang", "Snorlax", "Charmander", "Purin", "Pikachu"]
img_path = "D:\\Gongmo\\It_worked\\img\\"

label = [i for i in range(10)]

train_data = []
train_label = []

verfi_data = []
verfi_label = []

test_data = []
test_label = []

# 학습, 검증, 테스트 데이터 개수 조절
train_num = 600
verfi_num = 200
test_num = 100
# --------------------------------- #

verfi_num += train_num
test_num += verfi_num

for b in label:
    directory = img_path + str(b) + "\\" + poketmon[int(b)] + "_pp\\"
    files = glob.glob(directory + "*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        data = np.asarray(img) / 255
        if i <= train_num:
            train_data.append(np.array(data).flatten())
            train_label.append(b)
        elif train_num < i and i < verfi_num:
            verfi_data.append(np.array(data).flatten())
            verfi_label.append(b)
        elif i >= verfi_num and i < test_num :
            test_data.append(np.array(data).flatten())
            test_label.append(b)
        else:
            break
files = 0 # files에 trash 값 남아있음

# data -> numpy array, label -> one hot encoding
train_data = np.array(train_data)
train_label = to_categorical(train_label)

verfi_data = np.array(verfi_data)
verfi_label = to_categorical(verfi_label)

test_data = np.array(test_data)
test_label = to_categorical(test_label)
# ----------------------------------------- #

# 모델 구성
model = Sequential()
model.add(Dense(256, activation='relu', input_dim = 400 * 400))
model.add(Dense(256, activation='relu'))
model.add(Dense(256))
model.add(Dense(10, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
model.fit(train_data, train_label, epochs=5, batch_size=32, validation_data=(verfi_data, verfi_label))

# 5. 모델 평가하기
loss_and_metrics = model.evaluate(test_data, test_label, batch_size=32)
print('')
print('loss_and_metrics : ' + str(loss_and_metrics))

# 6. 모델 사용하기
xhat_idx = np.random.choice(test_data.shape[0], 5)
xhat = test_data[xhat_idx]
yhat = model.predict_classes(xhat)

for i in range(5):
    print('True : ' + str(np.argmax(test_label[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))
