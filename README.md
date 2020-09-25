# It_worked_yesterday

## 추가
* 2020.09.07 Model_Cifar10.py Callback 코드 추가
  * EarlyStopping : 일정 에포크 동안 정확도가 향상되지 않으면 학습 중단
  * ModelCheckpoint : 학습이 끝나면 가장 높은 정확도를 가지는 모델 저장  

* 2020.09.10 Image_generator 코드 추가
  * 이미지 증식

* 2020.09.17 Model_MobileNet.py 코드 추가
  * 코드 출처 : https://kau-deeperent.tistory.com/59
  
* 2020.09.25 remove_grayscale 코드 추가
  * grayscale 이미지 삭제

## 학습
| no | Model | epochs | batch_size |   img_size   | number_of_img_per_pokemon | EarlyStopping | EarlyStopping_patience | file_name(.h5) | val_loss | val_acc | test_acc |
|----|-------|--------|------------|--------------|---------------------------|---------------|------------------------|----------------|----------|---------|----------|
| 1  | CIFAR10 | 30  | 32 | 100 x 100 | 3000 | O | 3 | my_model | 1.0938 | 0.6433 | 5 / 5 |
| 2  | CIFAR10 | 125  | 32 | 100 x 100 | 3000 | O | 25 | my_model_1 | 1.0038 | 0.6906 | 3 / 5 |
| 3  | CIFAR10 | 500  | 64 | 100 x 100 | 3000 | X | X | my_model_2 | 1.0048 | 0.6973 | 5 / 5 |
