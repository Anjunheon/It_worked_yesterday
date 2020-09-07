# It_worked_yesterday

## 추가
* 2020.09.07 Model_Cifar10.py Callback 코드 추가
  * EarlyStopping : 일정 에포크 동안 정확도가 향상되지 않으면 학습 중단
  * ModelCheckpoint : 학습이 끝나면 가장 높은 정확도를 가지는 모델 저장

## 학습
| no | Model | epochs | batch_size | img_size | EarlyStopping | EarlyStopping_patience | ModelCheckpoint | file_name(.h5) |
|----|-------|--------|------------|----------|---------------|------------------------|----------------|
| 1 | CIFAR10 | 500 | 32 | 100 x 100 | O | 3 | O | my_model |
| 2 | CIFAR10 | 500 | 32 | 100 x 100 | O | 25 | O | my_model_1 |
| 3 | CIFAR10 | 500 | 64 | 100 x 100 | X | X | O | my_model_2 |
