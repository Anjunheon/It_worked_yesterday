#!/usr/bin/env python
# coding: utf-8

# In[ ]:


base_model = MobileNetV2 (weights = 'imagenet', include_top = False) # MobileNetV2 모델을 가져오고 마지막 1000 개의 뉴런 계층을 버립니다.
x = base_model.output
x = GlobalAveragePooling2D () (x)
x = Dense (1024, activation = 'relu') (x) # 모델이 더 복잡한 기능을 학습하고 더 나은 결과를 위해 분류 할 수 있도록 조밀 한 레이어를 추가합니다.
x = Dense (1024, activation = 'relu') (x) #Dense 레이어 2
x = Dense (512, activation = 'relu') (x) #Dense 레이어 3
preds = Dense (num_class, activation = 'softmax') (x) #N 클래스에 대한 소프트 맥스 활성화가있는 최종 레이어

model = Model (inputs = base_model.input, outputs = preds) # 입력 및 출력 지정

