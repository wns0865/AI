#7번

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Embedding
from tensorflow.keras import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import numpy as np

dic_siz=10000 # 사전의 크기(사전에 있는 단어 개수)
sample_siz=512 # 샘플의 크기
num_folds = 3 # 교차 검증을 위한 폴드 수

# tensorflow가 제공하는 간소한 버전의 IMDB 읽기
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=dic_siz)

embed_space_dim=16 # 16차원의 임베딩 공간

x_train=preprocessing.sequence.pad_sequences(x_train,maxlen=sample_siz)
x_test=preprocessing.sequence.pad_sequences(x_test,maxlen=sample_siz)

early=EarlyStopping(monitor='val_accuracy',patience=5,restore_best_weights=True)

# 신경망 모델의 설계와 학습(LSTM 층 포함)
embed=Sequential()
embed.add(Embedding(input_dim=dic_siz,output_dim=embed_space_dim,input_length=sample_siz))
embed.add(LSTM(units=32))
embed.add(Dense(1,activation='sigmoid'))
embed.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

# 교차 검증을 위한 인덱스 생성
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# 교차 검증을 수행하면서 모델을 평가
fold_idx = 1
acc_scores = []
for train_idxs, valid_idxs in kfold.split(x_train):
  
    fold_idx += 1
    
    # 각 폴드에서 사용할 학습 데이터와 검증 데이터 가져오기
    x_train_fold = x_train[train_idxs]
    y_train_fold = y_train[train_idxs]
    x_valid_fold = x_train[valid_idxs]
    y_valid_fold = y_train[valid_idxs]
    

    # 모델 학습
    hist=embed.fit(x_train_fold,y_train_fold,epochs=5,batch_size=64,validation_data=(x_valid_fold, y_valid_fold),verbose=2,callbacks=[early])
    
    # 검증 세트에서 정확도 계산
    scores = embed.evaluate(x_valid_fold, y_valid_fold, verbose=0)
    acc_scores.append(scores[1] * 100)
    print(f"Fold-{fold_idx} accuracy: {scores[1] * 100}")

# 교차 검증 평균 정확도 계산
print("평균 정확률은", np.mean(acc_scores))