
# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 김용석
- 리뷰어 : 


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [o] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [o] 2.주석을 보고 작성자의 코드가 이해되었나요?
  > 작성자가 세세한 주석을 달아줘서 함수 나 모델의 동작, 전처리부분을 쉽게 이해할 수 있었습니다. 
- [x] 3.코드가 에러를 유발할 가능성이 있나요?
  > 코드는 정상적으로 동작하였으며, 특이사항은 없었습니다. 
- [o] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 세세하게 코드가 작성되어 충분히 이해하고 있었습니다. 
- [o] 5.코드가 간결한가요?
  > 간결하게 작성되었습니다. 

# 예시

# 데이터 나누기
from sklearn.model_selection import train_test_split

x_train, val_x, y_train, val_y = train_test_split(tensor, filtered_target, test_size=0.2)
y_train = np.array(y_train)
val_y = np.array(val_y)

# LSTM 모델 설계
import tensorflow as tf

vocab_size = vocab_size
word_vector_dim = 300       # 임베딩 사이즈

model_lstm = tf.keras.Sequential()
model_lstm.add(tf.keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
model_lstm.add(tf.keras.layers.LSTM(128))
model_lstm.add(tf.keras.layers.Dense(32, activation='relu'))
model_lstm.add(tf.keras.layers.Dense(1, activation='sigmoid')) # 최종 출력은 긍정/부정을 나타내는 1dim 입니다.
model_lstm.summary()

-----------------------------------------------------------------------------------------------------------------

# 모델 훈련

epochs = 2

model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_lstm = model_lstm.fit(x_train,
                              y_train,
                              epochs=epochs,
                              batch_size=128,                # 배치사이즈 128
                              validation_data=(val_x, val_y),
                              verbose=1)

# loss : 0.2758
# acc : 0.8623



# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
