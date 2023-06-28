
# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 김용석
- 리뷰어 : 김다인


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [⭕] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  - 주어진 4가지 평가 기준을 잘 충족했습니다
  
- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?
```python
# 학습, 테스트 데이터 문자 변경 → 시퀀스로 변환한 것을 왜 다시 문자로 바꾸는지 (용현님 질문)

# 인덱스를 단어로 바꿔주는 딕셔너리 정의 
word_index = reuters.get_word_index(path="reuters_word_index.json")

index_to_word = { index+3 : word for word, index in word_index.items() }

# index_to_word에 숫자 0은 <pad>, 숫자 1은 <sos>, 숫자 2는 <unk>를 넣어줍니다.
for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
  index_to_word[index]=token
  
# 확인
print(' '.join([index_to_word[index] for index in x_train[0]]))

#  전체 데이터 텍스트로 변경 ##
decoded_train = []
for i in range(len(x_train)):
  t = ' '.join([index_to_word[index] for index in x_train[i]])  ## 
  decoded_train.append(t)

x_train = decoded_train

decoded_test = []
for i in range(len(x_test)):
  text = ' '.join([index_to_word[index] for index in x_test[i]])
  decoded_test.append(text)

x_test = decoded_test
```
특히 오늘 오전에 나온 질문을 적어주셔서 다시 한 번 생각해볼 수 있어서 좋았습니다   

- [❌] 3.코드가 에러를 유발할 가능성이 있나요?
  - 정상적으로 동작했으며 에러가 발생한 부분은 없었습니다   
  
- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
  - 질문에 대답이 가능할 만큼 이해하고 있습니다
  
- [⭕] 5.코드가 간결한가요?
```python
# tfidf 생성
# train
x_train_dtm = dtmvector.fit_transform(x_train)
x_train_tfidf = tfidf_matrix.fit_transform(x_train_dtm)

# test
x_test_dtm = dtmvector.transform(x_test)
x_test_tfidf = tfidf_matrix.transform(x_test_dtm)


# 모델 별 결과를 저장할 리스트 생성
result_acc = [] # 정확도 리스트
result_f1score = [] # f1score 리스트

# 모델 리스트
model_nb = MultinomialNB()
model_cnb = ComplementNB()
model_lr = LogisticRegression(C=10000, penalty='l2', max_iter=3000)
model_lsvc = LinearSVC(C=1000, penalty='l1', max_iter=3000, dual=False)
model_tree = DecisionTreeClassifier(max_depth=10, random_state=27)
model_forest = RandomForestClassifier(n_estimators = 5, random_state=27)
model_grbt = GradientBoostingClassifier(random_state=27, verbose=3)
voting_classifier = VotingClassifier(estimators=[
         ('lr', LogisticRegression(C=10000, max_iter=3000, penalty='l2')),
        ('cb', ComplementNB()),
        ('grbt', GradientBoostingClassifier(random_state=0))
], voting='soft')


model_list = [model_nb, model_cnb, model_lr, model_lsvc, model_forest, model_grbt, voting_classifier]

for model in model_list:
  model.fit(x_train_tfidf, y_train)
  y_pred = model.predict(x_test_tfidf)

  acc = accuracy_score(y_test, y_pred)
  f_score = f1_score(y_test, y_pred, average='weighted')

  result_acc.append(acc)
  result_f1score.append(f_score)


result_df = pd.DataFrame(zip(result_acc, result_f1score), index=model_list, columns=['accuracy', 'f1_score'])
result_df
```
- 모델을 하나로 묶어서 선언 후 학습한 것이 간결해보이고 인상적이었습니다

# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.

# 나이브 베이즈 분류기
print("[나이브 베이즈 분류기]")
mod = MultinomialNB()
mod.fit(tfidfv, y_train)

x_test_dtm = dtmvector.transform(x_test) #테스트 데이터를 DTM으로 변환
tfidfv_test = tfidf_transformer.transform(x_test_dtm) #DTM을 TF-IDF 행렬로 변환

predicted = mod.predict(tfidfv_test) #테스트 데이터에 대한 예측
print("- 정확도:", accuracy_score(y_test, predicted)) #예측값과 실제값 비교
print("\n")
print("[나이브 베이즈 분류기의 Classification Report]")
print(classification_report(y_test, mod.predict(tfidfv_test)))
```
이런 방식으로 각각의 모델을 따로 선언해 학습하는 방법도 생각해보시면 좋을 것 같습니다
