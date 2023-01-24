
### stacking 이용해서 값 구하기

```python
X_train, X_test, valid_train, valid_test = train_test_split(train_data, train_target, test_size = 0.2, stratify = train_target,random_state = 1)
dtc =  DecisionTreeClassifier()
rfc = RandomForestClassifier()
knn =  KNeighborsClassifier()
xgb = xgboost.XGBClassifier()
clf = [('dtc',dtc),('rfc',rfc),('knn',knn),('xgb',xgb)] #list of (str, estimator)
from sklearn.ensemble import StackingClassifier
lr = LogisticRegression()
stack_model = StackingClassifier( estimators = clf,final_estimator = lr)
stack_model.fit(X_train,valid_train)
valid_predict = stack_model.predict(X_test)
print(f1_score(valid_predict, valid_test, average = 'macro'))

```

### 랜덤 값 고정 후 분류 모델 전체 이용해보기

```python
X_train, X_test, valid_train, valid_test = train_test_split(train_data, train_target, test_size = 0.2, stratify = train_target,random_state = 1)

import lightgbm as lgb

knn =  KNeighborsClassifier(n_neighbors=15 )
cnb = CategoricalNB()
gnb = GaussianNB()
svc = SVC(kernel='linear',random_state = 1)
lgb = lgb.LGBMClassifier(random_state = 1, objective='multiclass')
etc = ExtraTreesClassifier(random_state = 1)
xgb = XGBClassifier(random_state = 1, learning_rate = 0.1)
rfc = RandomForestClassifier(random_state = 1)
gbc = GradientBoostingClassifier(random_state = 1)
dtc =  DecisionTreeClassifier(random_state = 1)
ada = AdaBoostClassifier(random_state = 1)
lr = LogisticRegression(random_state = 1)
clf = [('gbc',gbc),('rfc',rfc),('xgb',xgb),('ada',ada),('etc',etc), ('lgb',lgb),('dtc',dtc),('cnb',cnb),('gnb',gnb),('knn',knn)]
stack_model4 = StackingClassifier( estimators = clf,final_estimator = lr,cv = 5)
stack_model4.fit(X_train,valid_train)
valid_predict = stack_model4.predict(X_test)
print(f1_score(valid_predict, valid_test, average = 'macro'))
# 그닥 성능 좋지 않음
```

### gridsearchcv 로 하이퍼파라미터 값 구하기
```python
from sklearn.model_selection import GridSearchCV
param_grid_gbc = {'learning_rate': [0.1, 0.5, 1], 'n_estimators': [50, 100, 200]}
param_grid_rfc = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}
param_grid_xgb = {'learning_rate': [0.1, 0.5, 1], 'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}
param_grid_ada = {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.5, 1]}
param_grid_etc = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}
param_grid_lgb = {'learning_rate': [0.1, 0.5, 1], 'n_estimators': [50, 100, 200], 'num_leaves': [10, 30, 50], 'max_depth': [5, 10, 20]}
param_grid_lr = {'C': [0.1, 1, 10]}

X_train, X_test, valid_train, valid_test = train_test_split(train_data, train_target, test_size = 0.2, stratify = train_target,random_state = 1)

etc = ExtraTreesClassifier(random_state = 1)
grid_etc = GridSearchCV(etc, param_grid_etc, cv=5, scoring='f1_macro')
grid_etc.fit(X_train, valid_train)
print("Best parameters for RandomForestClassifier: ", grid_etc.best_params_)
print("Best score for RandomForestClassifier: ", grid_etc.best_score_)

import lightgbm as lgb
lgb = lgb.LGBMClassifier(random_state = 1)
grid_lgb = GridSearchCV(lgb, param_grid_lgb, cv=5, scoring='f1_macro')
grid_lgb.fit(X_train, valid_train)
print("Best parameters for RandomForestClassifier: ", grid_lgb.best_params_)
print("Best score for RandomForestClassifier: ", grid_lgb.best_score_)

lr = LogisticRegression(random_state = 1)
grid_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='f1_macro')
grid_lr.fit(X_train, valid_train)
print("Best parameters for RandomForestClassifier: ", grid_lr.best_params_)
print("Best score for RandomForestClassifier: ", grid_lr.best_score_)

ada = AdaBoostClassifier(random_state = 1)
grid_ada = GridSearchCV(ada, param_grid_ada, cv=5, scoring='f1_macro')
grid_ada.fit(X_train, valid_train)
print("Best parameters for RandomForestClassifier: ", grid_ada.best_params_)
print("Best score for RandomForestClassifier: ", grid_ada.best_score_)

xgb = XGBClassifier(random_state = 1)
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='f1_macro')
grid_xgb.fit(X_train, valid_train)
print("Best parameters for RandomForestClassifier: ", grid_xgb.best_params_)
print("Best score for RandomForestClassifier: ", grid_xgb.best_score_)
```

### 조정 이후 다시 stacking
```python
X_train, X_test, valid_train, valid_test = train_test_split(train_data, train_target, test_size = 0.2, stratify = train_target,random_state = 1)

import lightgbm as lgb
lgb = lgb.LGBMClassifier(random_state = 1,objective='multiclass',learning_rate=0.1, max_depth=10, n_estimators=100, num_leaves=10)
etc = ExtraTreesClassifier(random_state = 1,max_depth = 20, n_estimators= 200)
xgb = XGBClassifier(random_state = 1,n_estimators = 100,max_depth=10,learning_rate = 0.05)
rfc = RandomForestClassifier(random_state = 1,max_depth= 20, n_estimators= 200)
gbc = GradientBoostingClassifier(random_state = 1,learning_rate=0.1,n_estimators=200)
dtc =  DecisionTreeClassifier(random_state = 1)
ada = AdaBoostClassifier(random_state = 1,n_estimators = 200,learning_rate = 0.1)
lr = LogisticRegression(random_state = 1,C=0.1)
clf = [('gbc',gbc),('rfc',rfc),('xgb',xgb),('ada',ada),('etc',etc), ('lgb',lgb),('dtc',dtc)]
stack_model4 = StackingClassifier( estimators = clf,final_estimator = lr,cv = 3)
stack_model4.fit(X_train,valid_train)
valid_predict = stack_model4.predict(X_test)
print(f1_score(valid_predict, valid_test, average = 'macro'))
```

### 이 중 stack 모델 찾기 위해 계속 반복해서 모델 수정
```python
# 그나마 최적의 모델
import lightgbm as lgb
lgb = lgb.LGBMClassifier(random_state = 1,objective='multiclass')
etc = ExtraTreesClassifier(random_state = 7)
xgb = XGBClassifier(random_state = 1)
rfc = RandomForestClassifier(random_state = 1)
gbc = GradientBoostingClassifier(random_state = 1)
dtc =  DecisionTreeClassifier(random_state = 1)
ada = AdaBoostClassifier(random_state = 1)
lr = LogisticRegression(random_state = 1)
clf = [('gbc',gbc),('rfc',rfc),('xgb',xgb),('ada',ada),('etc',etc), ('lgb',lgb)]
stack_model4 = StackingClassifier( estimators = clf,final_estimator = lr)
stack_model4.fit(X_train,valid_train)
valid_predict = stack_model4.predict(X_test)
print(f1_score(valid_predict, valid_test, average = 'macro'))
```
