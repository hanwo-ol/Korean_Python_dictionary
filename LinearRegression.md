원본: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

## sklearn.linear_model.LinearRegression 

**`sklearn.linear_model.LinearRegression`** 클래스는 scikit-learn 라이브러리에서 제공하는 **선형 회귀** 모델입니다.  이 모델은 **최소 제곱법(Ordinary Least Squares)** 을 사용하여 데이터에 가장 적합한 선형 관계를 찾습니다.  즉, 입력 특성(features)과 출력 변수(target) 간의 선형 관계를 모델링하고, 예측값과 실제값의 차이(잔차 제곱의 합)를 최소화하는 계수(coefficients)를 추정합니다.

**주요 용도:**

*   **예측:**  주어진 입력 특성에 대한 출력 변수의 값을 예측합니다. (예: 주택 가격 예측, 판매량 예측)
*   **관계 분석:** 입력 특성과 출력 변수 간의 선형 관계의 강도와 방향을 파악합니다. (예: 광고 지출과 매출 간의 관계)

### 1. 클래스 생성

```python
from sklearn.linear_model import LinearRegression

# LinearRegression 객체 생성 (대부분의 경우 기본 설정으로 충분)
model = LinearRegression()
```

**생성자 매개변수 (Parameters):**

*   `fit_intercept` (bool, 기본값: `True`):  절편(intercept, y축과의 교점)을 계산할지 여부를 결정합니다.  `False`로 설정하면 데이터가 원점을 중심으로 분포한다고 가정합니다 (즉, 절편을 0으로 고정).  대부분의 경우 `True`로 유지하는 것이 좋습니다.

*   `copy_X` (bool, 기본값: `True`):  `True`이면 입력 데이터 `X`를 복사하여 사용합니다.  `False`로 설정하면 `X`가 덮어씌워질 수 있으므로 주의해야 합니다.  메모리 사용량이 매우 크지 않다면 `True`로 두는 것이 안전합니다.

*   `n_jobs` (int, 기본값: `None`):  계산에 사용할 CPU 코어 수를 지정합니다.  `None` (기본값)은 1을 의미하며, `-1`은 사용 가능한 모든 코어를 사용함을 의미합니다.  데이터 세트가 매우 크고, 여러 개의 대상 변수(multi-target)가 있거나, `positive=True`로 설정된 경우에만 속도 향상을 기대할 수 있습니다. `n_jobs`를 변경해도 결과는 달라지지 않습니다.

*   `positive` (bool, 기본값: `False`): `True`로 설정하면 회귀 계수(coefficients)를 양수로 제한합니다. 이 옵션은 밀집(dense) 배열에만 적용됩니다.  계수가 양수여야 한다는 사전 지식이 있는 경우에만 사용합니다.

### 2. 모델 학습 (fit)

```python
import numpy as np

# 예시 데이터 (2개의 특성을 가진 4개의 샘플)
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3  # y = 1 * x_0 + 2 * x_1 + 3

# 모델 학습
model.fit(X, y)
```

**`fit()` 메서드 매개변수:**

*   `X`:  입력 데이터 (특성 행렬).  NumPy 배열, Pandas DataFrame 또는 SciPy 희소 행렬 형태를 사용할 수 있습니다.  형태는 `(n_samples, n_features)`입니다.  `n_samples`는 데이터 샘플의 개수이고, `n_features`는 특성의 개수입니다.

*   `y`:  출력 데이터 (대상 변수).  NumPy 배열 또는 Pandas Series 형태를 사용할 수 있습니다.  형태는 `(n_samples,)` 또는 `(n_samples, n_targets)`입니다.  `n_targets`은 다중 출력(multi-output) 회귀의 경우 대상 변수의 개수입니다 (일반적으로 1).

*   `sample_weight` (선택 사항):  각 샘플에 대한 가중치를 지정합니다. NumPy 배열 형태이며, 형태는 `(n_samples,)`입니다.  특정 샘플에 더 큰 중요도를 부여하고 싶을 때 사용합니다.  기본값은 `None`이며, 모든 샘플에 동일한 가중치(1)를 부여합니다.

**`fit()` 메서드의 반환값:**

*   `self`:  학습된 `LinearRegression` 객체 자신을 반환합니다.

### 3. 예측 (predict)

```python
# 새로운 데이터에 대한 예측
X_new = np.array([[3, 5]])
y_pred = model.predict(X_new)
print(y_pred)  # 출력: [16.]
```

**`predict()` 메서드 매개변수:**

*  `X`: 예측에 사용할 입력 데이터. `fit()` 메서드에 사용된 `X`와 동일한 형태(`(n_samples, n_features)`)여야 합니다.

**`predict()` 메서드의 반환값:**

* 예측된 값. `y`가 1차원 배열인 경우 `(n_samples,)` 형태의 NumPy 배열, `y`가 2차원 배열인 경우 `(n_samples, n_targets)`형태의 NumPy 배열이 반환됩니다.

### 4. 모델 평가 (score)

```python
# R-squared (결정 계수) 계산
score = model.score(X, y)
print(score) # 출력: 1.0
```

**`score()` 메서드 매개변수:**

*   `X`:  입력 데이터. `fit()` 메서드에 사용된 `X`와 동일한 형태여야 합니다.
*   `y`:  실제 값 (대상 변수). `fit()` 메서드에 사용된 `y`와 동일한 형태여야 합니다.
*   `sample_weight` (선택 사항):  각 샘플에 대한 가중치 (선택 사항). `fit()` 메서드에서 사용한 것과 동일하게 적용됩니다.

**`score()` 메서드의 반환값:**

*   R-squared (결정 계수) 값을 반환합니다.  R-squared는 모델이 데이터를 얼마나 잘 설명하는지를 나타내는 지표입니다.  0과 1 사이의 값을 가지며, 1에 가까울수록 모델의 성능이 좋다는 것을 의미합니다.  R-squared는 음수가 될 수도 있으며, 이는 모델이 평균값으로 예측하는 것보다 더 나쁘다는 것을 의미합니다.

### 5. 주요 속성 (Attributes)

학습이 완료된 후(`fit()` 메서드 호출 후) 다음과 같은 속성을 통해 모델의 정보를 확인할 수 있습니다.

*   `coef_`:  추정된 회귀 계수(coefficients).  `y`가 1차원 배열인 경우 `(n_features,)` 형태의 NumPy 배열, `y`가 2차원 배열인 경우 `(n_targets, n_features)` 형태의 NumPy 배열입니다.  각 계수는 해당 특성이 출력 변수에 미치는 영향의 크기와 방향을 나타냅니다.

*   `intercept_`:  추정된 절편(intercept).  `y`가 1차원 배열인 경우 스칼라 값, `y`가 2차원 배열인 경우 `(n_targets,)` 형태의 NumPy 배열입니다.

*   `rank_`:  입력 데이터 행렬 `X`의 랭크(rank). `X`가 밀집 행렬일 때만 사용 가능합니다.

*   `singular_`:  입력 데이터 행렬 `X`의 특이값(singular values). `X`가 밀집 행렬일 때만 사용 가능합니다.

*   `n_features_in_`:  `fit()` 메서드에서 사용된 특성의 개수.

*  `feature_names_in_`: `fit` 하는 동안 표시되는 특성의 이름입니다. `X`에 문자열인 특성 이름이 있는 경우에만 정의됩니다.

### 6. 추가 메서드

* `get_params(deep=True)`:  모델의 매개변수를 딕셔너리 형태로 반환합니다.
* `set_params(**params)`:  모델의 매개변수를 설정합니다.
* `get_metadata_routing()`: 메타데이터 라우팅 정보를 가져옵니다. (일반적인 사용에서는 필요하지 않습니다.)
* `set_fit_request(*, sample_weight)`: `fit` 메서드에 대한 메타데이터 요청을 설정합니다.
* `set_score_request(*, sample_weight)`: `score` 메서드에 대한 메타데이터 요청을 설정합니다.

### 7. 예제 코드 (종합)

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 생성 (가상의 키와 몸무게 데이터)
np.random.seed(0)
X = np.random.rand(100, 1) * 50 + 150  # 키 (150cm ~ 200cm)
y = 0.6 * X + np.random.randn(100, 1) * 5 + 20  # 몸무게 (약간의 노이즈 추가)

# 훈련 세트와 테스트 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LinearRegression 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 테스트 세트에 대한 예측
y_pred = model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# 추정된 계수와 절편 확인
print(f"Coefficient: {model.coef_[0][0]:.2f}")  # 키에 대한 계수
print(f"Intercept: {model.intercept_[0]:.2f}")

# 새로운 키에 대한 몸무게 예측
new_height = np.array([[175]])  # 175cm 키
predicted_weight = model.predict(new_height)
print(f"Predicted weight for 175cm: {predicted_weight[0][0]:.2f} kg")
```

이 예제에서는 가상의 키와 몸무게 데이터를 생성하고, `LinearRegression` 모델을 사용하여 키로부터 몸무게를 예측하는 방법을 보여줍니다.  `train_test_split` 함수를 사용하여 데이터를 훈련 세트와 테스트 세트로 나누고, 모델을 훈련시킨 후 테스트 세트에서 모델의 성능을 평가합니다.  `mean_squared_error` (평균 제곱 오차)와 `r2_score` (결정 계수)를 사용하여 모델의 예측 정확도를 측정합니다.  마지막으로, `coef_`와 `intercept_` 속성을 통해 추정된 계수와 절편을 확인하고, 새로운 키 값에 대한 몸무게를 예측합니다.

이 설명서는 `sklearn.linear_model.LinearRegression`의 기본적인 사용법을 다루고 있습니다. 더 자세한 내용은 scikit-learn 공식 문서를 참조하시기 바랍니다.
