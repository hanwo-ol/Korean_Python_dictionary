## torch.cuda.amp.GradScaler

### 개요 (Overview)

`torch.cuda.amp.GradScaler`는 PyTorch의 자동 혼합 정밀도(Automatic Mixed Precision, AMP) 학습 시 발생하는 **그래디언트 언더플로우(gradient underflow)** 문제를 해결하기 위한 도구임.

혼합 정밀도 학습에서는 연산 속도를 높이기 위해 `float32` 대신 `float16`을 사용합니다. 
하지만 `float16`은 표현할 수 있는 수의 범위(dynamic range)가 `float32`보다 훨씬 좁기 때문에, 
매우 작은 그래디언트 값들이 0으로 처리되어 버리는 '언더플로우' 현상이 발생할 수 있습니다. 

**이 경우, 모델의 파라미터가 제대로 업데이트되지 않아 학습이 중단되거나 성능이 저하됩니다.**

* `GradScaler`는 이러한 문제를 방지하기 위해 **손실 값(loss)을 스케일링(scaling)**하는 기법을 사용합니다.

### 핵심 원리 (Core Principle)

`GradScaler`의 작동 원리는 간단합니다.

1.  **Loss Scaling**: `backward()`를 호출하기 전에, 손실 값에 큰 수(scale factor, $S$)를 곱합니다.
2.  **Gradient Calculation**: 연쇄 법칙(chain rule)에 의해, 그래디언트 값들도 동일하게 $S$배만큼 커집니다. 이로 인해 언더플로우가 발생할 가능성이 현저히 줄어듭니다.
3.  **Gradient Unscaling**: 옵티마이저가 파라미터를 업데이트하기 직전에, 스케일링되었던 그래디언트들을 다시 원래 크기로 되돌리기 위해 $S$로 나누어 줍니다.
4.  **Dynamic Scale Factor Update**: 학습 과정에서 그래디언트가 오버플로우(`inf` 또는 `NaN`)되는지 여부를 확인하며, 스케일 팩터 $S$를 동적으로 조절하여 최적의 학습 안정성을 유지합니다.

### 클래스 정의

```python
class torch.cuda.amp.GradScaler(
    init_scale=65536.0,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
    enabled=True
)
```

### 파라미터 (Parameters)

각 파라미터는 스케일 팩터를 동적으로 조절하는 정책을 결정합니다.

-   **`init_scale`** (float, 기본값: 65536.0)
    -   **설명**: 스케일 팩터의 초기값을 설정합니다. $2^{16}$인 65536.0은 `float16`의 최대값에 근접하지 않으면서도 충분히 큰 값으로, 대부분의 경우에 안정적인 시작점으로 사용됩니다.
    -   **영향**: 이 값이 너무 크면 첫 반복부터 그래디언트 오버플로우가 발생할 수 있고, 너무 작으면 언더플로우를 막지 못할 수 있습니다. 기본값을 사용하는 것이 일반적으로 권장됩니다.

-   **`growth_factor`** (float, 기본값: 2.0)
    -   **설명**: 그래디언트 오버플로우가 발생하지 않고 일정 기간(`growth_interval`) 동안 학습이 안정적으로 진행되었을 때, 스케일 팩터를 얼마나 증가시킬지를 결정하는 값입니다.
    -   **영향**: 값이 1.0보다 크면 스케일이 점차 증가합니다. 값이 클수록 스케일이 더 빠르게 커져 더 작은 그래디언트까지 표현할 수 있게 되지만, 오버플로우가 발생할 위험도 커집니다. 기본값인 2.0은 스케일을 2배씩 증가시킵니다.

-   **`backoff_factor`** (float, 기본값: 0.5)
    -   **설명**: 학습 중 그래디언트에서 오버플로우(`inf` 또는 `NaN`)가 감지되었을 때, 스케일 팩터를 얼마나 감소시킬지를 결정하는 값입니다.
    -   **영향**: 값이 1.0보다 작아야 합니다. 오버플로우가 발생했다는 것은 스케일이 너무 크다는 신호이므로, 스케일을 줄여 다음 반복에서 오버플로우를 방지합니다. 기본값인 0.5는 스케일을 절반으로 줄입니다.

-   **`growth_interval`** (int, 기본값: 2000)
    -   **설명**: 스케일 팩터를 증가시키기 전에, 오버플로우 없이 연속적으로 성공해야 하는 스텝(step)의 횟수입니다.
    -   **영향**: 이 값을 통해 스케일 팩터가 너무 성급하게 증가하는 것을 방지합니다. 예를 들어, 2000번의 스텝 동안 오버플로우가 한 번도 발생하지 않아야 스케일 팩터를 `growth_factor`만큼 증가시킵니다.

-   **`enabled`** (bool, 기본값: True)
    -   **설명**: `GradScaler`의 모든 기능을 활성화할지 여부를 결정합니다.
    -   **영향**: `False`로 설정하면 `GradScaler`의 모든 메소드(scale, step, update)는 아무 동작도 하지 않는 no-op 상태가 됩니다. 이는 AMP를 사용하지 않는 경우나 디버깅 시에 유용합니다.

### 수학적 원리 (Mathematical Principle)

1.  **Loss Scaling**

- 모델의 파라미터를 $\theta$, 손실 함수를 $L$이라고 할 때, 스케일링된 손실 $L_{scaled}$는 다음과 같습니다.

$$L_{scaled} = L \cdot S$$

- 여기서 $S$는 현재 스케일 팩터입니다.

2.  **Gradient Calculation (Chain Rule)**

- 스케일링된 손실에 대해 `backward()`를 호출하면, 연쇄 법칙에 의해 그래디언트 또한 스케일링됩니다.

$$\frac{\partial L_{scaled}}{\partial \theta} = \frac{\partial (L \cdot S)}{\partial \theta} = S \cdot \frac{\partial L}{\partial \theta}$$

- 따라서 원래 그래디언트($\frac{\partial L}{\partial \theta}$)가 $S$배만큼 커져 언더플로우를 방지합니다.

3.  **Gradient Unscaling**

- 옵티마이저가 파라미터를 업데이트하기 전, `scaler.step(optimizer)` 내부에서 스케일링된 그래디언트를 원래 값으로 되돌립니다.

$$\frac{\partial L}{\partial \theta} = \frac{1}{S} \cdot \frac{\partial L_{scaled}}{\partial \theta}$$

4.  **Scale Factor Update Logic**

- `scaler.update()` 메소드가 호출될 때, 다음 반복을 위한 스케일 팩터 $S_{new}$가 결정됩니다.

  - **오버플로우 발생 시**: `scaler.step()`에서 `inf`나 `NaN` 그래디언트가 발견되면, 옵티마이저 스텝은 건너뛰고 스케일 팩터를 줄입니다. 또한, `growth_interval` 카운터는 0으로 초기화됩니다.

$$S_{new} = S_{old} \cdot \text{backoff\_factor}$$

  - **오버플로우 미발생 시** : 옵티마이저 스텝이 성공적으로 실행되면, `growth interval` 카운터를 1 증가시킵니다. 만약 카운터가 `growth interval`에 도달하면, 스케일 팩터를 증가시킵니다. 카운터는 다시 0으로 초기화됩니다.

$$\text{if } (\text{counter} \pmod{\text{growth\_interval}} == 0): \\
  \quad S_{new} = S_{old} \cdot \text{growth\_factor}$$
  
        

### 사용 예제 (Usage Example)

일반적인 학습 루프에서 `GradScaler`는 다음과 같이 사용됩니다.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# 모델, 옵티마이저, 손실 함수 정의 (예시)
model = nn.Linear(10, 1).cuda()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# GradScaler 인스턴스 생성
scaler = GradScaler()

# 가상 데이터
data_loader = [(torch.randn(8, 10).cuda(), torch.randn(8, 1).cuda()) for _ in range(10)]

for epoch in range(num_epochs):
    for input, target in data_loader:
        optimizer.zero_grad()

        # 1. autocast 컨텍스트 내에서 순전파 실행
        #    float16으로 자동 형변환이 필요한 연산을 수행
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)

        # 2. scaler.scale()로 손실 값을 스케일링한 후, backward() 호출
        #    스케일링된 그래디언트가 생성됨
        scaler.scale(loss).backward()

        # 3. scaler.step()으로 옵티마이저 스텝 실행
        #    - 내부적으로 그래디언트를 unscale
        #    - inf/NaN 그래디언트가 없다면 optimizer.step() 호출
        #    - inf/NaN 그래디언트가 있다면 optimizer.step() 건너뜀
        scaler.step(optimizer)

        # 4. scaler.update()로 다음 반복을 위한 스케일 팩터 업데이트
        #    오버플로우 발생 여부에 따라 스케일을 조절
        scaler.update()

print("학습 완료!")
```

### 요약

`GradScaler`는 `float16` 혼합 정밀도 학습의 핵심 구성 요소로, 그래디언트 언더플로우를 방지하여 학습을 안정시키고 모델 성능을 유지하는 역할을 합니다. 손실 값을 동적으로 스케일링하고, 학습 상태를 모니터링하여 스케일 팩터를 자동으로 조절함으로써 사용자는 복잡한 설정 없이도 혼합 정밀도 학습의 이점을 누릴 수 있습니다.


---

## 사례

저 같은 경우엔 사용하는 loss function에도 언더플로우를 방지하려고 loss value에 compensation(=100)이라는 상수를 곱하는 기능이 내장되어 있었습니다. 그래서 그런지 상수에 init_scale까지 곱해지다보니 오버플로우가 생겼던 것 같네요.

### 문제 상황의 수학적 분석

먼저 관련된 변수를 다음과 같이 정의하겠습니다.

-   $L_{raw}$: 모델이 계산한 순수 손실 값 (보상 상수가 곱해지기 전)
-   $C$: 손실 함수 내의 보상 상수 (사용자 사례에서는 100)
-   $S$: `GradScaler`의 현재 스케일 팩터. 초기값은 `init_scale` ($S_{init}$)
-   $\theta$: 모델의 파라미터
-   $\frac{\partial L_{raw}}{\partial \theta}$: 파라미터에 대한 순수 그래디언트

#### 1. 손실 값의 이중 스케일링

제 학습 과정에서는 손실 값이 두 번에 걸쳐 스케일링됩니다.

1.  **손실 함수 내부 보상**: 먼저, 손실 함수가 순수 손실 값 $L_{raw}$에 보상 상수 $C$를 곱합니다. 이 값을 $L_{comp}$라고 하겠습니다.

$$L_{comp} = L_{raw} \cdot C$$

3.  **`GradScaler`의 스케일링**: 그 다음, `scaler.scale(loss)`가 호출될 때 `GradScaler`는 이미 보상된 손실 값 $L_{comp}$에 다시 스케일 팩터 $S$를 곱합니다. 최종적으로 스케일링된 손실 $L_{scaled}$는 다음과 같습니다.

$$L_{scaled} = L_{comp} \cdot S = (L_{raw} \cdot C) \cdot S$$

#### 2. 그래디언트의 과도한 증폭

`loss.backward()`가 호출되면, 연쇄 법칙(chain rule)에 따라 그래디언트가 계산됩니다. 최종 스케일링된 손실 $L_{scaled}$에 대한 그래디언트는 다음과 같이 증폭됩니다.

$$
\frac{\partial L_{scaled}}{\partial \theta} = \frac{\partial (L_{raw} \cdot C \cdot S)}{\partial \theta} = (C \cdot S) \cdot \frac{\partial L_{raw}}{\partial \theta}
$$

이 수식은 순수 그래디언트 $\frac{\partial L_{raw}}{\partial \theta}$가 **총 스케일링 팩터 $(C \cdot S)$** 만큼 곱해진다는 것을 의미합니다.

### `init_scale` 변경 전후 비교

`float16` 자료형이 표현할 수 있는 최대값은 약 65,504입니다. 중간 연산 과정에서 이 값을 초과하는 그래디언트가 발생하면 `Inf`가 됩니다.

#### 시나리오 A: `init_scale = 2**16 = 65536` (기본값)

-   보상 상수 $C = 100$
-   초기 스케일 팩터 $S_{init} = 65536$
-   **초기 총 스케일링 팩터**: $C \cdot S_{init} = 100 \times 65536 = 6,553,600$

이 경우, 순수 그래디언트 $\frac{\partial L_{raw}}{\partial \theta}$에 약 655만 배가 곱해집니다. 만약 순수 그래디언트의 크기가 매우 작더라도, 예를 들어 $10^{-5}$만 되어도 스케일링된 그래디언트는 $6,553,600 \times 10^{-5} \approx 65.5$가 됩니다.

만약 순수 그래디언트의 크기가 $0.01$만 되어도, 스케일링된 그래디언트는 $6,553,600 \times 0.01 = 65,536$이 됩니다. 이 값은 `float16`의 최대 표현 범위를 초과하므로 `Inf`가 발생할 가능성이 매우 높습니다.

#### 시나리오 B: `init_scale = 2**12 = 4096` (수정된 값)

-   보상 상수 $C = 100$
-   초기 스케일 팩터 $S_{init} = 4096$
-   **초기 총 스케일링 팩터**: $C \cdot S_{init} = 100 \times 4096 = 409,600$

이 경우, 순수 그래디언트에 약 41만 배가 곱해집니다. 총 스케일링 팩터가 시나리오 A에 비해 약 16배 작아졌습니다.

이제 순수 그래디언트의 크기가 $0.01$일 때 스케일링된 그래디언트는 $409,600 \times 0.01 = 4,096$이 됩니다. 이 값은 `float16`의 최대 표현 범위(65,504) 내에 있으므로 오버플로우가 발생하지 않습니다. 그래디언트가 `Inf`가 되지 않으므로 학습이 안정적으로 진행될 수 있습니다.

### 결론

**결론:**
제가 겪은 `Inf`/`NaN` 문제는 손실 함수 내부의 정적 보상 상수($C=100$)와 `GradScaler`의 동적 스케일 팩터($S$)가 중첩되어, 총 스케일링 팩터($C \cdot S$)가 과도하게 커졌기 때문입니다. 이로 인해 스케일링된 그래디언트가 `float16`의 표현 범위를 초과하여 오버플로우가 발생한 것입니다. `init_scale`을 $2^{16}$에서 $2^{12}$로 낮춤으로써 초기 총 스케일링 팩터를 줄였고, 그 결과 그래디언트가 `float16`의 표현 범위 내에 머무르게 되어 오버플로우 문제를 해결할 수 있었습니다.

