> 참고논문: [P. Grosche, M. Müller and F. Kurth, "Cyclic tempogram—A mid-level tempo representation for musicsignals," 2010 IEEE International Conference on Acoustics, Speech and Signal Processing, Dallas, TX, USA, 2010, pp. 5522-5525, doi: 10.1109/ICASSP.2010.5495219. keywords: {Multiple signal classification;Pulse measurements;Data mining;Robustness;Rhythm;Power harmonic filters;Music information retrieval;Audio recording;Signal analysis;Signal processing;tempo;tempogram;chroma;music signals;audio segmentation},]

---

### 전체 과정의 흐름

이 과정은 다음과 같은 흐름으로 진행됩니다.

**오디오 신호 → 1. Novelty Curve → 2a. Fourier Tempogram 또는 2b. Autocorrelation Tempogram → 3. Cyclic Tempogram**

### 1. Novelty Curve (Δ(t)) - 템포그램의 재료

이것은 템포그램은 아니지만, 모든 템포그램을 계산하기 위한 가장 기초적인 **입력 신호**입니다.

*   **역할:** 음악에서 리듬적인 사건, 즉 '온셋(onset)'이 발생하는 시점과 그 강도를 나타내는 1차원 시계열 데이터를 만듭니다. 드럼 비트, 피아노 타건 등 새로운 소리가 시작되는 순간에 높은 값을 가집니다. 즉, **음악의 '리듬 신호'를 추출**하는 단계입니다.
*   **수식 (개념적 설명):**
    1.  오디오 신호를 STFT(Short-Time Fourier Transform)하여 스펙트로그램을 얻습니다.
    2.  스펙트로그램의 각 주파수 빈(bin)에서 시간에 따른 에너지 변화량을 계산합니다.
    3.  특히 에너지가 **증가하는** 변화량만을 포착하여 모두 더합니다. (`|Y(t+1, k) – Y(t, k)|≥0`)
    4.  이를 통해 시간에 따른 '음악적 새로움(novelty)'의 강도를 나타내는 곡선, 즉 Novelty Curve를 만듭니다.
*   **잡아내는 특징:**
    *   음악의 비트, 리듬 등 시간적 패턴의 원재료가 되는 **리듬 이벤트의 발생 시점과 강도**.

---

### 2a. Fourier Tempogram (TF) - "하모닉스(배속)"를 잘 잡는 템포그램

Novelty Curve의 주기성을 분석하는 첫 번째 방법입니다.

*   **역할:** Novelty Curve에 대해 STFT(Short-Time Fourier Transform)를 적용하여 시간에 따른 주기성(템포)을 분석합니다.
*   **수식 (Eq. 5):** `TF(t, τ) = |F(t, τ/60)|`
    *   `Δ(n)`: Novelty Curve 신호
    *   `F(t, ω)`: Novelty Curve에 대한 STFT. 시간 `t`에서의 주파수 `ω` 성분의 크기를 나타냅니다.
    *   `τ/60`: 템포 `τ` (BPM, 분당 비트 수)를 주파수 `ω` (Hz, 초당 비트 수)로 변환하는 과정입니다.
    *   **해석:** 이 수식은 **"Novelty Curve를 하나의 오디오 신호처럼 보고, 스펙트로그램을 계산한 것"**과 같습니다. 스펙트로그램의 주파수 축(y축)을 Hz 대신 BPM으로 바꾼 것이 바로 Fourier Tempogram입니다.
*   **잡아내는 특징: 하모닉스 (Harmonics, 배속 템포)**
    *   만약 주된 템포(tactus)가 120 BPM이라면, 이 방법은 2배인 240 BPM, 3배인 360 BPM에서도 강한 에너지를 나타내는 경향이 있습니다.
    *   이는 120 BPM의 주기적인 펄스 신호가 수학적으로 240Hz, 360Hz... 주파수 성분을 포함하기 때문입니다.
    *   따라서 **빠른 비트나 세분화된 리듬(tatum)을 잘 감지**하지만, 곡의 큰 틀을 이루는 느린 템포(예: 마디 단위 템포)는 상대적으로 약하게 나타날 수 있습니다. (논문에서는 "subharmonics를 억제한다"고 표현)

---

### 2b. Autocorrelation Tempogram (TA) - "서브하모닉스(분할)"를 잘 잡는 템포그램

Novelty Curve의 주기성을 분석하는 두 번째 방법입니다.

*   **역할:** Novelty Curve에 대해 국소적 자기상관(local autocorrelation)을 계산하여 시간에 따른 주기성을 분석합니다.
*   **수식 (Eq. 6, 7):**
    1.  `A(t, l) = Σ [Δ(n) * Δ(n+l)]` (개념적 형태)
    2.  `TA(t, τ) = A(t, l)`
    *   `A(t, l)`: 시간 `t`를 중심으로 한 짧은 구간(window)에서, 시간차 `l`(lag)만큼 신호를 밀어서 원래 신호와 얼마나 유사한지를 측정합니다.
    *   `l`: 시간 지연(lag). 이 값이 주기와 일치할 때 자기상관 값 `A`가 커집니다.
    *   `l`과 템포 `τ`는 역수 관계입니다 (`τ ∝ 1/l`). 예를 들어, lag `l`이 0.5초이면 템포 `τ`는 120 BPM이 됩니다.
    *   **해석:** 이 수식은 **"리듬 신호를 특정 시간만큼 밀어서 자기 자신과 비교했을 때, 얼마나 일치하는지를 모든 시간대와 모든 시간차에 대해 계산한 것"**입니다.
*   **잡아내는 특징: 서브하모닉스 (Subharmonics, 분할 템포)**
    *   만약 주된 템포가 120 BPM이라면, 이 방법은 그 절반인 60 BPM (마디 단위 템포), 1/3인 40 BPM 등에서도 강한 에너지를 나타내는 경향이 있습니다.
    *   이는 120 BPM 패턴은 더 큰 60 BPM 패턴의 일부이기도 하기 때문입니다.
    *   따라서 **곡의 구조적인 리듬이나 마디(measure) 단위의 느린 템포를 잘 감지**하지만, 가장 빠른 비트의 하모닉스는 상대적으로 약하게 나타납니다. (논문에서는 "harmonics를 억제한다"고 표현)

---

### 3. Cyclic Tempogram (C) - 템포 모호성을 해결한 최종 템포그램

위 두 템포그램의 한계를 극복하기 위해 제안된 핵심 아이디어입니다.

*   **역할:** 템포의 '옥타브(octave)' 개념을 도입하여, 60 BPM, 120 BPM, 240 BPM 처럼 2의 거듭제곱 배수로 연관된 템포들을 하나의 "템포 클래스"로 묶어줍니다. 이는 음악에서 60 BPM이나 120 BPM을 종종 혼용해서 인식하는 모호성을 해결하기 위함입니다.
*   **수식 (Eq. 1):** `C(t, [τ]) = Σλ∈[τ] T(t, λ)`
    *   `T`: Fourier Tempogram 또는 Autocorrelation Tempogram.
    *   `[τ]`: 템포 `τ`의 등가 클래스(equivalence class). 예를 들어, `[120]` = {..., 30, 60, 120, 240, 480, ...}.
    *   **해석:** **"템포 축을 '옥타브' 단위로 잘라서 겹쳐서 더한 것"**입니다. 예를 들어, Cyclic Tempogram의 120 BPM에 해당하는 값을 계산하기 위해, 원래 Tempogram의 30, 60, 120, 240, 480 BPM 값을 모두 더합니다. 이는 마치 피아노 건반에서 모든 '도' 음을 하나로 묶어 'C'라는 Chroma로 표현하는 것과 같은 원리입니다.
*   **잡아내는 특징:**
    *   **템포의 옥타브 불변성(Octave Invariance):** 템포가 두 배로 빨라지거나 느려지는 변화에 강인한, 매우 안정적인 템포 표현입니다.
    *   **하모닉스와 서브하모닉스의 통합:** 어떤 템포그램을 기반으로 하든, 관련된 템포 성분들을 하나로 합치므로 주된 템포의 에너지가 더욱 명확하게 드러납니다.
    *   결과적으로 음악의 리듬 구조를 더 일관성 있고 강인하게 표현할 수 있습니다.

### 요약 표

| 템포그램 종류 | 역할 및 원리 | 잡아내는 특징 | 장점 / 단점 |
| :--- | :--- | :--- | :--- |
| **Fourier Tempogram** | Novelty Curve에 STFT를 적용. 주파수 분석 기반. | **하모닉스 (배속 템포)**<br>(예: 120 BPM → 240, 360 BPM) | **장점:** 빠른 리듬, 세분화된 비트 감지에 유리<br>**단점:** 구조적인 느린 템포를 놓칠 수 있음 |
| **Autocorrelation Tempogram** | Novelty Curve에 자기상관을 적용. 시간 지연(lag) 기반 유사도 측정. | **서브하모닉스 (분할 템포)**<br>(예: 120 BPM → 60, 40 BPM) | **장점:** 마디 단위 등 구조적이고 느린 템포 감지에 유리<br>**단점:** 빠른 템포의 배속 성분을 놓칠 수 있음 |
| **Cyclic Tempogram** | 위 템포그램들의 템포 축을 옥타브 단위로 접어서 합산. | **옥타브 불변 템포**<br>(예: 60, 120, 240 BPM을 하나로) | **장점:** 템포 인식 모호성을 해결하여 매우 강인하고 안정적임<br>**단점:** 실제 절대 템포(60 vs 120)의 구분은 사라짐 |
