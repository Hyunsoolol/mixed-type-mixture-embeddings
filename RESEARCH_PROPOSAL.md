# Interpretable Mixed-type Mixture Modeling

![Repo Name](https://img.shields.io/badge/Repo-interpretable--mixed--mixture-blueviolet)
![Status](https://img.shields.io/badge/Status-Research%20Proposal-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![Topic](https://img.shields.io/badge/Topic-Mixture%20Models%20%7C%20NLP%20%7C%20XAI-orange)

## Overview
본 프로젝트는 범주형 변수(예: 인구통계, 캠페인 속성)와 텍스트(비정형)로부터 얻어지는 고차원 임베딩을 하나의 잠재 클래스(혼합) 모형으로 결합하여 군집화/세그먼테이션을 수행하는 혼합형 데이터(Mixed-type data) 혼합모형 프레임워크를 제안합니다.

기존 연구들이 단어 가방 모형(Bag-of-Words, BoW)이나 강력한 독립성 가정(나이브 베이즈)에 의존했던 것과 달리, 본 연구는 **거대 언어 모델(LLM) 임베딩**을 활용하여 텍스트의 의미론적 문맥(Semantic Context)을 포착합니다. 또한, 임베딩의 "블랙박스" 특성과 "차원의 저주" 문제를 해결하기 위해, 모델 해석을 위한 **디베딩(De-embedding)** 전략과 안정적인 가우시안 혼합 모형(GMM) 추정을 위한 차원 축소 단계를 도입했습니다.

## Motivation

### 기존 방법론의 한계

[Shi et al. (2024)](https://doi.org/10.1214/24-AOAS1893)와 같은 최근 연구에서는 텍스트 $Z$를 이진(또는 BoW) 벡터로 두고, 클래스 $K_i=k$ 조건부에서 나이브 베이즈 형태를 가정합니다.

$$
P(Z_i | K_i = k) = \prod_{j=1}^{p} P(Z_{ij} | K_i = k)
$$

이때 주요 한계는 다음과 같습니다.
1.  **의미 손실 (Loss of Semantics):** 단어의 순서, 문맥, 그리고 미세한 의미론적 뉘앙스가 사라짐
2.  **비현실적 가정 (Unrealistic Assumption):** 자연어에서 단어 간의 독립성 가정은 현실적으로 위배되는 경우가 많습니다.

### 제안하는 접근법 (Our Approach)
우리는 이진 특성 벡터를 LLM(예: SBERT, OpenAI)이 생성한 **밀집 임베딩(Dense Embeddings)**으로 바꾸면, 문제는 **혼합형 혼합 모형(범주형 + 연속형)**으로 변환됩니다.
다만 임베딩은 고차원이고 해석이 어려우므로, **(차원축소 → 혼합모형 적합 → 디베딩 해석)**을 하나의 파이프라인으로 고정합니다.

## Methodology

관측치 $i$는 다음의 mixed-type 정보를 가진다고 둡니다.

범주형 블록: $\mathbf{x}^{(c)}_i$ (예: 인구통계, 국가/채널/소재유형)

텍스트: $\mathbf{z}_i$ (문서/문장/광고 카피)

(선택) 추가 연속형 지표: $\mathbf{y}_i$ (예: CTR, CVR, ROAS 등)

텍스트는 LLM 임베딩과 차원 축소를 거쳐 연속형 특징으로 변환합니다.

제안된 프레임워크는 크게 세 가지 단계로 구성됩니다:

```mermaid
graph LR
  A[원본 데이터 (x_cat, text, optional y)] --> B{전처리/정규화};
  B -->|x_cat| C[범주형 인코딩/결측 처리];
  B -->|text| D[LLM 임베딩 v in R^D];
  D --> E[PCA로 d차원 축소 x_emb in R^d];
  B -->|y (optional)| Y[연속형 지표 변환/스케일링];
  C --> F[결합 혼합모형 적합];
  E --> F;
  Y --> F;
  F --> G[잠재 클래스 posterior / 할당];
  G --> H[디베딩: 프로토타입+키워드+라벨링];
```
### 1. Feature Representation
관측된 대상 $i$의 데이터를 $(\mathbf{x}^{(c)}_i, \mathbf{z}_i)$라고 합시다. 여기서 $\mathbf{x}^{(c)}_i$는 범주형 벡터(예: 인구통계 변수), $\mathbf{z}_i$는 원본 텍스트 데이터입니다.

* **텍스트 임베딩 (Text Embedding):** 먼저 사전 학습된 LLM을 사용하여 원본 텍스트를 밀집 벡터로 변환합니다.

$$
\mathbf{v}_i = \text{LLM}(\mathbf{z}_i) \in \mathbb{R}^{D}
$$

*(여기서 D는 원본 임베딩 차원으로, 예: 768 또는 1536)*

* **차원 축소 (Dimensionality Reduction):** 가우시안 혼합 모형에서 공분산 행렬의 안정적인 추정을 위해(즉, $D \gg n$으로 인한 특이성 문제 방지), $\mathbf{v}_i$를 저차원 매니폴드 $\mathbb{R}^{d}$로 투영합니다 (예: $d \approx 20 \sim 50$).

$$
\mathbf{x}^{(e)}_i = \phi(\mathbf{v}_i) \in \mathbb{R}^d
$$

여기서 $\phi$는 GMM에 적합한 전역적 분산 구조를 보존하는 **주성분 분석(PCA)**과 같은 차원 축소 함수를 의미합니다.

### 2. Joint Mixture Model Specification
전체 모집단이 $K$개의 잠재 클래스(Latent Class)로 구성된다고 가정합니다. 혼합형 데이터 $(\mathbf{x}^{(c)}_i, \mathbf{x}^{(e)}_i)$에 대한 결합 우도(Likelihood)는 다음과 같이 정의됩니다.

$$
\mathcal{L}(\Theta) = \sum_{i=1}^{n} \log \left( \sum_{k=1}^{K} \pi_k \cdot f_{\text{cat}}(\mathbf{x}^{(c)}_i | \boldsymbol{\alpha}_k) \cdot f_{\text{cont}}(\mathbf{x}^{(e)}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)
$$

여기서:
* **혼합 비율 (Mixing Proportion):** $\pi_k$는 클래스 $k$의 사전 확률이며, $\sum_{k=1}^K \pi_k = 1$을 만족합니다.
* **범주형 부분 (Categorical Part):** $f_{\text{cat}}$은 파라미터 $\boldsymbol{\alpha}_k$를 갖는 **다항 분포(Multinomial distribution)**를 따르며, 각 클래스 내 인구통계학적 변수의 분포를 포착합니다.
* **연속형(임베딩) 부분 (Continuous Part):** $f_{\text{cont}}$는 **다변량 정규 분포(Multivariate Gaussian distribution)**를 따르며, 텍스트의 의미론적 군집을 포착합니다.

$$
f_{\text{cont}}(\mathbf{x}^{(e)}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) = (2\pi)^{-d/2}|\boldsymbol{\Sigma}_k|^{-1/2} \exp\left(-\frac{1}{2}(\mathbf{x}^{(e)}_i - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\mathbf{x}^{(e)}_i - \boldsymbol{\mu}_k)\right)
$$

### 3. Interpretation via De-embedding
임베딩의 주요 과제는 해석력 부족입니다. 군집의 중심(Centroid) $\boldsymbol{\mu}_k$가 축소된 임베딩 공간에 존재하기 때문에, 사람은 이를 직관적으로 이해할 수 없습니다(Black-box). 우리는 각 잠재 클래스 $k$의 의미를 복원하기 위해 두 가지 **"디베딩(De-embedding)"** 방법을 제안합니다.

#### 방법 A: 의미론적 앵커 (검색 기반)
추정된 군집 중심 $\boldsymbol{\mu}_k$와 기하학적으로 가장 가까운 원본 데이터셋의 **프로토타입 문서(Prototype Documents)**를 식별합니다.

$$
\text{Prototype}_k = \{ \mathbf{z}_j \mid \mathbf{z}_j \in \text{Dataset}, \text{argmax}_{j} \text{CosineSim}(\mathbf{x}^{(e)}_j, \boldsymbol{\mu}_k) \}
$$

* **활용:** 연구자가 실제 대표 텍스트를 읽음으로써 군집을 정성적으로 이해할 수 있게 합니다. (예: "이 군집은 '보이스피싱' 사건들을 대표한다.")

#### 방법 B: 선형 디코더 (키워드 추출)
축소된 임베딩을 다시 해석 가능한 단어 가방(BoW) 공간으로 매핑하는 전역 선형 디코더(또는 Lasso 회귀)를 학습합니다.

$$
\hat{\mathbf{W}} = \underset{\mathbf{W}}{\text{argmin}} \sum_{i=1}^n || \mathbf{y}_{\text{BoW}, i} - \mathbf{x}^{(e)}_i \mathbf{W} ||_2^2 + \lambda ||\mathbf{W}||_1
$$

추정된 가중치 행렬 $\hat{\mathbf{W}}$를 사용하여, 중심 벡터 $\boldsymbol{\mu}_k$를 키워드 중요도 벡터 $\mathbf{w}_k = \boldsymbol{\mu}_k \hat{\mathbf{W}}$로 변환하고, 각 클래스의 상위 가중치 단어를 추출합니다.
* **활용:** 군집에 대한 정량적 설명을 제공합니다. (예: "상위 키워드: *사기, 은행, 송금*")

## Key Contributions
1.  **의미 기반 군집화 (Semantic-Aware Clustering):** 문맥이 풍부한 LLM 임베딩을 활용하여 나이브 베이즈 가정과 이진 텍스트 표현의 한계를 극복했습니다.
2.  **통합 프레임워크 (Unified Framework):** 구조화된 인구통계 데이터와 비정형 텍스트 데이터를 결합하여 분석할 수 있는 엄밀한 통계적 모델을 제공합니다.
3.  **설명 가능성 (Explainability):** 제안된 디베딩 전략을 통해 "블랙박스" 신경망 임베딩과 "화이트박스" 통계적 추론 사이의 간극을 해소했습니다.

## References
* **Primary Reference:** Shi, J., Wang, F., Gao, Y., Song, X., & Wang, H. (2024). *Mixture conditional regression for estimating extralegal factor effects*. The Annals of Applied Statistics, 18(3), 2535-2550.
* **Mixture Models:** Scrucca, L., Fop, M., Murphy, T. B., & Raftery, A. E. (2016). *mclust 5: clustering, classification and density estimation using Gaussian finite mixture models*. The R Journal, 8(1), 289.
* **Embeddings:** Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*.


# 게임 마케팅 데이터에 방법론 적용 예시

## 1. 가상의 게임 마케팅 데이터 설정

예를 들어, 모바일 RPG 게임의 UA(User Acquisition) 캠페인 데이터가 있다고 가정합니다.

- 관측 단위: `캠페인 × 국가 × 채널`
  - 예: `Facebook_KR_10월1주차`
- 주요 변수들  
  - 범주형 변수
    - `Country` : KR / JP / US
    - `Channel` : Facebook / Google / TikTok
    - `Creative_Type` : Video / Image / Playable
  - 연속형 지표 (결과/설명 변수)
    - `CTR` (클릭률)
    - `CVR` (설치 전환율)
    - `7d_ROAS` (7일 수익/광고비 비율)
  - 텍스트 변수
    - `Ad_Text` : 예)  
      - “지금 접속하면 전설 장비 100% 지급!”  
      - “귀여운 캐릭터와 함께하는 힐링 RPG”

연구/분석 목표:

> 이 데이터를 바탕으로 **잠재 캠페인 유형(유저 세그먼트/마케팅 전략 패턴)** 을 찾고,  
> 그 잠재범주를 **텍스트 임베딩 + 디베딩(debedding)** 으로 “해석 가능한 라벨”로 붙이는 것.

---

## 2. 텍스트 임베딩 생성 단계: 광고 문구를 벡터로 변환

### 2.1. LLM 임베딩 적용

각 광고 문구 `Ad_Text` 에 대해 LLM 임베딩 모델(예: 768차원 벡터)을 적용한다고 가정합니다.

- 각 캠페인 $i$에 대해
  - $e_i \in \mathbb{R}^{768}$: 광고 문구 임베딩 벡터

개념적 예시는 다음과 같습니다.

| Campaign_ID | Ad_Text                          | Embedding (축약 표기)          |
|-------------|----------------------------------|---------------------------------|
| 1           | 전설 장비 100% 지급!             | $e_1 = (0.23, -0.11, \dots)$    |
| 2           | 귀여운 캐릭터 힐링 RPG           | $e_2 = (0.05, 0.19, \dots)$     |
| 3           | 실시간 PvP로 최강자를 증명하라! | $e_3 = (-0.12, 0.33, \dots)$    |

---

## 3. 차원축소: 고차원 임베딩을 저차원 특징으로 변환

임베딩(768차원)을 그대로 LCA에 넣기에는 차원이 너무 크므로,  
**PCA(주성분 분석)** 으로 차원을 줄입니다.

### 3.1. PCA 적용

- $E$: $(N \times 768)$ 임베딩 행렬
- PCA로 상위 $d$개 주성분만 사용  
  - 예: $d = 5$

결과적으로 각 캠페인 $i$에 대해:

- $z_i = (z_{i1}, \dots, z_{i5})$: 5차원 연속형 변수 벡터

예시 테이블:

| Campaign_ID | PC1 (Action) | PC2 (Casual) | PC3 (Reward) | PC4 | PC5 |
|-------------|--------------|--------------|--------------|-----|-----|
| 1           |  1.8         | -0.2         |  1.5         | …   | …   |
| 2           | -0.5         |  1.9         |  0.1         | …   | …   |
| 3           |  2.1         | -1.0         | -0.3         | …   | …   |

해석 예시(연구자가 사후 해석):

- PC1: 전투/경쟁/강한 어조 → 값이 클수록 “하드코어 액션” 느낌
- PC2: 귀여움/힐링/캐주얼 → 값이 클수록 “캐주얼 감성”
- PC3: 보상/혜택 강조 → 값이 클수록 “보상 프로모션”

---

## 4. 혼합 LCA 모형 적합: 범주형 + 연속형 결합

### 4.1. 모형 구조 설정

각 캠페인 $i$에 대해 잠재범주 $C_i \in \{1,\dots,K\}$ (예: $K = 3$)를 가정합니다.

각 잠재범주(클래스) $k$에서:

- 범주형 변수의 조건부 분포
  - $P(\text{Country} \mid C = k)$
  - $P(\text{Channel} \mid C = k)$
  - $P(\text{Creative Type} \mid C = k)$
- 연속형 변수의 조건부 분포  
  (텍스트에서 축약된 PC + 성과 지표)

  - $(z_{i1}, \dots, z_{i5}, \text{CTR}, \text{CVR}, \text{ROAS}) \mid C = k \sim \mathcal{N}(\mu_k, \Sigma_k)$

정리하면:

> **범주형 변수 + 연속형 변수** 를 한 번에 다루는 **혼합형 LCA 모형**으로  
> “잠재 마케팅 전략 타입”을 추정하는 구조입니다.

### 4.2. EM 알고리즘으로 파라미터 추정

모형 적합은 일반적인 EM 알고리즘으로 수행합니다.

1. 초기화
   - 랜덤 초기값 또는 k-means 등으로 잠재 클래스 초기 분류
2. E-step
   - 각 관측값에 대해 사후확률 계산
   - $P(C_i = k \mid \text{data}_i)$
3. M-step
   - 클래스별 파라미터 업데이트
     - 혼합비 $\pi_k$
     - 연속형 분포의 평균/공분산 $(\mu_k, \Sigma_k)$
     - 범주형 변수의 조건부 확률
4. 수렴할 때까지 2–3단계 반복
5. $K$ 선택
   - BIC, AIC 등을 사용하여 최적의 클래스 개수 $K$를 선택

---

## 5. 예시 결과: 잠재 캠페인 유형 3개 발견

아래는 가상의 결과 예시입니다.

### 5.1. 클래스별 특성 요약 (수치 기반)

| Class | 비율 ($\pi_k$)     | Country 분포              | Channel 분포            | Creative_Type 비중 | 텍스트 PC 특징                         | 성과 특징 (평균)       |
|-------|--------------------|---------------------------|-------------------------|--------------------|----------------------------------------|------------------------|
| 1     | 40%                | KR 70%, JP 20%, US 10%    | Facebook 60%, TikTok 30% | Video 80%         | PC1↑(액션), PC3↑(보상)                 | CTR↑, CVR↑, ROAS 중간 |
| 2     | 35%                | JP 60%, KR 20%, US 20%    | Google 70%              | Image 70%          | PC2↑(캐주얼), PC3↓(보상 강조 적음)    | CTR 중간, CVR↓, ROAS↓ |
| 3     | 25%                | US 50%, JP 30%, KR 20%    | TikTok 60%              | Playable 90%       | PC1↑(액션) + PC2↑(캐주얼 혼합)         | CTR↑, CVR↑, ROAS↑↑    |

해석 포인트:

- Class 1: 한국 중심, Facebook/Video, 액션+보상 메시지, 성과 양호
- Class 2: 일본 중심, Google/Image, 캐주얼 감성, 성과는 다소 낮음
- Class 3: 미국 비중 높음, TikTok/Playable, 액션+캐주얼 혼합, ROAS 매우 우수

---

## 6. 디베딩(debedding): 잠재범주에 자연어 해석 부여

이제 각 클래스별로 **대표 텍스트 임베딩 → 자연어 설명**을 만드는 과정입니다.

### 6.1. 클래스별 대표 임베딩 구성

각 클래스 $k$에 대해 다음과 같이 대표 임베딩을 만듭니다.

1. 클래스 $k$에 속할 확률이 높은 캠페인만 선택  
   - 예: $P(C_i = k) > 0.8$ 인 캠페인 집합
2. 이들 캠페인의 텍스트 임베딩을 이용해:
   - 평균 벡터 $\bar{e}_k$를 구하거나
   - $\bar{e}_k$와의 거리가 가장 가까운 광고 문구 상위 5개를 대표 샘플로 선택

정리:

- $\bar{e}_1$: Class 1의 평균 임베딩
- $\bar{e}_2$, $\bar{e}_3$도 동일하게 계산

또는:

- Class 1에서 평균 벡터 근처에 있는 대표 문구 3–5개를 뽑는 방식

### 6.2. LLM을 이용한 클래스 설명 생성 (프롬프트 예시)

각 클래스 $k$마다 대표 광고 문구들을 모아서 LLM에 넣고,  
해당 클래스의 타겟 유저·메시지 전략을 한두 문장으로 요약하도록 합니다.

예를 들어, Class 1의 대표 문구가 다음과 같다고 가정합니다.

- “전설 장비 100% 지급!”
- “지금 접속하면 한정 S급 무기 지급”
- “강력한 보스 레이드, 친구들과 함께 공략”

여기에 대해 LLM 프롬프트 예시는 다음과 같습니다.

> 다음 광고 문구들이 공통으로 타겟팅하는 유저 그룹과 마케팅 전략을 한두 문장으로 요약해 주세요.  
> - 전설 장비 100% 지급!  
> - 지금 접속하면 한정 S급 무기 지급  
> - 강력한 보스 레이드, 친구들과 함께 공략  

이때, Class 1에 대한 LLM 출력 예시는 다음과 같을 수 있습니다.

- Class 1 설명 (예)
  > 강한 전투 이미지와 높은 보상을 전면에 내세워,  
  > 하드코어 RPG 유저를 대상으로 즉각적인 참여를 유도하는 프로모션 중심 캠페인 유형입니다.

Class 2에 대해서는 캐주얼/힐링 감성의 문구들을 입력하면, 예를 들어:

- Class 2 설명 (예)
  > 귀엽고 편안한 분위기를 강조하며,  
  > 라이트 유저와 일상 속 힐링을 원하는 이용자를 타깃으로 한 감성 중심 브랜딩 캠페인 유형입니다.

Class 3에 대해서는 Playable, 체험형 문구들을 기반으로 할 때:

- Class 3 설명 (예)
  > 직접 플레이해보는 체험형 광고를 통해,  
  > 게임성 자체를 중시하는 글로벌 유저를 끌어들이는 퍼포먼스 중심 캠페인 유형입니다.

---

## 7. 최종 “해석 가능한 잠재범주” 요약 표

수치 기반 특성과 LLM 기반 자연어 설명을 결합하여,  
각 클래스에 사람이 이해하기 쉬운 레이블을 붙일 수 있습니다.

| Class | 비공식 이름                         | 자연어 설명 (요약)                                                                 |
|-------|-------------------------------------|--------------------------------------------------------------------------------------|
| 1     | 보상형 하드코어 RPG 캠페인         | 강한 전투·보상 메시지, KR 중심, Video/Facebook, CTR·CVR 양호, 단기 프로모션 효과가 높은 유형 |
| 2     | 캐주얼 감성 브랜딩 캠페인          | 귀여운/힐링 컨셉, JP·Image·Google 비중, 성과는 낮지만 브랜드 이미지 제고에 적합한 유형       |
| 3     | 글로벌 체험형 퍼포먼스 캠페인      | Playable/TikTok 중심, 액션+캐주얼 혼합, US 비중 높고 ROAS가 매우 우수한 퍼포먼스 유형       |

---

## 8. 마케팅 관점에서의 인사이트 예시

이제 위 결과를 바탕으로, 실제 마케팅 의사결정 측면에서 다음과 같은 논의를 할 수 있습니다.

### 8.1. 예산 배분 전략

- ROAS가 가장 높은 Class 3(글로벌 체험형 퍼포먼스 캠페인)에 예산을 우선 배분
- Class 1은 단기 성과 확보용으로 유지
- Class 2는 성과는 낮지만, 장기적인 브랜드 인지도/이미지 구축 목적이라면 최소 수준으로 유지

### 8.2. 크리에이티브 방향성 설계

- 신규 국가/채널 런칭 시, 어떤 문구·소재 조합이 어떤 잠재 클래스에 속하는지 예측 가능
- 예:
  - 액션/보상 메시지를 강화하면 Class 1/3 쪽에 속할 가능성 증가
  - 귀여움/힐링 중심 문구는 Class 2 성격을 강화

### 8.3. 캠페인 설계 자동 추천 시스템으로 확장

- 새 광고 문구 초안을 작성하면:
  1. 텍스트 임베딩 계산
  2. PCA로 차원 축소
  3. 적합된 LCA 모형으로 posterior 계산  
     - 예: “이 문구는 Class 1에 속할 확률 0.7, Class 3에 0.2”
- 이를 바탕으로:
  - 이 문구가 어떤 유저 세그먼트/성과 패턴과 연관될지 사전 예측
  - 내부 대시보드에서 “이 카피는 하드코어 프로모션형, 예상 ROAS는 중간~상”과 같은 설명 제공 가능

---

## 9. 전체 흐름 요약

본 예시에서의 전체 분석 파이프라인은 다음과 같이 정리할 수 있습니다.

1. 데이터 준비
   - 범주형 변수: 국가(Country), 채널(Channel), 소재 유형(Creative_Type)
   - 연속형 변수: 성과지표 (CTR, CVR, 7d_ROAS)
   - 텍스트 변수: 광고 문구(Ad_Text)

2. 임베딩
   - LLM 임베딩으로 광고 문구를 고차원 벡터로 변환

3. 차원축소
   - PCA를 이용해 텍스트 임베딩을 저차원 연속형 특징(PC1~PC5 등)으로 변환

4. 혼합 LCA 모형 적합
   - 범주형 + 연속형 변수를 함께 사용하는 잠재범주 모형
   - EM 알고리즘으로 추정, BIC 등으로 클래스 개수 선택

5. 디베딩(debedding)
   - 각 잠재 클래스에 대해 대표 임베딩/대표 문구를 추출
   - LLM에 입력하여 사람 친화적인 자연어 설명 생성

6. 결과 해석 및 활용
   - “보상형 하드코어”, “캐주얼 브랜딩”, “글로벌 체험형 퍼포먼스” 등  
     해석 가능한 캠페인 유형 도출
   - 예산 배분, 크리에이티브 전략, 신규 캠페인 설계 자동 추천 등  
     실무 의사결정에 연결


---
*Author: Hyunsoo Shin*

*Affiliation: Department of Statistics, Sungkyunkwan University*
