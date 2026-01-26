# Interpretable Mixed-type Mixture Modeling via Heterogeneity Pursuit

## 1. Overview

본 프로젝트는 범주형 변수(예: 인구통계, 캠페인 속성)와 텍스트(비정형)로부터 얻어지는 고차원 임베딩을 결합하여, **설명 가능한 군집화(Interpretable Clustering)**를 수행하는 새로운 통계적 프레임워크를 제안합니다.

기존 연구들이 임베딩의 모든 차원을 사용하여 군집화를 수행함으로써 "어떤 의미적 특징 때문에 군집이 나뉘었는지" 설명하기 어려웠던 문제(Black-box)를 해결하기 위해, 본 연구는 **이질성 탐색(Heterogeneity Pursuit)** 기법을 도입합니다. 이를 통해 군집을 구분 짓는 **핵심 의미 차원(Heterogeneous Feature)**과 모든 군집이 공유하는 **공통 차원(Common Feature)**을 통계적으로 분리하고, 선별된 차원만을 **디베딩(De-embedding)**하여 명확한 해석을 제공합니다.

---
## 2. Motivation

### 기존 방법론의 한계

1. **BoW/나이브 베이즈의 한계:** [Shi et al. (2024)](https://www.google.com/search?q=https://doi.org/10.1214/24-AOAS1893) 등은 텍스트를 이진 벡터로 가정하여 문맥(Context)과 의미론적 뉘앙스를 놓칩니다.
    
2. **임베딩 군집화의 한계:** LLM 임베딩을 그대로 GMM에 적용할 경우, 수백 개의 차원이 모두 군집 형성에 관여하므로 사후 해석이 불가능하며, 차원의 저주로 인한 추정 불안정성이 발생합니다.

### 제안하는 접근법 (Our Approach)

우리는 **"PCA 차원 축소"**와 **"벌점화된 혼합 모형(Penalized Mixture Model)"**을 결합한 통합 프레임워크를 제안합니다.

- **PCA:** 선형성을 유지하면서 고차원 임베딩을 '잠재 의미 단위(PC)'로 압축합니다.
    
- **Heterogeneity Pursuit:** 군집 간 차이가 없는 PC는 0으로 수축(Shrinkage)시키고, 실제 군집을 가르는 PC만 선별합니다.

---


## 3. Methodology: Heterogeneity-Pursuing Joint Mixture Model

관측치 $i$는 범주형 데이터 $\mathbf{x}^{(c)}_i$와 텍스트 데이터 $\mathbf{z}_i$로 구성됩니다. 제안된 프레임워크는 **특징 추출(Feature Representation)**, **모형 적합(Modeling)**, **해석(Interpretation)**의 3단계로 구성됩니다.

```mermaid
graph TD
    subgraph Phase 1: Feature Representation
    A[Raw Text Z] -->|SBERT Embedding| B(Dense Vectors V)
    B -->|PCA Reduction| C(Reduced Embeddings X_e)
    D[Categorical Data X_c] --> E{Input Data}
    C --> E
    end

    subgraph Phase 2: Heterogeneity-Pursuing Joint Modeling
    E --> F[Generalized EM Algorithm]
    F -->|E-Step| G(Calculate Posteriors)
    F -->|M-Step| H[Update Parameters with Penalty]
    H --> I{Sparsity Estimation}
    I -->|Delta approx 0| J[Common Feature]
    I -->|Delta != 0| K[Heterogeneous Feature]
    end

    subgraph Phase 3: Interpretation
    K -->|Targeted De-embedding| L[Extract Semantic Keywords]
    J --> M[Ignore (Background Noise)]
    L --> N[Final Segment Profiling]
    end
```

### 3.1. Feature Representation (PCA-based)

- **텍스트 임베딩:** 사전 학습된 LLM(SBERT 등)을 사용하여 텍스트 $\mathbf{z}_i$를 고차원 벡터 $\mathbf{v}_i \in \mathbb{R}^{768}$로 변환합니다.
    
- **선형 차원 축소 (PCA):** 이질성 탐색 모형의 선형성 가정($\mu_0 + \delta_k$)을 유지하고 공분산 추정의 안정성을 확보하기 위해, PCA를 사용하여 상위 $D$개(예: $D=50$)의 주성분으로 축소합니다.
    
    $$\mathbf{x}^{(e)}_i = \text{PCA}(\mathbf{v}_i) \in \mathbb{R}^{D}$$
    

### 3.2. Model Specification (Heterogeneity Pursuit)

전체 모집단이 $K$개의 잠재 클래스로 구성된다고 가정합니다. 결합 확률 밀도 함수는 다음과 같습니다.

$$f(\mathbf{x}_i | \Theta) = \sum_{k=1}^{K} \pi_k \cdot f_{\text{cat}}(\mathbf{x}^{(c)}_i | \boldsymbol{\alpha}_k) \cdot f_{\text{cont}}(\mathbf{x}^{(e)}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

여기서 **연속형 부분(Penalized GMM)**에 핵심 아이디어가 적용됩니다. 각 군집의 평균 벡터 $\boldsymbol{\mu}_k$를 다음과 같이 분해합니다.

$$\boldsymbol{\mu}_k = \boldsymbol{\mu}_0 + \boldsymbol{\delta}_k, \quad \text{subject to } \sum_{k=1}^{K} \boldsymbol{\delta}_k = \mathbf{0}$$

- $\boldsymbol{\mu}_0$: 전체 모집단의 **공통 평균 (Common Mean)**
    
- $\boldsymbol{\delta}_k$: 군집 $k$의 **특이 편차 (Cluster-specific Deviation)**. 즉, **이질성의 원인**입니다.
    

### 3.3. Penalized Estimation via Generalized EM

불필요한 차원의 편차를 0으로 만들기 위해, 로그 우도 함수에 **적응형 라소(Adaptive Lasso)** 페널티를 적용한 목적 함수 $\mathcal{Q}$를 최대화합니다.

$$\max_{\Theta} \left\{ \sum_{i=1}^{n} \log f(\mathbf{x}_i | \Theta) - n\lambda \sum_{j=1}^{D} \sum_{k=1}^{K} w_{jk} |\delta_{jk}| \right\}$$

이를 위해 **Generalized EM 알고리즘**을 수행합니다.

1. **E-step:** 각 관측치의 군집 소속 확률(Responsibility) $\gamma_{ik}$ 계산.
    
2. **M-step (Coordinate Descent):**
    
    - $\pi_k, \boldsymbol{\alpha}_k, \boldsymbol{\Sigma}_k$는 기존 방식대로 업데이트.
        
    - $\boldsymbol{\mu}_0, \boldsymbol{\delta}_k$는 제약 조건($\sum \delta=0$) 하에서 벌점화된 가중 최소자승 문제를 푸는 **좌표 하강법**으로 업데이트.
        
    - 결과적으로 중요하지 않은 차원 $j$에 대해 $\hat{\delta}_{jk}$는 정확히 **0으로 수축(Shrinkage)**됩니다.
        

---

## 4. Interpretation via Targeted De-embedding

모델링 결과 살아남은(Non-zero) $\delta$를 가진 차원만을 대상으로 해석을 수행합니다. 이는 해석의 노이즈를 획기적으로 줄여줍니다.

### 4.1. 이질성 원인 식별 (Identification)

- **Common Dimensions ($\hat{\delta} \approx 0$):** 군집 간 차이가 없는 배경 정보(Background Noise). 해석에서 제외합니다.
    
- **Heterogeneous Dimensions ($\hat{\delta} \neq 0$):** 군집을 나누는 핵심 요인. 집중 해석 대상입니다.
    

### 4.2. Targeted Method B: Linear Decoder

선별된 **이질성 차원(PC)**에 대해서만 원본 단어와의 관계를 역추적합니다.

$$\text{Keywords}(PC_j) = \text{Top-weighted words in } \mathbf{w}_j = \mathbf{V}^T \mathbf{e}_j$$

_(여기서 $\mathbf{V}$는 PCA 로딩 행렬)_

이를 통해 **"PC 3는 '보상'과 관련되며, 군집 1에서 양의 값($\delta_1 > 0$)을 가지므로 군집 1은 보상 지향 그룹이다"**와 같은 명확한 해석이 가능합니다.

---

## 5. Simulation Study Plan

본 방법론의 우수성을 검증하기 위해 다음의 비교 실험을 설계합니다.

1. **DGP (Data Generating Process):** * $p=50$개의 임베딩 차원 중 $q=5$개 차원만 군집별 평균이 다르고($\delta \neq 0$), 나머지는 동일($\delta = 0$)하게 데이터 생성.
    
2. **Comparison Models:**
    
    - **Model A (Standard GMM):** 모든 차원을 사용하여 군집화 (Over-fitting 예상).
        
    - **Model B (Two-step PCA+GMM):** 분산 기준 상위 PC만 선택 (중요하지만 분산이 작은 변수 누락 위험).
        
    - **Model C (Proposed):** PCA + Heterogeneity Pursuit 적용.
        
3. **Metrics:**
    
    - **Selection Consistency:** 실제 이질적 차원($q=5$)을 정확히 $\delta \neq 0$으로 추정했는지 여부 (TPR/FPR).
        
    - **Clustering Accuracy:** ARI (Adjusted Rand Index).
        

---

## 6. Case Study: 게임 마케팅 데이터 적용 예시

### 6.1. 가상의 분석 결과 (Expected Outcome)

모바일 RPG 게임 광고 데이터(1,000건)에 본 방법론을 적용했을 때의 예상 결과입니다.

**[Table 1] 이질성 탐색 결과 ($\hat{\delta}$ 추정치)**

|**임베딩 차원**|**δ^1​ (Cluster A)**|**δ^2​ (Cluster B)**|**δ^3​ (Cluster C)**|**판단 (Decision)**|
|---|---|---|---|---|
|**PC 1**|**+2.5**|**-1.2**|**-1.3**|**이질성 원인 (Heterogeneous)**|
|**PC 3**|-0.8|-0.2|**+1.0**|**이질성 원인 (Heterogeneous)**|
|**PC 4**|0.0|0.0|0.0|**공통 차원 (Common)** $\rightarrow$ _제외_|
|**PC 5**|0.0|0.0|0.0|**공통 차원 (Common)** $\rightarrow$ _제외_|
### 6.2. 최종 프로파일링 (Profiling)

선별된 PC 1, PC 3에 대해서만 디베딩을 수행하여 군집을 정의합니다.

- **PC 4, 5 (공통):** 'RPG', '설치', '레벨업' 등 일반적인 게임 용어로 판명됨. (세그먼트 구분에 무의미)
    
- **Cluster A (PC 1 $\uparrow$):** 키워드 '보상, 쿠폰, 100% 지급' $\rightarrow$ **"체리피커형 보상 그룹"**
    
- **Cluster C (PC 3 $\uparrow$):** 키워드 '타격감, 그래픽, 엔진' $\rightarrow$ **"비주얼 중시 하드코어 그룹"**

---

## 7. Conclusion & Contribution

본 연구는 **이질성 탐색(Heterogeneity Pursuit)**을 비지도 학습 영역으로 확장하여, 텍스트 임베딩 군집화의 고질적인 문제인 **해석 불가능성**을 해결했습니다.

1. **통계적 변수 선택:** 주관적인 판단이 아닌, 벌점화 우도(Penalized Likelihood)를 통해 군집을 가르는 핵심 의미 차원을 자동으로 선별합니다.
    
2. **효율적 해석:** 모든 차원을 해석하려 드는 비효율을 제거하고, 통계적으로 유의미한 차이($\delta \neq 0$)에만 집중하여 설명의 정확도(XAI)를 높였습니다.
    
3. **안정적 프레임워크:** PCA와 결합하여 고차원 데이터에서도 안정적인 공분산 추정과 선형성 유지를 달성했습니다.

---
_Author: Hyunsoo Shin_ _Affiliation: Department of Statistics, Sungkyunkwan University_
