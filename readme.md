# 서울 데이트 코스 추천 시스템 (Seoul Date Course Recommendation System)

> **콘텐츠 기반 필터링(Ridge)과 협업 필터링(KNNWithMeans, SVD++) 하이브리드 결합 및 도보 이동 시간 제약 기반 4스톱 데이트 코스 추천 파이프라인**

---

## 1. Introduction (프로젝트 개요)

본 프로젝트는 서울 지역 장소 데이터와 사용자의 취향·평점 이력을 결합하여 **최적의 4곳 데이트 코스를 추천하는 머신러닝 기반 추천 시스템**입니다.

### 주요 목표 및 해결 과제
* **개인화된 장소 평점 예측:** 사용자의 명시적 취향(카테고리 관심도, 선호 자치구)과 과거 평점 이력을 결합하여 미방문 장소에 대한 1~5점 척도의 예상 평점을 산출합니다.
* **하이브리드 추천 모델링:** 콘텐츠 기반 필터링(Content-Based)과 협업 필터링(Collaborative Filtering)의 장점을 결합하고, 검증셋(Validation) 기반으로 최적의 앙상블 가중치를 탐색합니다.
* **신규 사용자(Cold Start) 대응:** 평점 기록이 없는 신규 사용자 입력 시에도 콘텐츠 기반 예측 및 유사 취향 기존 사용자 기반 대리 추정(CF Proxy)을 통해 즉시 추천을 제공합니다.
* **실용적인 코스 동선 제약 반영:** 단순히 평점이 높은 장소 4곳을 나열하는 것이 아니라, 구간별 도보 이동 시간과 총 이동 시간 제약을 충족하는 후보 코스 중 총 예측 평점이 가장 높은 코스를 선정합니다.

---

## 2. System Architecture & Pipeline (전체 시스템 구조)

전체 파이프라인은 크게 **① 개별 장소 평점 예측 단계**와 **② 4스톱 코스 필터링 및 랭킹 단계**로 구성됩니다.

```text
[입력 데이터]
  ├── 장소 메타데이터 (places_for_recsys.csv)
  ├── 사용자 프로필 (users_synthetic.csv)
  ├── 평점 이력 (ratings_train / validation / test.csv)
  └── 4곳 후보 코스 풀 (four_stop_courses_estimated.csv)
         │
         ▼
[1단계: 개별 장소 예측 평점 산출]
  ├── 콘텐츠 기반 모델 (Ridge 회귀: 카테고리 유사도 + 지역 근접도 + 평점 후함 정도)
  └── 협업 필터링 모델 (Item-based KNNWithMeans + SVD++ 가중 앙상블)
         │
         ▼  (Validation RMSE 기준 최적 가중치 β 탐색 및 결합)
[2단계: 코스 필터링 및 최종 랭킹]
  ├── 기방문(평가) 장소 제외 및 4곳 중복 제거
  ├── 이동 시간 제약 적용 (구간별 도보 ≤ 40분 & 총 도보 ≤ 75분)
  └── 예측 평점 합산(rating_sum) 내림차순, 총 이동 시간 오름차순 정렬
```

### 주요 모듈 구성
| 파일명 | 역할 및 주요 기능 |
| :--- | :--- |
| **`contents_rec.py`** | 콘텐츠 기반 필터링 코어 모듈. 피처 엔지니어링, 모델 변형(A~F) 비교, Leave-One-Out 누수 방지, 동점 처리(Tie-break) 및 Top-K 평가, 신규 사용자 추천 함수(`recommend_for_new_user`) 제공 |
| **`final_course_pipeline.py`** | 기존 사용자 대상 통합 파이프라인. 콘텐츠 기반(Ridge) + 협업 필터링(KNNWithMeans, SVD++) 하이브리드 가중치 탐색 및 특정 사용자(`--user-id`)의 최적 4곳 도보 코스 산출 |
| **`new_user_both_models.py`** | 신규 사용자 대상 대화형 하이브리드 추천. 카테고리 선호도 및 선호 자치구를 입력받아 콘텐츠 점수 + 유사 이웃 기반 CF 대리 추정 점수를 결합하고 단계별 지역 확장을 통해 코스 추천 |
| **`new_user_course.py`** | 신규 사용자 대상 경량 콘텐츠 기반 추천. 평점 이력 없이 오직 입력받은 취향(명소/문화/쇼핑)과 선호 자치구만으로 콜드 스타트 코스 추천 수행 |

---

## 3. Dataset & Preprocessing (데이터셋 및 전처리)

### 입력 데이터 명세
* **`places_for_recsys.csv`**: 서울 지역 장소 1,047개 데이터 (`place_id`, `name`, `category`, `district`, `latitude`, `longitude`, `address`)
* **`users_synthetic.csv`**: 가상 사용자 2,000명 프로필 (`user_id`, `interest_명소`, `interest_문화`, `interest_쇼핑`, `preferred_district`, `rating_generosity`)
* **`ratings_train.csv` / `ratings_validation.csv` / `ratings_test.csv`**: 사용자-장소 평점 데이터 (1~5점)
* **`four_stop_courses_estimated.csv`**: 사전에 생성된 4개 장소 조합 후보 코스 및 구간별 추정 도보 시간 (`leg1_min`, `leg2_min`, `leg3_min`, `total_estimated_minutes`)

### 전처리 및 피처 엔지니어링
1. **결측치 보존 처리:** 자치구(`district`, `preferred_district`) 결측값을 제거하지 않고 `"미상"`으로 대체하여 추천 대상에서 누락되지 않도록 처리하되, 원-핫 인코딩 차원에서는 제외하여 미상끼리의 오매칭을 방지했습니다.
2. **멀티 블록 원-핫 벡터화:** 카테고리는 3차원(`명소`, `문화`, `쇼핑`), 자치구는 25차원으로 분리 구성하여 각 블록별 코사인 유사도를 독립적으로 산출합니다.
3. **지수 감쇠 기반 지역 근접도 (`exp_near`):** 동일 자치구 여부만 따질 때 발생하는 경계 지역 불이익을 보완하기 위해, 자치구별 중앙값(Median) 좌표를 기준으로 지수 감쇠 근접도를 계산합니다.
   $$\text{Proximity} = \exp\left(-\frac{d_{\text{km}}}{s_{\text{km}}}\right)$$
   *(검증셋 실험을 통해 최적 기준 거리 $s_{\text{km}} = 3\text{km}$ 도출)*

---

## 4. Recommendation Methodology (추천 모델링 및 코스 최적화)

### 4.1 콘텐츠 기반 필터링 (`contents_rec.py`)
* **명시적(Explicit) & 학습된(Learned) 프로필:** 사용자의 초기 관심도 벡터 외에도 과거 평점 편차($r_{u,i} - \bar{r}_u$)를 가중치로 합산한 학습 프로필을 구축했습니다. 학습 시 타깃 행이 피처에 포함되는 데이터 누수(Data Leakage)를 막기 위해 **Leave-One-Out(LOO)** 연산을 적용했습니다.
* **모델 변형(A~F) 비교 및 Ridge 회귀:** 피처 조합별(모델 A~F)로 `Ridge` 회귀를 학습하고, 검증 RMSE 기준으로 최적의 피처 조합(`exp_cat`, `exp_near`, `generosity`)과 규제 계수($\alpha$)를 선택합니다.

### 4.2 협업 필터링 및 하이브리드 앙상블 (`final_course_pipeline.py`)
* **CF 내부 앙상블:** 아이템 기반 `KNNWithMeans`(코사인 유사도, $k=40$)와 잠재 요인 모델 `SVD++`($n_{\text{factors}}=40$)의 예측값을 가중 평균($\alpha_{\text{cf}} \cdot \hat{r}_{\text{KNN}} + (1 - \alpha_{\text{cf}}) \cdot \hat{r}_{\text{SVD++}}$)합니다.
* **최종 하이브리드 결합:** 검증셋 그리드 탐색(0.05 단위)을 통해 콘텐츠 기반 예측($\hat{r}_{\text{CB}}$)과 협업 필터링 예측($\hat{r}_{\text{CF}}$)의 최적 결합 비율 $\beta$를 도출합니다.
  $$\hat{r}_{\text{final}} = \beta \cdot \hat{r}_{\text{CB}} + (1 - \beta) \cdot \hat{r}_{\text{CF}}$$

### 4.3 신규 사용자 콜드 스타트 대응 (`new_user_both_models.py`)
* **CF 대리 추정 (Proxy Estimation):** 신규 사용자에게 임의의 기존 ID를 부여하지 않고, 입력받은 카테고리 취향 유사도(80%)와 동일 자치구 여부(20%)로 상위 $k$명의 유사 기존 사용자를 찾아 이들의 CF 예측값을 유사도 가중 평균하여 반영합니다.
* **지역 우선순위 단계별 탐색:** 선택한 자치구 내에 이동 제약을 만족하는 4곳 코스가 없을 경우, ① 선택 자치구 내부 $\rightarrow$ ② 중심점 기준 인접 5개 자치구 포함 $\rightarrow$ ③ 서울 전체로 탐색 범위를 자동 완화합니다.

### 4.4 4곳 데이트 코스 선정 조건
* **필터링 제약:** 4개 장소 모두 서로 다른 장소일 것, 이미 평가한 장소를 포함하지 않을 것, 각 구간 도보 이동 시간 $\le 40$분, 총 도보 이동 시간 $\le 75$분.
* **최적화 목적함수:** 제약 조건을 만족하는 코스 중 4곳의 예측 평점 합(`rating_sum`)을 최대화하며, 동점일 경우 총 이동 시간(`total_estimated_minutes`)이 짧은 코스를 우선 추천합니다.

---

## 5. Evaluation & Results (실험 및 평가 지표)

모델의 객관적 성능 검증을 위해 평점 예측 정확도와 Top-K 랭킹 품질을 다각도로 평가합니다. (최종 평가 전 `Train + Validation` 통합 재학습 후 `Test` 셋은 단 1회만 평가)

* **평점 예측 지표:** `MAE` (Mean Absolute Error), `RMSE` (Root Mean Squared Error)
* **랭킹 및 추천 다양성 지표:** `Precision@10`, `Recall@10`, `NDCG@10`, `HitRate@10`, `Coverage@10` (4점 이상을 정답(Relevant)으로 간주)
* **기준선(Baseline) 비교:** 전체 평균 예측, 사용자별 평균 예측, 무작위 추천, 인기순(평가 수 기준) 추천과 대비하여 성능 향상폭 검증
* **동점 처리(Tie-break) 최적화:** 동일 예측 점수 발생 시 `np.lexsort`를 활용해 보조 기준(평균 평점순 / 평가 수순)에 따른 검증 `NDCG@10` 최적화

---

## 6. Project Structure (폴더 및 파일 구조)

```text
project_root/
├── contents_rec.py                 # 콘텐츠 기반 추천 학습, 평가 및 점수표 생성
├── final_course_pipeline.py        # 기존 사용자 하이브리드 추천 및 4곳 코스 선정
├── new_user_both_models.py         # 신규 사용자 대화형 추천 (콘텐츠 + CF 대리 추정)
├── new_user_course.py              # 신규 사용자 대화형 추천 (콘텐츠 기반 단독)
│
├── 02_콘텐츠기반_추천/
│   ├── places_for_recsys.csv       # 서울 장소 메타데이터
│   ├── users_synthetic.csv         # 가상 사용자 프로필 데이터
│   ├── ratings_validation.csv      # 검증용 평점 데이터
│   ├── ratings_test.csv            # 테스트용 평점 데이터
│   └── output/                     # contents_rec.py 실행 결과 (점수표, 비교표, metrics.json)
│
├── 03_협업필터링_하이브리드/
│   └── ratings_train.csv           # 학습용 평점 데이터
│
└── four_stop_courses_estimated.csv # 4개 장소 후보 코스 및 추정 이동 시간 데이터
```

---

## 7. Getting Started / Usage (설치 및 실행 방법)

### 환경 설정 및 필수 패키지 설치
Python 3.8 이상 환경을 권장하며, 아래 라이브러리가 필요합니다.
```bash
pip install numpy pandas scikit-learn scikit-surprise
```

### 실행 방법

**1. 콘텐츠 기반 모델 학습·비교 및 전체 점수표 생성**
```bash
python3 contents_rec.py
```
* 실행 결과는 `02_콘텐츠기반_추천/output/` 디렉터리에 CSV 및 JSON 형태로 저장됩니다.

**2. 기존 사용자 대상 하이브리드 모델 평가 및 4곳 데이트 코스 추천**
```bash
python3 final_course_pipeline.py --data-root . --user-id U0001 --output-dir final_course_output
```
* `--user-id`: 추천을 생성할 기존 사용자 ID (기본값: `U0001`)

**3. 신규 사용자 대상 대화형 코스 추천 (하이브리드 결합 버전)**
```bash
python3 new_user_both_models.py --data-root . --neighbors 20 --output-dir new_user_both_output
```
* 실행 후 터미널 안내에 따라 `명소 / 문화 / 쇼핑` 선호도(0~10)와 `선호 자치구`를 입력합니다.

**4. 신규 사용자 대상 대화형 코스 추천 (콘텐츠 기반 단독 버전)**
```bash
python3 new_user_course.py --data-root . --output-dir new_user_output
```

---

## 8. Limitations & Future Work (한계점 및 개선 방향)

* **도보 이동 시간 추정치의 한계:** 현재 후보 코스의 도보 시간(`leg1_min`~`leg3_min`)은 실제 도로망 기반 길찾기 검증값이 아닌 직선거리 보정 추정치입니다. 향후 카카오모빌리티, TMAP, OSRM 등 실제 보행자 경로 API와 연동하여 동선 정확도를 높일 수 있습니다.
* **합성 사용자 데이터 기반 학습:** `users_synthetic.csv` 및 합성 평점 분포를 기반으로 검증되었으므로, 실제 서비스 로그(클릭, 체류 시간, 실제 방문 리뷰) 수집 후 온라인 A/B 테스트를 통한 가중치 재보정이 필요합니다.
* **영업시간 및 동적 제약 반영:** 장소별 운영 시간, 휴무일, 식사/카페/활동 등 카테고리 순서 배치(예: 명소 $\rightarrow$ 식사 $\rightarrow$ 카페) 규칙을 코스 필터링 단계에 추가하면 추천 품질을 한층 고도화할 수 있습니다.