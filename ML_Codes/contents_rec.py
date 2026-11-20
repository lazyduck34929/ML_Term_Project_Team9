"""
서울 데이트 코스 추천 - 콘텐츠 기반 추천 파트 (ML 텀프로젝트 9조)

대략적인 흐름
  1. 장소를 벡터로 바꿈 (카테고리 원-핫 + 자치구 원-핫)
  2. 사용자도 같은 형태의 벡터로 바꿈
       - 명시적 프로필: users 파일에 있는 관심도, 선호 자치구를 그대로 사용
       - 학습된 프로필: 과거 평점을 보고 취향을 거꾸로 추정한 것
  3. 사용자-장소 코사인 유사도 계산 + 선호 지역에서 장소까지 거리로 근접도 계산
  4. 3번 값들을 Ridge 회귀에 넣어서 1~5점짜리 예측 평점으로 변환
  5. 예측 평점 높은 순으로 Top-K 추천
  6. MAE, RMSE, Precision/Recall/NDCG@K, Coverage로 평가
  7. 하이브리드/경로 파트에서 쓸 수 있게 점수표를 csv로 저장

실행: python3 contents_rec.py

읽는 파일
  02_콘텐츠기반_추천/places_for_recsys.csv    장소 1,047개
  02_콘텐츠기반_추천/users_synthetic.csv      가상 사용자 2,000명
  02_콘텐츠기반_추천/ratings_validation.csv   검증용 평점
  02_콘텐츠기반_추천/ratings_test.csv         테스트용 평점
  03_협업필터링_하이브리드/ratings_train.csv  학습용 평점
    -> 내 폴더엔 train이 없어서 협업필터링 폴더 걸 가져다 씀.
       validation/test는 두 폴더 파일이 똑같아서 협업필터링이랑 같은 조건으로 비교 가능

만드는 파일 (02_콘텐츠기반_추천/output/)
  cb_scores_all.csv             전체 (사용자, 장소) 점수표
  cb_top20_per_user.csv         사용자별 추천 20개
  cb_model_comparison.csv       모델 A~F 검증 성능
  cb_tie_break_comparison.csv   동점 처리 방식별 검증 성능
  cb_near_scale_comparison.csv  근접도 기준거리별 검증 성능
  cb_final_metrics.json         최종 모델 테스트 성능, 계수 등
"""

import json
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use("Agg")  # Save PNG files without opening an interactive window
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 설정값
# ---------------------------------------------------------------------------
# 경로는 이 파일 위치 기준으로 잡음. 다른 폴더에서 실행해도 안 깨지게 하려고.
BASE_DIR = Path(__file__).resolve().parent
CB_DIR = BASE_DIR / "02_콘텐츠기반_추천"       # 내 폴더
CF_DIR = BASE_DIR / "03_협업필터링_하이브리드"  # train 평점은 여기 있음
OUT_DIR = CB_DIR / "output"                    # 결과 저장 위치

CATEGORIES = ["명소", "문화", "쇼핑"]  # users 파일의 interest_명소/문화/쇼핑이랑 순서 맞춤
UNKNOWN = "미상"                      # 자치구 비어 있을 때 채우는 값

RELEVANT_THRESHOLD = 4   # 4점 이상이면 좋아한 장소로 침 (Top-K 평가 정답 기준)
TOP_K = 10               # 평가할 때 K
TOP_N_SAVE = 20          # 사용자마다 저장할 추천 개수
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]  # Ridge alpha 후보. 검증셋으로 고름
EPS = 1e-12              # 0으로 나누기 방지용
NEAR_SCALES_KM = [1, 2, 3, 5, 8, 10]  # 근접도 기준거리(km) 후보. 이것도 검증셋으로 고름
KM_PER_DEGREE = 111.0    # 위도 1도 ≈ 111km
TIE_DECIMALS = 9         # 소수 9자리까지 같으면 동점으로 봄 (부동소수점 오차 흡수용)

# 맥에서 numpy 행렬곱 할 때 divide by zero / overflow 경고가 뜨는데 결과는 멀쩡함.
# NaN, inf 없는지는 main()에서 assert로 따로 확인하니까 이 경고만 꺼둠.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

# 한글은 터미널에서 2칸을 차지하는데 pandas는 1칸으로 계산해서 표가 삐뚤게 나옴.
# 이 옵션 켜면 한글 폭까지 고려해서 맞춰줌.
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 200)  # 표가 중간에 잘려서 줄바꿈되는 것 방지


def pad(text, width):
    """
    터미널에 보이는 폭 기준으로 오른쪽에 공백을 채움.
    f"{text:<14}" 로 하면 글자 수로만 세서 한글 있는 줄이 밀림. 한글은 2칸으로 계산.
    """
    text = str(text)
    shown = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - shown, 0)


# ---------------------------------------------------------------------------
# 1. 데이터 불러오기
# ---------------------------------------------------------------------------
def load_data():
    """csv 5개 읽고 결측만 간단히 처리해서 반환."""
    places = pd.read_csv(CB_DIR / "places_for_recsys.csv")
    users = pd.read_csv(CB_DIR / "users_synthetic.csv")
    train = pd.read_csv(CF_DIR / "ratings_train.csv")
    valid = pd.read_csv(CB_DIR / "ratings_validation.csv")
    test = pd.read_csv(CB_DIR / "ratings_test.csv")

    # 자치구 없는 장소가 10곳, 선호 자치구 없는 사용자가 90명 있음.
    # 지워버리면 그 장소/사용자는 추천 자체가 안 되니까 "미상"으로 채워서 살려둠.
    # (미상끼리 같은 지역으로 묶이면 안 되니까 벡터 만들 때 미상은 빼고 만듦)
    places["district"] = places["district"].fillna(UNKNOWN)
    users["preferred_district"] = users["preferred_district"].fillna(UNKNOWN)

    # data_origin, generation_seed는 전부 같은 값이라 쓸모없음. 필요한 컬럼만 남김
    cols = ["user_id", "place_id", "rating"]
    return places, users, train[cols], valid[cols], test[cols]


# ---------------------------------------------------------------------------
# 2. 장소 벡터, 사용자 벡터(명시적 프로필)
# ---------------------------------------------------------------------------
def build_place_features(places):
    """
    장소를 원-핫 벡터 두 덩어리로 만듦.
      카테고리: [명소, 문화, 쇼핑] 3차원  (문화면 [0, 1, 0])
      자치구  : 25차원

    둘을 한 벡터로 안 합친 이유: 합치면 카테고리랑 지역 중에 뭐가 더 중요한지를
    내가 임의로 정해야 함. 따로 유사도를 구해놓고 회귀가 가중치를 알아서 찾게 하려고 나눔.
    """
    # 미상은 실제 구가 아니니까 차원에서 제외 (미상인 장소는 자치구 벡터가 전부 0)
    districts = sorted(d for d in places["district"].unique() if d != UNKNOWN)

    # (장소 수, 3)
    cat_mat = np.stack(
        [(places["category"] == c).to_numpy(dtype=float) for c in CATEGORIES], axis=1
    )
Draggable item chatgpt:project:g-p-6adb15f1f2408191ac039d4059cc4154 was dropped over droppable area chatgpt:conversation:6af955a1-5628-83ec-8bab-ece30d9487c8




