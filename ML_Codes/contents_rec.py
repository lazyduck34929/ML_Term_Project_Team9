"""
=============================================================================
 서울 데이트 코스 추천 시스템 - 콘텐츠 기반 추천 (Content-Based Filtering)
 Machine Learning Term Project / 9조
=============================================================================

[이 파일이 하는 일]
  1) 장소 속성(카테고리, 자치구)을 벡터로 만든다.                -> 장소 벡터
  2) 사용자 취향을 같은 공간의 벡터로 만든다.                    -> 사용자 벡터
       - 명시적 프로필 : 사용자가 직접 밝힌 관심도 / 선호 자치구
       - 학습된 프로필 : 과거 평점으로부터 추정한 취향
  3) 사용자 벡터와 장소 벡터의 코사인 유사도를 계산한다.         -> 유사도 피처
     선호 지역 중심에서 장소까지의 거리로 지역 근접도도 계산한다. -> 근접도 피처
  4) 유사도 피처를 Ridge 회귀에 넣어 1~5점 예측 평점으로 바꾼다. -> 예측 평점
  5) 예측 평점이 높은 순서로 Top-K 장소를 추천한다. (동점은 검증으로 고른 규칙으로 정렬)
  6) MAE / RMSE / Precision@K / Recall@K / NDCG@K / Coverage 로 평가한다.
  7) 하이브리드 담당 / 경로 담당이 쓸 수 있도록 점수표를 CSV로 저장한다.

[실행 방법]
  python3 contents_rec.py

[입력 파일]
  02_콘텐츠기반_추천/places_for_recsys.csv   : 장소 1,047개
  02_콘텐츠기반_추천/users_synthetic.csv     : 합성 사용자 2,000명
  02_콘텐츠기반_추천/ratings_validation.csv  : 검증용 평점
  02_콘텐츠기반_추천/ratings_test.csv        : 테스트용 평점
  03_협업필터링_하이브리드/ratings_train.csv : 학습용 평점
      (내 폴더에는 train 파일이 없어서 협업필터링 폴더의 것을 함께 쓴다.
       두 폴더의 validation / test 파일은 완전히 같은 파일이므로
       협업필터링 담당과 동일한 분할로 성능을 비교할 수 있다.)

[출력 파일]  (02_콘텐츠기반_추천/output/ 폴더)
  cb_scores_all.csv       : 모든 (사용자, 장소) 쌍의 콘텐츠 기반 점수
  cb_top20_per_user.csv   : 사용자별 추천 상위 20개 장소
  cb_model_comparison.csv : 모델 변형별 검증 성능 비교표
  cb_tie_break_comparison.csv : 동점 처리 규칙별 검증 Top-K 성능 비교표
  cb_near_scale_comparison.csv : 지역 근접도 기준거리별 검증 성능 비교표
  cb_final_metrics.json   : 최종 모델의 테스트 성능과 설정값
"""

import json
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# =============================================================================
# 0. 설정값
# =============================================================================
# 이 스크립트 파일이 있는 폴더를 기준으로 경로를 잡는다.
# (어느 위치에서 실행해도 같은 파일을 찾도록 하기 위함)
BASE_DIR = Path(__file__).resolve().parent
CB_DIR = BASE_DIR / "02_콘텐츠기반_추천"       # 내 담당 폴더
CF_DIR = BASE_DIR / "03_협업필터링_하이브리드"  # train 평점이 들어 있는 폴더
OUT_DIR = CB_DIR / "output"                    # 결과 저장 폴더

CATEGORIES = ["명소", "문화", "쇼핑"]  # 장소 카테고리 3종 (users의 interest_* 컬럼과 대응)
UNKNOWN = "미상"                      # 자치구 결측값을 대신할 이름

RELEVANT_THRESHOLD = 4   # 평점이 4점 이상이면 "사용자가 좋아한 장소"로 본다 (Top-K 평가용)
TOP_K = 10               # Top-K 평가에 쓸 K
TOP_N_SAVE = 20          # 사용자별로 저장할 추천 개수
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]  # Ridge 정규화 세기 후보 (검증셋으로 고른다)
EPS = 1e-12              # 0으로 나누는 것을 막기 위한 아주 작은 값
NEAR_SCALES_KM = [1, 2, 3, 5, 8, 10]  # 지역 근접도의 거리 감쇠 기준(km) 후보 (검증셋으로 고른다)
KM_PER_DEGREE = 111.0    # 위도 1도에 해당하는 거리(km)
TIE_SCALE = 1e-6         # 동점 처리용 보조 점수에 곱하는 아주 작은 수 (본 점수의 순서는 바꾸지 않는다)

# macOS 의 일부 numpy 버전은 행렬 곱(matmul)에서 실제로는 문제가 없는데도
# "divide by zero / overflow / invalid value" 경고를 출력한다.
# 계산 결과에 NaN·무한대가 없음을 main() 안에서 직접 검사하므로 이 경고만 숨긴다.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

# 한글은 터미널에서 영문·숫자보다 두 배 넓게(2칸) 표시된다.
# pandas 는 기본적으로 모든 글자를 1칸으로 계산해서 표를 맞추기 때문에
# 한글이 섞인 표는 열이 어긋나 보인다. 아래 옵션을 켜면 글자 폭을 반영해 정렬한다.
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 200)  # 표가 중간에 줄바꿈되지 않도록 한 줄 폭을 넉넉히 잡는다


def pad(text, width):
    """
    문자열 뒤에 공백을 붙여 터미널에서 보이는 폭을 width 칸으로 맞춘다.
    한글처럼 넓은 글자(W, F)는 2칸, 나머지는 1칸으로 센다.
    (파이썬의 f"{text:<14}" 는 글자 '개수'로 맞추기 때문에 한글이 있으면 어긋난다.)
    """
    text = str(text)
    shown = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - shown, 0)


# =============================================================================
# 1. 데이터 불러오기
# =============================================================================
def load_data():
    """CSV 5개를 읽고 결측값을 정리해서 돌려준다."""
    places = pd.read_csv(CB_DIR / "places_for_recsys.csv")
    users = pd.read_csv(CB_DIR / "users_synthetic.csv")
    train = pd.read_csv(CF_DIR / "ratings_train.csv")
    valid = pd.read_csv(CB_DIR / "ratings_validation.csv")
    test = pd.read_csv(CB_DIR / "ratings_test.csv")

    # 자치구가 비어 있는 장소(10곳)와 선호 자치구가 없는 사용자(90명)가 있다.
    # 행을 지우면 그 장소/사용자를 추천에서 아예 못 쓰게 되므로,
    # 지우지 않고 "미상"이라는 별도 값으로 채운다.
    # ("미상"끼리는 지역이 일치한다고 보지 않도록 뒤에서 따로 처리한다.)
    places["district"] = places["district"].fillna(UNKNOWN)
    users["preferred_district"] = users["preferred_district"].fillna(UNKNOWN)

    # 평점 파일에서 실제로 쓰는 컬럼만 남긴다.
    # (data_origin, generation_seed 는 모든 행이 같은 값이라 정보가 없다.)
    cols = ["user_id", "place_id", "rating"]
    return places, users, train[cols], valid[cols], test[cols]


# =============================================================================
# 2. 장소 벡터 / 사용자 명시적 프로필 벡터 만들기
# =============================================================================
def build_place_features(places):
    """
    장소를 두 개의 원-핫(one-hot) 블록으로 표현한다.

      카테고리 블록 : [명소, 문화, 쇼핑]            -> 3차원
      자치구 블록   : [강남구, 강동구, ..., 중랑구] -> 25차원

    원-핫이란 해당하는 칸만 1이고 나머지는 0인 벡터다.
    예) 문화 카테고리 -> [0, 1, 0]

    카테고리와 자치구를 한 벡터로 합치지 않고 블록을 나눠 두는 이유:
    블록별 유사도를 따로 계산한 뒤 "카테고리가 얼마나 중요한지",
    "지역이 얼마나 중요한지"를 회귀 모델이 데이터에서 직접 배우게 하기 위해서다.
    """
    # 자치구 목록 ("미상"은 실제 지역이 아니므로 차원에서 뺀다 -> 해당 장소는 0벡터)
    districts = sorted(d for d in places["district"].unique() if d != UNKNOWN)

    # 카테고리 원-핫 : (장소 수 x 3) 행렬
    cat_mat = np.stack(
        [(places["category"] == c).to_numpy(dtype=float) for c in CATEGORIES], axis=1
    )
    # 자치구 원-핫 : (장소 수 x 25) 행렬
    dist_mat = np.stack(
        [(places["district"] == d).to_numpy(dtype=float) for d in districts], axis=1
    )
    return cat_mat, dist_mat, districts


def build_explicit_user_profiles(users, districts):
    """
    사용자가 직접 밝힌 취향으로 만드는 "명시적 프로필".
    평점 이력이 하나도 없는 신규 사용자에게도 만들 수 있다는 것이 장점이다.

      카테고리 블록 : [interest_명소, interest_문화, interest_쇼핑] (합이 1인 관심도)
      자치구 블록   : 선호 자치구의 원-핫
    """
    cat_prof = users[[f"interest_{c}" for c in CATEGORIES]].to_numpy(dtype=float)
    dist_prof = np.stack(
        [(users["preferred_district"] == d).to_numpy(dtype=float) for d in districts],
        axis=1,
    )
    return cat_prof, dist_prof


# =============================================================================
# 3. 학습된 프로필 (과거 평점으로 추정한 취향)
# =============================================================================
class LearnedProfiles:
    """
    사용자가 과거에 매긴 평점으로 취향 벡터를 추정한다.

      사용자 프로필 = Σ (평점 - 그 사용자의 평균 평점) × 장소 벡터

    평균을 빼는 이유:
      평점을 후하게 주는 사람의 4점과 짜게 주는 사람의 4점은 의미가 다르다.
      자기 평균보다 높게 준 장소는 +방향, 낮게 준 장소는 -방향으로 반영해야
      "이 사용자가 상대적으로 무엇을 좋아하는지"가 드러난다.

    [중요] 정보 누수(leakage) 방지 - Leave-One-Out
      학습 데이터의 (사용자 u, 장소 i, 평점 r) 한 줄에 대한 피처를 만들 때
      프로필 안에 바로 그 평점 r 이 들어 있으면, 모델이 정답을 미리 본 셈이 된다.
      그러면 학습 성능만 비정상적으로 좋아지고 새로운 데이터에서는 성능이 떨어진다.
      그래서 학습 행에 대해서는 "그 행 하나를 뺀" 프로필을 쓴다.

      매 행마다 프로필을 새로 계산하면 느리므로 합계를 미리 구해 두고 빼는 방식으로 처리한다.
        A_u = Σ 평점 × 장소벡터,  B_u = Σ 장소벡터,  S_u = Σ 평점,  N_u = 평점 개수
        전체 프로필       = A_u - (S_u / N_u) × B_u
        한 행을 뺀 프로필 = (A_u - r·x_i) - ((S_u - r) / (N_u - 1)) × (B_u - x_i)
    """

    def __init__(self, train_idx, n_users, place_blocks):
        """
        train_idx    : (사용자 번호 배열, 장소 번호 배열, 평점 배열)
        n_users      : 전체 사용자 수
        place_blocks : {"cat": 카테고리 원-핫 행렬, "dist": 자치구 원-핫 행렬}
        """
        u, p, r = train_idx
        self.place_blocks = place_blocks

        # 사용자별 평점 합계와 개수
        self.S = np.bincount(u, weights=r, minlength=n_users)   # Σ 평점
        self.N = np.bincount(u, minlength=n_users).astype(float)  # 평점 개수

        # 블록별로 A_u, B_u 를 미리 누적해 둔다.
        self.A, self.B = {}, {}
        for name, X in place_blocks.items():
            A = np.zeros((n_users, X.shape[1]))
            B = np.zeros((n_users, X.shape[1]))
            # np.add.at : 같은 사용자 번호가 여러 번 나와도 빠짐없이 더해 준다.
            np.add.at(A, u, r[:, None] * X[p])
            np.add.at(B, u, X[p])
            self.A[name], self.B[name] = A, B

    def user_mean(self, global_mean):
        """사용자별 평균 평점. 평점이 하나도 없는 사용자는 전체 평균으로 대신한다."""
        return np.where(self.N > 0, self.S / np.maximum(self.N, 1), global_mean)

    def full(self, name):
        """모든 학습 평점을 사용한 프로필 (검증/테스트/최종 추천에 사용)."""
        mean = self.S / np.maximum(self.N, 1)
        return self.A[name] - mean[:, None] * self.B[name]

    def leave_one_out(self, name, u, p, r):
        """학습 행 (u, p, r) 각각에 대해, 그 행을 뺀 프로필을 돌려준다."""
        X = self.place_blocks[name][p]                    # 각 행의 장소 벡터
        mean_excl = (self.S[u] - r) / np.maximum(self.N[u] - 1, 1)  # 그 행을 뺀 평균
        return (self.A[name][u] - r[:, None] * X) - mean_excl[:, None] * (
            self.B[name][u] - X
        )

    def user_mean_leave_one_out(self, u, r, global_mean):
        """학습 행 각각에 대해, 그 행을 뺀 사용자 평균 평점."""
        n = self.N[u] - 1
        return np.where(n > 0, (self.S[u] - r) / np.maximum(n, 1), global_mean)


# =============================================================================
# 4. 코사인 유사도
# =============================================================================
def cosine_rows(a, b):
    """
    같은 줄끼리의 코사인 유사도.  a[k] 와 b[k] 를 비교한다.

      cos(a, b) = (a · b) / (|a| × |b|)

    두 벡터의 "방향"이 얼마나 비슷한지를 -1 ~ 1 로 나타낸다.
    길이(크기)의 영향은 없애고 취향의 방향만 비교하므로,
    평점을 많이 남긴 사용자와 적게 남긴 사용자를 같은 기준으로 다룰 수 있다.
    어느 한쪽이 0벡터(정보 없음)이면 유사도는 0으로 둔다.
    """
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.where(den > EPS, num / np.maximum(den, EPS), 0.0)


def cosine_all_pairs(user_mat, place_mat):
    """
    모든 사용자 × 모든 장소의 코사인 유사도 행렬 (사용자 수 x 장소 수).
    각 벡터를 길이 1로 정규화한 뒤 행렬 곱을 하면 한 번에 계산된다.
    """
    un = user_mat / np.maximum(np.linalg.norm(user_mat, axis=1, keepdims=True), EPS)
    pn = place_mat / np.maximum(np.linalg.norm(place_mat, axis=1, keepdims=True), EPS)
    return un @ pn.T


# =============================================================================
# 4-2. 지역 근접도 (선호 지역에서 얼마나 가까운가)
# =============================================================================
def district_centers(places):
    """
    자치구별 기준 위치(위도, 경도)를 구한다.
    그 자치구에 속한 장소들의 좌표 중앙값을 기준 위치로 쓴다.
    (평균 대신 중앙값을 쓰면 외곽에 떨어진 장소 몇 곳에 덜 끌려간다.
     데이터 처리 담당이 합성 평점을 만들 때 쓴 기준 위치 정의와도 같다.)
    """
    known = places[places["district"] != UNKNOWN]
    return known.groupby("district")[["latitude", "longitude"]].median()


def proximity_matrix(ref_lat, ref_lon, places, scale_km):
    """
    기준 위치(사용자별 선호 지역 중심)와 각 장소 사이의 근접도 행렬을 만든다.

      근접도 = exp(-거리 / scale_km)

    거리 0km 이면 1이고, 멀어질수록 0에 가까워진다.
    scale_km 이 클수록 먼 장소까지 점수가 천천히 줄어든다.
    (예: scale_km = 3 일 때  0km -> 1.00,  3km -> 0.37,  10km -> 0.04)

    "같은 자치구인가(1/0)"만 보는 exp_dist 와 달리,
    구 경계 바로 건너편의 가까운 장소에도 높은 점수를 주고
    같은 구 안에서도 가까운 장소와 먼 장소를 구분한다는 점이 다르다.

    거리는 위도·경도 차이를 km 로 바꾼 평면 근사 거리다.
    서울 정도의 좁은 범위에서는 Haversine 거리와 차이가 거의 없다.
    선호 지역이 없는 사용자(기준 위치가 NaN)는 근접도를 0으로 둔다.

    ref_lat, ref_lon : 사용자별 기준 위치 배열 (길이 = 사용자 수)
    돌려주는 값      : (사용자 수 x 장소 수) 행렬
    """
    p_lat = places["latitude"].to_numpy(dtype=float)
    p_lon = places["longitude"].to_numpy(dtype=float)
    # 경도 1도의 실제 거리는 위도에 따라 달라지므로 cos(위도)를 곱해 보정한다.
    lon_km = KM_PER_DEGREE * np.cos(np.radians(np.mean(p_lat)))
    dy = (ref_lat[:, None] - p_lat[None, :]) * KM_PER_DEGREE
    dx = (ref_lon[:, None] - p_lon[None, :]) * lon_km
    dist_km = np.sqrt(dx ** 2 + dy ** 2)
    return np.nan_to_num(np.exp(-dist_km / scale_km), nan=0.0)


# =============================================================================
# 5. 회귀 모델에 넣을 피처 정의
# =============================================================================
# 피처 이름과 의미
#   exp_cat    : 명시적 관심도 벡터 vs 장소 카테고리 벡터의 코사인 유사도
#   exp_dist   : 선호 자치구 벡터 vs 장소 자치구 벡터의 코사인 유사도 (일치 1 / 불일치 0)
#   exp_near   : 선호 자치구 중심과 장소 사이의 근접도 exp(-거리/기준거리) (0 ~ 1)
#   lrn_cat    : 학습된 카테고리 프로필 vs 장소 카테고리 벡터의 코사인 유사도
#   lrn_dist   : 학습된 자치구 프로필 vs 장소 자치구 벡터의 코사인 유사도
#   generosity : 사용자의 평점 후함 정도 (users_synthetic.csv 에 주어진 값)
#   user_mean  : 사용자의 과거 평균 평점 (학습 평점에서 계산)
#
# 아래 변형들을 검증셋으로 비교해서 가장 좋은 것을 최종 모델로 고른다.
# 변형을 나눠 보는 이유는 "어떤 정보가 성능에 얼마나 기여하는지"를 보고서에서 보여 주기 위해서다.
MODEL_VARIANTS = {
    "A. 명시적 프로필 (카테고리)":          ["exp_cat"],
    "B. 명시적 프로필 (카테고리+지역)":     ["exp_cat", "exp_dist"],
    "C. B + 평점 후함 정도":                ["exp_cat", "exp_dist", "generosity"],
    "D. 학습된 프로필":                     ["lrn_cat", "lrn_dist", "user_mean"],
    "E. 명시적 + 학습된 프로필 (전체)":     ["exp_cat", "exp_dist", "generosity",
                                             "lrn_cat", "lrn_dist", "user_mean"],
    "F. C의 지역 일치를 지역 근접도로 교체": ["exp_cat", "exp_near", "generosity"],
}
# 평점 이력 없이도 만들 수 있는 피처들.
# 이 피처만 쓰는 변형 중 검증 성능이 가장 좋은 것을 신규 사용자용 모델로 쓴다.
EXPLICIT_FEATURES = {"exp_cat", "exp_dist", "exp_near", "generosity"}


def to_index(ratings, user_pos, place_pos):
    """평점 표의 user_id / place_id 문자열을 행렬 번호(0, 1, 2, ...)로 바꾼다."""
    u = ratings["user_id"].map(user_pos).to_numpy()
    p = ratings["place_id"].map(place_pos).to_numpy()
    r = ratings["rating"].to_numpy(dtype=float)
    return u, p, r


def pair_features(u, p, ctx, loo_ratings=None):
    """
    (사용자, 장소) 쌍 목록에 대한 피처 표를 만든다.

    u, p        : 사용자 번호 배열, 장소 번호 배열 (같은 길이)
    ctx         : 미리 만들어 둔 벡터들을 담은 딕셔너리
    loo_ratings : 학습 행의 피처를 만들 때만 그 행의 평점을 넘긴다.
                  넘기면 Leave-One-Out 프로필을 쓰고(누수 방지),
                  None 이면 전체 학습 평점으로 만든 프로필을 쓴다(검증/테스트).
    """
    lp = ctx["learned"]
    if loo_ratings is None:
        lrn_cat_vec = lp.full("cat")[u]
        lrn_dist_vec = lp.full("dist")[u]
        user_mean = lp.user_mean(ctx["global_mean"])[u]
    else:
        lrn_cat_vec = lp.leave_one_out("cat", u, p, loo_ratings)
        lrn_dist_vec = lp.leave_one_out("dist", u, p, loo_ratings)
        user_mean = lp.user_mean_leave_one_out(u, loo_ratings, ctx["global_mean"])

    return {
        "exp_cat": cosine_rows(ctx["u_cat"][u], ctx["p_cat"][p]),
        "exp_dist": cosine_rows(ctx["u_dist"][u], ctx["p_dist"][p]),
        "exp_near": ctx["near"][u, p],
        "lrn_cat": cosine_rows(lrn_cat_vec, ctx["p_cat"][p]),
        "lrn_dist": cosine_rows(lrn_dist_vec, ctx["p_dist"][p]),
        "generosity": ctx["generosity"][u],
        "user_mean": user_mean,
    }


def all_pair_features(ctx):
    """모든 사용자 × 모든 장소에 대한 피처. 각 값은 (사용자 수 x 장소 수) 행렬."""
    lp = ctx["learned"]
    n_places = ctx["p_cat"].shape[0]
    ones = np.ones((1, n_places))  # 사용자별 값을 장소 방향으로 늘리기 위한 보조 행렬
    return {
        "exp_cat": cosine_all_pairs(ctx["u_cat"], ctx["p_cat"]),
        "exp_dist": cosine_all_pairs(ctx["u_dist"], ctx["p_dist"]),
        "exp_near": ctx["near"],
        "lrn_cat": cosine_all_pairs(lp.full("cat"), ctx["p_cat"]),
        "lrn_dist": cosine_all_pairs(lp.full("dist"), ctx["p_dist"]),
        "generosity": ctx["generosity"][:, None] * ones,
        "user_mean": lp.user_mean(ctx["global_mean"])[:, None] * ones,
    }


def stack(feats, names):
    """피처 딕셔너리에서 필요한 피처만 골라 (행 수 x 피처 수) 행렬로 만든다."""
    return np.stack([np.ravel(feats[n]) for n in names], axis=1)


# =============================================================================
# 6. 평가 지표
# =============================================================================
def clip_rating(x):
    """예측값을 실제 평점 범위인 1~5 안으로 자른다."""
    return np.clip(x, 1.0, 5.0)


def mae(y_true, y_pred):
    """MAE (평균 절대 오차) : 예측이 평균적으로 몇 점 빗나갔는지."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """RMSE (평균 제곱근 오차) : 큰 오차에 더 큰 벌점을 주는 지표."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def tie_break_scores(kind, place_cnt, place_mean):
    """
    동점 처리용 장소별 보조 점수를 0~1 범위로 만들어 돌려준다.

    왜 필요한가:
      장소 피처가 카테고리(3종)와 자치구(25개)뿐이라서, 한 사용자에게
      "같은 카테고리 + 같은 구"인 장소들은 예측 평점이 완전히 똑같다.
      이 동점 묶음 안에서 무엇을 먼저 보여 줄지 정하는 규칙이 필요하다.

    kind
      "없음"      : 보조 점수 없음 (동점이면 장소 번호 순서)
      "평균 평점" : 학습 평점의 장소별 평균이 높은 순
      "평가 수"   : 학습 평점에서 평가를 많이 받은(인기 있는) 순
    """
    if kind == "평균 평점":
        return (place_mean - 1.0) / 4.0
    if kind == "평가 수":
        return place_cnt / max(float(place_cnt.max()), 1.0)
    return np.zeros_like(place_mean)


def topk_metrics(score_mat, train_idx, eval_idx, k=TOP_K, tie=None):
    """
    Top-K 추천 품질을 평가한다.

    tie : 장소별 동점 처리 보조 점수 (tie_break_scores 의 결과). None 이면 쓰지 않는다.

    방법
      - 각 사용자에 대해, 학습 때 이미 평가한 장소를 뺀 나머지 전체 장소를 점수순으로 줄 세운다.
      - 평가셋(검증 또는 테스트)에서 그 사용자가 4점 이상 준 장소를 "정답"으로 본다.
      - 상위 K개 안에 정답이 얼마나 들어 있는지 센다.

    지표
      Precision@K : 추천한 K개 중 정답 비율
      Recall@K    : 정답 중 추천 K개 안에 들어온 비율
      NDCG@K      : 정답이 위쪽 순위에 있을수록 높은 점수 (순위까지 반영)
      HitRate@K   : 정답을 1개라도 맞힌 사용자 비율
      Coverage@K  : 전체 장소 중 누군가의 Top-K에 한 번이라도 등장한 장소 비율
                    (추천이 소수 장소에만 쏠리는지 확인하는 지표)

    주의: 사용자 한 명의 평가셋 정답은 평균 1~2개뿐인데 후보는 약 1,000개라서
          Precision/Recall 의 절대값은 구조적으로 작게 나온다.
          절대값보다는 무작위·인기순 기준선과의 "상대 비교"로 해석해야 한다.
    """
    tu, tp, _ = train_idx
    eu, ep, er = eval_idx

    scores = score_mat.copy()
    if tie is not None:
        # 동점 처리: 0~1 범위의 보조 점수에 아주 작은 수를 곱해 더한다.
        # 예측 평점은 소수 넷째 자리 수준에서 차이가 나므로 본 점수의 순서는 그대로다.
        scores = scores + TIE_SCALE * tie[None, :]
    scores[tu, tp] = -np.inf  # 이미 평가한 장소는 추천 후보에서 제외

    # 사용자별 정답 장소 집합
    relevant = {}
    for u, p, r in zip(eu, ep, er):
        if r >= RELEVANT_THRESHOLD:
            relevant.setdefault(u, set()).add(p)

    # 점수가 높은 순으로 상위 K개 장소 번호
    # (kind="stable" : 점수가 같으면 장소 번호 순서를 유지해 결과가 매번 같게 나오도록)
    topk = np.argsort(-scores, axis=1, kind="stable")[:, :k]

    # NDCG 계산용 순위별 할인값 : 1위 1.0, 2위 0.63, 3위 0.5, ...
    discount = 1.0 / np.log2(np.arange(2, k + 2))

    precisions, recalls, ndcgs, hits = [], [], [], []
    for u, rel in relevant.items():
        hit_flags = np.array([p in rel for p in topk[u]], dtype=float)
        n_hit = hit_flags.sum()
        precisions.append(n_hit / k)
        recalls.append(n_hit / len(rel))
        # DCG : 실제 추천 순서의 점수 / IDCG : 정답을 맨 위부터 채웠을 때의 최고 점수
        dcg = float(np.sum(hit_flags * discount))
        idcg = float(np.sum(discount[: min(len(rel), k)]))
        ndcgs.append(dcg / idcg)
        hits.append(1.0 if n_hit > 0 else 0.0)

    return {
        f"Precision@{k}": float(np.mean(precisions)),
        f"Recall@{k}": float(np.mean(recalls)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        f"HitRate@{k}": float(np.mean(hits)),
        f"Coverage@{k}": float(len(np.unique(topk)) / score_mat.shape[1]),
    }


# =============================================================================
# 7. 모델 학습과 선택
# =============================================================================
def fit_and_select(variant_names, f_train, y_train, f_valid, y_valid):
    """
    주어진 피처 조합으로 Ridge 회귀를 학습하고, 검증 RMSE가 가장 낮은 alpha 를 고른다.

    Ridge 회귀 :  예측 평점 = 절편 + Σ (계수 × 피처)
      일반 선형회귀에 "계수가 너무 커지지 않게 하는 벌점(alpha)"을 더한 모델이다.
      여기서 계수는 곧 "카테고리 유사도 / 지역 유사도에 얼마의 가중치를 줄 것인가"이므로,
      블록별 가중치를 사람이 정하지 않고 데이터에서 학습하는 셈이다.
    """
    X_train, X_valid = stack(f_train, variant_names), stack(f_valid, variant_names)
    best = None
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        score = rmse(y_valid, clip_rating(model.predict(X_valid)))
        if best is None or score < best["valid_rmse"]:
            best = {"model": model, "alpha": alpha, "valid_rmse": score}
    best["valid_mae"] = mae(y_valid, clip_rating(best["model"].predict(X_valid)))
    return best


def predict_matrix(model, variant_names, f_all, shape):
    """모든 (사용자, 장소) 쌍의 예측 평점 행렬을 만든다. (1~5 범위로 자름)"""
    return clip_rating(model.predict(stack(f_all, variant_names))).reshape(shape)


# =============================================================================
# 8. 신규 사용자 추천 (콜드 스타트)
# =============================================================================
def recommend_for_new_user(interest, preferred_district, artifacts, top_n=10,
                           generosity=None):
    """
    평점 이력이 없는 새 사용자에게 장소를 추천한다.
    (제안서 시스템 구조의 "User Input -> Recommendation Engine" 단계에 해당)

    interest           : {"명소": 0.2, "문화": 0.5, "쇼핑": 0.3} 형태의 관심도
    preferred_district : 선호 자치구 이름 (예: "종로구"). 없으면 None
    artifacts          : main() 이 돌려주는 학습 결과 묶음
    generosity         : 평점 후함 정도. 모르면 전체 사용자의 평균값을 쓴다.

    돌려주는 값 : 추천 장소 표 (place_id, name, category, district, 예측 평점)
    """
    ctx, places = artifacts["ctx"], artifacts["places"]
    model, names = artifacts["cold_model"], artifacts["cold_features"]

    # 새 사용자의 명시적 프로필 벡터 (1명이므로 1행짜리 행렬)
    u_cat = np.array([[interest.get(c, 0.0) for c in CATEGORIES]], dtype=float)
    u_dist = np.array(
        [[1.0 if d == preferred_district else 0.0 for d in artifacts["districts"]]]
    )
    g = float(np.mean(ctx["generosity"])) if generosity is None else float(generosity)

    # 선호 자치구의 기준 위치. 선호 지역이 없거나 모르는 이름이면 NaN -> 근접도 0
    centers = artifacts["centers"]
    if preferred_district in centers.index:
        ref_lat = np.array([centers.loc[preferred_district, "latitude"]])
        ref_lon = np.array([centers.loc[preferred_district, "longitude"]])
    else:
        ref_lat = ref_lon = np.array([np.nan])

    n_places = len(places)
    feats = {
        "exp_cat": cosine_all_pairs(u_cat, ctx["p_cat"]),
        "exp_dist": cosine_all_pairs(u_dist, ctx["p_dist"]),
        "exp_near": proximity_matrix(ref_lat, ref_lon, places, artifacts["near_scale_km"]),
        "generosity": np.full((1, n_places), g),
    }
    pred = clip_rating(model.predict(stack(feats, names)))

    out = places[["place_id", "name", "category", "district"]].copy()
    out["cb_pred_rating"] = np.round(pred, 4)
    # 예측 평점이 완전히 같은 장소가 있을 수 있으므로(예: 좌표가 같은 장소),
    # 동점일 때는 검증에서 고른 동점 처리 규칙의 보조 점수가 높은 장소를 먼저 보여 준다.
    out["_tie"] = artifacts["tie"]
    out = out.sort_values(["cb_pred_rating", "_tie"], ascending=False)
    return out.drop(columns="_tie").head(top_n).reset_index(drop=True)


# =============================================================================
# 9. 전체 실행 흐름
# =============================================================================
def main():
    # ---- 9-1. 데이터 준비 --------------------------------------------------
    places, users, train, valid, test = load_data()
    n_users, n_places = len(users), len(places)
    print(f"[데이터] 장소 {n_places:,}개 / 사용자 {n_users:,}명 / "
          f"평점 train {len(train):,} · valid {len(valid):,} · test {len(test):,}")

    # 문자열 ID -> 행렬 번호 변환표
    user_pos = {uid: i for i, uid in enumerate(users["user_id"])}
    place_pos = {pid: i for i, pid in enumerate(places["place_id"])}
    tr = to_index(train, user_pos, place_pos)
    va = to_index(valid, user_pos, place_pos)
    te = to_index(test, user_pos, place_pos)

    # ---- 9-2. 벡터 만들기 --------------------------------------------------
    p_cat, p_dist, districts = build_place_features(places)
    u_cat, u_dist = build_explicit_user_profiles(users, districts)
    global_mean = float(tr[2].mean())  # 전체 학습 평점의 평균

    # 사용자별 기준 위치 = 선호 자치구의 중심 좌표 (선호 지역이 "미상"이면 NaN)
    centers = district_centers(places)
    ref_lat = users["preferred_district"].map(centers["latitude"]).to_numpy(dtype=float)
    ref_lon = users["preferred_district"].map(centers["longitude"]).to_numpy(dtype=float)
    # 거리 감쇠 기준 후보별 근접도 행렬을 미리 만들어 둔다. (평점과 무관하므로 한 번만 계산)
    near_by_scale = {s: proximity_matrix(ref_lat, ref_lon, places, s) for s in NEAR_SCALES_KM}
    near_holder = {"near": near_by_scale[NEAR_SCALES_KM[0]]}  # 아래 9-3b 에서 고른 값으로 바뀐다

    def make_context(train_idx):
        """주어진 학습 평점으로 학습된 프로필까지 만든 '재료 묶음'을 돌려준다."""
        return {
            "p_cat": p_cat, "p_dist": p_dist,
            "u_cat": u_cat, "u_dist": u_dist,
            "generosity": users["rating_generosity"].to_numpy(dtype=float),
            "near": near_holder["near"],
            "global_mean": float(train_idx[2].mean()),
            "learned": LearnedProfiles(train_idx, n_users, {"cat": p_cat, "dist": p_dist}),
        }

    ctx = make_context(tr)

    # ---- 9-3. 피처 만들기 --------------------------------------------------
    f_train = pair_features(tr[0], tr[1], ctx, loo_ratings=tr[2])  # 학습 행: 누수 방지(LOO)
    f_valid = pair_features(va[0], va[1], ctx)                     # 검증 행: 전체 프로필
    f_all = all_pair_features(ctx)                                 # 모든 쌍 (Top-K 평가용)

    # ---- 9-3b. 지역 근접도의 거리 감쇠 기준 선택 ----------------------------
    # exp(-거리/기준거리) 의 기준거리를 몇 km 로 둘지 검증 RMSE 로 고른다.
    # 근접도 피처만 바꿔 끼우면서 변형 F 의 피처 조합으로 비교한다.
    f_feats = MODEL_VARIANTS["F. C의 지역 일치를 지역 근접도로 교체"]
    scale_rows, best_scale, best_scale_rmse = [], None, np.inf
    for s in NEAR_SCALES_KM:
        f_train["exp_near"] = near_by_scale[s][tr[0], tr[1]]
        f_valid["exp_near"] = near_by_scale[s][va[0], va[1]]
        res = fit_and_select(f_feats, f_train, tr[2], f_valid, va[2])
        scale_rows.append({"기준거리(km)": s, "MAE": res["valid_mae"], "RMSE": res["valid_rmse"]})
        if res["valid_rmse"] < best_scale_rmse:
            best_scale, best_scale_rmse = s, res["valid_rmse"]
    scale_comparison = pd.DataFrame(scale_rows)
    print("\n[지역 근접도 기준거리 비교 - 검증셋]")
    print(scale_comparison.round(4).to_string(index=False))
    print(f"  -> 선택: {best_scale}km")

    # 고른 기준거리의 근접도를 이후 모든 단계에서 쓰도록 반영한다.
    near_holder["near"] = near_by_scale[best_scale]
    ctx["near"] = near_holder["near"]
    f_train["exp_near"] = ctx["near"][tr[0], tr[1]]
    f_valid["exp_near"] = ctx["near"][va[0], va[1]]
    f_all["exp_near"] = ctx["near"]

    # 피처에 NaN 이나 무한대가 섞여 있으면 여기서 바로 멈춘다. (계산 오류 조기 발견)
    for feats in (f_train, f_valid, f_all):
        assert all(np.isfinite(v).all() for v in feats.values()), "피처에 비정상 값이 있습니다."

    # ---- 9-4. 기준선(baseline) --------------------------------------------
    # 모델이 의미 있는지 보려면 "아무 것도 안 한 경우"와 비교해야 한다.
    print("\n[검증셋 성능 비교]")
    rows = []

    # (기준선 1) 모든 평점을 전체 평균으로 예측
    pred = np.full(len(va[2]), global_mean)
    rows.append({"모델": "기준선: 전체 평균", "피처": "-", "alpha": "-",
                 "MAE": mae(va[2], pred), "RMSE": rmse(va[2], pred)})

    # (기준선 2) 각 사용자의 평균 평점으로 예측
    pred = ctx["learned"].user_mean(global_mean)[va[0]]
    rows.append({"모델": "기준선: 사용자 평균", "피처": "-", "alpha": "-",
                 "MAE": mae(va[2], pred), "RMSE": rmse(va[2], pred)})

    # Top-K 기준선용 점수 행렬
    rng = np.random.default_rng(42)                       # 재현 가능하도록 시드 고정
    random_scores = rng.random((n_users, n_places))       # 무작위 추천
    place_sum = np.bincount(tr[1], weights=tr[2], minlength=n_places)
    place_cnt = np.bincount(tr[1], minlength=n_places)
    place_mean = np.where(place_cnt > 0, place_sum / np.maximum(place_cnt, 1), global_mean)
    popular_scores = np.tile(place_cnt.astype(float), (n_users, 1))  # 평가 많은 순 추천

    rows[0].update(topk_metrics(random_scores, tr, va))
    rows[0]["모델"] = "기준선: 전체 평균 (Top-K는 무작위 추천)"
    rows[1].update(topk_metrics(popular_scores, tr, va))
    rows[1]["모델"] = "기준선: 사용자 평균 (Top-K는 인기순 추천)"

    # ---- 9-5. 모델 변형 비교 -----------------------------------------------
    fitted = {}
    for name, feat_names in MODEL_VARIANTS.items():
        res = fit_and_select(feat_names, f_train, tr[2], f_valid, va[2])
        score_mat = predict_matrix(res["model"], feat_names, f_all, (n_users, n_places))
        fitted[name] = res
        rows.append({"모델": name, "피처": ", ".join(feat_names), "alpha": res["alpha"],
                     "MAE": res["valid_mae"], "RMSE": res["valid_rmse"],
                     **topk_metrics(score_mat, tr, va)})

    comparison = pd.DataFrame(rows)
    print(comparison.drop(columns="피처").round(4).to_string(index=False))

    # 검증 RMSE가 가장 낮은 변형을 최종 모델로 선택
    best_name = min(fitted, key=lambda n: fitted[n]["valid_rmse"])
    best_feats, best_alpha = MODEL_VARIANTS[best_name], fitted[best_name]["alpha"]

    # ---- 9-5b. 동점 처리 규칙 선택 -----------------------------------------
    # 선택된 모델의 예측 평점은 그대로 두고, 동점일 때의 순서 규칙만 바꿔 가며
    # 검증 NDCG@K 가 가장 높은 규칙을 고른다. (MAE/RMSE 에는 영향이 없다.)
    best_scores = predict_matrix(fitted[best_name]["model"], best_feats, f_all,
                                 (n_users, n_places))
    tie_rows, best_tie, best_ndcg = [], "없음", -1.0
    for kind in ["없음", "평균 평점", "평가 수"]:
        m = topk_metrics(best_scores, tr, va, tie=tie_break_scores(kind, place_cnt, place_mean))
        tie_rows.append({"동점 처리": kind, **m})
        if m[f"NDCG@{TOP_K}"] > best_ndcg:
            best_tie, best_ndcg = kind, m[f"NDCG@{TOP_K}"]
    tie_comparison = pd.DataFrame(tie_rows)
    print("\n[동점 처리 규칙 비교 - 검증셋]")
    print(tie_comparison.round(4).to_string(index=False))
    print(f"  -> 선택: {best_tie}")
    print(f"\n[선택된 모델] {best_name} (alpha={best_alpha})")
    coef = dict(zip(best_feats, fitted[best_name]["model"].coef_))
    print("  절편:", round(float(fitted[best_name]["model"].intercept_), 4))
    for k, v in coef.items():
        print(f"  계수 {pad(k, 10)}: {v:+.4f}")

    # ---- 9-6. 테스트셋 최종 평가 -------------------------------------------
    # 테스트셋은 모델과 alpha 를 다 정한 뒤 "딱 한 번만" 사용한다.
    # (테스트 결과를 보고 설정을 바꾸면 테스트 성능이 낙관적으로 부풀려지기 때문)
    # 최종 모델은 train + validation 을 합쳐서 다시 학습한다. (쓸 수 있는 데이터를 모두 활용)
    tv = tuple(np.concatenate([a, b]) for a, b in zip(tr, va))
    ctx_final = make_context(tv)
    f_tv = pair_features(tv[0], tv[1], ctx_final, loo_ratings=tv[2])
    f_test = pair_features(te[0], te[1], ctx_final)
    f_all_final = all_pair_features(ctx_final)

    final_model = Ridge(alpha=best_alpha).fit(stack(f_tv, best_feats), tv[2])
    test_pred = clip_rating(final_model.predict(stack(f_test, best_feats)))
    final_scores = predict_matrix(final_model, best_feats, f_all_final, (n_users, n_places))

    # 테스트용 기준선
    gm_tv = float(tv[2].mean())
    um_tv = ctx_final["learned"].user_mean(gm_tv)
    cnt_tv = np.bincount(tv[1], minlength=n_places).astype(float)
    # 장소별 평균 평점 (train + validation 기준)
    pm_tv = np.where(cnt_tv > 0, np.bincount(tv[1], weights=tv[2], minlength=n_places)
                     / np.maximum(cnt_tv, 1), gm_tv)
    final_tie = tie_break_scores(best_tie, cnt_tv, pm_tv)  # 검증에서 고른 동점 규칙
    test_metrics = {
        "모델": best_name, "alpha": best_alpha, "동점 처리": best_tie,
        "MAE": mae(te[2], test_pred), "RMSE": rmse(te[2], test_pred),
        **topk_metrics(final_scores, tv, te, tie=final_tie),
    }
    test_baselines = {
        "전체 평균": {"MAE": mae(te[2], np.full(len(te[2]), gm_tv)),
                      "RMSE": rmse(te[2], np.full(len(te[2]), gm_tv))},
        "사용자 평균": {"MAE": mae(te[2], um_tv[te[0]]), "RMSE": rmse(te[2], um_tv[te[0]])},
        "무작위 추천": topk_metrics(random_scores, tv, te),
        "인기순 추천": topk_metrics(np.tile(cnt_tv, (n_users, 1)), tv, te),
    }
    print("\n[테스트셋 최종 성능]")
    for k, v in test_metrics.items():
        print(f"  {pad(k, 14)}: {v if isinstance(v, str) else round(v, 4)}")
    print("  (기준선) 전체 평균 RMSE", round(test_baselines["전체 평균"]["RMSE"], 4),
          "/ 사용자 평균 RMSE", round(test_baselines["사용자 평균"]["RMSE"], 4))

    # ---- 9-7. 신규 사용자용 모델 -------------------------------------------
    # 명시적 정보(관심도, 선호 지역, 후함 정도)만 쓰는 변형 중 검증 RMSE 가 가장 낮은 것을
    # train+validation 으로 다시 학습해 둔다.
    explicit_variants = [n for n, feats in MODEL_VARIANTS.items()
                         if set(feats) <= EXPLICIT_FEATURES]
    cold_name = min(explicit_variants, key=lambda n: fitted[n]["valid_rmse"])
    cold_feats = MODEL_VARIANTS[cold_name]
    cold_model = Ridge(alpha=fitted[cold_name]["alpha"]).fit(stack(f_tv, cold_feats), tv[2])

    # ---- 9-8. 결과 저장 ----------------------------------------------------
    OUT_DIR.mkdir(exist_ok=True)

    # (1) 모든 (사용자, 장소) 쌍의 점수표  ->  하이브리드 / 경로 담당에게 전달
    #   cb_pred_rating : 예측 평점 (1~5).  협업필터링의 예측 평점과 같은 단위라서
    #                    RMSE 비교와 가중 평균 결합에 그대로 쓸 수 있다. (기본으로 쓸 점수)
    #   cb_score_norm  : (예측 평점 - 1) / 4 로 0~1 범위로 바꾼 값.
    #                    거리·시간 같은 다른 단위의 점수와 섞을 때 쓰기 편하다.
    #                    모든 사용자에게 같은 식을 적용하므로 사용자 간 비교도 가능하다.
    #   already_rated  : 그 사용자가 train/validation 에서 이미 평가한 장소면 1
    rated = np.zeros((n_users, n_places), dtype=np.int8)
    rated[tv[0], tv[1]] = 1
    scores_df = pd.DataFrame({
        "user_id": np.repeat(users["user_id"].to_numpy(), n_places),
        "place_id": np.tile(places["place_id"].to_numpy(), n_users),
        "cb_pred_rating": np.round(final_scores.ravel(), 4),
        "cb_score_norm": np.round((final_scores.ravel() - 1.0) / 4.0, 4),
        "already_rated": rated.ravel(),
    })
    scores_df.to_csv(OUT_DIR / "cb_scores_all.csv", index=False, encoding="utf-8-sig")

    # (2) 사용자별 Top-N 추천 목록 (이미 평가한 장소 제외)
    masked = final_scores.copy()
    masked[tv[0], tv[1]] = -np.inf
    # 동점 처리: 평가 때와 똑같은 규칙(검증에서 고른 것)을 적용한다.
    order = np.argsort(-(masked + TIE_SCALE * final_tie[None, :]), axis=1,
                       kind="stable")[:, :TOP_N_SAVE]
    top_df = pd.DataFrame({
        "user_id": np.repeat(users["user_id"].to_numpy(), TOP_N_SAVE),
        "rank": np.tile(np.arange(1, TOP_N_SAVE + 1), n_users),
        "place_id": places["place_id"].to_numpy()[order.ravel()],
        "cb_pred_rating": np.round(np.take_along_axis(final_scores, order, axis=1).ravel(), 4),
    }).merge(places[["place_id", "name", "category", "district", "latitude", "longitude"]],
             on="place_id", how="left")
    top_df.to_csv(OUT_DIR / "cb_top20_per_user.csv", index=False, encoding="utf-8-sig")

    # (3) 모델 비교표, (4) 최종 성능과 설정값
    comparison.to_csv(OUT_DIR / "cb_model_comparison.csv", index=False, encoding="utf-8-sig")
    tie_comparison.to_csv(OUT_DIR / "cb_tie_break_comparison.csv", index=False,
                          encoding="utf-8-sig")
    scale_comparison.to_csv(OUT_DIR / "cb_near_scale_comparison.csv", index=False,
                            encoding="utf-8-sig")
    with open(OUT_DIR / "cb_final_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "selected_model": best_name,
            "features": best_feats,
            "ridge_alpha": best_alpha,
            "tie_break": best_tie,
            "near_scale_km": best_scale,
            "intercept": float(final_model.intercept_),
            "coefficients": dict(zip(best_feats, map(float, final_model.coef_))),
            "test_metrics": test_metrics,
            "test_baselines": test_baselines,
            "cold_start_model": {
                "variant": cold_name,
                "features": cold_feats,
                "intercept": float(cold_model.intercept_),
                "coefficients": dict(zip(cold_feats, map(float, cold_model.coef_))),
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[저장 완료] {OUT_DIR}")

    # ---- 9-9. 추천 예시 출력 -----------------------------------------------
    artifacts = {
        "ctx": ctx_final, "places": places, "districts": districts,
        "centers": centers, "near_scale_km": best_scale,
        "cold_model": cold_model, "cold_features": cold_feats, "tie": final_tie,
    }
    first_user = users["user_id"].iloc[0]
    print(f"\n[예시 1] 기존 사용자 {first_user} 의 추천 Top 5")
    print(top_df[top_df["user_id"] == first_user]
          .head(5)[["rank", "name", "category", "district", "cb_pred_rating"]]
          .to_string(index=False))

    print("\n[예시 2] 신규 사용자 (문화 0.6 / 명소 0.3 / 쇼핑 0.1, 선호 지역 종로구) 추천 Top 5")
    print(recommend_for_new_user({"문화": 0.6, "명소": 0.3, "쇼핑": 0.1}, "종로구",
                                 artifacts, top_n=5).to_string(index=False))
    return artifacts


if __name__ == "__main__":
    main()
