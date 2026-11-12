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
TIE_SCALE = 1e-6         # 동점 처리 점수에 곱하는 값. 작아서 원래 순위는 안 건드림

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
    # (장소 수, 25)
    dist_mat = np.stack(
        [(places["district"] == d).to_numpy(dtype=float) for d in districts], axis=1
    )
    return cat_mat, dist_mat, districts


def build_explicit_user_profiles(users, districts):
    """
    users 파일에 있는 값으로 바로 만드는 프로필.
      카테고리: interest_명소/문화/쇼핑 (합이 1)
      자치구  : 선호 자치구 원-핫
    평점 기록이 없는 신규 사용자도 이건 만들 수 있음.
    """
    cat_prof = users[[f"interest_{c}" for c in CATEGORIES]].to_numpy(dtype=float)
    dist_prof = np.stack(
        [(users["preferred_district"] == d).to_numpy(dtype=float) for d in districts],
        axis=1,
    )
    return cat_prof, dist_prof


# ---------------------------------------------------------------------------
# 3. 학습된 프로필 (평점 보고 취향 추정)
# ---------------------------------------------------------------------------
class LearnedProfiles:
    """
    사용자가 준 평점으로 취향 벡터를 추정.

      프로필 = Σ (평점 - 그 사람 평균 평점) × 장소 벡터

    평균을 빼는 이유는 사람마다 점수 주는 기준이 달라서임. 원래 후하게 주는 사람의 4점이랑
    짜게 주는 사람의 4점은 다르니까, 자기 평균보다 높게 준 곳은 +, 낮게 준 곳은 -로 반영.

    * 누수 주의 (leave-one-out)
      학습 데이터 한 줄 (u, i, r)의 피처를 만드는데 프로필 안에 그 r이 이미 들어가 있으면
      정답을 보고 푸는 거랑 같음. train 성능만 좋아지고 val/test에서는 떨어짐.
      그래서 학습 행은 "자기 자신을 뺀" 프로필로 피처를 만듦.

      행마다 다시 계산하면 너무 느려서 합계를 미리 구해두고 빼는 식으로 함.
        A_u = Σ 평점×장소벡터, B_u = Σ 장소벡터, S_u = Σ 평점, N_u = 평점 개수
        전체 프로필 = A_u - (S_u/N_u)·B_u
        한 줄 뺀 것 = (A_u - r·x_i) - ((S_u - r)/(N_u - 1))·(B_u - x_i)
    """

    def __init__(self, train_idx, n_users, place_blocks):
        """
        train_idx: (사용자 번호, 장소 번호, 평점) 배열 3개
        place_blocks: {"cat": 카테고리 원-핫, "dist": 자치구 원-핫}
        """
        u, p, r = train_idx
        self.place_blocks = place_blocks

        self.S = np.bincount(u, weights=r, minlength=n_users)   # 사용자별 평점 합
        self.N = np.bincount(u, minlength=n_users).astype(float)  # 사용자별 평점 개수

        # A_u, B_u 미리 계산 (cat, dist 각각)
        self.A, self.B = {}, {}
        for name, X in place_blocks.items():
            A = np.zeros((n_users, X.shape[1]))
            B = np.zeros((n_users, X.shape[1]))
            # A[u] += ... 로 쓰면 같은 u가 여러 번 나올 때 한 번만 더해져서 add.at 사용
            np.add.at(A, u, r[:, None] * X[p])
            np.add.at(B, u, X[p])
            self.A[name], self.B[name] = A, B

    def user_mean(self, global_mean):
        """사용자별 평균 평점. 평점이 아예 없으면 전체 평균으로 대체."""
        return np.where(self.N > 0, self.S / np.maximum(self.N, 1), global_mean)

    def full(self, name):
        """학습 평점 전부 써서 만든 프로필. 검증/테스트/최종 추천용."""
        mean = self.S / np.maximum(self.N, 1)
        return self.A[name] - mean[:, None] * self.B[name]

    def leave_one_out(self, name, u, p, r):
        """학습 행마다 그 행만 빼고 만든 프로필."""
        X = self.place_blocks[name][p]                    # 그 행의 장소 벡터
        mean_excl = (self.S[u] - r) / np.maximum(self.N[u] - 1, 1)  # 그 행 빼고 구한 평균
        return (self.A[name][u] - r[:, None] * X) - mean_excl[:, None] * (
            self.B[name][u] - X
        )

    def user_mean_leave_one_out(self, u, r, global_mean):
        """위랑 같은 이유로, 그 행을 뺀 사용자 평균."""
        n = self.N[u] - 1
        return np.where(n > 0, (self.S[u] - r) / np.maximum(n, 1), global_mean)


# ---------------------------------------------------------------------------
# 4. 코사인 유사도
# ---------------------------------------------------------------------------
def cosine_rows(a, b):
    """
    a[k]와 b[k]를 한 줄씩 짝지어서 코사인 유사도 계산.
      cos = (a·b) / (|a||b|)
    벡터 크기는 무시하고 방향만 비교하는 값이라, 평점을 많이 남긴 사람이든 적게 남긴
    사람이든 같은 기준으로 비교됨. 한쪽이 0벡터면 그냥 0으로 처리.
    """
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.where(den > EPS, num / np.maximum(den, EPS), 0.0)


def cosine_all_pairs(user_mat, place_mat):
    """
    전체 사용자 × 전체 장소 코사인 유사도, (사용자 수, 장소 수) 행렬.
    길이를 1로 맞춘 다음 행렬곱 한 번으로 계산.
    """
    un = user_mat / np.maximum(np.linalg.norm(user_mat, axis=1, keepdims=True), EPS)
    pn = place_mat / np.maximum(np.linalg.norm(place_mat, axis=1, keepdims=True), EPS)
    return un @ pn.T


# ---------------------------------------------------------------------------
# 4-2. 지역 근접도
# ---------------------------------------------------------------------------
def district_centers(places):
    """
    자치구별 중심 좌표. 그 구에 있는 장소들 위경도의 중앙값으로 잡음.
    평균으로 하면 구 외곽에 동떨어진 장소 때문에 중심이 쏠려서 중앙값 사용.
    합성 평점 만들 때 기준 위치도 이렇게 잡았다고 해서 똑같이 맞춤.
    """
    known = places[places["district"] != UNKNOWN]
    return known.groupby("district")[["latitude", "longitude"]].median()


def proximity_matrix(ref_lat, ref_lon, places, scale_km):
    """
    선호 지역 중심에서 각 장소까지 얼마나 가까운지. (사용자 수, 장소 수) 행렬.

      근접도 = exp(-거리 / scale_km)

    거리가 0이면 1, 멀어질수록 0에 가까워짐.
    scale_km=3이면 3km에서 0.37, 10km에서 0.04 정도.

    처음엔 같은 구인지 아닌지(exp_dist)만 봤는데, 그러면 구 경계 바로 옆에 있는 가까운
    장소도 0이 되고 같은 구 안에서는 거리 차이가 전혀 반영이 안 돼서 거리 기반으로 바꿈.

    거리는 위경도 차이를 km로 환산한 평면 거리. 서울 안에서는 haversine이랑 거의 같음.
    선호 지역이 없는 사용자(ref가 NaN)는 0으로 둠.
    """
    p_lat = places["latitude"].to_numpy(dtype=float)
    p_lon = places["longitude"].to_numpy(dtype=float)
    # 경도 1도 길이는 위도에 따라 달라서 cos(위도)를 곱해줘야 함
    lon_km = KM_PER_DEGREE * np.cos(np.radians(np.mean(p_lat)))
    dy = (ref_lat[:, None] - p_lat[None, :]) * KM_PER_DEGREE
    dx = (ref_lon[:, None] - p_lon[None, :]) * lon_km
    dist_km = np.sqrt(dx ** 2 + dy ** 2)
    return np.nan_to_num(np.exp(-dist_km / scale_km), nan=0.0)


# ---------------------------------------------------------------------------
# 5. 회귀에 넣을 피처
# ---------------------------------------------------------------------------
# exp_cat    관심도 벡터 vs 장소 카테고리 벡터 코사인 유사도
# exp_dist   선호 자치구 vs 장소 자치구 (같은 구면 1, 아니면 0)
# exp_near   선호 자치구 중심 ~ 장소 근접도 (0~1)
# lrn_cat    학습된 카테고리 프로필 vs 장소 카테고리 벡터
# lrn_dist   학습된 자치구 프로필 vs 장소 자치구 벡터
# generosity 평점 후하게 주는 정도 (users 파일에 있는 값)
# user_mean  사용자의 과거 평균 평점
#
# 어떤 정보가 실제로 도움이 되는지 보려고 조합을 여러 개 만들어서 비교함.
# 검증 RMSE가 제일 낮은 걸 최종 모델로 씀.
MODEL_VARIANTS = {
    "A. 명시적 프로필 (카테고리)":          ["exp_cat"],
    "B. 명시적 프로필 (카테고리+지역)":     ["exp_cat", "exp_dist"],
    "C. B + 평점 후함 정도":                ["exp_cat", "exp_dist", "generosity"],
    "D. 학습된 프로필":                     ["lrn_cat", "lrn_dist", "user_mean"],
    "E. 명시적 + 학습된 프로필 (전체)":     ["exp_cat", "exp_dist", "generosity",
                                             "lrn_cat", "lrn_dist", "user_mean"],
    "F. C의 지역 일치를 지역 근접도로 교체": ["exp_cat", "exp_near", "generosity"],
}
# 평점 기록 없이도 계산되는 피처들. 신규 사용자용 모델은 이것만 쓰는 조합 중에서 고름
EXPLICIT_FEATURES = {"exp_cat", "exp_dist", "exp_near", "generosity"}


def to_index(ratings, user_pos, place_pos):
    """U0001, P0001 같은 문자열 id를 행렬 인덱스(0, 1, 2...)로 변환."""
    u = ratings["user_id"].map(user_pos).to_numpy()
    p = ratings["place_id"].map(place_pos).to_numpy()
    r = ratings["rating"].to_numpy(dtype=float)
    return u, p, r


def pair_features(u, p, ctx, loo_ratings=None):
    """
    (사용자, 장소) 쌍들에 대한 피처 계산.

    loo_ratings: 학습 행 피처를 만들 때만 그 행들의 평점을 넘겨줌.
                 넘기면 leave-one-out 프로필을 쓰고, None이면 전체 프로필 사용(검증/테스트).
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
    """전체 사용자 × 전체 장소 피처. 값 하나하나가 (사용자 수, 장소 수) 행렬."""
    lp = ctx["learned"]
    n_places = ctx["p_cat"].shape[0]
    ones = np.ones((1, n_places))  # 사용자별 값을 장소 개수만큼 옆으로 늘릴 때 씀
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
    """피처 dict에서 names에 있는 것만 뽑아서 (행 수, 피처 수) 행렬로."""
    return np.stack([np.ravel(feats[n]) for n in names], axis=1)


# ---------------------------------------------------------------------------
# 6. 평가 지표
# ---------------------------------------------------------------------------
def clip_rating(x):
    """평점은 1~5라서 예측값도 그 범위로 자름."""
    return np.clip(x, 1.0, 5.0)


def mae(y_true, y_pred):
    """평균 절대 오차. 평균적으로 몇 점 틀렸는지."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """평균 제곱근 오차. 크게 틀린 예측에 벌점을 더 줌."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def tie_break_scores(kind, place_cnt, place_mean):
    """
    예측 평점이 똑같은 장소끼리 순서를 정할 때 쓰는 보조 점수 (0~1).

    카테고리랑 구만 쓰는 모델(A~C)은 같은 카테고리 + 같은 구면 점수가 완전히 같아서
    동점이 엄청 많이 나옴. 그 안에서 뭘 먼저 보여줄지 정해야 해서 만듦.

      "없음"      그냥 장소 번호 순
      "평균 평점" 장소 평균 평점 높은 순
      "평가 수"   평가 많이 받은 순 (인기순)
    """
    if kind == "평균 평점":
        return (place_mean - 1.0) / 4.0
    if kind == "평가 수":
        return place_cnt / max(float(place_cnt.max()), 1.0)
    return np.zeros_like(place_mean)


def topk_metrics(score_mat, train_idx, eval_idx, k=TOP_K, tie=None):
    """
    Top-K 추천 평가.

    사용자마다 이미 평가한 장소는 빼고 나머지를 점수순으로 정렬한 뒤,
    평가셋에서 4점 이상 준 장소(정답)가 상위 K개에 몇 개 들어갔는지 확인.

      Precision@K  추천 K개 중 정답 비율
      Recall@K     정답 중 K개 안에 들어온 비율
      NDCG@K       정답이 위쪽 순위에 있을수록 높음
      HitRate@K    하나라도 맞힌 사용자 비율
      Coverage@K   전체 장소 중 누군가의 Top-K에 한 번이라도 나온 비율
                   (추천이 몇몇 장소로만 쏠리는지 보려고 넣음)

    tie: tie_break_scores() 결과. None이면 동점 처리 안 함.

    참고로 사용자 한 명당 정답이 1~2개밖에 없는데 후보는 1,000개쯤 돼서
    precision, recall 숫자 자체는 엄청 작게 나옴. 랜덤/인기순 대비 얼마나 나은지로 봐야 함.
    """
    tu, tp, _ = train_idx
    eu, ep, er = eval_idx

    scores = score_mat.copy()
    if tie is not None:
        # 보조 점수(0~1)에 1e-6을 곱해서 더함. 동점일 때만 순서에 영향이 감
        scores = scores + TIE_SCALE * tie[None, :]
    scores[tu, tp] = -np.inf  # 이미 평가한 장소는 추천에서 제외

    # 사용자별 정답 장소
    relevant = {}
    for u, p, r in zip(eu, ep, er):
        if r >= RELEVANT_THRESHOLD:
            relevant.setdefault(u, set()).add(p)

    # 점수 높은 순 상위 K개. stable로 해야 동점일 때 실행할 때마다 순서가 안 바뀜
    topk = np.argsort(-scores, axis=1, kind="stable")[:, :k]

    # NDCG용 순위 가중치. 1등 1.0, 2등 0.63, 3등 0.5 ...
    discount = 1.0 / np.log2(np.arange(2, k + 2))

    precisions, recalls, ndcgs, hits = [], [], [], []
    for u, rel in relevant.items():
        hit_flags = np.array([p in rel for p in topk[u]], dtype=float)
        n_hit = hit_flags.sum()
        precisions.append(n_hit / k)
        recalls.append(n_hit / len(rel))
        # idcg는 정답이 전부 맨 위에 있을 때의 최대값
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


# ---------------------------------------------------------------------------
# 7. 모델 학습
# ---------------------------------------------------------------------------
def fit_and_select(variant_names, f_train, y_train, f_valid, y_valid):
    """
    피처 조합 하나를 받아서 Ridge를 학습하고, 검증 RMSE가 제일 낮은 alpha를 고름.

    Ridge는 선형회귀에 계수가 너무 커지지 않게 벌점(alpha)을 붙인 것.
      예측 평점 = 절편 + Σ 계수×피처
    계수가 곧 카테고리 유사도, 지역 근접도에 주는 가중치라서
    가중치를 손으로 정하지 않고 데이터로 구하는 셈.
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
    """전체 (사용자, 장소) 예측 평점 행렬. 1~5로 자른 값."""
    return clip_rating(model.predict(stack(f_all, variant_names))).reshape(shape)


# ---------------------------------------------------------------------------
# 8. 신규 사용자 추천 (cold start)
# ---------------------------------------------------------------------------
def recommend_for_new_user(interest, preferred_district, artifacts, top_n=10,
                           generosity=None):
    """
    평점 기록이 없는 새 사용자한테 추천. 제안서 구조도에서 User Input 받아서
    Recommendation Engine으로 넘기는 부분.

    interest: {"명소": 0.2, "문화": 0.5, "쇼핑": 0.3} 이런 식으로 넣으면 됨
    preferred_district: "종로구" 같은 구 이름. 없으면 None
    artifacts: main()이 리턴하는 것
    generosity: 모르면 비워두면 전체 사용자 평균값으로 들어감

    place_id, name, category, district, 예측 평점이 담긴 표를 돌려줌.
    """
    ctx, places = artifacts["ctx"], artifacts["places"]
    model, names = artifacts["cold_model"], artifacts["cold_features"]

    # 사용자 1명이라 1행짜리 행렬로 만듦
    u_cat = np.array([[interest.get(c, 0.0) for c in CATEGORIES]], dtype=float)
    u_dist = np.array(
        [[1.0 if d == preferred_district else 0.0 for d in artifacts["districts"]]]
    )
    g = float(np.mean(ctx["generosity"])) if generosity is None else float(generosity)

    # 선호 자치구 중심 좌표. 구 이름이 없거나 목록에 없으면 NaN으로 두고 근접도는 0 처리
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
    # 좌표가 같은 장소처럼 점수가 완전히 같은 경우가 있어서 동점이면 보조 점수 순으로 정렬
    out["_tie"] = artifacts["tie"]
    out = out.sort_values(["cb_pred_rating", "_tie"], ascending=False)
    return out.drop(columns="_tie").head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 9. 실행
# ---------------------------------------------------------------------------
def main():
    # --- 데이터 준비 ---
    places, users, train, valid, test = load_data()
    n_users, n_places = len(users), len(places)
    print(f"[데이터] 장소 {n_places:,}개 / 사용자 {n_users:,}명 / "
          f"평점 train {len(train):,} · valid {len(valid):,} · test {len(test):,}")

    # id -> 인덱스 매핑
    user_pos = {uid: i for i, uid in enumerate(users["user_id"])}
    place_pos = {pid: i for i, pid in enumerate(places["place_id"])}
    tr = to_index(train, user_pos, place_pos)
    va = to_index(valid, user_pos, place_pos)
    te = to_index(test, user_pos, place_pos)

    # --- 벡터 만들기 ---
    p_cat, p_dist, districts = build_place_features(places)
    u_cat, u_dist = build_explicit_user_profiles(users, districts)
    global_mean = float(tr[2].mean())  # train 평점 전체 평균

    # 사용자별 기준 위치 = 선호 자치구 중심. 선호 지역이 미상이면 NaN이 들어감
    centers = district_centers(places)
    ref_lat = users["preferred_district"].map(centers["latitude"]).to_numpy(dtype=float)
    ref_lon = users["preferred_district"].map(centers["longitude"]).to_numpy(dtype=float)
    # 근접도는 평점이랑 상관없는 값이라 기준거리 후보별로 미리 다 계산해둠
    near_by_scale = {s: proximity_matrix(ref_lat, ref_lon, places, s) for s in NEAR_SCALES_KM}
    near_holder = {"near": near_by_scale[NEAR_SCALES_KM[0]]}  # 일단 첫 후보로 넣고 밑에서 교체

    def make_context(train_idx):
        """피처 계산에 필요한 것들을 dict 하나로 묶음. 학습 평점이 바뀌면 다시 만들어야 함."""
        return {
            "p_cat": p_cat, "p_dist": p_dist,
            "u_cat": u_cat, "u_dist": u_dist,
            "generosity": users["rating_generosity"].to_numpy(dtype=float),
            "near": near_holder["near"],
            "global_mean": float(train_idx[2].mean()),
            "learned": LearnedProfiles(train_idx, n_users, {"cat": p_cat, "dist": p_dist}),
        }

    ctx = make_context(tr)

    # --- 피처 계산 ---
    f_train = pair_features(tr[0], tr[1], ctx, loo_ratings=tr[2])  # 학습 행은 LOO (누수 방지)
    f_valid = pair_features(va[0], va[1], ctx)                     # 검증 행은 전체 프로필
    f_all = all_pair_features(ctx)                                 # 전체 쌍, Top-K 평가할 때 씀

    # --- 근접도 기준거리 고르기 ---
    # exp(-거리/기준거리)에서 기준거리를 몇 km로 할지 검증 RMSE로 결정.
    # 모델 F 피처 조합에서 exp_near만 바꿔가면서 비교함
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

    # 고른 기준거리로 근접도를 확정. 이 뒤로는 전부 이 값 사용
    near_holder["near"] = near_by_scale[best_scale]
    ctx["near"] = near_holder["near"]
    f_train["exp_near"] = ctx["near"][tr[0], tr[1]]
    f_valid["exp_near"] = ctx["near"][va[0], va[1]]
    f_all["exp_near"] = ctx["near"]

    # 혹시 NaN이나 inf가 섞였으면 여기서 바로 멈추게 함
    for feats in (f_train, f_valid, f_all):
        assert all(np.isfinite(v).all() for v in feats.values()), "피처에 비정상 값이 있습니다."

    # --- 기준선 ---
    # 모델이 의미가 있는지 보려면 아무것도 안 한 경우랑 비교해야 함
    print("\n[검증셋 성능 비교]")
    rows = []

    # 전부 전체 평균으로 찍는 경우
    pred = np.full(len(va[2]), global_mean)
    rows.append({"모델": "기준선: 전체 평균", "피처": "-", "alpha": "-",
                 "MAE": mae(va[2], pred), "RMSE": rmse(va[2], pred)})

    # 사용자별 평균으로 찍는 경우
    pred = ctx["learned"].user_mean(global_mean)[va[0]]
    rows.append({"모델": "기준선: 사용자 평균", "피처": "-", "alpha": "-",
                 "MAE": mae(va[2], pred), "RMSE": rmse(va[2], pred)})

    # Top-K 쪽 기준선은 랜덤 추천이랑 인기순 추천
    rng = np.random.default_rng(42)                       # 시드 고정 (돌릴 때마다 같은 결과)
    random_scores = rng.random((n_users, n_places))       # 랜덤 추천
    place_sum = np.bincount(tr[1], weights=tr[2], minlength=n_places)
    place_cnt = np.bincount(tr[1], minlength=n_places)
    place_mean = np.where(place_cnt > 0, place_sum / np.maximum(place_cnt, 1), global_mean)
    popular_scores = np.tile(place_cnt.astype(float), (n_users, 1))  # 평가 많이 받은 순

    rows[0].update(topk_metrics(random_scores, tr, va))
    rows[0]["모델"] = "기준선: 전체 평균 (Top-K는 무작위 추천)"
    rows[1].update(topk_metrics(popular_scores, tr, va))
    rows[1]["모델"] = "기준선: 사용자 평균 (Top-K는 인기순 추천)"

    # --- 모델 A~F 비교 ---
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

    # 검증 RMSE 제일 낮은 걸 최종 모델로
    best_name = min(fitted, key=lambda n: fitted[n]["valid_rmse"])
    best_feats, best_alpha = MODEL_VARIANTS[best_name], fitted[best_name]["alpha"]

    # --- 동점 처리 방식 고르기 ---
    # 예측 평점은 그대로 두고 동점일 때 순서 규칙만 바꿔서 검증 NDCG가 제일 높은 걸 고름.
    # 순서만 바뀌는 거라 MAE, RMSE는 안 변함
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

    # --- 테스트셋 평가 ---
    # test는 모델, alpha 다 정하고 나서 마지막에 한 번만 봄.
    # test 결과를 보고 설정을 바꾸면 성능이 실제보다 좋게 나와서 의미가 없어짐.
    # 최종 모델은 train + validation을 합쳐서 다시 학습 (있는 데이터는 다 쓰려고)
    tv = tuple(np.concatenate([a, b]) for a, b in zip(tr, va))
    ctx_final = make_context(tv)
    f_tv = pair_features(tv[0], tv[1], ctx_final, loo_ratings=tv[2])
    f_test = pair_features(te[0], te[1], ctx_final)
    f_all_final = all_pair_features(ctx_final)

    final_model = Ridge(alpha=best_alpha).fit(stack(f_tv, best_feats), tv[2])
    test_pred = clip_rating(final_model.predict(stack(f_test, best_feats)))
    final_scores = predict_matrix(final_model, best_feats, f_all_final, (n_users, n_places))

    # test용 기준선도 train + validation 기준으로 다시 계산
    gm_tv = float(tv[2].mean())
    um_tv = ctx_final["learned"].user_mean(gm_tv)
    cnt_tv = np.bincount(tv[1], minlength=n_places).astype(float)
    # 장소별 평균 평점
    pm_tv = np.where(cnt_tv > 0, np.bincount(tv[1], weights=tv[2], minlength=n_places)
                     / np.maximum(cnt_tv, 1), gm_tv)
    final_tie = tie_break_scores(best_tie, cnt_tv, pm_tv)  # 위에서 고른 동점 처리 방식
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

    # --- 신규 사용자용 모델 ---
    # 관심도, 선호 지역, 후함 정도처럼 평점 없이도 아는 정보만 쓰는 조합 중에서
    # 검증 RMSE가 제일 낮은 걸 골라 train + validation으로 다시 학습해둠
    explicit_variants = [n for n, feats in MODEL_VARIANTS.items()
                         if set(feats) <= EXPLICIT_FEATURES]
    cold_name = min(explicit_variants, key=lambda n: fitted[n]["valid_rmse"])
    cold_feats = MODEL_VARIANTS[cold_name]
    cold_model = Ridge(alpha=fitted[cold_name]["alpha"]).fit(stack(f_tv, cold_feats), tv[2])

    # --- 결과 저장 ---
    OUT_DIR.mkdir(exist_ok=True)

    # 1) 전체 점수표. 하이브리드, 경로 파트에 넘길 파일
    #   cb_pred_rating  예측 평점(1~5). 협업필터링 예측이랑 단위가 같아서
    #                   RMSE 비교나 가중 평균할 때 그대로 쓰면 됨. 기본은 이걸 쓰면 됨
    #   cb_score_norm   (예측 평점 - 1) / 4 해서 0~1로 바꾼 값.
    #                   거리, 시간처럼 단위가 다른 값이랑 섞을 때 쓰라고 넣음.
    #                   사용자마다 따로 정규화한 게 아니라서 사용자끼리 비교해도 됨
    #   already_rated   train/validation에서 이미 평가한 장소면 1
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

    # 2) 사용자별 Top-N. 이미 평가한 장소는 제외
    masked = final_scores.copy()
    masked[tv[0], tv[1]] = -np.inf
    # 동점 처리는 평가할 때랑 똑같은 방식으로 맞춤
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

    # 3) 비교표들이랑 최종 성능
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

    # --- 추천 예시 출력 ---
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
