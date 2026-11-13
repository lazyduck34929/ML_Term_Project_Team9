# -*- coding: utf-8 -*-
"""신규 사용자 입력 -> 원본 콘텐츠 기반 Ridge + 원본 CF(KNNWithMeans/SVD++) 결합 -> 4곳 추천.
실행: python new_user_both_models.py --data-root .
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import contents_rec as cb
from final_course_pipeline import cf_fit, cf_predict

CATS = ('명소', '문화', '쇼핑')

def ask_preferences():
    print('\n=== 신규 사용자 서울 데이트 코스 추천 (콘텐츠 + 협업 필터링) ===')
    weights = []
    for cat in CATS:
        while True:
            try:
                val = float(input(f'{cat} 선호도 (0~10): ').strip())
                if not np.isfinite(val) or not 0 <= val <= 10: raise ValueError
                weights.append(val)
                break
            except ValueError:
                print('0~10 사이의 숫자를 입력해 주세요.')
    weights = np.asarray(weights, dtype=float)
    if weights.sum() == 0:
        weights[:] = 1
        print('모두 0이므로 세 유형을 동일하게 반영합니다.')
    return weights / weights.sum()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='.')
    ap.add_argument('--output-dir', default='new_user_both_output')
    ap.add_argument('--neighbors', type=int, default=20)
    args = ap.parse_args()
    root = Path(args.data_root)
    def load(name):
        path = root / name
        if not path.is_file(): raise FileNotFoundError(f'필요한 파일이 없습니다: {path}')
        return pd.read_csv(path)
    places = load('places_for_recsys.csv')
    users = load('users_synthetic.csv')
    train = load('ratings_train.csv')
    valid = load('ratings_validation.csv')
    test = load('ratings_test.csv')
    courses = load('four_stop_courses_estimated.csv')
    places['place_id'] = places.place_id.astype(str)
    places['district'] = places.district.fillna(cb.UNKNOWN)
    users['user_id'] = users.user_id.astype(str)
    users['preferred_district'] = users.preferred_district.fillna(cb.UNKNOWN)
    for d in (train, valid, test):
        d['user_id'] = d.user_id.astype(str)
        d['place_id'] = d.place_id.astype(str)

    pref = ask_preferences()
    districts = sorted(d for d in places.district.unique() if d != cb.UNKNOWN)
    print('선택 가능한 지역:', ', '.join(districts))
    while True:
        district = input('선호 자치구: ').strip()
        if district in districts: break
        print('위 목록의 자치구 이름을 입력해 주세요.')

    print('\nTrain으로 두 원본 모델 학습 및 Validation 가중치 탐색 중...')
    up = {uid:i for i,uid in enumerate(users.user_id)}
    pp = {pid:i for i,pid in enumerate(places.place_id)}
    tr, va, te = [cb.to_index(df, up, pp) for df in (train, valid, test)]
    pcat, _, dnames = cb.build_place_features(places)
    ucat, _ = cb.build_explicit_user_profiles(users, dnames)
    centers = cb.district_centers(places)
    near = cb.proximity_matrix(
        users.preferred_district.map(centers.latitude).to_numpy(dtype=float),
        users.preferred_district.map(centers.longitude).to_numpy(dtype=float), places, 3)
    generosity = users.rating_generosity.to_numpy(dtype=float)
    def features(idx):
        u,p,_ = idx
        return np.column_stack([cb.cosine_rows(ucat[u], pcat[p]), near[u,p], generosity[u]])
    xtr, xva = features(tr), features(va)
    ridge_trials = []
    for a in cb.RIDGE_ALPHAS:
        model = Ridge(alpha=a).fit(xtr, tr[2])
        pred = np.clip(model.predict(xva),1,5)
        ridge_trials.append((cb.rmse(va[2],pred),a,pred))
    _, ridge_alpha, cb_val = min(ridge_trials, key=lambda x:x[0])
    cf_models = cf_fit(train)
    knn_val = cf_predict(cf_models, valid.user_id, valid.place_id, 1.0)
    svd_val = cf_predict(cf_models, valid.user_id, valid.place_id, 0.0)
    cf_trials = [(cb.rmse(va[2], a*knn_val+(1-a)*svd_val), float(a))
                 for a in np.round(np.arange(0,1.001,0.05),2)]
    _, cf_alpha = min(cf_trials)
    cf_val = cf_alpha*knn_val + (1-cf_alpha)*svd_val
    trials = []
    for beta in np.round(np.arange(0,1.001,0.05),2):
        pred = beta*cb_val+(1-beta)*cf_val
        trials.append({'content_weight':float(beta),'cf_weight':float(1-beta),
                       'validation_rmse':cb.rmse(va[2],pred),'validation_mae':cb.mae(va[2],pred)})
    weights_df = pd.DataFrame(trials)
    beta = float(weights_df.loc[weights_df.validation_rmse.idxmin(),'content_weight'])

    tv = pd.concat([train,valid], ignore_index=True)
    tvi = cb.to_index(tv,up,pp)
    cb_model = Ridge(alpha=ridge_alpha).fit(features(tvi),tvi[2])
    cf_models = cf_fit(tv)
    cb_test = np.clip(cb_model.predict(features(te)),1,5)
    cf_test = cf_predict(cf_models,test.user_id,test.place_id,cf_alpha)
    blended_test = beta*cb_test+(1-beta)*cf_test
    metrics = pd.DataFrame([
        {'model':'Content Ridge','rmse':cb.rmse(te[2],cb_test),'mae':cb.mae(te[2],cb_test)},
        {'model':'CF KNN + SVD++','rmse':cb.rmse(te[2],cf_test),'mae':cb.mae(te[2],cf_test)},
        {'model':'Combined','rmse':cb.rmse(te[2],blended_test),'mae':cb.mae(te[2],blended_test)},
    ])

    # 콘텐츠 기반: 입력 취향으로 새로운 프로필 생성 (기존 모델의 동일한 3개 특징)
    n = len(places)
    center = centers.loc[district]
    near_new = cb.proximity_matrix(np.array([center.latitude]),np.array([center.longitude]),places,3)[0]
    new_features = np.column_stack([
        cb.cosine_rows(np.repeat(pref[None,:],n,axis=0),pcat),
        near_new, np.full(n,float(np.nanmean(generosity)))])
    content_scores = np.clip(cb_model.predict(new_features),1,5)

    # 협업 필터링: 입력 취향과 유사한 기존 사용자의 KNN+SVD++ 예측값을 가중 평균.
    # 신규 사용자를 Surprise의 기존 ID로 가장하지 않으며, '대리 추정'임을 명시.
    norms = np.linalg.norm(ucat,axis=1)*np.linalg.norm(pref)
    cat_sim = np.divide(ucat@pref,norms,out=np.zeros(len(users)),where=norms>0)
    # 카테고리 유사도 80%, 동일 자치구 보너스 20% (신규 사용자 유사도 설계값)
    similarity = 0.8*cat_sim + 0.2*(users.preferred_district.to_numpy()==district).astype(float)
    k = min(max(1,args.neighbors),len(users))
    neighbors = np.argsort(-similarity,kind='stable')[:k]
    sim_weights = np.maximum(similarity[neighbors],0)
    if sim_weights.sum()==0: sim_weights=np.ones(k)
    sim_weights=sim_weights/sim_weights.sum()
    cf_scores = np.zeros(n)
    for idx,weight in zip(neighbors,sim_weights):
        uid=users.iloc[idx].user_id
        cf_scores += weight*cf_predict(cf_models,[uid]*n,places.place_id,cf_alpha)
    cf_scores = np.clip(cf_scores,1,5)
    combined = beta*content_scores+(1-beta)*cf_scores

    scored=places[['place_id','name','category','district','address']].copy()
    scored['content_score']=content_scores
    scored['cf_proxy_score']=cf_scores
    scored['combined_score']=combined
    score_map=scored.set_index('place_id').combined_score
    cols=[f'place_{i}_id' for i in range(1,5)]
    for col in cols: courses[col]=courses[col].astype(str)
    valid_courses=courses.loc[(courses[['leg1_min','leg2_min','leg3_min']]<=40).all(axis=1)
        &(courses.total_estimated_minutes<=75)
        &(courses[cols].nunique(axis=1)==4)].copy()
    # 지역 우선순위: 선택 구 4곳 -> 선택 구와 가까운 5개 구 -> 서울 전체.
    # '가까운 구'는 장소 좌표로 계산한 자치구 중심점의 직선거리 순서이며,
    # 행정구역 경계의 실제 인접 여부를 보장하지 않습니다.
    place_district = places.set_index('place_id')['district']
    for i, col in enumerate(cols, 1):
        valid_courses[f'district_{i}'] = valid_courses[col].map(place_district)
    region_cols = [f'district_{i}' for i in range(1, 5)]
    same_district = valid_courses[region_cols].eq(district).all(axis=1)

    lat0 = np.radians(float(center.latitude))
    lon0 = np.radians(float(center.longitude))
    other_centers = centers.drop(index=district, errors='ignore').dropna(subset=['latitude','longitude'])
    lat = np.radians(other_centers.latitude.to_numpy(dtype=float))
    lon = np.radians(other_centers.longitude.to_numpy(dtype=float))
    hav = np.sin((lat-lat0)/2)**2 + np.cos(lat0)*np.cos(lat)*np.sin((lon-lon0)/2)**2
    nearby = other_centers.index[np.argsort(2*6371*np.arcsin(np.sqrt(np.clip(hav,0,1))))[:5]].tolist()
    allowed = set([district] + nearby)
    near_district = valid_courses[region_cols].isin(allowed).all(axis=1)

    if same_district.any():
        region_stage = '선택한 자치구 내 4곳'
        valid_courses = valid_courses.loc[same_district].copy()
    elif near_district.any():
        region_stage = '선택 자치구 + 중심점이 가까운 5개 자치구'
        valid_courses = valid_courses.loc[near_district].copy()
    else:
        region_stage = '서울 전체 (지역 조건 완화)'
    print(f'지역 우선 탐색 단계: {region_stage}')
    print(f'인근 지역 후보(중심점 거리 기준): {", ".join(nearby)}')
    print(f'해당 단계의 이동 제약 충족 코스: {len(valid_courses)}개')
    if region_stage != '선택한 자치구 내 4곳':
        print('선택 자치구 내부에 이동 조건을 만족하는 4곳 코스가 없어 탐색 범위를 확장했습니다.')
    for i,col in enumerate(cols,1): valid_courses[f'rating_{i}']=valid_courses[col].map(score_map)
    valid_courses=valid_courses.dropna(subset=[f'rating_{i}' for i in range(1,5)])
    valid_courses['rating_sum']=valid_courses[[f'rating_{i}' for i in range(1,5)]].sum(axis=1)
    valid_courses['rating_mean']=valid_courses.rating_sum/4
    valid_courses=valid_courses.sort_values(['rating_sum','total_estimated_minutes'],ascending=[False,True])
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    weights_df.to_csv(out/'validation_weight_search.csv',index=False,encoding='utf-8-sig')
    metrics.to_csv(out/'test_metrics.csv',index=False,encoding='utf-8-sig')
    scored.sort_values('combined_score',ascending=False).to_csv(out/'new_user_place_scores.csv',index=False,encoding='utf-8-sig')
    valid_courses.to_csv(out/'new_user_ranked_courses.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({'similar_user_id':users.iloc[neighbors].user_id.to_numpy(),
                  'similarity':similarity[neighbors], 'cf_proxy_weight':sim_weights}).to_csv(
                  out/'similar_users.csv',index=False,encoding='utf-8-sig')
    print(f'\n최적 가중치: 콘텐츠 {beta:.2f} / 협업 필터링 {1-beta:.2f}')
    print(f'협업 필터링 내부: KNN {cf_alpha:.2f} / SVD++ {1-cf_alpha:.2f}')
    print('\nTest 성능 (기존 가상 사용자 대상):\n',metrics.to_string(index=False))
    print(f'\n신규 사용자: {district}, 취향 {dict(zip(CATS,np.round(pref,3)))}')
    print(f'협업 필터링 점수: 유사 사용자 {k}명의 모델 예측값을 가중 평균한 대리 추정')
    print(f'이동 제약 충족 후보: {len(valid_courses)}개')
    if valid_courses.empty:
        print('추천 가능한 코스가 없습니다.');return
    top=valid_courses.iloc[0]
    print('\n=== 최종 추천 4개 장소 ===')
    for i in range(1,5):
        pid=top[f'place_{i}_id']
        p=scored.set_index('place_id').loc[pid]
        print(f'{i}. {p["name"]} ({p["category"]}, {p["district"]}) | 콘텐츠 {p["content_score"]:.3f}, 협업 {p["cf_proxy_score"]:.3f}, 결합 {p["combined_score"]:.3f}')
        if i<4: print(f'   ↓ 추정 도보 {top[f"leg{i}_min"]:.1f}분')
    print(f'총 추정 도보: {top.total_estimated_minutes:.1f}분 / 평균 결합 점수: {top.rating_mean:.3f}')
    print('주의: 도보 시간은 실제 길찾기 검증값이 아닌 직선거리 보정 추정값입니다.')
    print('결과 CSV:',out.resolve())

if __name__=='__main__': main()
