# -*- coding: utf-8 -*-
"""신규 사용자 취향 입력 → 콘텐츠 기반 4곳 추천 (콜드 스타트 대응).
실행: python new_user_course.py --data-root .
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import contents_rec as cb


def ask_weights():
    print('\n[1] 관심도 입력: 명소 / 문화 / 쇼핑 (0~10, 높을수록 선호)')
    vals=[]
    for c in ('명소','문화','쇼핑'):
        while True:
            try:
                v=float(input(f'  {c} 관심도 (0~10): ').strip())
                if not 0<=v<=10: raise ValueError
                vals.append(v);break
            except ValueError: print('  0부터 10까지 숫자로 입력해 주세요.')
    if sum(vals)==0:
        print('모두 0이므로 세 유형을 동일하게 취급합니다.')
        vals=[1.,1.,1.]
    return np.array(vals,dtype=float)/sum(vals)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',default='.')
    ap.add_argument('--output-dir',default='new_user_output')
    args=ap.parse_args()
    root=Path(args.data_root)
    def read(name):
        path=root/name
        if not path.exists(): raise FileNotFoundError(f'{path} 파일이 필요합니다.')
        return pd.read_csv(path)
    places=read('places_for_recsys.csv');users=read('users_synthetic.csv')
    train=read('ratings_train.csv');valid=read('ratings_validation.csv')
    courses=read('four_stop_courses_estimated.csv')
    places['place_id']=places.place_id.astype(str)
    places['district']=places.district.fillna(cb.UNKNOWN)
    users['preferred_district']=users.preferred_district.fillna(cb.UNKNOWN)
    print('=== 서울 데이트 코스 추천: 신규 사용자 ===')
    interest=ask_weights()
    districts=sorted(places.district.dropna().unique())
    districts=[d for d in districts if d!=cb.UNKNOWN]
    print('\n[2] 선호 지역을 입력하세요. 예: 종로구 / 강남구 / 마포구')
    print('  선택 가능 지역:', ', '.join(districts))
    while True:
        district=input('  선호 자치구: ').strip()
        if district in districts: break
        print('  데이터에 있는 자치구 이름을 정확하게 입력해 주세요.')
    print('\n모델 학습 및 코스 계산 중...')
    pcat,_,_=cb.build_place_features(places)
    ucat,_=cb.build_explicit_user_profiles(users,districts)
    centers=cb.district_centers(places)
    near=cb.proximity_matrix(users.preferred_district.map(centers.latitude).to_numpy(dtype=float),
                             users.preferred_district.map(centers.longitude).to_numpy(dtype=float),places,3)
    up={uid:i for i,uid in enumerate(users.user_id.astype(str))}
    pp={pid:i for i,pid in enumerate(places.place_id)}
    for df in (train,valid):
        df['user_id']=df.user_id.astype(str);df['place_id']=df.place_id.astype(str)
    tr=cb.to_index(train,up,pp);va=cb.to_index(valid,up,pp)
    generosity=users.rating_generosity.to_numpy(dtype=float)
    def features(idx):
        u,p,_=idx
        return np.column_stack([cb.cosine_rows(ucat[u],pcat[p]),near[u,p],generosity[u]])
    xtr=features(tr);xva=features(va)
    best=min(((cb.rmse(va[2],np.clip(Ridge(alpha=a).fit(xtr,tr[2]).predict(xva),1,5)),a)
              for a in cb.RIDGE_ALPHAS),key=lambda x:x[0])[1]
    both=pd.concat([train,valid],ignore_index=True)
    tv=cb.to_index(both,up,pp)
    model=Ridge(alpha=best).fit(features(tv),tv[2])
    n=len(places)
    cat=np.repeat(interest[None,:],n,axis=0)
    near_new=cb.proximity_matrix(np.array([centers.loc[district,'latitude']]),
                                 np.array([centers.loc[district,'longitude']]),places,3)[0]
    # 신규 사용자는 평점 부여 성향을 모르므로 기존 사용자 평균을 대체값으로 사용
    gen=float(np.nanmean(generosity))
    xnew=np.column_stack([cb.cosine_rows(cat,pcat),near_new,np.full(n,gen)])
    scores=np.clip(model.predict(xnew),1,5)
    scored=places[['place_id','name','category','district','address']].copy()
    scored['predicted_rating']=scores
    score_map=scored.set_index('place_id').predicted_rating
    cols=[f'place_{i}_id' for i in range(1,5)]
    for col in cols: courses[col]=courses[col].astype(str)
    valid_courses=courses.loc[(courses[['leg1_min','leg2_min','leg3_min']]<=40).all(axis=1)
                              &(courses.total_estimated_minutes<=75)
                              &(courses[cols].nunique(axis=1)==4)].copy()
    for i,col in enumerate(cols,1):
        valid_courses[f'rating_{i}']=valid_courses[col].map(score_map)
    valid_courses=valid_courses.dropna(subset=[f'rating_{i}' for i in range(1,5)])
    valid_courses['rating_sum']=valid_courses[[f'rating_{i}' for i in range(1,5)]].sum(axis=1)
    valid_courses['rating_mean']=valid_courses.rating_sum/4
    valid_courses=valid_courses.sort_values(['rating_sum','total_estimated_minutes'],ascending=[False,True])
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    valid_courses.to_csv(out/'ranked_courses_new_user.csv',index=False,encoding='utf-8-sig')
    scored.sort_values('predicted_rating',ascending=False).to_csv(out/'place_scores_new_user.csv',index=False,encoding='utf-8-sig')
    print('\n=== 입력한 취향 ===')
    print(f'명소 {interest[0]:.1%}, 문화 {interest[1]:.1%}, 쇼핑 {interest[2]:.1%} / 선호 지역: {district}')
    print(f'이동 제약을 충족하는 코스: {len(valid_courses)}개')
    if valid_courses.empty:
        print('조건을 충족하는 코스가 없습니다.');return
    best_course=valid_courses.iloc[0]
    print('\n=== 추천 데이트 코스: 4곳 ===')
    for i in range(1,5):
        pid=best_course[f'place_{i}_id']
        place=scored.loc[scored.place_id==pid].iloc[0]
        print(f'{i}. {place["name"]} | {place["category"]} | {place["district"]} | 예측 평점 {best_course[f"rating_{i}"]:.3f}')
        if i<4: print(f'   ↓ 예상 도보 {best_course[f"leg{i}_min"]:.1f}분')
    print(f'총 예상 도보 시간: {best_course.total_estimated_minutes:.1f}분 / 평균 예측 평점: {best_course.rating_mean:.3f}')
    print('주의: 도보 시간은 직선거리 보정 추정치이며 실제 도보 경로 검증값이 아닙니다.')
    print('신규 사용자는 평점 이력이 없어 협업 필터링을 사용하지 않고 콘텐츠 기반 모델로 추천합니다.')
    print('결과 CSV:',out.resolve())

if __name__=='__main__':main()
