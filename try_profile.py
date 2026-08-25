import sqlite3
from collections import Counter

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

con = sqlite3.connect("cosmetic.db")

model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 64}
)

product_rows = con.execute("""
    SELECT product_id, name, brand, category, price, skin_type, ingredient, concern, tags, description 
    FROM products ORDER BY product_id
""").fetchall()


# ===============================================
# 1. 상품아이디 리스트, 상품아이디별 벡터화할 상품정보 리스트
# ===============================================
# 상품 아이디만 리스트로 반환
product_ids = [r[0] for r in product_rows]
# print(product_ids)

# 상품 아이디를 제외한 나머지 상품정보를 하나의 문자열로 묶어서 리스트로 반환
# 반환된 각각의 리스트를 벡터화 처리
product_vectors = np.array(model.embed_documents(
[
    " / ".join(str(x) for x in r[1:]) # 첫번재 제품아이디를 제외한 나머지 값들을 문자화 시킨뒤 하나의 문자열로 이어붙임
    for r in product_rows # 각레코드를 하나씩 추출해서 
]
), dtype="float32") 

# print(product_vectors[0])

# ===============================================
# 2. 고객id별 기본정보 분리해서 가져옴
# ===============================================
# [c, *r in 리스트]
# [
#     ("C001", 25, "여성", "건성", "서울"), 
#     ("C001", 25, "여성", "건성", "서울"), 
# ]
# [
#     ("C001","나머지값")
# ]
# 기본 고객 정보를 가져옴 (이때 고객아이디와, 나머지 고객정보를 분리해서 저장)
customer_rows = con.execute("SELECT customer_id, age, gender, skin_type, city FROM customers")
customer_info = {c: r for c, *r in customer_rows}
# print(customer_info["C002"])
# print("-------")

# ===============================================
# 3. 고객별 구매 이력을 가져와서 딕서녀리 형태로 카테고라이징
# ===============================================
history = {}

for cid, name, cat, ing, concern, rating, review in con.execute("""
    SELECT purchases.customer_id, products.name, products.category, products.ingredient, products.concern, 
    purchases.rating, purchases.review
    FROM purchases JOIN products ON products.product_id = purchases.product_id
    WHERE purchases.is_holdout = 0
    ORDER BY purchases.customer_id, purchases.purchase_id
"""):
    history.setdefault(cid,[]).append((name, cat, ing, concern, rating, review))

# ("고객아이디", [제품명, 카테고리명, 성분, 피부걱정, 별점, 리뷰])
# print(history["C002"])

# 지금까지 찾아놓은 고객별 구매 정보를 반환받음

# ===============================================
# 4. 고객별 인기 구매 상품 비교
# ===============================================
# is_holdout=1로 숨겨놓은 제품을 위의 정보와 비교하면서 해당고객이 구매한 제품중 인기 제품이 얼마나 많이 있는지 비교
# 해당 고객이 구매한 항목중 인기상품 항목만 가져옴 (이고객의 취향 비교를 위해 이고객의 정답 제품만 가져옴)
hits_product = dict(con.execute("SELECT customer_id, product_id FROM purchases WHERE is_holdout=1"))

# 고객아이디별로 구매한 제품명이 등록될 빈 딕셔너리
bought = {}
for cid, pid in con.execute("SELECT customer_id, product_id FROM purchases WHERE is_holdout=0"):
    bought.setdefault(cid, set()).add(pid)


cids = [c for c in sorted(history) if c in hits_product]



# ===============================================
# 4. 고객id를 이용해서 고객취향을 문장덩어리로 반환함수 등록
# ===============================================
# 리스트에사 특정 값의 갯수를 카운트에서 가장 많이 나온 value값을 상위 n번째까지 반환해주는 함수
def top(items, n=3):
    # 리스트에 각 값의 갯수를 카운트해서 가장 많이 나온값의 상위 n번째 값까지 반환하는 함수
    return "/".join(x for x, _ in Counter(items).most_common(n))


# 인자로 전달받은 고객아이디를 통해서 해당 고객의 구매이력을 선호도와 함께 반환하는 함수
def taste(cid):
    h = history[cid]

    #특정 고객의 구매이력에서 별점만 모두 더한뒤, 해당 별점의 총합을 구매 건수로 나면 해당 고객의 평균 별점 반환
    avg = sum(x[4] for x in h) / len(h)

    # 고객의 피부타입과 구매이력에서 자주 등장한 항목들을 추출해서 하나의 소개 문장으로 만든다.
    # customer_info[cid][2]: 고객의 피부타입

    # 해당 고객이 제일 많이 구매한 제품 카테고리 명이 반환
    skin_type = customer_info[cid][2] # 해당 사용자의 스킨 타입
    t_category = top(x[1] for x in h) # 해당 사용자가 구매한 제품중 가장 많이 언급된 제품 카테고리 명
    t_ingredient = top(x[2] for x in h) # 해당 사용자가 구매한 제품중 가장 많이 언급된 성분명
    t_concern = top(x[3] for x in h)

    return (f"스킨타입:{skin_type} / 선호제품 카테고리: {t_category} / 선호 성분: {t_ingredient} / 주요관심사: {t_concern} / 평균별점: {avg:.1f} ")

# print("특정 고객 취향 분석 문장", taste("C002"))



# =========================================================
# 4. 다양한 정보값 조합을 위해서 정보 카테고리별로 람다함수 리스트에 등록
# =========================================================
VARIANTS = [
    #("함수라벨", 특정 값을 반환하는 함수의 호출문이 아닌 정의형태가 들어가 있음)
    ("1.후기만 이어붙임", lambda c: " ".join(f"{x[0]} (별점 {x[4]}) {x[5]}" for x in history[c])),
    ("2.취향만 가져옴", taste),
    ("3.취향+나이,성별,거주지", lambda c: f"{customer_info[c][0] // 10 * 10}대  /  {'여성' if customer_info[c][1] =="F" else '남성'} / {customer_info[c][3]} /" + taste(c) ),
    # 기본고객정보 / 후기1, 후기2, 후기3, 후기4
    ("4.취향 + 후기", lambda c: taste(c) + " / " +" ".join(x[5] for x in history[c])[:200]),
]


# ===============================================================
# 5. 위에서 람다식으로 등록한 프리셋 함수를 반복돌면서 고객정보 청킹 데이터를 추출
# ===============================================================
# 위의 variants값을 이용해서 해당 프리셋을 반복돌면서 각 결과정보를 문자열로 출력 
for label, func in VARIANTS: 
    texts = [ func(c) for c in cids]
    customer_vectors = np.array(model.embed_documents(texts), dtype="float32")
    hits = [0, 0, 0]

    # 고객별 순번과 고객 id하나씩 가져와서 코사인유사도로 추천 제품 목록 줄세우기
    for i, cid in enumerate(cids):
        order = np.argsort( -(product_vectors @ customer_vectors[i]))
        # 아래 구문은 고객이 구매하지 않은 제품 목록중에서 추천할만한 제품 정보를 반환받고 싶을때 사용
        # ranked = [ product_ids[j] for j in order if product_ids[j] not in bought.get(cid, ())]
        ranked = [ product_ids[j] for j in order]
        # print(ranked)

        # 각 고객 정보의 추천도 카운트 정보를 hits에 옮겨담음
        for slot, k in enumerate((1,3,5)):
            hits[slot] += hits_product[cid] in ranked[:k]

    # 평가에 활용된 전체 고객수
    n = len(cids)

    print(f"{label} / 평균글자: {sum(len(t) for t in texts) / n:.0f} / hit@1: {hits[0] / n * 100:.1f}% /  hit@3: {hits[1] / n * 100:.1f}% /  hit@5: {hits[2] / n * 100:.1f}%")

# 1.후기만 이어붙임      / 평균글자: 260  / hit@1: 3.7%  /  hit@3: 7.0%  /  hit@5: 10.3%
# 2.취향만 가져옴       / 평균글자: 82   / hit@1: 2.0%  /  hit@3: 6.7%  /  hit@5: 9.3%
# 3.취향+나이,성별,거주지 / 평균글자: 99   / hit@1: 2.3%  /  hit@3: 7.7%  /  hit@5: 10.3%
# 4.취향 + 후기         / 평균글자: 238  / hit@1: 1.7%  /  hit@3: 4.7%  /  hit@5: 7.0%

# 우선적으로 파악해야할 정보값
# 1-실제 구매 상품중 추천도 1위로 맞힌 비율
# 2-실제 구매 상품중 추천도 3위안에 맞힌 비율
# 3-실제 구매 상품중 추천도 5위안에 맞힌 비율
# 4-전달한 토큰수 대비 추천도 맞힌 비율이 높은 데이터셋 분석




