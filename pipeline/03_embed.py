"""
  전체 코드 흐름 정리

  1. 임베딩에 필요한 준비를 한다.
    - SQLite 데이터베이스에 연결한다.
    - 텍스트를 벡터로 바꾸는 임베딩 모델을 사용할 준비를 한다.

  2. 검색에 사용할 텍스트를 만든다.
    - 문서 조각(chunk)은 chunks 테이블의 text를 그대로 사용한다.
    - 상품(product)은 이름, 브랜드, 카테고리, 가격, 피부 타입, 성분 등의
      정보를 embedding.product_text()로 연결해 검색용 문장으로 만든다.
    - 고객(customer)은 피부 타입과 과거 구매 내역을 embedding.customer_text()로
      요약하여 고객의 취향을 나타내는 문장으로 만든다.
    - 후기(review)는 purchases 테이블의 후기 한 건을 문장 하나로 사용한다.

  3. 위에서 준비한 정보를 targets에 모은다.
    - targets에는 데이터 종류별로 다음 네 가지 정보가 들어간다.
    - (ID 컬럼 이름, 원본 테이블 이름, ID 목록, 임베딩할 텍스트 목록)
    - 예: 상품은 ("product_id", "products", 상품 ID 목록, 상품 문장 목록)이다.

  4. 텍스트를 벡터로 변환한다.
    - targets에서 조각, 상품, 고객, 후기를 한 종류씩 꺼낸다.
    - embedding.embed_documents(texts)가 각 텍스트를 숫자 목록인 벡터로 바꾼다.
    - IDs와 vectors는 같은 순서이므로 ids[0]의 벡터는 vectors[0]이다.

  5. 변환한 벡터를 데이터베이스에 저장한다.
    - storage.save_vectors()가 종류에 맞는 벡터 테이블을 만들고 저장한다.
    - chunk_vectors, product_vectors, customer_vectors, review_vectors가 만들어진다.
    - 각 벡터는 원본 데이터의 ID와 연결되어 나중에 검색 결과를 찾을 수 있다.

  6. 처리 결과를 확인하고 종료한다.
    - 데이터 개수, 평균 글자 수, 처리 시간과 처리 속도를 출력한다.
    - 실제로 저장된 상품 벡터와 고객 취향 문장을 예시로 보여 준다.
    - 마지막으로 데이터베이스 연결을 닫는다.

  한 줄 요약:
  이 파일은 DB에서 검색에 사용할 글을 가져와 종류별 문장으로 정리하고,
  그 문장들을 임베딩 벡터로 변환한 다음 SQLite의 벡터 테이블에 저장한다.
"""


# ============================================================
# 1단계. 임베딩에 필요한 모듈, 설정, 데이터베이스 준비
# ============================================================
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(errors="replace")

from huggingface_hub.utils import disable_progress_bars
from huggingface_hub.utils import logging as hub_logging


hub_logging.set_verbosity_error()
disable_progress_bars()


from app.core.config import DB_PATH, EMBED_DIM
from pipeline.prep import embedding, storage

con = sqlite3.connect(DB_PATH)
con.execute("PRAGMA foreign_keys = ON")


# ============================================================
# 2단계. DB에서 데이터를 가져와 검색용 문장으로 만들기
# ============================================================

# 임베딩할 네 종류의 정보를 모아 둘 딕셔너리
targets = {}
"""
targets = {
  "chunk": ("chunk_id", "chunks", [제품상세설명 순번], [제품상세설명 조각들],),
  "product": ("product_id", "product", [제품아이디], [제품한줄설명]),
  "customer": ("customer_id", "customers", [고객아이디], [고객설명]),
  "review": ("purchase_id", "purchases", [후기순번], [후기내용])
}
"""


# 제품 상세 설명의 조각들 targets["chunk"]에 등록 
r = con.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
# x[0]으로 ID 목록을, x[1]로 임베딩할 텍스트 목록을 만든다.
targets["chunk"] = ("chunk_id", "chunks", [x[0] for x in r], [x[1] for x in r])


# 상품정보 한줄 설명을 targets["product"]에 등록
r = con.execute("""
    SELECT product_id, name, brand, category, price, skin_type, ingredient, concern, tags, description
    FROM products ORDER BY product_id
""").fetchall()
# 첫 값은 상품 ID로 쓰고, 나머지는 embedding.product_text()로 한 문장으로 만든다.
# embedding.product_text() 함수 활용
targets["product"] = ("product_id", "products", [x[0] for x in r], [embedding.product_text(x[1:]) for x in r])


# 고객별 구매 이력을 모아 둘 딕셔너리
history = {}

# 학습용 구매 내역과 해당 상품의 카테고리·성분·고민·별점을 함께 저장
for cid, category, ingredient, concern, rating in con.execute("""
  SELECT purchases.customer_id, products.category, products.ingredient, products.concern, purchases.rating
    FROM purchases JOIN products ON products.product_id = purchases.product_id
    WHERE purchases.is_holdout = 0
    ORDER BY purchases.customer_id, purchases.purchase_id
"""):
  # 고객 ID가 처음 나오면 빈 목록을 만들고, 그 고객의 구매 정보를 추가한다.
  history.setdefault(cid, []).append(( category, ingredient, concern, rating))


# {고객 ID: 피부 타입} 형태의 딕셔너리를 생성
skin_of = dict(con.execute("SELECT customer_id, skin_type FROM customers"))

# 구매 이력이 있는 고객의 ID만 순서대로 리스트로 생성
cids = [c for (c,) in con.execute("SELECT customer_id FROM customers ORDER BY customer_id") if c in history]

# 각 고객의 피부 타입과 구매 이력을 tragets["customer"]에 등록
# embedding.customer_text() 함수 활용
targets["customer"] = ("customer_id", "customers", cids, [embedding.customer_text(skin_of[c], history[c]) for c in cids])


# 구매목록에서 후기가 있는 정보만 targets["review"]에 등록
r = con.execute("""
    SELECT purchase_id, review FROM purchases
    WHERE is_holdout = 0 AND review IS NOT NULL AND review != ''
    ORDER BY purchase_id
""").fetchall()
# 구매 ID와 후기 텍스트를 각각 목록으로 나누어 저장한다.
targets["review"] = ("purchase_id", "purchases", [x[0] for x in r], [x[1] for x in r])




# ============================================================
# 3단계. 네 종류의 ID와 검색용 텍스트가 준비되었는지 확인
# ============================================================
# targets에는 chunk, product, customer, review 정보가 준비되어 있다.
print("임베딩용 문장확인", f"  임베딩할 데이터 준비: {', '.join(targets)}")




# ============================================================
# 4단계. 임베딩 모델을 준비하고 텍스트를 벡터로 변환
# ============================================================
# 임베딩 모델 준비를 시작한 시간을 기록
started = time.perf_counter()

# 임베딩 모델을 미리 불러와 사용할 준비
# 실제 get_embeedings함수는 core폴더의 안쪽의 embedder.py에 있지만
# embeddings.py 자체가 해당 임데더자체를 import해서 참조시키 때문에
# 아래구문으로 임베더기 호출 가능
embedding.get_embeddings()


# 모델 준비에 걸린 시간을 출력한다.
print(f"  임베딩기 준비 {time.perf_counter() - started:.1f}초\n")

# 처리 결과 표의 제목 줄 출력
print(f"  {'무엇':10s} {'개수':>7s} {'평균 글자':>9s} {'걸린 시간':>10s} {'초당':>9s}")

# targets의 값을 반복 돌며 처리
for kind, (key, parent, ids, texts) in targets.items():
  """
    kind = "product"

    key = "product_id"
    parent = "products"
    ids = ["P001", "P002", ...]
    texts = ["상품 검색용 문장1", "상품 검색용 문장2", ...]
  """
  # 현재 종류의 모든 텍스트를 벡터로 바꾸고 걸린 시간도 받는다.
  vectors, elapsed = embedding.embed_documents(texts)

  # 첫 번째 벡터에 숫자가 몇 개 있는지 세어 실제 벡터 차원을 구한다.
  dim = len(vectors[0])
  # 설정된 예상 차원과 실제 차원이 다르면 경고를 보여 준다.
  if dim != EMBED_DIM:
    print(f"  경고: config 의 EMBED_DIM({EMBED_DIM}) 과 실제 차원({dim}) 이 다르다")




  # ==========================================================
  # 5단계. 현재 종류의 ID와 벡터를 DB 벡터 테이블에 저장
  # ==========================================================
  # ID와 벡터를 연결해 종류별 벡터 테이블에 저장
  storage.save_vectors(con, kind, key, parent, ids, vectors, dim)

  # 처리한 개수, 평균 글자 수, 걸린 시간, 초당 처리량을 출력한다.
  print(f"  {kind:10s} {len(ids):>7,} {sum(len(t) for t in texts) / len(texts):>9.0f} {elapsed:>9.1f}초 {len(ids) / elapsed:>8.0f}개")




# ============================================================
# 6단계. 저장 결과를 예시로 확인하고 DB 연결 종료
# ============================================================

# 저장된 상품 벡터 중 첫 번째 값을 하나 가져온다.
peek = con.execute("SELECT vector FROM product_vectors LIMIT 1").fetchone()[0]
# 벡터 전체가 길기 때문에 앞의 60글자만 예시로 출력한다.
print(f"\n  실제로 이렇게 들어 있다:\n    {peek[:60]}...")

# 저장된 고객 벡터 중 첫 번째 고객 ID를 하나 가져온다.
sample = con.execute("SELECT customer_id FROM customer_vectors LIMIT 1").fetchone()[0]
# 예시로 보여 줄 고객 ID를 출력한다.
print(f"\n  고객 한 명을 이렇게 적었다 ({sample}):")
# 고객 ID가 있는 위치를 찾아 그 고객의 취향 요약 문장을 출력한다.
print(f"    {targets['customer'][3][targets['customer'][2].index(sample)]}")
# 모든 작업이 끝났으므로 데이터베이스 연결을 닫는다.
con.close()

