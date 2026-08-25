"""
  파이프라인 결과를 실제로 검사하는 함수들을 모아 둔다.

  이 파일은 검증 방법을 담당한다.
  04_verify.py는 필요한 값을 준비하고 이 함수들을 순서대로 호출한다.
"""

import time

import numpy as np

from app.core.db import load_vectors
from pipeline.prep import chunking, embedding


# 검사 조건을 출력하고 실패 메시지를 문제 목록에 추가하는 함수
def check(ok, error_message, problems):
  """
  인자로 들어오는 값 예시:
    ok = False
    error_message = "상품 벡터 개수가 다름 (195/200)"
    problems = []

  반환값:
    없음(None)

  실행 후 problems 예시:
    ["상품 벡터 개수가 다름 (195/200)"]

  핵심 개념:
    이 함수는 검사 방법을 알지 못하고, 전달받은 ok가 참인지 거짓인지만 본다.
    problems는 리스트이므로 append()한 결과가 이 함수를 호출한 곳에도 그대로 남는다.
    따라서 여러 검사 함수가 발견한 오류를 하나의 목록에 계속 모을 수 있다.
  """
  print(f"  [{'OK  ' if ok else '문제'}] {error_message}")
  if not ok:
    problems.append(error_message)


# 테이블별 행 개수와 데이터 연결 상태를 검사하는 함수
def check_table_data(con, table_names, problems):
  """
  인자로 들어오는 값 예시:
    con = sqlite3.connect(DB_PATH)
    table_names = ("customers", "products", "chunks", "chunk_vectors")
    problems = []

  반환값 예시:
    {
      "customers": 300,
      "products": 200,
      "chunks": 1560,
      "chunk_vectors": 1560,
    }

  검사 실패 메시지는 전달받은 problems 리스트에도 추가한다.

  이 함수가 필요한 이유:
    원본 행은 있는데 벡터가 빠졌거나, 존재하지 않는 부모 데이터를 가리키는 행이
    있으면 이후 검색 결과가 누락되거나 잘못 연결될 수 있다. 그래서 벡터 계산 전에
    데이터 개수와 외래 키 연결 상태를 먼저 확인한다.

  주의할 점:
    원본 개수와 벡터 개수가 같다는 사실만으로 모든 ID가 정확히 대응한다고 완전히
    증명되는 것은 아니다. 기본 키와 외래 키 제약 조건을 함께 검사해야 더 안전하다.
  """
  counts = {}

  for table in table_names:
    counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table:18s} {counts[table]:>7,}")

  print()
  check(
    counts["chunk_vectors"] == counts["chunks"],
    f"조각 벡터 개수가 다름 ({counts['chunk_vectors']:,}/{counts['chunks']:,})",
    problems,
  )
  check(
    counts["product_vectors"] == counts["products"],
    f"상품 벡터 개수가 다름 ({counts['product_vectors']:,}/{counts['products']:,})",
    problems,
  )

  # 원본 섹션테이블에 존재하지 않는 section_id값이 조각 테이블에 하나도 없으면 
  # 원본 섹션과 연결이 끊어진 조각이 없으니 통과
  orphan = con.execute("""
      SELECT COUNT(*) FROM chunks
      WHERE section_id NOT IN (SELECT section_id FROM sections)
  """).fetchone()[0]
  check(orphan == 0, f"원문 섹션과 끊긴 조각이 있음 ({orphan}개)", problems)

  fk_errors = con.execute("PRAGMA foreign_key_check").fetchall()
  check(len(fk_errors) == 0, f"외래 키 위반이 있음 ({len(fk_errors)}개)", problems)

  return counts


# 종류별 벡터를 불러와 차원과 모델 이름을 검사하는 함수
def check_vector_data(con, kinds, expected_dim, expected_model, problems):
  """
  인자로 들어오는 값 예시:
    con = sqlite3.connect(DB_PATH)
    kinds = (("chunk", "chunk_id"), ("product", "product_id"))
    expected_dim = 384
    expected_model = "intfloat/multilingual-e5-small"
    problems = []

  반환값 예시:
    {
      "chunk": ([1, 2, 3], np.array([[0.01, -0.02, ...], ...])),
      "product": (["P001", "P002"], np.array([[0.03, 0.04, ...], ...])),
    }

  딕셔너리의 값은 (ID 목록, NumPy 벡터 행렬) 구조이다.

  핵심 개념:
    matrix.shape[0]은 벡터 개수이고 matrix.shape[1]은 벡터 하나의 차원이다.
    서로 다른 차원의 벡터는 행렬 곱셈을 할 수 없다. 또한 차원이 같더라도 서로 다른
    임베딩 모델로 만든 벡터는 숫자의 의미 체계가 달라 직접 비교하면 안 된다.

  이 함수가 필요한 이유:
    추천과 검색을 시작하기 전에 모든 벡터가 같은 차원과 같은 모델을 사용하는지
    확인하여, 실행 중 오류나 의미 없는 유사도 비교를 미리 막는다.
  """
  vectors = {}

  for kind, key in kinds:
    ids, matrix = load_vectors(
      f"{kind}_vectors", key, connection=con,
    )
    vectors[kind] = (ids, matrix)
    print(
      f"  벡터 종류: {kind:8s} / 개수: {matrix.shape[0]:>6,} / "
      f"차원: {matrix.shape[1]}"
    )

  all_dims = {matrix.shape[1] for _, matrix in vectors.values()}
  check(
    all_dims == {expected_dim},
    f"벡터 차원이 설정값과 다름 (설정 {expected_dim}, 실제 {all_dims})",
    problems,
  )

  all_models = {
    model
    for kind, _ in kinds
    for (model,) in con.execute(f"SELECT DISTINCT model FROM {kind}_vectors")
  }
  check(
    all_models == {expected_model},
    f"벡터 모델이 설정값과 다름 (설정 {expected_model}, 실제 {all_models})",
    problems,
  )

  return vectors


# 벡터의 TEXT 저장 크기와 BLOB 예상 크기를 계산하는 함수
def check_vector_storage(con, kinds, vectors, embed_dim):
  """
  인자로 들어오는 값 예시:
    con = sqlite3.connect(DB_PATH)
    kinds = (("chunk", "chunk_id"), ("product", "product_id"))
    vectors = check_vector_data(...)가 반환한 딕셔너리
    embed_dim = 384

  반환값 예시:
    {
      "vector_count": 3260,
      "text_bytes": 13003161,
      "blob_bytes": 5007360,
    }

  text_bytes는 벡터 문자열 길이의 합이고 blob_bytes는 float32 기준 예상값이다.

  핵심 개념:
    현재 벡터는 "[0.12, -0.03, ...]" 같은 TEXT 문자열로 저장된다.
    BLOB은 같은 숫자를 이진 형식으로 저장하는 방식이며 보통 더 작고 빠르지만,
    사람이 SELECT 결과를 바로 읽기는 어렵다.

  주의할 점:
    SQLite의 LENGTH(TEXT)는 기본적으로 문자 수를 센다. 벡터 문자열은 대부분 숫자와
    ASCII 기호라서 바이트 크기와 비슷하지만 정확한 DB 파일 크기와 완전히 같지는 않다.
    blob_bytes 역시 인덱스와 테이블 부가 공간을 제외한 순수 벡터 크기 예상값이다.
  """
  vector_count = sum(matrix.shape[0] for _, matrix in vectors.values())
  text_bytes = sum(
    con.execute(
      f"SELECT COALESCE(SUM(LENGTH(vector)), 0) FROM {kind}_vectors"
    ).fetchone()[0]
    for kind, _ in kinds
  )

  # float32 숫자 하나는 4바이트를 사용한다.
  blob_bytes = vector_count * embed_dim * 4

  print(f"  전체 벡터 개수: {vector_count:,}개")
  print(f"  TEXT 문자열 길이 합계: {text_bytes:,}")
  print(f"  float32 BLOB 예상 크기: {blob_bytes:,}바이트")

  return {
    "vector_count": vector_count,
    "text_bytes": text_bytes,
    "blob_bytes": blob_bytes,
  }


# 문서 조각이 임베딩 모델의 토큰 상한을 넘는지 검사하는 함수
def check_token_sizes(con, max_tokens, problems):
  """
  인자로 들어오는 값 예시:
    con = sqlite3.connect(DB_PATH)
    max_tokens = 512
    problems = []

  반환값 예시:
    {
      "chunk_tokens": [42, 57, 81, ...],
      "section_tokens": [120, 245, 310, ...],
      "over": 0,
      "average_chunk_tokens": 63.4,
    }

  over는 max_tokens를 초과한 문서 조각의 개수이다.
  average_chunk_tokens는 문서 조각 하나당 평균 토큰 수이다.

  핵심 개념:
    토큰은 단순한 글자 수나 단어 수가 아니라 임베딩 모델이 문장을 나누는 단위이다.
    모델의 최대 토큰 수를 넘으면 문장 뒤쪽이 잘릴 수 있어 중요한 정보가 벡터에
    반영되지 않을 수 있다.

  section과 chunk의 차이:
    section은 사람이 구분한 원문 단위이고, chunk는 실제 임베딩에 넣는 더 작은 단위다.
    비용과 검색 품질을 판단할 때는 실제 입력인 chunk의 토큰 수가 더 직접적인 기준이다.
  """
  chunk_tokens = sorted(
    n for (n,) in con.execute("SELECT n_tokens FROM chunks")
  )
  section_tokens = [
    n for (n,) in con.execute("SELECT n_tokens FROM sections")
  ]

  over = sum(n > max_tokens for n in chunk_tokens)
  average_chunk_tokens = sum(chunk_tokens) / len(chunk_tokens)
  check(
    over == 0,
    f"토큰 상한({max_tokens})을 넘는 조각이 있음 ({over}개)",
    problems,
  )

  remaining = (1 - max(chunk_tokens) / max_tokens) * 100
  print(f"  조각당 평균 토큰 수: {average_chunk_tokens:.1f}")
  print(f"  가장 긴 조각도 토큰 여유가 {remaining:.0f}% 남음")

  return {
    "chunk_tokens": chunk_tokens,
    "section_tokens": section_tokens,
    "over": over,
    "average_chunk_tokens": average_chunk_tokens,
  }


# 고객과 상품·조각의 유사도를 세 가지 추천 점수로 계산하는 함수
def calculate_scores(customer_vectors, product_vectors, chunk_vectors, chunk_ids, product_ids, product_of):
  """
  인자로 들어오는 값 예시:
    customer_vectors = np.array([[0.01, -0.02, ...], ...])
    product_vectors = np.array([[0.03, 0.04, ...], ...])
    chunk_vectors = np.array([[0.05, -0.01, ...], ...])
    chunk_ids = [1, 2, 3, ...]
    product_ids = ["P001", "P002", ...]
    product_of = {1: "P001", 2: "P001", 3: "P002"}

  반환값 예시:
    {
      "상품 요약 벡터 (기준선)": np.array([[0.81, 0.42, ...], ...]),
      "조각 벡터 · max 로 합치기": np.array([[0.87, 0.51, ...], ...]),
      "조각 벡터 · mean 으로 합치기": np.array([[0.72, 0.39, ...], ...]),
    }

  각 점수 배열의 모양은 (고객 수, 상품 수)이다.
  """
  
  """
  세 가지 방식의 의미:
    1. 상품 요약 벡터
       상품의 이름, 브랜드, 카테고리, 성분 등을 한 문장으로 합쳐 만든 벡터 하나와
       고객 벡터를 직접 비교한다. 가장 단순하므로 다른 방식의 기준선으로 사용한다.

    2. 조각 벡터 max
       한 상품에 속한 여러 조각 중 고객과 가장 비슷한 조각의 점수를 상품 점수로 쓴다.
       조각 하나라도 강하게 관련되면 상품 점수가 높아지지만, 우연히 높은 조각 하나에
       지나치게 영향을 받을 수 있다. 조각이 많은 상품이 유리해질 가능성도 있다.

    3. 조각 벡터 mean
       한 상품에 속한 모든 조각 점수의 평균을 상품 점수로 쓴다. 상품 내용이 전반적으로
       고객과 비슷한지 볼 수 있지만, 중요한 조각 하나가 관련 없는 조각들에 묻힐 수 있다.

  사용할 수 있는 데이터 조건:
    이 비교는 "부모 대상 하나 + 그 대상에 속한 여러 조각" 구조에서 사용할 수 있다.
    예를 들어 상품과 상세 문단, 논문과 각 절, 문서와 각 페이지에 적용할 수 있다.
    부모·조각 관계가 없거나 부모 요약 텍스트가 없다면 세 방식을 그대로 비교할 수 없다.

  주의할 점:
    summary, max, mean 중 언제나 가장 좋은 정답은 없다. 데이터 특성과 추천 목적에 따라
    hit@k 같은 실제 평가 결과를 보고 선택해야 한다.
  """
  # 1단계. 모든 고객 × 모든 상품 요약 벡터의 유사도 → (고객 300, 상품 200)
  # .T 로 상품 벡터를 뒤집어야 가운데 384차원이 맞물려 곱셈이 된다.

  """
      (300, 384)  @  (384, 200)
        ↑   └───-----─┘    ↑
      고객정보    백터 차원    상품정보

    상품정보의 행과열으 바꿔 고객과 상품의 백터 차원을 맞춘다
    결과에는 모든 고객과 모든 상품 사이의 유사도 점수가 담긴다

    product_scores = customer_vectors @ product_vectors.T


    상품 벡터 배열의 행과 열을 바꾸는 이유는,
    고객 벡터와 상품 벡터를 같은 벡터 차원을 기준으로 비교하기 위해서이다.
    그 결과, 각 고객과 각 상품의 유사도 점수가 다음과 같은 표로 만들어진다.

             상품1  상품2  상품3
    고객1     점수   점수   점수
    고객2     점수   점수   점수
    고객3     점수   점수   점수

    product_scores = customer_vectors @ product_vectors.T
  """

  product_scores = customer_vectors @ product_vectors.T

  # 2단계. 같은 방법으로 고객 × 조각 → (고객 300, 조각 1560)
  chunk_scores = customer_vectors @ chunk_vectors.T

  # 3단계. "P001의 조각은 chunk_scores의 몇 번째 열인가"를 미리 모아 둔다.
  # 만들려는 것: {"P001": [0, 1, 2], "P002": [3, 4], ...}
  chunk_positions = {}

  for position, chunk_id in enumerate(chunk_ids):
    product_id = product_of[chunk_id]
    chunk_positions.setdefault(product_id, []).append(position)

  # 4단계. 상품마다 자기 조각 점수만 뽑아 상품 점수 하나로 합친다.
  # 상품 한 개당 배열 하나가 쌓이고, 그 배열에는 고객 300명의 점수가 들어 있다.
  max_score_columns = []
  mean_score_columns = []

  for product_id in product_ids:
    positions = chunk_positions[product_id]

    # 이 상품의 조각 열만 뽑는다 → (고객 300, 이 상품의 조각 수)
    #          조각0  조각1  조각2         max    mean
    #   고객1   0.81   0.42   0.90  →   0.90   0.71
    #   고객2   0.22   0.60   0.15  →   0.60   0.32
    #         └─ axis=1: 이 가로 방향의 대표값을 뽑는다 ─┘
    product_chunk_scores = chunk_scores[:, positions]
    

    # 고객마다 가장 잘 맞는 조각 하나의 점수
    max_score_columns.append(product_chunk_scores.max(axis=1))

    # 고객마다 모든 조각 점수의 평균
    mean_score_columns.append(product_chunk_scores.mean(axis=1))

  # 5단계. 지금은 상품별 점수 배열이 200개로 따로 놀고 있다. 이걸 표 하나로 합친다.
  #   합치기 전 : [ P001 점수배열, P002 점수배열, ... ]
  #                       P001  P002
  #   합친 후   :  고객1   0.90  0.55
  #               고객2   0.60  0.81
  # axis=1 은 상품을 열에 놓으라는 뜻. 빼면 행과 열이 반대로 만들어진다.
  max_scores = np.stack(max_score_columns, axis=1)
  mean_scores = np.stack(mean_score_columns, axis=1)

  # 세 방식의 점수표를 이름표와 함께 돌려준다. 셋 다 (고객 300, 상품 200)이다.
  return {
    "상품 요약 벡터 (기준선)": product_scores,
    "조각 벡터 · max 로 합치기": max_scores,
    "조각 벡터 · mean 으로 합치기": mean_scores,
  }


# 추천 점수로 hit@1·3·5 성공률을 계산하는 함수
def hit_at(scores, customer_ids, product_ids, bought, answers, ks=(1, 3, 5)):
  """
  인자로 들어오는 값 예시:
    scores = np.array([[0.91, 0.35], [0.22, 0.88]])
    customer_ids = ["C001", "C002"]
    product_ids = ["P001", "P002"]
    bought = {"C001": {"P002"}, "C002": {"P001"}}
    answers = {"C001": "P001", "C002": "P002"}
    ks = (1, 3, 5)

  반환값 예시:
    {1: 3.0, 3: 6.3, 5: 11.0}

  반환된 숫자는 각 추천 범위에서 정답을 맞힌 고객 비율(%)이다.

  핵심 개념:
    hit@1은 추천 1개 안에 정답이 있는 비율, hit@3은 상위 3개 안에 정답이 있는 비율,
    hit@5는 상위 5개 안에 정답이 있는 비율이다. k가 커지면 맞힐 기회가 늘어나므로
    보통 hit@1보다 hit@3, hit@3보다 hit@5가 높다.

  평가 방법:
    is_holdout=1인 최신 구매 상품을 숨겨 둔 정답으로 사용한다. 과거에 이미 구매한
    상품은 추천 후보에서 제외하고, 모델이 숨겨 둔 상품을 상위 k 안에 추천하는지 본다.

  주의할 점:
    hit@k는 정답 상품이 목록 안에 있는지만 보며 순위 사이의 세밀한 차이나 추천의
    다양성은 측정하지 않는다. 필요하면 MRR, NDCG 같은 지표를 추가할 수 있다.
  """
  hits = {k: 0 for k in ks}

  for row, customer_id in enumerate(customer_ids):
    order = np.argsort(-scores[row])
    ranked = [
      product_ids[i] for i in order
      if product_ids[i] not in bought.get(customer_id, ())
    ]

    for k in ks:
      hits[k] += answers[customer_id] in ranked[:k]

  return {k: count / len(customer_ids) * 100 for k, count in hits.items()}


# 세 가지 추천 방식의 hit@1·3·5 결과와 평균 토큰 수를 비교하는 함수
def compare_recommendations(con, vectors, token_result):
  """
  인자로 들어오는 값 예시:
    con = sqlite3.connect(DB_PATH)
    vectors = {
      "customer": (고객 ID 목록, 고객 벡터 행렬),
      "product": (상품 ID 목록, 상품 벡터 행렬),
      "chunk": (조각 ID 목록, 조각 벡터 행렬),
    }
    token_result = {
      "average_chunk_tokens": 63.4,
      ...
    }

  반환값 예시:
    {
      "상품 요약 벡터 (기준선)": {1: 3.0, 3: 6.3, 5: 11.0},
      "조각 벡터 · max 로 합치기": {1: 2.0, 3: 7.0, 5: 10.0},
      "조각 벡터 · mean 으로 합치기": {1: 2.7, 3: 6.0, 5: 10.0},
    }

  출력 표에는 각 방식이 사용하는 검색 대상 텍스트의 평균 토큰 수도 표시한다.
  상품 방식은 product_text() 결과를, 조각 방식은 chunks.n_tokens를 기준으로 한다.

  이 함수가 하는 일:
    calculate_scores()로 세 방식의 고객별 상품 점수를 만들고, 각 점수를 hit_at()에
    전달해 hit@1·3·5를 계산한다. 즉 실제 점수 계산과 성능 평가를 이어 주는 함수다.

  평균 토큰을 함께 보는 이유:
    hit 값만 보면 추천 품질만 알 수 있다. 평균 토큰을 함께 보면 한 데이터 벡터를
    만드는 데 들어간 입력 길이와 성능을 같이 비교할 수 있어 비용 대비 결과를 판단하기
    쉽다. 상품 요약은 상품 검색문장의 평균, max와 mean은 같은 조각의 평균을 사용한다.

  해석할 때 주의할 점:
    평균 토큰은 벡터 하나당 입력 길이다. 전체 임베딩 비용은 평균 토큰뿐 아니라 벡터
    개수에도 영향을 받는다. 상품당 벡터 하나인 요약 방식과 상품당 여러 벡터가 필요한
    조각 방식의 총비용을 정확히 비교하려면 전체 토큰 합계와 벡터 개수도 함께 봐야 한다.

  범용성:
    이 세 방식은 모든 벡터 데이터에 무조건 적용되는 표준이 아니다. 하나의 부모 데이터에
    요약 벡터 하나와 여러 조각 벡터가 있을 때 사용할 수 있는 기본 비교 후보들이다.
  """
  customer_ids, customer_vectors = vectors["customer"]
  product_ids, product_vectors = vectors["product"]
  chunk_ids, chunk_vectors = vectors["chunk"]

  answers = dict(con.execute("SELECT customer_id, product_id FROM purchases WHERE is_holdout = 1"))
  bought = {}
  for customer_id, product_id in con.execute("SELECT customer_id, product_id FROM purchases WHERE is_holdout = 0"):
    bought.setdefault(customer_id, set()).add(product_id)

  product_of = dict(
    con.execute("SELECT chunk_id, product_id FROM chunks")
  )

  # 03_embed.py와 같은 방법으로 상품 검색문장을 다시 만들어 토큰 수를 센다.
  product_rows = con.execute("""
      SELECT name, brand, category, price, skin_type, ingredient, concern, tags, description
      FROM products ORDER BY product_id
  """).fetchall()
  product_texts = [embedding.product_text(row) for row in product_rows]
  average_product_tokens = sum(chunking.count_tokens(text) for text in product_texts) / len(product_texts)

  # max와 mean 방식은 같은 조각 벡터를 사용하므로 평균 토큰 수가 같다.
  average_chunk_tokens = token_result["average_chunk_tokens"]
  average_tokens = {
    "상품 요약 벡터 (기준선)": average_product_tokens,
    "조각 벡터 · max 로 합치기": average_chunk_tokens,
    "조각 벡터 · mean 으로 합치기": average_chunk_tokens,
  }

  started = time.perf_counter()
  score_sets = calculate_scores(customer_vectors, product_vectors, chunk_vectors,  chunk_ids, product_ids, product_of,)
  elapsed = time.perf_counter() - started

  print(f"  고객 {len(customer_ids)}명 · 상품 {len(product_ids)}개 · 조각 {len(chunk_ids):,}개 비교: {elapsed * 1000:.0f}ms\n")
  print(f"  {'무엇으로 찾나':28s} {'평균 토큰':>10s} {'hit@1':>7s} {'hit@3':>7s} {'hit@5':>7s}")

  hit_results = {}
  for label, scores in score_sets.items():
    hits = hit_at(scores, customer_ids, product_ids, bought, answers)
    hit_results[label] = hits
    print(f"  {label:28s} {average_tokens[label]:>10.1f} {hits[1]:>6.1f}% {hits[3]:>6.1f}% {hits[5]:>6.1f}%" )

  return hit_results


# 예시 질문과 의미가 가까운 문서 조각을 찾아 출력하는 함수
def inspect_search_results(con, chunk_ids, chunk_vectors, questions):
  """
  인자로 들어오는 값 예시:
    con = sqlite3.connect(DB_PATH)
    chunk_ids = [1, 2, 3, ...]
    chunk_vectors = np.array([[0.01, -0.02, ...], ...])
    questions = ["배송은 얼마나 걸리나요", "환불은 어떻게 하나요"]

  반환값 예시:
    {
      "배송은 얼마나 걸리나요": ["배송 및 교환", "배송 및 교환", "상품 설명"],
      "환불은 어떻게 하나요": ["배송 및 교환", "자주 묻는 질문", "주의사항"],
    }

  각 질문의 값에는 유사도 상위 3개 문서 조각의 섹션 이름이 들어간다.

  핵심 개념:
    질문 벡터와 모든 조각 벡터의 내적을 계산하고 점수가 높은 순서대로 3개를 고른다.
    현재 임베딩은 정규화되어 있으므로 내적값을 코사인 유사도처럼 사용할 수 있다.

  이 함수가 필요한 이유:
    hit@k 같은 숫자 평가만으로는 실제 검색 문장이 자연스러운지 알기 어렵다. 대표 질문의
    검색 결과를 직접 읽어 보면 모델이 관련 의미를 제대로 연결하는지 확인할 수 있다.

  주의할 점:
    정규화되지 않은 벡터라면 단순 내적이 벡터 길이의 영향을 받는다. 그 경우 별도의
    코사인 유사도 계산이나 벡터 정규화가 필요하다.
  """
  from app.core.embedder import get_embeddings

  model = get_embeddings()
  question_vectors = np.array(model.embed_documents(questions), dtype="float32")

  meta = {
    chunk_id: (product_id, section, body)
    for chunk_id, product_id, section, body
    in con.execute("SELECT chunk_id, product_id, section, body FROM chunks")
  }

  top_sections = {}
  for question, question_vector in zip(questions, question_vectors):
    scores = chunk_vectors @ question_vector
    ranks = np.argsort(-scores)[:3]

    print(f"\n  Q. {question}")
    top_sections[question] = [meta[chunk_ids[rank]][1] for rank in ranks]

    for rank in ranks:
      product_id, section, body = meta[chunk_ids[rank]]
      print(
        f"     {scores[rank]:.3f}  [{product_id} > {section}] "
        f"{body[:44].replace(chr(10), ' ')}..."
      )

  print()
  for word in ("환불", "반품", "교환"):
    count = con.execute(
      "SELECT COUNT(*) FROM chunks WHERE body LIKE ?",
      (f"%{word}%",),
    ).fetchone()[0]
    print(f"  '{word}'이 들어간 조각: {count:,}개")

  refund_question = "환불하고 싶은데 어떻게 하나요"
  refund_top = top_sections.get(refund_question, [])
  if refund_top and refund_top[0] == "배송 및 교환":
    print("  임베딩이 '환불'과 '교환·반품'을 연결함 → 1위 [배송 및 교환]")
  elif refund_top:
    print(f"  임베딩이 두 의미를 연결하지 못함 → 1위 [{refund_top[0]}]")

  return top_sections


# 여섯 단계에서 발견된 문제를 마지막에 모아서 출력하는 함수
def print_final_result(problems):
  """
  인자로 들어오는 값 예시:
    problems = []
    또는
    problems = ["상품 벡터 개수가 다름 (195/200)"]

  반환값:
    없음(None)

  출력 예시:
    problems가 비어 있으면 "전부 통과"
    문제가 있으면 문제 개수와 각 오류 메시지를 출력한다.

  핵심 개념:
    각 단계는 같은 problems 리스트에 실패 메시지를 추가한다. 이 함수는 모든 단계가
    끝난 뒤 그 목록을 한 번에 확인하는 최종 보고 역할만 하며 새로운 검사는 하지 않는다.
  """
  print()
  print("=" * 70)

  if problems:
    print(f"문제 {len(problems)}건 ― 앱을 붙이기 전에 고친다")
    for message in problems:
      print(f"  - {message}")
  else:
    print("전부 통과")