"""
전체 흐름 요약

  1. 테이블의 데이터 개수와 연결 상태를 검사한다.
  2. 벡터를 불러와 차원과 모델 이름을 검사한다.
  3. 벡터의 현재 저장 크기와 BLOB 예상 크기를 계산한다.
  4. 임베딩 토큰 상한을 넘는 문서 조각이 있는지 검사한다.
  5. 세 가지 추천 방식의 hit@1·3·5 결과를 비교한다.
  6. 예시 질문으로 실제 검색 결과를 확인한다.

  04_verify.py는 검사 순서와 입력값을 보여 주고,
  verifying.py는 각 검사를 실제로 수행한다.
"""

import sqlite3
import sys
from pathlib import Path

# 프로젝트 최상위 폴더를 파이썬 모듈 검색 경로에 추가한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(errors="replace")

from app.core.config import DB_PATH, EMBED_DIM, EMBED_MAX_TOKENS, EMBED_MODEL
from pipeline.prep import verifying

# 모든 단계에서 함께 사용할 DB 연결과 문제 목록을 준비한다.
con = sqlite3.connect(DB_PATH)
problems = []

# 벡터 종류와 각 벡터 테이블의 ID 컬럼 이름이다.
KINDS = (
  ("chunk", "chunk_id"),
  ("product", "product_id"),
  ("customer", "customer_id"),
  ("review", "purchase_id"),
)


# ============================================================
# 1단계. 테이블 개수와 연결 상태 검사
# ============================================================

# 검사할 테이블 이름을 준비한다.
TABLE_NAMES = (
  "customers", "products", "purchases", "product_details",
  "sections", "chunks", "chunk_vectors", "product_vectors",
  "customer_vectors", "review_vectors",
)

# 테이블별 개수와 잘못 연결된 데이터가 있는지 검사한다.
# counts = verifying.check_table_data(con, TABLE_NAMES, problems)


# ============================================================
# 2단계. 벡터 차원과 모델 검사
# ============================================================

# 벡터를 불러오고 네 종류의 차원과 모델이 같은지 검사한다.
vectors = verifying.check_vector_data(
  con, KINDS, EMBED_DIM, EMBED_MODEL, problems,
)


# ============================================================
# 3단계. 벡터 저장 크기 확인
# ============================================================

# TEXT로 저장한 현재 크기와 BLOB으로 저장할 때의 예상 크기를 계산한다.
# storage_result = verifying.check_vector_storage(
#   con, KINDS, vectors, EMBED_DIM,
# )


# ============================================================
# 4단계. 조각 토큰 수 검사
# ============================================================

# 임베딩 모델의 토큰 상한을 넘는 조각이 있는지 검사한다.
# token_result = verifying.check_token_sizes(
#   con, EMBED_MAX_TOKENS, problems,
# )


# ============================================================
# 5단계. 추천 방식별 hit@1·3·5 비교
# ============================================================

# 상품 요약, 조각 최고점, 조각 평균 방식의 추천 성능을 비교한다.
# hit_results = verifying.compare_recommendations(
#   con, vectors, token_result,
# )
# print("hit_results", hit_results)


# ============================================================
# 6단계. 예시 질문으로 검색 결과 확인
# ============================================================

# # 검색 검사에 필요한 조각 ID와 벡터를 준비한다.
# chunk_ids, chunk_vectors = vectors["chunk"]
# questions = [ "환불하고 싶은데 어떻게 하나요"]

# # 질문별로 가장 비슷한 문서 조각을 찾아 눈으로 확인한다.
# top_sections = verifying.inspect_search_results(
#   con, chunk_ids, chunk_vectors, questions,
# )

print(verifying.search_any(con, "review", ["환불하고 싶은데 어떻게 하나요?"]))


# 조각이 아닌 다른 자료를 근거로 찾고 싶으면 search_any()에 종류 이름만 넘긴다.
# 원본 테이블과 ID 컬럼은 벡터 테이블의 외래 키를 보고 알아서 찾는다.
# verifying.search_any(con, "review", ["배송이 너무 느렸어요"])
# verifying.search_any(con, "product", ["건성 피부에 좋은 수분 크림"])

# 여섯 단계에서 발견한 문제를 모아 최종 출력한다.
# verifying.print_final_result(problems)
con.close()

