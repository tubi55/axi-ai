
import time
from collections import Counter

# 이 파일은 "무엇을 어떤 글로 적어 벡터로 만드나" 를 다룬다.
# 임베딩기(모델) 자체를 만드는 일은 app/core/embedder.py 가 한다 ― 이름이 비슷하니 주의.
# 앱과 파이프라인이 각자 만들면 언젠가 조건이 어긋나고,
# 그때 검색 결과는 오류 없이 이상해진다.
# 이 파일이 하는 일은 "어떤 글로 적나" 와 "그 글을 숫자로" 두 가지뿐이다.
from app.core.embedder import get_embeddings


# 상품 정보를 검색용 한 문장으로 만들어서 반환
def product_text(row):
  """
  인자로 들어오는 값 예시:
    row = ("수분 크림", "스킨랩", "크림", 18000, "건성","히알루론산", "건조함", "수분·보습", "촉촉한 보습 크림")

  반환값 예시:
    "수분 크림 · 스킨랩 · 크림 · 18000원 · 건성 · 히알루론산 · 건조함 · 수분·보습 · 촉촉한 보습 크림"

  """
  name, brand, category, price, skin_type, ingredient, concern, tags, desc = row
  return (f"{name} · {brand} · {category} · {price}원 · {skin_type} · "
          f"{ingredient} · {concern} · {tags} · {desc}")


# 가장 자주 나온 값 n개를 구해서 문자열로 반환
def top(values, n=3):
  """
  인자로 들어오는 값 예시:
    values = ["크림", "토너", "크림", "세럼", "크림", "토너"]
    n = 2

  반환값 예시:
    "크림 · 토너"

  n을 생략하면 기본값 3이 사용된다.
  """
  return " · ".join(value for value, _ in Counter(values).most_common(n))


# 고객의 피부 타입과 구매 이력을 취향 한 문장으로 반환
def customer_text(skin_type, purchases):
  """
  인자로 들어오는 값 예시:
    skin_type = "건성"
    purchases = [
      ("크림", "히알루론산", "건조함", 5),
      ("토너", "세라마이드", "민감성", 4),
    ]

  purchases 한 항목의 구조:
    (카테고리, 성분, 피부 고민, 별점)
    03_embed.py 가 이 모양으로 넣어 준다. 순서가 곧 약속이라 한쪽만 바꾸면
    엉뚱한 칸을 읽고, 오류 없이 이상한 문장이 만들어진다.

  반환값 예시:
    "건성 피부 · 선호 카테고리 크림 · 토너 · "
    "자주 쓴 성분 히알루론산 · 세라마이드 · "
    "관심 고민 건조함 · 민감성 · 평균 별점 4.5"
  """
  # 각 구매 정보의 마지막 값인 별점만 꺼낸다.
  ratings = [rating for *_, rating in purchases]

  return (f"{skin_type} 피부 · 선호 카테고리 {top([p[0] for p in purchases])} · "
          f"자주 쓴 성분 {top([p[1] for p in purchases])} · "
          f"관심 고민 {top([p[2] for p in purchases])} · "
          f"평균 별점 {sum(ratings) / len(ratings):.1f}")


# 글 목록을 벡터 목록으로 반환해주는 함수 (벡터라이징에 걸린 시간 출력)
def embed_documents(texts):
  """
  인자로 들어오는 값 예시:
    texts = ["수분 크림", "민감성 피부용 토너"]

  반환값 예시:
    (
      [
        [0.018, -0.024, 0.031, ...],
        [0.012, 0.007, -0.015, ...],
      ],
      0.42,
    )

  첫 번째 값은 벡터 목록이고 두 번째 값은 걸린 시간(초)이다.
  """
  started = time.perf_counter()
  vectors = get_embeddings().embed_documents(texts)
  return vectors, time.perf_counter() - started
