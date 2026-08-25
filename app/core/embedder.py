"""
  임베딩기(모델) 한 개를 만들어 준다. 그게 이 파일이 하는 일의 전부다.

  왜 한 벌이어야 하나:
    예전에는 앱과 파이프라인이 각자 하나씩 들고 있었다. 그러면 한쪽만 고쳐도
    아무도 모른다. 질문 벡터와 문서 벡터가 서로 다른 조건으로 만들어지고,
    검색 결과가 이상해지는데 오류는 나지 않는다.
    "같은 조건으로 만들자" 를 주석으로 약속하는 대신 같은 함수를 부르게 했다.

  왜 pipeline/ 이 아니라 app/ 에 두나:
    앱은 파이프라인 없이도 떠야 한다. 서비스를 배포할 때 pipeline/ 은 따라가지 않는다.
    그래서 app/ 이 pipeline/ 을 import 하면 안 된다. 반대 방향(pipeline -> app)은
    이미 하고 있다. 둘이 같이 쓰는 것은 app/ 쪽 아래층에 있어야 한다.

  왜 features/ 가 아니라 core/ 에 두나:
    features/ 의 여러 파일이 이 모델을 쓴다. 그중 한 파일에 두면 나머지가 그 파일을
    import 해야 하고, 서로를 import 하는 고리가 생겨 ImportError 로 죽는다.
    아무도 가져다 쓰지 않는 아래층(core/)에 두면 고리가 안 생긴다.
"""

from app.core.config import EMBED_MODEL

_embeddings = None       # 임베딩기는 무겁다. 첫 질문이 들어올 때 올린다


def get_embeddings():
  """질문을 숫자로 바꾸는 모델. 처음 부를 때 한 번만 올린다."""
  # 함수 밖의 _embeddings 변수에 모델을 저장하기 위해 global을 사용한다.
  global _embeddings

  # 모델이 이미 만들어져 있으면 새로 만들지 않고 그대로 재사용한다.
  if _embeddings is not None:
    return _embeddings

  # 모델을 불러올 때 표시되는 진행 바와 불필요한 안내 문구를 숨길 도구이다.
  from huggingface_hub.utils import disable_progress_bars
  from huggingface_hub.utils import logging as hub_logging

  hub_logging.set_verbosity_error()
  disable_progress_bars()


  from langchain_huggingface import HuggingFaceEmbeddings

  # DB의 기존 벡터를 만들 때 사용한 것과 같은 임베딩 모델을 준비한다.
  _embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
  )
  
  return _embeddings