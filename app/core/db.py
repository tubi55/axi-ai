import json
import sqlite3

from app.core.config import DB_PATH

con = sqlite3.connect(DB_PATH)


def query(sql, params=()):
    """여러 줄을 꺼낸다. 튜플의 목록이 온다."""
    return con.execute(sql, params).fetchall()


def one(sql, params=()):
    """한 줄만 꺼낸다. 없으면 None 이 온다."""
    return con.execute(sql, params).fetchone()


def dicts(sql, params=()):
    """컬럼 이름이 붙은 딕셔너리 목록으로 꺼낸다."""
    cur = con.execute(sql, params)
    columns = [c[0] for c in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


# 문자로 넣어둔 백터정보를 Numpy 행렬로 숫자화해서 되살리는 함수
def load_vectors(table, key, connection=None):
    import numpy as np

    # 만약 해당 함수를 호출하는 파일에 con접속객체가 있으면 그걸 재활용하고 없으면 새로 만들어서 전달
    active_con = connection if connection is not None else con

    # DB에 가지고온 id값과 벡터 좌표값을 담을 빈 리스트 2개 생성
    ids, rows = [], []

    # 인수로 전달된 테이블에서 ID열과 vector 열을 한 행씩 가져옴
    for row_id, vector in active_con.execute(f"SELECT {key}, vector FROM {table}"):
        ids.append(row_id)
        # 리스트에 따옴표가 붙어있어서 통짜로 문자화되어 있는 데이터를 json객체형태로 변경
        rows.append(json.loads(vector))

    # 객체안쪽에 있는 vector안쪽의 좌표값을 다시 숫자형태로 변경
    return ids, np.array(rows, dytpe="float3")







if __name__ == "__main__":
    print("고객 수:", one("SELECT COUNT(*) FROM customers")[0])
    print("상품 수:", one("SELECT COUNT(*) FROM products")[0])
    for row in dicts("SELECT name, age FROM customers LIMIT 3"):
        print(f"  {row['name']} ({row['age']}세)")
