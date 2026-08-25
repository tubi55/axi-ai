import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(errors="replace")

from huggingface_hub.utils import disable_progress_bars
from huggingface_hub.utils import logging as hub_logging


hub_logging.set_verbosity_error()
disable_progress_bars()

from app.core.config import DB_PATH
from pipeline.prep import chunking, storage


con = sqlite3.connect(DB_PATH)
con.execute("PRAGMA foreign_keys = ON")



details = con.execute("""
      SELECT product_details.product_id, products.name, product_details.detail
      FROM product_details
      JOIN products ON product_details.product_id = products.product_id
      ORDER BY product_details.product_id
""").fetchall()

# 이 한 줄이 자르기의 전부다. 어떻게 자르는지는 prep/chunking.py 가 안다
sections, chunks, n_resplit = chunking.split_details(details)

#  이 한 줄이 저장의 전부다. 어떻게 넣는지는 prep/storage.py 가 안다
storage.save_sections_and_chunks(con, sections, chunks)
con.close()
