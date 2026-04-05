# indexing/loader.py
import os
import re
import json
from langchain_core.documents import Document

def clean_text(text):
    # 去掉 \r \n \t 等控制符號
    text = text.replace("\r", "").replace("\n", "").replace("\t", "")
    # 去掉多餘空格
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def data_loader_by_years(base_dir, case_type=None, n_years=None):
    docs = []

    # 取得年份資料夾
    year_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    # 排序（新 → 舊）
    year_dirs = sorted(year_dirs, reverse=True)

    # 決定要取幾年
    if n_years is not None:
        year_dirs = year_dirs[:n_years]

    # 讀 JSON（每個年份資料夾內）
    for year in year_dirs:
        year_path = os.path.join(base_dir, year)

        for filename in os.listdir(year_path):
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(year_path, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            text = clean_text(data.get("JFULL").strip())

            docs.append(
                Document( 
                    page_content=text, 
                    metadata={ 
                        "JID": data.get("JID"), 
                        "JTITLE": data.get("JTITLE"), 
                        "JYEAR": data.get("JYEAR"), 
                        "JCASE": data.get("JCASE"), 
                        "JDATE": data.get("JDATE"), 
                        "JPDF": data.get("JPDF"), 
                        "COURT": "最高法院", 
                        "TYPE": case_type,  
                    } 
                ) 
            )

    return docs
