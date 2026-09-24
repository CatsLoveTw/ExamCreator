# =========================================================================
# 題庫資料庫雙向融合仲裁、幽靈清理與知識點淨化引擎
# =========================================================================
import os
import re
import json
import shutil
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATABASE_DIR = "./exam_database_output"
TICKETS_FILE = "exam_issue_tickets.json"

GHOST_KEYWORDS_EXTENDED = [
    "本題於原試卷中不存在", "全卷掃描漏失", "無完整題目文本", 
    "題目文字未在影像中提供", "此頁面為試卷封面", "本頁為「大學入學考試中心",
    "作答注意事項", "無實質試題文字", "本題不存在", "本頁為封面", "此頁面為試題封面",
    "題目內容缺失", "題目闕如", "無法判定", "未包含第", "無實質試題", "免予計分"
]

def clean_latex_corruptions(text: str) -> str:
    if not isinstance(text, str) or not text: return text
    text = re.sub(r'\\?fty\b', r'\\infty', text)
    text = re.sub(r'(?<!\\)\bimplies\b', r'\\implies', text)
    text = re.sub(r'(?<!\\)\biff\b', r'\\iff', text)
    text = re.sub(r'\\?bsim\b', r'\\approx', text)
    text = re.sub(r'\\bar\{log\}', r'\\log', text)
    text = text.replace(r"\r\right", r"\right").replace(r"\r\left", r"\left").replace(r"\r", "")
    text = re.sub(r'[\x0c]rac(?![a-zA-Z])', r'\\frac', text)
    text = re.sub(r'(?<=\$)rac(?=\{)', r'\\frac', text)
    return text

def evaluate_solution_score(q: dict) -> int:
    sol = str(q.get("detailed_solution", "")).strip()
    score = len(sol)
    for kw in GHOST_KEYWORDS_EXTENDED:
        if kw in sol: score -= 80000
    if "超時或失敗" in sol or "崩潰" in sol: score -= 50000
    if "### 【標準解法】" in sol or "【標準解法】" in sol: score += 1000
    if "### 【另解" in sol or "【另解" in sol: score += 1200
    if "### 【速解" in sol or "【速解" in sol: score += 800
    if "![圖" in sol or "diagram_" in sol: score += 600
    opt_ana = str(q.get("options_analysis", ""))
    if opt_ana and "本題為非選擇題" not in opt_ana and len(opt_ana) > 50: score += 800
    return score

def is_valid_question(q: dict) -> bool:
    """精準過濾幽靈題與無效題"""
    q_text = str(q.get("question_text", "")).strip()
    ans_text = str(q.get("answer", "")).strip()
    cat_text = str(q.get("topic_category", "")) + " " + " ".join(q.get("topic_categories", []))
    ana_text = str(q.get("question_analysis", ""))
    
    if len(q_text) < 5 and not q.get("has_image"): return False
    combined = f"{q_text} | {ans_text} | {cat_text} | {ana_text}"
    if any(gk in combined for gk in GHOST_KEYWORDS_EXTENDED): return False
    if ans_text in ["/", "／"] and len(q_text) < 15 and not q.get("has_image"): return False
    return True

def natural_sort_key(s):
    cn_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
    s_str = str(s).strip()
    for cn, num in cn_map.items(): s_str = s_str.replace(cn, f"{num:02d}")
    parts = re.split(r'(\d+)', s_str)
    return [int(text) if text.isdigit() else text.lower() for text in parts]

def sanitize_knowledge_point(cat_str: str) -> str:
    """清洗知識點，確保第 1 階層必為『必修』或『選修』，並剔除題目闕如等垃圾文字"""
    cat_str = str(cat_str).strip()
    if any(gk in cat_str for gk in ["題目闕如", "無法判定", "未包含", "缺失", "無效", "說明頁"]):
        return "必修_綜合主題_綜合概念應用"
        
    if cat_str.startswith("生物_"):
        cat_str = "選修_" + cat_str[3:]
        
    parts = cat_str.split("_")
    if len(parts) >= 2 and parts[0] not in ["必修", "選修"]:
        cat_str = "必修_" + "_".join(parts[1:])
    elif len(parts) < 2:
        cat_str = "必修_綜合主題_核心概念"
        
    return cat_str

def reconcile_and_merge_database_pair(db_path: str, partial_path: str, raw_path: str):
    """
    核心融合機制：
    1. 同時載入 _database.json 與 _partial_database.json
    2. 取兩者題目之聯集，同題號保留品質得分較高者
    3. 清洗知識點與 LaTeX 殘損
    4. 依完整度裁決存為 _database 還是 _partial
    """
    db_questions = []
    partial_questions = []
    expected_total = 0
    
    # 讀取 Stage 1 藍圖（若有）以獲取真實總題數
    if os.path.exists(raw_path):
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                expected_total = len([q for q in raw_data if is_valid_question(q)])
        except Exception: pass

    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                db_questions = json.load(f)
        except Exception: pass

    if os.path.exists(partial_path):
        try:
            with open(partial_path, "r", encoding="utf-8") as f:
                partial_questions = json.load(f)
        except Exception: pass

    # 1. 雙向融合池 (Union Pool)
    merged_map = {}
    
    for q in db_questions + partial_questions:
        if not is_valid_question(q):
            continue
        q_num = str(q.get("question_number", "")).strip()
        q_type = str(q.get("question_type", "")).strip()
        key = (q_num, q_type)
        
        if key not in merged_map:
            merged_map[key] = q
        else:
            # 兩邊都有該題，比對詳解品質！保留高分者
            old_score = evaluate_solution_score(merged_map[key])
            new_score = evaluate_solution_score(q)
            if new_score > old_score:
                merged_map[key] = q

    merged_list = list(merged_map.values())
    
    # 2. 清洗知識點、年代與格式
    for q in merged_list:
        # 清洗知識點
        main_cat = sanitize_knowledge_point(q.get("topic_category", ""))
        cats = q.get("topic_categories", [])
        if not isinstance(cats, list) or not cats:
            cats = [main_cat]
        else:
            cats = [sanitize_knowledge_point(c) for c in cats]
        q["topic_categories"] = list(dict.fromkeys(cats))
        q["topic_category"] = q["topic_categories"][0]
        
        # 清洗年代 (< 111 年改指考)
        year_raw = str(q.get("academic_year", ""))
        digits = "".join(filter(str.isdigit, year_raw))
        if digits and int(digits) < 111:
            q["academic_year"] = q.get("academic_year", "").replace("分科", "指考")
            q["exam_source"] = q.get("exam_source", "").replace("分科測驗", "指定科目考試")
            
        # 清洗 LaTeX
        for k in ["question_text", "detailed_solution", "question_analysis", "solving_strategy"]:
            if k in q: q[k] = clean_latex_corruptions(q[k])
            
        # 清除選擇題偽造手寫評分
        if q.get("question_type") in ["單選題", "多選題"]:
            det = q.get("detailed_solution", "")
            if "### 【手寫題評分對照】" in det:
                q["detailed_solution"] = det.split("### 【手寫題評分對照】")[0].strip()

    # 排序
    def safe_sort(x):
        try: p = int(x.get("page_number", 1))
        except: p = 1
        return (p, natural_sort_key(x.get("question_number", "0")))
    merged_list.sort(key=safe_sort)
    
    current_count = len(merged_list)
    
    # 3. 🚨 狀態裁決 (State Adjudication)
    # 判定標準：若有 raw 則必須達標 raw 題數；若無 raw，則考量學科最低合理題數（數學大考至少 8 題，其餘至少 15 題）
    min_threshold = 8 if any(k in db_path for k in ["數", "math"]) else 15
    is_complete = False
    
    if expected_total > 0:
        is_complete = (current_count >= expected_total)
    else:
        is_complete = (current_count >= min_threshold)
        
    if is_complete:
        # ✅ 真實完工：寫入唯一 _database.json，物理銷毀 _partial
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=4)
        if os.path.exists(partial_path):
            os.remove(partial_path)
            logging.info(f"🏆 [{os.path.basename(db_path)}] 雙向融合完工 (共 {current_count} 題)，已清除 partial 暫存！")
    else:
        # ⚠️ 尚未完整（如只有 2 題或未達標）：降級保留為 _partial，物理刪除殘缺的 _database！
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=4)
        if os.path.exists(db_path):
            os.remove(db_path)
            logging.warning(f"⚠️ [{os.path.basename(db_path)}] 題目殘缺 (僅 {current_count} 題，預期 {expected_total or min_threshold} 題)，已降級為 partial 並刪除假完成 database，以利主程序接力重跑！")

def run_database_repair_suite():
    if not os.path.exists(DATABASE_DIR):
        print(f"❌ 找不到目錄: {DATABASE_DIR}")
        return

    print("=" * 70)
    print("🚀 啟動【題庫資料庫雙向融合與狀態機仲裁修復引擎】...")
    print("=" * 70)

    # 收集所有考卷前綴
    exam_bases = set()
    for root, dirs, files in os.walk(DATABASE_DIR):
        for f in files:
            if f.endswith("_database.json") or f.endswith("_partial_database.json"):
                base_name = f.replace("_partial_database.json", "").replace("_database.json", "")
                exam_bases.add(os.path.join(root, base_name))

    for base in exam_bases:
        db_p = f"{base}_database.json"
        partial_p = f"{base}_partial_database.json"
        raw_p = f"{base}_raw_extracted.json"
        reconcile_and_merge_database_pair(db_p, partial_p, raw_p)

    print("=" * 70)
    print("🎉 資料庫修復與狀態機校準圓滿完成！")
    print("=" * 70)

if __name__ == "__main__":
    run_database_repair_suite()