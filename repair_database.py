# =========================================================================
# 題庫資料庫全能離線修復、頁碼錨點回歸與知識庫淨化引擎
# =========================================================================
import os
import re
import json
import shutil
import logging
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATABASE_DIR = "./exam_database_output"
SUBJECT_FILE = "subject.json"
TICKETS_FILE = "exam_issue_tickets.json"

GHOST_KEYWORDS_EXTENDED = [
    "本題於原試卷中不存在", "全卷掃描漏失", "無完整題目文本", 
    "題目文字未在影像中提供", "此頁面為試卷封面", "本頁為「大學入學考試中心",
    "作答注意事項", "無實質試題文字", "本題不存在", "本頁為封面", "此頁面為試題封面",
    "題目內容缺失", "題目闕如", "無法判定", "未包含第", "無實質試題", "免予計分",
    "不適用（此頁為試題封面"
]

PURGE_TOPIC_KEYWORDS = [
    "題目闕如", "無法判定", "placeholder", "請補件", "說明頁", 
    "內容缺失", "無法分類", "無實質試題", "資訊缺失", "未包含第",
    "系統測試題", "綜合主題", "待補充", "因題目內容", "本題因",
    "以下為", "注意**", "待確認", "重新判定"
]

# -------------------------------------------------------------------------
# 1. LaTeX 深層語法與 \log 格式清洗器
# -------------------------------------------------------------------------
def clean_latex_corruptions(text: str) -> str:
    if not isinstance(text, str) or not text: return text
    
    # 救回常見被損毀的符號
    text = re.sub(r'\\?fty\b', r'\\infty', text)
    text = re.sub(r'(?<!\\)\bimplies\b', r'\\implies', text)
    text = re.sub(r'(?<!\\)\biff\b', r'\\iff', text)
    text = re.sub(r'\\?bsim\b', r'\\approx', text)
    
    # 救回 \log 變異字元
    text = re.sub(r'\\bar\{log\}', r'\\log', text)
    text = re.sub(r'(?<!\\)\blog\b', r'\\log', text)
    # 標準化對數下標：將 \log_2 x 補全為 \log_{2} x，強化前端 KaTeX 渲染穩定度
    text = re.sub(r'\\log_([0-9a-zA-Z])(?![0-9a-zA-Z{])', r'\\log_{\1}', text)
    
    # 救回括號與控制字元
    text = text.replace(r"\r\right", r"\right").replace(r"\r\left", r"\left").replace(r"\r", "")
    text = re.sub(r'[\x0c]rac(?![a-zA-Z])', r'\\frac', text)
    text = re.sub(r'(?<=\$)rac(?=\{)', r'\\frac', text)
    
    return text

# -------------------------------------------------------------------------
# 2. 知識點鋼鐵淨化器
# -------------------------------------------------------------------------
def sanitize_knowledge_point(cat_str: str) -> str:
    cat_str = re.sub(r'^[\[\"\'\s]+|[\]\"\'\s]+$', '', str(cat_str)).strip()
    if any(gk in cat_str for gk in PURGE_TOPIC_KEYWORDS):
        return "必修_綜合主題_核心概念應用"
        
    if cat_str.startswith("生物_"):
        cat_str = "選修_" + cat_str[3:]
        
    parts = cat_str.split("_")
    if len(parts) >= 2 and parts[0] not in ["必修", "選修"]:
        if any(kw in cat_str for kw in ["細胞", "遺傳", "演化", "生態", "植物", "動物", "生理"]):
            cat_str = "必修_" + "_".join(parts[1:])
        elif any(kw in cat_str for kw in ["多項式", "微積分", "機率", "向量", "矩陣", "曲線", "三角"]):
            cat_str = "必修_" + "_".join(parts[1:])
        else:
            cat_str = "必修_" + "_".join(parts)
    elif len(parts) < 2:
        cat_str = "必修_綜合主題_核心概念"
        
    return cat_str

def sanitize_subject_json():
    """自動清理並規範化 subject.json，杜絕單字碎屑"""
    if not os.path.exists(SUBJECT_FILE): return
    try:
        with open(SUBJECT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cleaned = {}
        purged_cnt = 0
        for subj, content in data.items():
            cleaned[subj] = {"topics": [], "techniques": []}
            for t in content.get("topics", []):
                t_str = str(t).strip()
                if len(t_str) >= 4 and len(t_str) <= 50 and not any(k in t_str for k in PURGE_TOPIC_KEYWORDS) and "\n" not in t_str:
                    cleaned[subj]["topics"].append(sanitize_knowledge_point(t_str))
                else: purged_cnt += 1
            for tech in content.get("techniques", []):
                tech_str = str(tech).strip()
                if len(tech_str) >= 4 and len(tech_str) <= 60 and not any(k in tech_str for k in PURGE_TOPIC_KEYWORDS) and "\n" not in tech_str:
                    cleaned[subj]["techniques"].append(tech_str)
                else: purged_cnt += 1
            cleaned[subj]["topics"] = sorted(list(dict.fromkeys(cleaned[subj]["topics"])))
            cleaned[subj]["techniques"] = sorted(list(dict.fromkeys(cleaned[subj]["techniques"])))
        with open(SUBJECT_FILE, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=4)
        logging.info(f"🧹 [知識庫洗淨] 已成功從 {SUBJECT_FILE} 拔除 {purged_cnt} 筆垃圾考點與單字碎屑！")
    except Exception as e:
        logging.error(f"淨化 {SUBJECT_FILE} 失敗: {e}")

# -------------------------------------------------------------------------
# 3. 實體文字錨點校準器（修正題目與整頁預覽錯位）
# -------------------------------------------------------------------------
def verify_and_fix_page_alignment(q: dict) -> bool:
    """若原卷 PDF 存在，搜尋題幹文字指紋，將錯位的 page_number 與 full_page_image_path 扶正"""
    pdf_path = q.get("question_pdf_path", "")
    if not pdf_path or not os.path.exists(pdf_path):
        return False
        
    q_text = q.get("question_text", "")
    anchor_text = re.sub(r'[^\w\u4e00-\u9fa5]', '', q_text)[:18]
    if len(anchor_text) < 4: return False
    
    try:
        with fitz.open(pdf_path) as doc:
            for p_idx in range(len(doc)):
                page_raw = doc[p_idx].get_text("text")
                page_clean = re.sub(r'[^\w\u4e00-\u9fa5]', '', page_raw)
                if anchor_text in page_clean:
                    real_page = p_idx + 1
                    if q.get("page_number") != real_page:
                        old_p = q.get("page_number")
                        q["page_number"] = real_page
                        # 修正整頁大圖路徑
                        if q.get("full_page_image_path"):
                            q["full_page_image_path"] = re.sub(r'page_\d+\.png', f'page_{real_page:02d}.png', q["full_page_image_path"])
                            q["full_page_image_path"] = re.sub(r'q_full_page_\d+\.png', f'q_full_page_{real_page:02d}.png', q["full_page_image_path"])
                        return True
                    break
    except Exception: pass
    return False

# -------------------------------------------------------------------------
# 4. 幽靈題判定、品質評分與自然排序
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# 5. 雙向融合仲裁主引擎
# -------------------------------------------------------------------------
def reconcile_and_merge_database_pair(db_path: str, partial_path: str, raw_path: str) -> Dict[str, Any]:
    stats = {"file": os.path.basename(db_path), "merged": 0, "ghosts": 0, "aligned_pages": 0, "status": ""}
    db_questions = []
    partial_questions = []
    expected_total = 0
    
    if os.path.exists(raw_path):
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                expected_total = len([q for q in raw_data if is_valid_question(q)])
        except Exception: pass

    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f: db_questions = json.load(f)
        except Exception: pass

    if os.path.exists(partial_path):
        try:
            with open(partial_path, "r", encoding="utf-8") as f: partial_questions = json.load(f)
        except Exception: pass

    # 雙向融合池 (Union Pool)
    merged_map = {}
    for q in db_questions + partial_questions:
        if not is_valid_question(q):
            stats["ghosts"] += 1
            continue
        q_num = str(q.get("question_number", "")).strip()
        q_type = str(q.get("question_type", "")).strip()
        key = (q_num, q_type)
        
        if key not in merged_map:
            merged_map[key] = q
        else:
            if evaluate_solution_score(q) > evaluate_solution_score(merged_map[key]):
                merged_map[key] = q

    merged_list = list(merged_map.values())
    stats["merged"] = len(merged_list)

    for q in merged_list:
        # 1. 頁碼文字錨點校準
        if verify_and_fix_page_alignment(q):
            stats["aligned_pages"] += 1

        # 2. 多知識點規範化
        main_cat = sanitize_knowledge_point(q.get("topic_category", ""))
        cats = q.get("topic_categories", [])
        if not isinstance(cats, list) or not cats: cats = [main_cat]
        else: cats = [sanitize_knowledge_point(c) for c in cats]
        q["topic_categories"] = list(dict.fromkeys(cats))
        q["topic_category"] = q["topic_categories"][0]
        
        # 3. 年代指考校正 (<111年)
        year_raw = str(q.get("academic_year", ""))
        digits = "".join(filter(str.isdigit, year_raw))
        if digits and int(digits) < 111:
            q["academic_year"] = q.get("academic_year", "").replace("分科", "指考")
            q["exam_source"] = q.get("exam_source", "").replace("分科測驗", "指定科目考試")
            
        # 4. LaTeX 與 \log 格式清洗
        for k in ["question_text", "detailed_solution", "question_analysis", "solving_strategy", "scoring_rubric", "concept_review", "traps_and_warnings"]:
            if k in q: q[k] = clean_latex_corruptions(q[k])
            
        # 5. 選擇題防偽評分清洗
        if q.get("question_type") in ["單選題", "多選題"]:
            det = q.get("detailed_solution", "")
            if "### 【手寫題評分對照】" in det:
                q["detailed_solution"] = det.split("### 【手寫題評分對照】")[0].strip()
            # 清除 (3)(5) [即 (C)(E)] 污染
            ans_raw = str(q.get("answer", "")).strip()
            if "即" in ans_raw or "[" in ans_raw:
                tokens = re.findall(r'[A-Za-z0-9]', ans_raw)
                if tokens: q["answer"] = "".join(sorted(list(dict.fromkeys([t.upper() for t in tokens]))))

    # 依實體印刷頁碼與題號自然排序
    def safe_sort(x):
        try: p = int(x.get("page_number", 1))
        except: p = 1
        return (p, natural_sort_key(x.get("question_number", "0")))
    merged_list.sort(key=safe_sort)
    
    current_count = len(merged_list)
    min_threshold = 8 if any(k in db_path for k in ["數", "math"]) else 15
    is_complete = (current_count >= expected_total) if expected_total > 0 else (current_count >= min_threshold)

    if is_complete:
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=4)
        if os.path.exists(partial_path): os.remove(partial_path)
        stats["status"] = f"✅ 100% 完工存檔 (共 {current_count} 題，已銷毀 partial)"
    else:
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=4)
        if os.path.exists(db_path): os.remove(db_path)
        stats["status"] = f"⏳ 題數不足 ({current_count}/{expected_total or min_threshold} 題)，保留 partial 並清除假完成 database"

    return stats

def run_database_repair_suite():
    print("=" * 75)
    print("🚀 啟動【100+ 份資料庫雙向融合、圖文回歸與全能無損修復引擎】...")
    print("=" * 75)

    # 1. 先徹底淨化知識庫
    sanitize_subject_json()

    if not os.path.exists(DATABASE_DIR):
        print(f"❌ 找不到題庫目錄: {DATABASE_DIR}")
        return

    exam_bases = set()
    for root, dirs, files in os.walk(DATABASE_DIR):
        for f in files:
            if f.endswith("_database.json") or f.endswith("_partial_database.json"):
                base_name = f.replace("_partial_database.json", "").replace("_database.json", "")
                exam_bases.add(os.path.join(root, base_name))

    total_exams = len(exam_bases)
    total_ghosts_purged = 0
    total_pages_fixed = 0

    for base in exam_bases:
        db_p = f"{base}_database.json"
        partial_p = f"{base}_partial_database.json"
        raw_p = f"{base}_raw_extracted.json"
        res = reconcile_and_merge_database_pair(db_p, partial_p, raw_p)
        total_ghosts_purged += res["ghosts"]
        total_pages_fixed += res["aligned_pages"]
        print(f"📦 [{res['file']}] -> {res['status']}")
        if res["ghosts"] > 0: print(f"    * 拔除幽靈題: {res['ghosts']} 題")
        if res["aligned_pages"] > 0: print(f"    * 扶正錯位圖文頁碼: {res['aligned_pages']} 題")

    print("=" * 75)
    print("🎉 【全庫修復與狀態機校準圓滿完成】統計摘要：")
    print(f"  * 總掃描並仲裁試卷數: {total_exams} 份")
    print(f"  * 全庫拔除幽靈假題數: {total_ghosts_purged} 題")
    print(f"  * 實體文字錨點扶正頁碼: {total_pages_fixed} 題")
    print(f"  * 雲端單一狀態保證：完工者僅保留 _database，殘缺者僅保留 _partial 供續跑！")
    print("=" * 75)

if __name__ == "__main__":
    run_database_repair_suite()