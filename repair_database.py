# =========================================================================
# 題庫資料庫全能離線修復、實體 PDF 題數校準與跨科淨化引擎
# =========================================================================
import os
import re
import json
import logging
from typing import List, Dict, Any, Tuple, Set
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATABASE_DIR = "./exam_database_output"
SUBJECT_FILE = "subject.json"
PURGE_FILE = "pending_cloud_purges.txt"

GHOST_KEYWORDS_EXTENDED = [
    "本題於原試卷中不存在", "全卷掃描漏失", "無完整題目文本", 
    "題目文字未在影像中提供", "此頁面為試卷封面", "本頁為「大學入學考試中心",
    "作答注意事項", "無實質試題文字", "本題不存在", "本頁為封面", "此頁面為試題封面",
    "題目內容缺失", "題目闕如", "無法判定", "未包含第", "無實質試題", "免予計分",
    "不適用（此頁為試題封面", "題目內容未提供", "選項內容未提供", "請補充",
    "作答示例", "第壹部分作答示例", "劃記方式之說明", "例：若第", "無法從提供的影像中"
]

PURGE_TOPIC_KEYWORDS = [
    "題目闕如", "無法判定", "placeholder", "請補件", "說明頁", 
    "內容缺失", "無法分類", "無實質試題", "資訊缺失", "未包含第",
    "系統測試題", "綜合主題", "待補充", "因題目內容", "本題因",
    "以下為", "注意**", "待確認", "重新判定", "考試作答規範", "讀卡格式"
]

# 各大考學科的實體題數合理上限 (容許 2~4 題緩衝，但絕不可能跑到 40~50 題)
SUBJECT_HARD_LIMITS = {
    "指考_物理": 26, "指考_化學": 26, "指考_生物": 26,
    "分科_物理": 28, "分科_化學": 28, "分科_生物": 28,
    "指考_數甲": 18, "指考_數乙": 18, "分科_數甲": 20,
    "學測_數學": 22, "學測_數A": 22, "學測_數B": 22,
    "指考_國文": 26, "指考_英文": 56, "指考_歷史": 44, "指考_地理": 42, "指考_公民": 42,
    "學測_國文": 44, "學測_英文": 56, "學測_自然": 68, "學測_社會": 72
}

def register_cloud_purge(local_path: str):
    """登記待銷毀之雲端暫存檔路徑"""
    try:
        rel_path = os.path.relpath(local_path, DATABASE_DIR).replace("\\", "/")
        cloud_target = f"gdrive:exam_database_output/{rel_path}"
        with open(PURGE_FILE, "a", encoding="utf-8") as f:
            f.write(f"{cloud_target}\n")
    except Exception:
        pass

def find_actual_pdf(db_path: str, candidate_pdf_path: str = "") -> str:
    """若 JSON 記錄的路徑失效，跨目錄自動搜尋真正的實體 PDF"""
    if candidate_pdf_path and os.path.exists(candidate_pdf_path):
        return candidate_pdf_path
        
    base_file = os.path.basename(db_path).replace("_database.json", "").replace("_partial_database.json", "")
    year_m = re.search(r'(\d+)', base_file)
    year_str = year_m.group(1) if year_m else ""
    
    subj_kw = ""
    for s_name in ["數學", "數甲", "數乙", "物理", "化學", "生物", "地科", "地球科學", "國文", "國寫", "英文", "歷史", "地理", "公民"]:
        if s_name in base_file:
            subj_kw = s_name; break

    search_dirs = ["ast_exam_papers_only", "gsat_exam_papers_only", "mock_exam_papers_only", "school_exam_papers_only"]
    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            for root, dirs, files in os.walk(s_dir):
                for f in files:
                    if f.endswith(".pdf"):
                        if year_str and year_str in f:
                            if subj_kw and (subj_kw in f or subj_kw in root):
                                return os.path.join(root, f)
    return ""
    
# -------------------------------------------------------------------------
# 1. 深度 LaTeX 與 Markdown 格式修復器
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
    text = re.sub(r'\\log_([0-9a-zA-Z])(?![0-9a-zA-Z{])', r'\\log_{\1}', text)
    
    # 救回括號與控制字元
    text = text.replace(r"\r\right", r"\right").replace(r"\r\left", r"\left").replace(r"\r", "")
    text = re.sub(r'[\x0c]rac(?![a-zA-Z])', r'\\frac', text)
    text = re.sub(r'(?<=\$)rac(?=\{)', r'\\frac', text)
    
    # 🚨 自動修復裸露未加 $ 的公式（如 \vec{V}、\frac{...}、等號方程式）
    def wrap_naked_math(m):
        raw_eq = m.group(0).strip()
        return f" ${raw_eq}$ "
    
    # 匹配未在 $ 內部的向量、分數與帶有下標的等式
    naked_pattern = r'(?<!\$)(?<!\\)\b(\\vec\{[^\}]+\}(?:_[0-9a-zA-Z]+)?\s*=\s*\\frac\{[^\}]+\}\{[^\}]+\})(?!\$)'
    text = re.sub(naked_pattern, wrap_naked_math, text)
    
    # 修復粗體與公式擠在一起產生的渲染失敗，如 **綜上所述，本題正確答案為：$公式$**
    text = re.sub(r'\*\*(.+?)\$', r'**\1** $', text)
    text = re.sub(r'\$(.+?)\*\*', r'$ **\1**', text)

    return text

# -------------------------------------------------------------------------
# 2. 知識點鋼鐵淨化器
# -------------------------------------------------------------------------
def sanitize_knowledge_point(cat_str: str) -> str:
    if not isinstance(cat_str, str): return "必修_綜合主題_核心概念"
    cat_str = re.sub(r'^[\[\]\'\"\s\\]+|[\[\]\'\"\s\\]+$', '', cat_str).strip()
    cat_str = re.sub(r'^[\[\]\'\"]*(?:必修|選修)[_/\\]+[\[\]\'\"]*(必修|選修)', r'\1', cat_str)
    cat_str = re.sub(r'[_/\\]+[\[\]\'\"]*(?:必修|選修)[_/\\]+', '_', cat_str)
    cat_str = re.sub(r'[\[\]\'\"]+', '', cat_str)

    if any(gk in cat_str for gk in PURGE_TOPIC_KEYWORDS):
        return "必修_綜合主題_核心概念應用"

    if cat_str.startswith("生物_"):
        cat_str = "選修_" + cat_str[3:]

    parts = [p.strip() for p in cat_str.split("_") if p.strip()]
    if not parts: return "必修_綜合主題_核心概念"
    if parts[0] in ["必修", "選修"]:
        return "_".join(parts) if len(parts) >= 2 else f"{parts[0]}_綜合主題_核心概念"
    return "必修_" + "_".join(parts)

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
# 3. 實體 PDF 題數真理掃描器 (Ground Truth Scanner)
# -------------------------------------------------------------------------
def get_real_pdf_max_question_info(pdf_path: str, exam_name: str = "") -> Tuple[int, Set[str]]:
    """以極度嚴謹的段落排版分析真實題目卷題數（自動排除 80分鐘、100分、年份等雜訊）"""
    if not pdf_path or not os.path.exists(pdf_path):
        return 0, set()
        
    # 國寫專用防禦：國寫永遠只有 2 大題
    if "國寫" in exam_name or "寫作" in exam_name:
        return 2, {"1", "2", "一", "二"}

    detected_nums = set()
    try:
        doc = fitz.open(pdf_path)
        for p_idx in range(len(doc)):
            # 封面頁（通常含大考作答範例與考試時間80分鐘）一律不納入題號計算
            if p_idx == 0 and len(doc) > 1 and "作答注意事項" in doc[0].get_text("text"):
                continue

            txt = doc[p_idx].get_text("text")
            for line in txt.splitlines():
                line_str = line.strip()
                if not line_str:
                    continue
                    
                # 🚨 剛性過濾：凡是包含時間、總分、配分說明的行，絕不是題號！
                if any(bad in line_str for bad in ["分鐘", "考試時間", "滿分", "總分", "學年度", "共計", "作答示例", "答案卡"]):
                    continue

                # 匹配行首標準題號：如 "1."、"12．"、"第 5 題"、"24 "
                m = re.match(r'^(?:第\s*)?(\d{1,2})\s*(?:題|[.．、\s])', line_str)
                if m:
                    # 確保緊接著數字後面的不是「分」或「秒」（防止 "1. 10分" 抓成 10）
                    rest = line_str[m.end():].strip()
                    if not rest.startswith(("分", "％", "%")):
                        val = int(m.group(1))
                        # 排除 0 或明顯是日期的數字
                        if 1 <= val <= 75:
                            detected_nums.add(val)
        doc.close()
    except Exception as e:
        logging.warning(f"掃描 PDF {pdf_path} 實體題數失敗: {e}")

    if not detected_nums:
        return 0, set()

    # 🚨 連續性濾網：找出真實考題的最大題號（杜絕中間斷層 30 題的假數字，例如有 1~24 題，後面突然跳出一個 75）
    sorted_nums = sorted(list(detected_nums))
    real_max = sorted_nums[0]
    for i in range(len(sorted_nums) - 1):
        curr_n = sorted_nums[i]
        next_n = sorted_nums[i+1]
        # 正常考卷題號連續或頂多題組跳 1~3 題；若突然跨度大於 12，代表後面是雜訊（如分數或圖表編號）
        if next_n - curr_n <= 12:
            real_max = next_n
        else:
            break

    valid_q_set = {str(n) for n in sorted_nums if n <= real_max}
    return real_max, valid_q_set

# -------------------------------------------------------------------------
# 4. 跨科嚴重污染檢驗 (如生物卷出現數學/物理)
# -------------------------------------------------------------------------
def is_cross_subject_contamination(q: dict, expected_subject: str) -> bool:
    sub_subj = q.get("sub_subject", "")
    q_txt = q.get("question_text", "")
    sol_txt = q.get("detailed_solution", "")
    combined = q_txt + " " + sol_txt
    
    # 1. 考科名稱明顯矛盾
    if expected_subject in ["生物"] and sub_subj in ["數學", "物理", "公民與社會", "歷史"]:
        return True
    if expected_subject in ["物理"] and sub_subj in ["生物", "歷史", "地理", "國文"]:
        return True
    if expected_subject in ["數學"] and sub_subj in ["生物", "化學", "歷史", "英文"]:
        return True

    # 2. 內容實質特徵過濾
    if "生物" in expected_subject:
        # 生物卷絕對不會出現電流磁效應、安培右手、動量守恆、矩陣、求極限
        math_phys_clues = ["電流的磁效應", "直角坐標中", "帶電質點", "電磁場中運動", "\\begin{bmatrix}", "轉移矩陣", "二階導數", "外接圓半徑"]
        if any(c in combined for c in math_phys_clues):
            return True
            
    if "物理" in expected_subject:
        bio_clues = ["孟德爾遺傳", "光敏素", "葉綠體囊狀體", "有絲分裂", "聚合酶連鎖反應"]
        if any(c in combined for c in bio_clues):
            return True

    return False

# -------------------------------------------------------------------------
# 5. 綜合嚴格題目審查
# -------------------------------------------------------------------------
def is_valid_question_strict(q: dict, max_allowed_q: int, physical_q_set: Set[str], expected_subject: str, exam_tag: str) -> bool:
    q_text = str(q.get("question_text", "")).strip()
    ans_text = str(q.get("answer", "")).strip()
    cat_text = str(q.get("topic_category", "")) + " " + " ".join(q.get("topic_categories", []))
    ana_text = str(q.get("question_analysis", ""))
    q_num_raw = str(q.get("question_number", "")).strip()
    opts = q.get("options", [])
    combined = f"{q_text} | {ans_text} | {cat_text} | {ana_text}"

    # 🚨 1. 實體原卷題數裁決（允許 +2 題的合理結構偏差，如手寫小題拆分；但絕不放過 Q49 這類跨度超過 15 題的假題）
    digits = re.findall(r'^\d+$', q_num_raw)
    if digits:
        val = int(digits[0])
        limit_ceiling = max_allowed_q if max_allowed_q > 0 else STANDARD_MAX_QUESTIONS.get(expected_subject, 0)
        
        # 只要題號在合法天花板 + 2 之內，視為安全範圍，絕對不刪！
        # 只有像 24 題的考卷卻出現 40 幾題，這種跨度離譜的假題才剔除！
        if limit_ceiling > 0 and val > (limit_ceiling + 2):
            return False

    # 🚨 若該題有實體附圖，且題幹長度充實 (>35字)，絕對視為珍貴真題，嚴禁誤刪！
    if q.get("has_image") and len(q_text) >= 35:
        # 僅防禦明顯的跨科污染
        if is_cross_subject_contamination(q, expected_subject):
            return False
        return True
        

    # 🚨 2. 跨科污染檢驗（消滅生物卷裡的數學/物理題）
    if is_cross_subject_contamination(q, expected_subject):
        return False

    # 🚨 3. 攔截作答規範與封面說明
    instruction_keywords = [
        "作答示例", "作答注意事項", "劃記方式之說明", "作答說明", "答案卡第", 
        "考試作答規範", "讀卡格式與欄位", "選填題電子讀卡", "例：若第", "答題卷劃記",
        "本試題共", "作答範例", "範例題"
    ]
    if any(ik in combined for ik in instruction_keywords):
        return False

    # 🚨 4. 攔截幽靈關鍵字
    if any(gk in combined for gk in GHOST_KEYWORDS_EXTENDED):
        return False

    # 🚨 5. 題幹只是重複題號標題
    if re.match(r'^(?:[一二三四五六七八九十\d]+年?學?年?度?)?(?:[^\n]{2,15}考科)?第?\s*\d+\s*題$', q_text):
        return False

    # 🚨 6. 虛擬選項 A/B/C/D 過濾
    dummy_opt_values = {"選項A", "選項B", "選項C", "選項D", "選項E", "A", "B", "C", "D", "E", "(選項內容未提供)", "選項內容未提供"}
    if opts and all(str(opt.get("value", "")).strip() in dummy_opt_values for opt in opts):
        return False

    if len(q_text) < 6 and not q.get("has_image"):
        return False

    return True

def natural_sort_key(s):
    cn_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
    s_str = str(s).strip()
    for cn, num in cn_map.items(): s_str = s_str.replace(cn, f"{num:02d}")
    parts = re.split(r'(\d+)', s_str)
    return [int(text) if text.isdigit() else text.lower() for text in parts]

# -------------------------------------------------------------------------
# 6. 雙向融合仲裁主引擎
# -------------------------------------------------------------------------
def reconcile_and_merge_database_pair(db_path: str, partial_path: str, raw_path: str) -> Dict[str, Any]:
    stats = {"file": os.path.basename(db_path), "merged": 0, "ghosts": 0, "status": ""}
    db_questions, partial_questions = [], []
    
    # 判定考科與考卷標籤
    expected_subject = "未知"
    for s_name in ["數學", "物理", "化學", "生物", "歷史", "地理", "公民與社會", "英文", "國文", "地科", "自然", "社會"]:
        if s_name in db_path:
            expected_subject = s_name; break

    exam_tag = ""
    if "指考" in db_path: exam_tag = f"指考_{expected_subject}"
    elif "分科" in db_path: exam_tag = f"分科_{expected_subject}"
    elif "學測" in db_path: exam_tag = f"學測_{expected_subject}"

    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f: db_questions = json.load(f)
        except Exception: pass

    if os.path.exists(partial_path):
        try:
            with open(partial_path, "r", encoding="utf-8") as f: partial_questions = json.load(f)
        except Exception: pass

    # 尋找實體 PDF（確保 actual_pdf 一定被定義）
    actual_pdf = ""
    for q_cand in db_questions + partial_questions:
        cand_p = q_cand.get("question_pdf_path", "")
        if cand_p and os.path.exists(cand_p):
            actual_pdf = cand_p
            break
            
    # 若 JSON 內記錄的路徑失效（例如換了環境），自動跨資料夾搜尋實體 PDF
    if not actual_pdf:
        actual_pdf = find_actual_pdf(db_path, "")

    max_pdf_q, physical_q_set = get_real_pdf_max_question_info(actual_pdf, exam_name=os.path.basename(db_path))
    if max_pdf_q > 0:
        logging.info(f"📄 [{os.path.basename(db_path)}] 經題目卷校準，實體題數上限為第 {max_pdf_q} 題")
        
    # 🚨 關鍵去重：以題號為唯一鍵，消除下載兩份同名資料庫導致的雙倍重複題
    merged_map = {}
    for q in db_questions + partial_questions:
        if not is_valid_question_strict(q, max_pdf_q, physical_q_set, expected_subject, exam_tag):
            stats["ghosts"] += 1
            continue
            
        q_num = str(q.get("question_number", "")).strip()
        q_type = str(q.get("question_type", "")).strip()
        key = (q_num, q_type)
        
        if key not in merged_map:
            merged_map[key] = q
        else:
            if len(str(q.get("detailed_solution", ""))) > len(str(merged_map[key].get("detailed_solution", ""))):
                merged_map[key] = q

    merged_list = list(merged_map.values())
    stats["merged"] = len(merged_list)

    # 執行分類清洗、年代校驗與 LaTeX 格式修復
    for q in merged_list:
        main_cat = sanitize_knowledge_point(q.get("topic_category", ""))
        cats = [sanitize_knowledge_point(c) for c in q.get("topic_categories", []) if c]
        q["topic_categories"] = list(dict.fromkeys(cats)) if cats else [main_cat]
        q["topic_category"] = q["topic_categories"][0]
        
        # 深度 LaTeX 清洗（消滅裸露公式）
        for k in ["question_text", "detailed_solution", "question_analysis", "solving_strategy", "scoring_rubric", "concept_review", "traps_and_warnings"]:
            if k in q: q[k] = clean_latex_corruptions(q[k])

    def safe_sort(x):
        try: p = int(x.get("page_number", 1))
        except: p = 1
        return (p, natural_sort_key(x.get("question_number", "0")))
    merged_list.sort(key=safe_sort)

    current_count = len(merged_list)
    min_threshold = max_pdf_q if max_pdf_q > 0 else (8 if "數" in db_path else 15)
    is_complete = (current_count >= min_threshold) and (current_count > 0)

    if is_complete:
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=4)
        if os.path.exists(partial_path): 
            os.remove(partial_path)
            register_cloud_purge(partial_path)
        stats["status"] = f"✅ 100% 完工存檔 (真題收錄 {current_count}/{min_threshold} 題)"
    else:
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=4)
        if os.path.exists(db_path): 
            try: 
                os.remove(db_path)
                register_cloud_purge(db_path) # 登記刪除雲端偽完工 database
            except Exception: pass
            
        if os.path.exists(raw_path):
            try:
                os.remove(raw_path)
                register_cloud_purge(raw_path)
            except Exception: pass

        stats["status"] = f"⏳ 題數不足 ({current_count}/{min_threshold} 題)，已降級 partial、清除舊 database 與髒快取，待補齊真題"
        
    return stats

def run_database_repair_suite():
    print("=" * 75)
    print("🚀 啟動【實體 PDF 題數裁決、跨科淨化與同名去重修復引擎】...")
    print("=" * 75)

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

    total_ghosts_purged = 0
    for base in exam_bases:
        db_p = f"{base}_database.json"
        partial_p = f"{base}_partial_database.json"
        raw_p = f"{base}_raw_extracted.json"
        res = reconcile_and_merge_database_pair(db_p, partial_p, raw_p)
        total_ghosts_purged += res["ghosts"]
        print(f"📦 [{res['file']}] -> {res['status']}")
        if res["ghosts"] > 0: print(f"    * 成功斬除幽靈/跨科假題: {res['ghosts']} 題")

    print("=" * 75)
    print(f"🎉 清理完畢！全庫總共拔除 {total_ghosts_purged} 筆多餘幽靈題與跨科污染題目。")
    print("=" * 75)

if __name__ == "__main__":
    run_database_repair_suite()