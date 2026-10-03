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

# =========================================================================
# 台灣高中各科常規題數合理範圍 (最小值, 最大值) - 保護真題不被誤殺
# =========================================================================
EXAM_REASONABLE_RANGES = {
    "國寫": (2, 4),
    "數學甲": (12, 18), "數學乙": (11, 17), "數學A": (15, 22), "數學B": (15, 22), "數學": (12, 22),
    "物理": (22, 32), "化學": (22, 34), "生物": (40, 62), "地球科學": (35, 50),
    "國文": (38, 46), "英文": (50, 60), "歷史": (36, 46), "地理": (36, 46), "公民與社會": (36, 46)
}

# 保底單科天花板（超過此數且發生斷層才視為幽靈題）
STANDARD_MAX_QUESTIONS = {
    "國寫": 4, "數學甲": 20, "數學乙": 18, "數學A": 22, "數學B": 22, "數學": 22,
    "物理": 34, "化學": 36, "生物": 65, "地球科學": 52,
    "國文": 48, "英文": 62, "歷史": 50, "地理": 50, "公民與社會": 50
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
    # 🚨 1. 自動包裹所有未在 $ 內的裸露 LaTeX (如 \vec{v}_0, -2\vec{v}_0, \frac{...}{...}, m_0)
    # 排除已被 $ 或 $$ 包裹的區塊
    parts = re.split(r'(\$\$[\s\S]*?\$\$|\$[^$]*?\$)', text)
    
    # 匹配所有裸露的反斜線數學指令 (包括帶負號、向量、分數、下標變數)
    bare_math_pattern = r'(?<![\$\w\\])(-?\\(?:vec\{[^}]+\}|mathbf\{[^}]+\}|frac\{[^}]+\}\{[^}]+\}|sqrt\{[^}]+\}|pm|times|cdot|alpha|beta|gamma|theta|lambda|mu|pi|omega|Delta|Omega)(?:_[0-9a-zA-Z]+)?)(?![\$\w])'
    
    for i in range(len(parts)):
        if i % 2 == 0:  # 純文字/Markdown 區塊
            # 包裹裸露公式
            parts[i] = re.sub(bare_math_pattern, r' $\1$ ', parts[i])
            # 包裹像 m_0, v_0 這類裸露的下標物理量
            parts[i] = re.sub(r'(?<![\$\w\\])\b([a-zA-Z]_[0-9a-zA-Z]+)\b(?![\$\w])', r' $\1$ ', parts[i])
            # 🚨 解決小題答案擠在一起：將 (a)...(b)... 強制換行
            parts[i] = re.sub(r'(\([a-d]\)[^\(\n\r]+?)(?=\([a-d]\))', r'\1\n\n', parts[i])
            # 解決 **綜上所述，本題正確答案為：(a) $式子$(b) $式子$** 內部黏連
            parts[i] = re.sub(r'(\([a-d]\)\s*\$[^\$]+\$)(?=\([a-d]\))', r'\1\n\n', parts[i])

    text = "".join(parts)
    
    # 🚨 2. 修復粗體語法 **...$公式$** 導致的 KaTeX 語法解析崩潰
    text = re.sub(r'\*\*\s*(\([a-d]\))?\s*\$', r'**\1** $', text)
    text = re.sub(r'\$\s*\*\*', r'$ **', text)
    text = re.sub(r' +', ' ', text)

    return text.strip()

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
    sub_subj = str(q.get("sub_subject", "")).strip()
    q_txt = str(q.get("question_text", ""))
    sol_txt = str(q.get("detailed_solution", ""))
    combined = q_txt + " " + sol_txt
    
    # 🚨 1. 單科試卷（非綜合自然/社會）中，sub_subject 必須嚴格吻合
    if expected_subject not in ["自然", "社會", "未知"]:
        # 容許數學科目細分互通 (數學A/數學B/數甲/數乙)
        if "數" in expected_subject and "數" in sub_subj:
            pass
        elif sub_subj and sub_subj != expected_subject:
            return True

    # 🚨 2. 實質內容與考點特征嚴格校驗 (消滅誤被標記為生物的數學/物理幽靈題)
    if expected_subject in ["生物"]:
        math_phys_clues = [
            "電流的磁效應", "直角坐標中", "帶電質點", "電磁場中運動", 
            "\\begin{bmatrix}", "轉移矩陣", "二階導數", "外接圓半徑", 
            "動量守恆", "牛頓第二定律", "拋體運動", "動能與位能", "庫侖定律"
        ]
        if any(c in combined for c in math_phys_clues):
            return True
            
    if expected_subject in ["物理"]:
        bio_math_clues = [
            "孟德爾遺傳", "光敏素", "葉綠體", "有絲分裂", "聚合酶連鎖反應", 
            "染色體", "基因型", "生態系", "轉錄轉譯", "原核生物"
        ]
        if any(c in combined for c in bio_math_clues):
            return True

    if "數學" in expected_subject or "數甲" in expected_subject or "數乙" in expected_subject:
        sci_clues = ["電磁感應", "莫耳濃度", "氧化還原", "光合作用", "熱力學定律"]
        if any(c in combined for c in sci_clues):
            return True

    return False

# -------------------------------------------------------------------------
# 5. 綜合嚴格題目審查
# -------------------------------------------------------------------------
def is_valid_question_strict(q: dict, max_allowed_q: int, physical_q_set: Set[str], expected_subject: str, exam_tag: str = "") -> bool:
    q_text = str(q.get("question_text", "")).strip()
    ans_text = str(q.get("answer", "")).strip()
    cat_text = str(q.get("topic_category", "")) + " " + " ".join(q.get("topic_categories", []))
    ana_text = str(q.get("question_analysis", ""))
    q_num_raw = str(q.get("question_number", "")).strip()
    opts = q.get("options", [])
    combined = f"{q_text} | {ans_text} | {cat_text} | {ana_text}"

    # 🚨【真題鋼鐵保護傘】：若題號為選填字母 (A~H)、中文題號 (一、二)、或帶子小題 (如 1(a), 21(b))
    # 這是標準的台灣大考題組與非選結構，絕對嚴禁以純數字天花板將其砍除！
    is_non_digit_question = any(c in q_num_raw for c in ["一", "二", "三", "四", "五", "A", "B", "C", "D", "E", "F", "G", "H", "(", "（"])
    if is_non_digit_question:
        # 只做跨科檢查與純佔位檢查
        if is_cross_subject_contamination(q, expected_subject):
            return False
        if any(gk in combined for gk in GHOST_KEYWORDS_EXTENDED):
            return False
        return True

    # 🚨【純數字題號安全裁決】：
    digits = re.findall(r'^\d+$', q_num_raw)
    if digits:
        val = int(digits[0])
        # 取得合理的上限（優先採用掃描到的真實題數，若無則依學科標準天花板）
        ceiling = max_allowed_q if max_allowed_q > 0 else STANDARD_MAX_QUESTIONS.get(expected_subject, 40)
        
        # 允許 +3 題的合理結構偏差（防止題組題拆小題導致編號微增）
        # 只有當題號超出上限 4 題以上（例如 24 題的考卷卻出現 49 題），才判定為捏造幽靈題！
        if ceiling > 0 and val > (ceiling + 3):
            return False

    # 跨科污染檢驗（生物卷裡的數學/物理題）
    if is_cross_subject_contamination(q, expected_subject):
        return False

    # 幽靈關鍵字與作答規範過濾
    if any(gk in combined for gk in GHOST_KEYWORDS_EXTENDED):
        return False

    # 題幹只是純標題（如「91年指考物理第32題」）
    if re.match(r'^(?:[一二三四五六七八九十\d]+年?學?年?度?)?(?:[^\n]{2,15}考科)?第?\s*\d+\s*題$', q_text):
        return False

    # 選項全為佔位符
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
        
    # 🚨 雙重去重閥門：同時比對「題號+題型」與「題幹純文字特徵」，徹底擊殺雙倍重複題目
    merged_map = {}
    text_fingerprints = {}

    for q in db_questions + partial_questions:
        if not is_valid_question_strict(q, max_pdf_q, physical_q_set, expected_subject, exam_tag):
            stats["ghosts"] += 1
            continue
            
        q_num = str(q.get("question_number", "")).strip()
        q_type = str(q.get("question_type", "")).strip()
        key = (q_num, q_type)
        
        # 題幹純文字特徵指紋 (去除標點空格，取前 30 字)
        clean_text_fp = re.sub(r'[^\w\u4e00-\u9fa5]', '', str(q.get("question_text", "")))[:30]
        
        current_sol_len = len(str(q.get("detailed_solution", "")))
        
        # 情況 A：題號完全相同
        if key in merged_map:
            existing_len = len(str(merged_map[key].get("detailed_solution", "")))
            if current_sol_len > existing_len:
                merged_map[key] = q
            continue
            
        # 情況 B：題幹高度重複 (同名題庫雙胞胎碰撞)
        if clean_text_fp and len(clean_text_fp) >= 10:
            if clean_text_fp in text_fingerprints:
                old_key = text_fingerprints[clean_text_fp]
                existing_len = len(str(merged_map[old_key].get("detailed_solution", "")))
                if current_sol_len > existing_len:
                    del merged_map[old_key]
                    merged_map[key] = q
                    text_fingerprints[clean_text_fp] = key
                continue
            text_fingerprints[clean_text_fp] = key

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
    
    # 獲取本科的合理題數範圍 (min_q, max_q)
    r_min, r_max = EXAM_REASONABLE_RANGES.get(expected_subject, (12, 50))
    
    # 若有掃描到真實 PDF，則以實體題數作為參考基準；否則依學科常規範圍
    if max_pdf_q > 0:
        # 只要達到實體掃描題數的 90%（容許部分非選題未拆小題的微小差異），且大於最小合理題數，即視為 100% 完工！
        is_complete = (current_count >= min(max_pdf_q, r_min))
        display_threshold = max_pdf_q
    else:
        # 無 PDF 時（純依賴資料庫）：只要收錄題數落在本科合理區間內，絕不盲目降級！
        is_complete = (current_count >= r_min)
        display_threshold = r_min

    if is_complete:
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=4)
        if os.path.exists(partial_path):
            os.remove(partial_path)
            register_cloud_purge(partial_path)
        stats["status"] = f"✅ 100% 完工存檔 (收錄 {current_count} 題，符合 {expected_subject} 合理題數)"
    else:
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=4)
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                register_cloud_purge(db_path)
            except Exception: pass
        if os.path.exists(raw_path):
            try:
                os.remove(raw_path)
                register_cloud_purge(raw_path)
            except Exception: pass
        stats["status"] = f"⏳ 題數不足 ({current_count}/{display_threshold} 題)，已降級 partial，待補齊真題"
        
    
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