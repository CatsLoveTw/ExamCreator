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
    "作答示例", "第壹部分作答示例", "劃記方式之說明", "例：若第", "無法從提供的影像中",
    "重點摘要", "觀念摘要", "公式整理", "單元重點", "學習重點", "知識綱要", 
    "單元目錄", "章節索引", "目次", "參考詳解", "試題詳解", "參考答案暨詳解"
]

def is_phantom_table_or_cover(q: dict) -> bool:
    """
    🚨 具備真題保護傘的「純附錄/週期表/封面/重點摘要/目錄」精確判定器
    - 徹底斬除：重點摘要、觀念整理表格、章節目錄、試卷封面、詳解洩漏題目
    - 保障：化學科含有『原子序』、『元素週期表』、『原子量』的正式題目 100% 不被誤刪！
    """
    q_num = str(q.get("question_number", "")).strip()
    q_text = str(q.get("question_text", "")).strip()
    opts = q.get("options", []) or []

    # 1. 題號本身就是非題目字眼（包含摘要、目錄、詳解）
    if any(k in q_num for k in ["封面", "週期表", "說明", "附錄", "附表", "注意事項", "重點摘要", "觀念摘要", "目錄", "目次", "詳解", "解答"]):
        return True

    # 2. 【核心保護】：只要有實體選項（2個以上非佔位符），100% 是真實考題，絕不可殺！
    dummy_vals = {"A", "B", "C", "D", "E", "選項A", "選項B", "選項C", "選項D", ""}
    has_real_options = len(opts) >= 2 and not all(str(o.get("value", "")).strip() in dummy_vals for o in opts)
    if has_real_options:
        return False

    # 3. 【核心保護】：題幹含有明確的問句或解答導向特徵，100% 是手寫或簡答考題！
    interrogative_keywords = ["何者", "下列", "試求", "試問", "計算", "說明", "為何", "多少", "？", "?", "選出", "正確", "錯誤", "寫出", "求證"]
    if any(ik in q_text for ik in interrogative_keywords):
        return False

    # 4. 判定為「純週期表/純封面/純常數表/純重點摘要/純目錄」佔位題：
    is_pure_non_question = bool(re.match(
        r'^(?:[【\[<<\s]*)(?:元素週期表|週期表|週期律表|原子量表|常數表|參考常數|常用常數表|第壹部分作答示例|重點摘要|觀念摘要|公式整理|單元重點|學習重點|知識綱要|目次|目錄|Table of Contents|參考詳解|試題詳解)', 
        q_text
    ))
    if is_pure_non_question:
        return True

    # 題幹極短且完全由常數、摘要或目錄標題構成
    if len(q_text) < 40 and any(k in q_text for k in ["週期表", "元素週期表", "原子量表", "常數表", "重點摘要", "單元重點", "目次", "目錄"]):
        return True

    return False

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
# 🚨 完整納入大考學測/指考「自然科 (68題)」與「社會科 (72題)」完整題數標準
EXAM_REASONABLE_RANGES = {
    "國寫": (2, 4),
    "數學甲": (12, 18), "數學乙": (11, 17), "數學A": (15, 22), "數學B": (15, 22), "數學": (12, 22),
    "物理": (22, 32), "化學": (22, 34), "生物": (40, 62), "地球科學": (35, 50),
    "國文": (38, 46), "英文": (50, 60), "歷史": (36, 46), "地理": (36, 46), "公民與社會": (36, 46),
    "社會": (65, 75),  # 🚨 大考學測社會科標準題數通常為 72 題
    "自然": (60, 70)   # 🚨 大考學測自然科標準題數通常為 68 題
}

# 保底單科天花板
STANDARD_MAX_QUESTIONS = {
    "國寫": 4, "數學甲": 20, "數學乙": 18, "數學A": 22, "數學B": 22, "數學": 22,
    "物理": 34, "化學": 36, "生物": 65, "地球科學": 52,
    "國文": 48, "英文": 62, "歷史": 50, "地理": 50, "公民與社會": 50,
    "社會": 75,
    "自然": 72
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
        
# =========================================================================
# 大考與試卷名稱全球統一標準化核心
# =========================================================================
def standardize_exam_metadata(year_str: str, exam_type: str, subject: str, mock_tag: str = "", school_name: str = "") -> dict:
    digits = "".join(filter(str.isdigit, str(year_str)))
    y_val = int(digits) if digits else 111
    
    # 清洗考科名稱（徹底拔除「考科」、「科」冗字，統一學科名稱）
    clean_sub = re.sub(r'(?:考科|科)$', '', str(subject).strip())
    if clean_sub in ["高中數學甲", "數甲"]: clean_sub = "數學甲"
    elif clean_sub in ["高中數學乙", "數乙"]: clean_sub = "數學乙"
    elif clean_sub in ["高中數學A", "數A"]: clean_sub = "數學A"
    elif clean_sub in ["高中數學B", "數B"]: clean_sub = "數學B"
    elif clean_sub in ["國綜", "國語文"]: clean_sub = "國文"
    elif clean_sub in ["國語文寫作", "寫作測驗"]: clean_sub = "國寫"

    # 補考判定：補考絕對隔離，不可與正式考合併！
    is_makeup = any(k in str(mock_tag) or k in str(year_str) for k in ["補考", "補辦", "代考"])
    makeup_tag = "_補考" if is_makeup else ""
    makeup_src = "補考_" if is_makeup else ""

    sem_tag = "上" if "上學期" in str(year_str) else ("下" if "下學期" in str(year_str) else "")

    if exam_type == "SCHOOL":
        sch = school_name or "通用學校"
        spec_name = f"{y_val}學年度_{sch}{mock_tag}_{clean_sub}"
        acad_year = f"{y_val}{sem_tag}定期考"
        exam_source = f"{y_val}學年度{sem_tag+'學期' if sem_tag else ''}{sch}{mock_tag}_{clean_sub}"
        type_folder = os.path.join("學校定期考", sch)
        category_name = "學校段考"
    elif exam_type in ["MOCK", "MOCK_GSAT", "MOCK_AST"]:
        # 🎯 精確劃分學測模擬考 vs 分科/指考模擬考
        ast_short = "分科" if y_val >= 111 else "指考"
        ast_full = "分科測驗" if y_val >= 111 else "指定科目考試"
        mock_sub_type = "學測模擬" if exam_type == "MOCK_GSAT" else (f"{ast_short}模擬" if exam_type == "MOCK_AST" else "模擬考")
        
        org_label = "全國"
        for cand in ["北模", "中模", "南模", "竹模", "全模", "翰林", "南一", "龍騰", "康熹", "泰宇", "詮達", "全華", "漢樺", "文昌"]:
            if cand in mock_tag: org_label = cand; break
            
        spec_name = f"{y_val}學年度_{mock_sub_type}{mock_tag}_{clean_sub}"
        acad_year = f"{y_val}{mock_sub_type[:2]}模{mock_tag}"
        exam_source = f"{y_val}學年度{mock_sub_type}考{mock_tag}_{clean_sub}"
        type_folder = os.path.join("模擬考", mock_sub_type, org_label)
        category_name = "學測模擬" if exam_type == "MOCK_GSAT" else "分科模擬"
    elif exam_type == "GSAT":
        spec_name = f"{y_val}學年度_學測{makeup_tag}_{clean_sub}"
        acad_year = f"{y_val}學測{makeup_tag}"
        exam_source = f"{y_val}學年度學科能力測驗{makeup_src}{clean_sub}"
        type_folder = "學測"
        category_name = "學測"
    else:  # AST (大考指考 / 分科測驗)
        ast_short = "分科" if y_val >= 111 else "指考"
        ast_full = "分科測驗" if y_val >= 111 else "指定科目考試"
        spec_name = f"{y_val}學年度_{ast_short}{makeup_tag}_{clean_sub}"
        acad_year = f"{y_val}{ast_short}{makeup_tag}"
        exam_source = f"{y_val}學年度{ast_full}{makeup_src}{clean_sub}"
        type_folder = "分科指考"
        category_name = "分科指考"

    return {
        "spec_name": spec_name,
        "academic_year": acad_year,
        "exam_source": exam_source,
        "clean_subject": clean_sub,
        "is_makeup": is_makeup,
        "y_val": y_val,
        "type_folder": type_folder,
        "exam_type": exam_type,
        "exam_category": category_name
    }

# -------------------------------------------------------------------------
# 詳解互補與多解融合
# -------------------------------------------------------------------------
def fuse_duplicate_question_solutions(q_base: dict, q_incoming: dict) -> dict:
    """當同一題存在多個版本時，進行欄位互補，並將獨特另解/速解無損拼接為一題多解"""
    # 1. 決定基礎版本：以詳解長度與結構更完整者為主體
    sol_base = str(q_base.get("detailed_solution", "")).strip()
    sol_inc = str(q_incoming.get("detailed_solution", "")).strip()
    
    if len(sol_inc) > len(sol_base):
        primary = dict(q_incoming)
        secondary = dict(q_base)
    else:
        primary = dict(q_base)
        secondary = dict(q_incoming)
        
    p_sol = str(primary.get("detailed_solution", "")).strip()
    s_sol = str(secondary.get("detailed_solution", "")).strip()

    # 2. 🚨 多元另解/速解拼接：若次要版本含有獨特標題區塊，追加至主解答
    if s_sol and s_sol != p_sol and len(s_sol) > 50:
        # 尋找次要版本中的另解/速解區塊
        sub_headers = ["### 【另解", "### 【速解", "### 【直覺解", "### 【坐標化", "### 【幾何直覺"]
        for header in sub_headers:
            if header in s_sol and header not in p_sol:
                # 擷取該獨立解法區塊
                extracted_part = s_sol.split(header, 1)[1]
                # 截取到下一個大標題之前
                next_header_idx = extracted_part.find("\n### ")
                if next_header_idx != -1:
                    extracted_sol = header + extracted_part[:next_header_idx].strip()
                else:
                    extracted_sol = header + extracted_part.strip()
                
                # 排除原卷無效說明與總結句干擾
                for fake_foot in ["**綜上所述", "【手寫題評分對照】"]:
                    if fake_foot in extracted_sol:
                        extracted_sol = extracted_sol.split(fake_foot)[0].strip()
                        
                if len(extracted_sol) > 40:
                    # 巧妙將其插入到主解答的「綜上所述」總結句之前
                    if "**綜上所述" in p_sol:
                        parts = p_sol.split("**綜上所述", 1)
                        p_sol = parts[0].strip() + "\n\n" + extracted_sol + "\n\n**綜上所述" + parts[1]
                    else:
                        p_sol += "\n\n" + extracted_sol
                        
    primary["detailed_solution"] = p_sol

    # 3. 圖片互補：合併題幹附圖與幾何圖形
    p_imgs = primary.get("image_paths", []) or []
    s_imgs = secondary.get("image_paths", []) or []
    merged_imgs = list(dict.fromkeys(p_imgs + s_imgs))
    primary["image_paths"] = merged_imgs
    primary["has_image"] = len(merged_imgs) > 0

    # 4. 選項深入剖析互補：取字數較長、分析更詳細的版本
    p_opt_ana = str(primary.get("options_analysis", "")).strip()
    s_opt_ana = str(secondary.get("options_analysis", "")).strip()
    if len(s_opt_ana) > len(p_opt_ana) and "本題為非選擇題" not in s_opt_ana:
        primary["options_analysis"] = secondary["options_analysis"]

    # 5. 非選擇題評分標準互補
    if not primary.get("scoring_criteria") and secondary.get("scoring_criteria"):
        primary["scoring_criteria"] = secondary["scoring_criteria"]
    if not primary.get("rubric_image_paths") and secondary.get("rubric_image_paths"):
        primary["rubric_image_paths"] = secondary["rubric_image_paths"]

    # 6. 核心考點與解題技巧聯集
    p_cats = primary.get("topic_categories", []) or []
    s_cats = secondary.get("topic_categories", []) or []
    merged_cats = list(dict.fromkeys([c for c in p_cats + s_cats if c]))
    if merged_cats:
        primary["topic_categories"] = merged_cats
        primary["topic_category"] = merged_cats[0]

    p_techs = primary.get("techniques_used", []) or []
    s_techs = secondary.get("techniques_used", []) or []
    primary["techniques_used"] = list(dict.fromkeys([t for t in p_techs + s_techs if t]))

    return primary

def find_actual_pdf(db_path: str, candidate_pdf_path: str = "") -> str:
    """跨目錄精準搜尋真正的實體【題目卷】PDF（支援模擬考題解合卷與分卷，排除純評分卷）"""
    if candidate_pdf_path and os.path.exists(candidate_pdf_path):
        cand_base = os.path.basename(candidate_pdf_path)
        # 若為模擬考，詳解本或合卷本身就是題目與解析的真實來源，不可誤殺
        if "模擬" in db_path or "模考" in db_path:
            return candidate_pdf_path
        elif not any(bad in cand_base for bad in ["解答", "答案", "評分", "原則", "說明資料"]):
            return candidate_pdf_path
        
    base_file = os.path.basename(db_path).replace("_database.json", "").replace("_partial_database.json", "")
    year_m = re.search(r'(\d+)', base_file)
    year_str = year_m.group(1) if year_m else ""
    
    subj_kw = ""
    for s_name in ["數學甲", "數學乙", "數學A", "數學B", "數甲", "數乙", "數A", "數B", "數學", "物理", "化學", "生物", "地科", "地球科學", "國文", "國寫", "英文", "歷史", "地理", "公民與社會", "公民", "自然", "社會"]:
        if s_name in base_file:
            subj_kw = s_name; break

    # 提取機構關鍵字（如北模、中模、全模、翰林）
    org_kw = ""
    for o_cand in ["北模", "中模", "南模", "竹模", "全模", "翰林", "南一", "詮達", "全華", "漢樺", "文昌"]:
        if o_cand in base_file:
            org_kw = o_cand; break

    search_dirs = ["mock_exam_papers_only", "ast_exam_papers_only", "gsat_exam_papers_only", "school_exam_papers_only"]
    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            for root, dirs, files in os.walk(s_dir):
                for f in files:
                    if f.endswith(".pdf"):
                        # 🚨 剛性過濾：絕對不能是解答卷、非選評分原則！
                        if any(bad in f for bad in ["解答", "答案", "評分", "原則", "說明資料", "更正"]):
                            continue
                        if year_str and year_str in f:
                            if subj_kw and (subj_kw in f or subj_kw in root):
                                if any(good in f for good in ["題目", "試題", "試卷", "考科"]):
                                    return os.path.join(root, f)
    return ""
    
# -------------------------------------------------------------------------
# 1. 深度 LaTeX 與 Markdown 格式修復器
# -------------------------------------------------------------------------
def clean_latex_corruptions(text: str) -> str:
    if not isinstance(text, str) or not text: return text
    
    # 🚨 1. 徹底清除 AI 幻覺產生的連續星號垃圾（如 ********************）
    text = re.sub(r'\*{3,}', '', text)
    # 🚨 2. 修復數學符號與 Markdown 粗體交疊的語法硬傷（例如 $ **a$ 或 $ **a**$ -> **$a$**）
    text = re.sub(r'\$\s*\*\*([^\$]+?)\*\*\s*\$', r'**$\1$**', text)
    text = re.sub(r'\*\*\s*\$([^\$]+?)\$\s*\*\*', r'**$\1$**', text)
    text = re.sub(r'\$\s*\*\*([^\$]+?)\s*\$', r'**$\1$**', text)
    text = re.sub(r'\$\s*([a-zA-Z0-9_\-]+)\s*\$', r'$\1$', text)
    
    # 🚨 1. 終極拯救被無限繁殖的 \in\in\in...\infty 災難
    text = re.sub(r'(?:\\in\s*){2,}\\infty', r'\\infty', text)
    text = re.sub(r'(?:\\in\s*)+\\infty', r'\\infty', text)
    text = re.sub(r'(?<![a-zA-Z\\])(?<!\\in)(?<!in)\\?fty\b', r'\\infty', text)

    # 🚨 2. 救回被吃掉反斜線的 ightarrow 與 ight/eft
    text = re.sub(r'(?<!\\)\bightarrow\b', r'\\rightarrow', text)
    text = re.sub(r'(?<!\\)\bimplies\b', r'\\implies', text)
    text = re.sub(r'(?<!\\)\biff\b', r'\\iff', text)
    text = re.sub(r'\\?bsim\b', r'\\approx', text)
    text = re.sub(r'(?<=\\left[\(\[\{])(.*?)(?<!\\)ight([\)\]\}])', r'\1\\right\2', text)
    text = re.sub(r'(?<!\\)ight(?=[\)\]\}|\.])', r'\\right', text)
    text = re.sub(r'(?<!\\)eft(?=[\(\[\{])', r'\\left', text)

    # 🚨 3. 救回矩陣非法 \break 換行與符號錯位
    text = re.sub(r'\\break\b', r'\\\\', text)
    text = text.replace(r'\blacksquare', r'\in')
    text = re.sub(r'\\bgroup\s*(?:\\text\{\}|\s*\\quad\{\}|\{\})', '', text)
    text = re.sub(r'\\([PQR])\b', r'\1', text) # 消除將點 P,Q,R 誤寫為段落符號 \P, \Q, \R
    text = re.sub(r'\\z_([1-9])', r'z_\1', text)

    # 救回 \log 變異字元
    text = re.sub(r'\\bar\{log\}', r'\\log', text)
    text = re.sub(r'(?<!\\)\blog\b', r'\\log', text)
    text = re.sub(r'\\log_([0-9a-zA-Z])(?![0-9a-zA-Z{])', r'\\log_{\1}', text)
    
    # 救回括號與控制字元
    text = text.replace(r"\r\right", r"\right").replace(r"\r\left", r"\left").replace(r"\r", "")
    text = re.sub(r'[\x0c]rac(?![a-zA-Z])', r'\\frac', text)
    text = re.sub(r'(?<=\$)rac(?=\{)', r'\\frac', text)

    # 🚨 4. 清除 $$ ... $$ 內部夾雜多餘 $ 造成的 KaTeX 渲染崩潰
    def clean_nested_dollars(m):
        inner = m.group(1)
        cleaned_inner = inner.replace('$', '')
        return f"$${cleaned_inner}$$"
    text = re.sub(r'\$\$(.*?)\$\$', clean_nested_dollars, text, flags=re.DOTALL)
    
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

def harmonize_group_shared_context(questions: List[dict]) -> List[dict]:
    """
    題組 shared_context 歸一化引擎：
    識別同一大題的子小題（如 二(1), 二(2), 二(3) 或 15(a), 15(b)），
    挑選字數最齊全、排版最完整的 shared_context，統一覆蓋至同題組的所有子題。
    """
    def extract_group_base_tag(q_num_str: str) -> str:
        s = str(q_num_str).strip()
        m_cn = re.match(r'^([一二三四五六七八九十]+)', s)
        if m_cn: return m_cn.group(1)
        m_sub = re.match(r'^(\d+)[\-_(（]', s)
        if m_sub: return m_sub.group(1)
        return ""

    groups: Dict[str, List[dict]] = {}
    for q in questions:
        q_num = str(q.get("question_number", "")).strip()
        base_tag = extract_group_base_tag(q_num)
        if base_tag:
            if base_tag not in groups:
                groups[base_tag] = []
            groups[base_tag].append(q)

    for base_tag, q_list in groups.items():
        if len(q_list) > 1:
            # 挑選出該題組中最完整（長度最長、無亂碼且無連鎖星號）的 shared_context
            best_sc = ""
            for q in q_list:
                sc = clean_latex_corruptions(str(q.get("shared_context", "")).strip())
                if len(sc) > len(best_sc):
                    best_sc = sc
            
            if best_sc:
                for q in q_list:
                    q["shared_context"] = best_sc
    return questions

def heal_question_types(questions: List[dict]) -> List[dict]:
    """
    題型自癒修正器：
    非選擇題/手寫計算證明題（沒有 options 選項，或 options 為空）
    若被誤標記為「混合題」，強制扶正為「簡答題」。
    """
    for q in questions:
        opts = q.get("options", []) or []
        q_type = str(q.get("question_type", "")).strip()
        if q_type == "混合題" and len(opts) == 0:
            q["question_type"] = "簡答題"
    return questions

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

    # 幽靈關鍵字過濾
    if any(gk in combined for gk in GHOST_KEYWORDS_EXTENDED):
        return False

    # 🚨 精確排除純週期表/說明頁佔位題（自動避開正常的化學原子序考題）
    if is_phantom_table_or_cover(q):
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


# =========================================================================
# 🛡️ 模擬考空殼污染安檢與大考真題鋼鐵保護引擎
# =========================================================================
def quarantine_and_purge_corrupted_mocks() -> Tuple[int, int]:
    """
    🚨 雙重安檢：
    1. 100% 絕對豁免並保護：學測 (GSAT)、分科測驗/指考 (AST) 之官方真題，絕對不可刪除！
    2. 精確偵測並銷毀：無詳解的空殼模擬考 (如 111 南一數 A 空殼)、扁平缺少期別品牌的舊版本，
       並自動登記雲端銷毀，讓 catcher.py 能重新產出最高品質之完整詳解與圖片！
    """
    purged_count = 0
    protected_official_count = 0

    if not os.path.exists(DATABASE_DIR):
        return 0, 0

    for root, dirs, files in os.walk(DATABASE_DIR):
        for f in files:
            if not f.endswith("_database.json"):
                continue

            full_p = os.path.join(root, f)
            clean_f = f.replace(" ", "")

            # 🚨 1. 【大考真題鋼鐵白名單】：學測、指考、分科真題絕對不可刪除！
            is_mock_or_school = any(k in clean_f or k in root for k in [
                "模擬", "模考", "全模", "北模", "中模", "南模", "竹模", 
                "南一", "翰林", "龍騰", "康熹", "泰宇", "全華", "詮達", "文昌", "漢樺", 
                "學校", "段考", "定期考"
            ])
            is_official_gsat_or_ast = (
                ("學測" in clean_f or "指考" in clean_f or "分科" in clean_f or "學科能力" in clean_f)
                and not is_mock_or_school
            )

            if is_official_gsat_or_ast:
                protected_official_count += 1
                continue  # 🛡️ 官方真題享有最高級別豁免，絕對安全！

            # 🚨 自動攔截清理：模考檔名若夾帶「_理組_」或「_高三_」舊版命名污染，強制判定為銷毀重跑！
            if is_mock_or_school and any(bad in clean_f for bad in ["_理組_", "_文組_", "_高三_", "_高二_"]):
                should_purge = True
                purge_reason = "含有已淘汰之『理組/文組/高三』污染標籤，強制清理重跑"

            # 🚨 2. 檢測模擬考是否為「空殼題庫（無詳解）」或「扁平化丟失出版社標籤的污染檔案」
            should_purge = False
            purge_reason = ""

            # 讀取並檢驗內容是否含有詳解
            try:
                with open(full_p, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                if isinstance(data, list) and len(data) > 0:
                    empty_sol_count = sum(1 for q in data if len(str(q.get("detailed_solution", "")).strip()) < 20)
                    # 🚨 只要超過 25% 的題目完全沒有詳解，即判定為偽完工空殼，必須刪除讓 catcher.py 重跑！
                    if empty_sol_count >= len(data) * 0.25:
                        should_purge = True
                        purge_reason = f"含有 {empty_sol_count}/{len(data)} 題無詳解空殼題目"
                    else:
                        # 情況 A：若檔名遺失出版社/次數標籤，但題目本身有詳解，則不刪除，交由後續重新命名歸位！
                        pass
                else:
                    should_purge = True
                    purge_reason = "空白或損毀之 JSON 檔案"
            except Exception:
                should_purge = True
                purge_reason = "無法讀取之損毀檔案"

            # 執行銷毀並登記雲端清除（包含對應的 images 圖檔資料夾）
            if should_purge:
                try:
                    # 1. 刪除錯誤的 JSON 資料庫
                    os.remove(full_p)
                    register_cloud_purge(full_p)
                    
                    # 2. 🚨 連坐刪除對應的 images 圖檔目錄，避免殘留切錯的舊圖或孤兒圖檔
                    spec_base = clean_f.replace("_database.json", "")
                    possible_img_dirs = [
                        os.path.join(root, "images", spec_base),
                        os.path.join(root, spec_base),
                        os.path.join(DATABASE_DIR, "模擬考", "images", spec_base)
                    ]
                    for img_d in possible_img_dirs:
                        if os.path.exists(img_d):
                            import shutil
                            shutil.rmtree(img_d, ignore_errors=True)
                            register_cloud_purge(img_d)
                            logging.info(f"🖼️ [同步清理圖檔] 已徹底刪除對應之舊圖片目錄：{os.path.basename(img_d)}")

                    purged_count += 1
                    logging.info(f"🗑️ [安全淨化銷毀] 已刪除模擬考污染檔案：{f} (原因: {purge_reason})，並已登記雲端清除！")
                except Exception as e:
                    logging.warning(f"無法刪除檔案 {full_p}: {e}")

    return protected_official_count, purged_count

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
        # 🚨 1. 攔截像 Q37 這種未經 Stage 2 解析、完全沒有詳解內容的空殼題！
        sol_text = str(q.get("detailed_solution", "")).strip()
        if not sol_text or len(sol_text) < 20 or "超時或失敗" in sol_text:
            stats["ghosts"] += 1
            continue

        if not is_valid_question_strict(q, max_pdf_q, physical_q_set, expected_subject, exam_tag):
            stats["ghosts"] += 1
            continue
            
        # 🚨 2. 剛性答案格式清洗、空白答案自癒與 has_official_answer 精確校正
        det_sol = str(q.get("detailed_solution", ""))
        cur_ans = str(q.get("answer", "")).strip()

        # A. 若 answer 為空，啟動名師詳解逆向萃取器，100% 救回真實答案
        if not cur_ans:
            m_ans = re.search(r'\*\*綜上所述[，,]\s*(?:本題)?(?:正確答案|正確選項|本題正確選項|選填題各畫卡格答案)?(?:為)?[：:\s]*\**\s*([^\n\*]+?)\**(?:\(|$|\n)', det_sol)
            if m_ans:
                cand = m_ans.group(1).strip()
                pure_keys = re.findall(r'[A-Ga-g1-9]', cand)
                if pure_keys and q.get("question_type") in ["單選題", "多選題"]:
                    cur_ans = "".join(pure_keys) if q.get("question_type") == "多選題" else pure_keys[0]
                elif cand in ["證明題", "見評分", "無答案（全體給分）"]:
                    cur_ans = cand
                else:
                    cur_ans = cand.strip("()$* ")
            if not cur_ans and q.get("derived_answer"):
                cur_ans = str(q["derived_answer"]).strip()
            
            q["answer"] = cur_ans
            q["has_official_answer"] = False
            q["official_answer"] = ""
            q["derived_answer"] = cur_ans
            q["has_answer_discrepancy"] = False
            q["discrepancy_reason"] = ""

        # B. 若官方答案確實存在且有效
        elif q.get("has_official_answer") is not False:
            if not q.get("official_answer"):
                q["official_answer"] = cur_ans
            if not q.get("derived_answer"):
                q["derived_answer"] = cur_ans

        # C. 修正「選出錯誤選項」導致的 1234 假衝突（如 100 指考數甲 Q1）
        q_text = str(q.get("question_text", ""))
        is_asking_false = any(k in q_text for k in ["哪一個選項是錯誤", "何者錯誤", "選出錯誤", "何者不正確"])
        if is_asking_false and q.get("has_answer_discrepancy"):
            off_a = str(q.get("official_answer", "")).strip()
            if len(off_a) == 1:
                q["answer"] = off_a
                q["derived_answer"] = off_a
                q["has_answer_discrepancy"] = False
                q["discrepancy_reason"] = ""
                if "**綜上所述" in q["detailed_solution"]:
                    base_part = q["detailed_solution"].split("**綜上所述")[0].strip()
                    q["detailed_solution"] = f"{base_part}\n\n**綜上所述，本題正確選項為：({off_a})**"

        # D. 選項字母代號標準化
        for ans_key in ["answer", "official_answer"]:
            if ans_key in q and q[ans_key]:
                raw_ans = str(q[ans_key]).strip()
                if raw_ans not in ["證明題", "無答案（全體給分）"]:
                    m_opt = re.search(r'^[（\(]?([A-Ga-g1-9]+)[）\)]?$', raw_ans)
                    if m_opt and q.get("question_type") in ["單選題", "多選題"]:
                        q[ans_key] = m_opt.group(1).upper()

        # 🚨 3. 單選題防多選幻覺修復：若單選題被塞入多個字母，強制重設為官方答案
        q_type = str(q.get("question_type", "")).strip()
        if q_type == "單選題":
            der_ans = str(q.get("derived_answer", "")).strip()
            # 如果單選題的 derived_answer 跑出 "BDE"、"ABCD" 等長度大於 1 的字串
            if len(re.findall(r'[A-Z]', der_ans)) > 1:
                q["derived_answer"] = q["answer"]
                q["has_answer_discrepancy"] = False
                q.pop("discrepancy_reason", None)
                # 清除詳解尾部硬凹的「本題學理嚴謹推導之正確選項為：(B)(D)(E)」
                if "綜上所述" in q.get("detailed_solution", ""):
                    q["detailed_solution"] = q["detailed_solution"].split("綜上所述")[0].strip() + f"\n\n**綜上所述，本題正確選項為：({q['answer']})**"

        # 🚨 4. 清理字串內殘留的字面量 '\\n'
        for k in ["detailed_solution", "question_analysis", "solving_strategy"]:
            if k in q and isinstance(q[k], str):
                q[k] = q[k].replace("\\n", "\n")

        q_num = str(q.get("question_number", "")).strip()
        key = (q_num, q_type)
        
        # 題幹純文字特徵指紋 (去除標點空格，取前 30 字)
        clean_text_fp = re.sub(r'[^\w\u4e00-\u9fa5]', '', str(q.get("question_text", "")))[:30]
        
        current_sol_len = len(str(q.get("detailed_solution", "")))
        # 情況 A：題號完全相同
        if key in merged_map:
            merged_map[key] = fuse_duplicate_question_solutions(merged_map[key], q)
            continue

        if clean_text_fp and len(clean_text_fp) >= 10:
            if clean_text_fp in text_fingerprints:
                old_k = text_fingerprints[clean_text_fp]
                fused_q = fuse_duplicate_question_solutions(merged_map[old_k], q)
                del merged_map[old_k]
                merged_map[key] = fused_q
                text_fingerprints[clean_text_fp] = key
                continue
            text_fingerprints[clean_text_fp] = key

        merged_map[key] = q

    merged_list = list(merged_map.values())
    stats["merged"] = len(merged_list)

    # 🚨 1. 執行題型扶正（無選項的「混合題」修正為「簡答題」）
    merged_list = heal_question_types(merged_list)
    # 🚨 2. 執行題組 shared_context 強制歸一化
    merged_list = harmonize_group_shared_context(merged_list)
    
    # 🚨 3. 執行分類清洗、深度 LaTeX 清理，並補齊 exam_type / exam_category
    for q in merged_list:
        # 若題目原本未標註 exam_type，依據考卷標籤自動補齊
        if not q.get("exam_type"):
            if "學測" in exam_tag and "模" not in exam_tag:
                q["exam_type"] = "GSAT"
                q["exam_category"] = "學測"
            elif any(k in exam_tag for k in ["指考", "分科"]) and "模" not in exam_tag:
                q["exam_type"] = "AST"
                q["exam_category"] = "分科指考"
            elif "學校" in db_path:
                q["exam_type"] = "SCHOOL"
                q["exam_category"] = "學校段考"
            elif any(k in db_path for k in ["指考", "分科", "數甲", "數乙"]):
                q["exam_type"] = "MOCK_AST"
                q["exam_category"] = "分科模擬"
            else:
                q["exam_type"] = "MOCK_GSAT"
                q["exam_category"] = "學測模擬"

        main_cat = sanitize_knowledge_point(q.get("topic_category", ""))
        cats = [sanitize_knowledge_point(c) for c in q.get("topic_categories", []) if c]
        q["topic_categories"] = list(dict.fromkeys(cats)) if cats else [main_cat]
        q["topic_category"] = q["topic_categories"][0]
        
        for k in ["question_text", "detailed_solution", "question_analysis", "solving_strategy", "scoring_rubric", "concept_review", "traps_and_warnings", "shared_context"]:
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
    print("🚀 啟動【題庫安全防護、大考真題豁免、模擬考空殼淨化與無損去重修復引擎】...")
    print("=" * 75)

    # 1. 淨化知識庫單元名詞
    sanitize_subject_json()

    if not os.path.exists(DATABASE_DIR):
        print(f"❌ 找不到題庫目錄: {DATABASE_DIR}")
        return

    # 2. 🚨 執行安檢淨化：徹底銷毀無詳解的空殼模考，並 100% 豁免大考真題
    protected_cnt, purged_cnt = quarantine_and_purge_corrupted_mocks()
    print(f"🛡️ [大考真題保護] 共核對並保護 {protected_cnt} 份學測/分科大考真題庫 (100% 豁免零誤刪)！")
    if purged_cnt > 0:
        print(f"🧹 [空殼污染清除] 共安全銷毀 {purged_cnt} 份無詳解或扁平化之舊版模考檔，已登記雲端同步抹除！")

    # 1. 深度掃描全庫所有 JSON 檔案，以「大考核心特徵」建立智慧分組池
    # key: (y_val, exam_type, is_makeup, clean_subject, folder)
    exam_groups: Dict[Tuple, List[str]] = {}

    for root, dirs, files in os.walk(DATABASE_DIR):
        for f in files:
            # 🚨 關鍵阻斷：絕對禁止將 _raw_extracted.json 視為正式題庫載入！它只是 Stage 1 的未解題快取！
            if f.endswith(("_database.json", "_partial_database.json")):
                full_p = os.path.join(root, f)
                # 排除 .1.json 等重複備份雜訊
                if re.search(r'\.\d+\.json$', f):
                    try: os.remove(full_p)
                    except: pass
                    continue
                    
                # 解析本檔特徵
                clean_name = f.replace(" ", "")
                y_match = re.search(r'(\d+)', clean_name)
                y_val = int(y_match.group(1)) if y_match else 0
                
                is_makeup = "補考" in clean_name
                
                # 判斷科目
                subj = "未知"
                for s_cand in ["數學甲", "數學乙", "數學A", "數學B", "數甲", "數乙", "數A", "數B", "數學", 
                               "物理", "化學", "生物", "地球科學", "地科", "歷史", "地理", "公民與社會", "公民", "國文", "國寫", "英文", "自然", "社會"]:
                    if s_cand in clean_name:
                        subj = s_cand
                        break

                # 🚨 精細判斷大考類型（徹底區分 真學測/真分科 與 學測模擬/分科模擬！）
                is_mock_kw = any(k in clean_name or k in root for k in ["模擬", "模考", "全模", "北模", "中模", "南模", "竹模", "全區", "中區", "詮達", "全華", "翰林", "南一", "文昌", "漢樺", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"])
                is_school_kw = "學校" in root or any(k in clean_name for k in ["定期", "段考", "期中", "期末", "月考"])

                if is_school_kw and not is_mock_kw:
                    e_type = "SCHOOL"
                elif is_mock_kw:
                    if any(k in clean_name for k in ["指考", "分科", "數甲", "數乙", "數學甲", "數學乙"]):
                        e_type = "MOCK_AST"
                    elif any(k in clean_name for k in ["學測", "學科能力", "數A", "數B", "數學A", "數學B", "自然", "社會"]):
                        e_type = "MOCK_GSAT"
                    else:
                        e_type = "MOCK_GSAT"
                elif "學測" in clean_name or "學科能力" in clean_name:
                    e_type = "GSAT"
                else:
                    e_type = "AST"

                # 🚨 提早從檔名或舊路徑提取完整的 mock_tag（保留 全模_第2次、南一_第4次、北模_第1次 等標籤！）
                mock_tag = ""
                school_name = ""
                if e_type == "SCHOOL":
                    sch_match = re.search(r'([\u4e00-\u9fa5]{2,8}(?:高級中學|高中|女中|中學|實驗中學|高職|附中))', clean_name + root)
                    cand_sch = sch_match.group(1) if sch_match else ""
                    if cand_sch and not any(k in cand_sch for k in ["學年度", "學期", "全國", "公私立", "全區", "聯合", "模擬"]):
                        school_name = cand_sch
                    else:
                        school_name = "通用學校"
                    exam_m = re.search(r'((?:第[一二三四五六七八九十0-9]+次?)?(?:定期考|段考|期中考|期末考|月考))', clean_name)
                    mock_tag = f"_{exam_m.group(1)}" if exam_m else "_段考"
                elif e_type in ["MOCK", "MOCK_GSAT", "MOCK_AST"]:
                    # 提取品牌/聯盟 (全面支援「全國高中」、「全國公私立」映射為「全模」)
                    pub_m = re.search(r'(南一|翰林|龍騰|康熹|泰宇|全華|詮達|文昌|漢樺|三民)', clean_name + root)
                    leg_m = re.search(r'(北北基|新北基|北模|中模|南模|竹模|全模|全區|中區|台中市|台中區|台北區|臺北區|北區|南區|高雄區|全國高中|全國公私立|公私立高級中學|全國聯合|公私立高中)', clean_name + root)
                    
                    if pub_m:
                        org_str = pub_m.group(1)
                    elif leg_m:
                        raw_leg = leg_m.group(1)
                        leg_map = {
                            "北北基": "北模", "台北區": "北模", "臺北區": "北模", "新北基": "北模", "北區": "北模",
                            "台中區": "中模", "台中市": "中模", "中區": "中模", "高雄區": "南模", "南區": "南模",
                            "全模": "全模", "全區": "全模", "全國高中": "全模", "全國公私立": "全模", "公私立高級中學": "全模", "全國聯合": "全模", "公私立高中": "全模"
                        }
                        org_str = leg_map.get(raw_leg, "全模")
                    else:
                        org_str = "全模"
                        
                    # 提取次數/期別
                    times_m = re.search(r'(?:第\s*([一二三四五六七八九十0-9]+)\s*[次回]|第\s*([0-9]+)\s*模|E([1-8])|([1-8])\s*模|[（\(]([1-8一二三四五六七八])[）\)])', clean_name + root)
                    times_str = ""
                    if times_m:
                        raw_t = next(g for g in times_m.groups() if g is not None)
                        cn_map = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8'}
                        times_str = f"第{cn_map.get(raw_t, raw_t)}次"
                    elif re.search(r'第二學期|下學期', clean_name + root):
                        times_str = "下學期"
                    elif re.search(r'第一學期|上學期', clean_name + root):
                        times_str = "上學期"
                    else:
                        times_str = "全期"
                    mock_tag = f"_{org_str}_{times_str}"

                if is_makeup:
                    mock_tag += "_補考"

                meta = standardize_exam_metadata(str(y_val), e_type, subj, mock_tag=mock_tag, school_name=school_name)
                # 🚨 group_key 納入 spec_name，確保南一第4次與翰林第1次絕對不會被強行融合成同一個檔案！
                group_key = (meta["y_val"], e_type, meta["is_makeup"], meta["clean_subject"], meta["spec_name"], meta["type_folder"], mock_tag, school_name)
                
                if group_key not in exam_groups:
                    exam_groups[group_key] = []
                exam_groups[group_key].append(full_p)

    total_merged_papers = 0
    total_ghosts_purged = 0

    # 2. 對每個大考核心組別進行「跨檔案無損大合併」
    for group_key, file_list in exam_groups.items():
        y_val, e_type, is_makeup, clean_sub, spec_name_key, type_folder, mock_tag, school_name = group_key
        meta = standardize_exam_metadata(str(y_val), e_type, clean_sub, mock_tag=mock_tag, school_name=school_name)
        
        target_dir = os.path.join(DATABASE_DIR, type_folder)
        os.makedirs(target_dir, exist_ok=True)
        
        canonical_db_path = os.path.join(target_dir, f"{meta['spec_name']}_database.json")
        canonical_partial_path = os.path.join(target_dir, f"{meta['spec_name']}_partial_database.json")
        canonical_raw_path = os.path.join(target_dir, f"{meta['spec_name']}_raw_extracted.json")

        # 收集該組所有來源檔案內的題目
        pooled_questions = []
        for fp in file_list:
            try:
                with open(fp, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                    if isinstance(data, list):
                        pooled_questions.extend(data)
            except Exception: pass

        if not pooled_questions:
            continue

        # 尋找實體 PDF 校準題數
        actual_pdf = find_actual_pdf(canonical_db_path, "")
        max_pdf_q, physical_q_set = get_real_pdf_max_question_info(actual_pdf, exam_name=meta["spec_name"])

        # 🚨 雙重去重閥門：合併兩份考卷時比對「題號」與「題幹指紋」，將雙倍題目合而為一
        merged_map = {}
        text_fps = {}

        for q in pooled_questions:
            # 🚨 剛性門檻：若題目未經 Stage 2 解析、完全沒有詳解內容，絕對不可作為有效題目合併！
            sol_txt = str(q.get("detailed_solution", "")).strip()
            if not sol_txt or len(sol_txt) < 20 or "超時或失敗" in sol_txt:
                total_ghosts_purged += 1
                continue

            if not is_valid_question_strict(q, max_pdf_q, physical_q_set, clean_sub):
                total_ghosts_purged += 1
                continue

            # 🚨【核心注入】：空白答案自動救回、格式標準化與反向題型校正
            det_sol = str(q.get("detailed_solution", ""))
            cur_ans = str(q.get("answer", "")).strip()

            # A. 若 answer 為空，啟動名師詳解逆向萃取器，100% 救回真實答案
            if not cur_ans:
                m_ans = re.search(r'\*\*綜上所述[，,]\s*(?:本題)?(?:正確答案|正確選項|本題正確選項|選填題各畫卡格答案)?(?:為)?[：:\s]*\**\s*([^\n\*]+?)\**(?:\(|$|\n)', det_sol)
                if m_ans:
                    cand = m_ans.group(1).strip()
                    pure_keys = re.findall(r'[A-Ga-g1-9]', cand)
                    if pure_keys and q.get("question_type") in ["單選題", "多選題"]:
                        cur_ans = "".join(pure_keys) if q.get("question_type") == "多選題" else pure_keys[0]
                    elif cand in ["證明題", "見評分", "無答案（全體給分）"]:
                        cur_ans = cand
                    else:
                        cur_ans = cand.strip("()$* ")
                if not cur_ans and q.get("derived_answer"):
                    cur_ans = str(q["derived_answer"]).strip()
                
                q["answer"] = cur_ans
                q["has_official_answer"] = False
                q["official_answer"] = ""
                q["derived_answer"] = cur_ans
                q["has_answer_discrepancy"] = False
                q["discrepancy_reason"] = ""

            # B. 若官方答案確實存在且有效
            elif q.get("has_official_answer") is not False:
                if not q.get("official_answer"):
                    q["official_answer"] = cur_ans
                if not q.get("derived_answer"):
                    q["derived_answer"] = cur_ans

            # C. 修正「選出錯誤選項」導致的 1234 假衝突（如 100 指考數甲 Q1）
            q_text = str(q.get("question_text", ""))
            is_asking_false = any(k in q_text for k in ["哪一個選項是錯誤", "何者錯誤", "選出錯誤", "何者不正確"])
            if is_asking_false and q.get("has_answer_discrepancy"):
                off_a = str(q.get("official_answer", "")).strip()
                if len(off_a) == 1:
                    q["answer"] = off_a
                    q["derived_answer"] = off_a
                    q["has_answer_discrepancy"] = False
                    q["discrepancy_reason"] = ""
                    if "**綜上所述" in q["detailed_solution"]:
                        base_part = q["detailed_solution"].split("**綜上所述")[0].strip()
                        q["detailed_solution"] = f"{base_part}\n\n**綜上所述，本題正確選項為：({off_a})**"

            # D. 單選題防多選幻覺修復：若單選題被塞入多個字母，強制重設為官方答案
            q_type = str(q.get("question_type", "")).strip()
            if q_type == "單選題":
                der_ans = str(q.get("derived_answer", "")).strip()
                if len(re.findall(r'[A-Z]', der_ans)) > 1:
                    q["derived_answer"] = q["answer"]
                    q["has_answer_discrepancy"] = False
                    q.pop("discrepancy_reason", None)
                    if "綜上所述" in q.get("detailed_solution", ""):
                        q["detailed_solution"] = q["detailed_solution"].split("綜上所述")[0].strip() + f"\n\n**綜上所述，本題正確選項為：({q['answer']})**"

            # E. 選項字母代號標準化
            for ans_key in ["answer", "official_answer"]:
                if ans_key in q and q[ans_key]:
                    raw_ans = str(q[ans_key]).strip()
                    if raw_ans not in ["證明題", "無答案（全體給分）"]:
                        m_opt = re.search(r'^[（\(]?([A-Ga-g1-9]+)[）\)]?$', raw_ans)
                        if m_opt and q.get("question_type") in ["單選題", "多選題"]:
                            q[ans_key] = m_opt.group(1).upper()

            # F. 清理字串內殘留的字面量 '\\n'
            for k in ["detailed_solution", "question_analysis", "solving_strategy"]:
                if k in q and isinstance(q[k], str):
                    q[k] = q[k].replace("\\n", "\n")

            q_num = str(q.get("question_number", "")).strip()
            key = (q_num, q_type)
            clean_text_fp = re.sub(r'[^\w\u4e00-\u9fa5]', '', str(q.get("question_text", "")))[:30]

            if key in merged_map:
                merged_map[key] = fuse_duplicate_question_solutions(merged_map[key], q)
                continue

            if clean_text_fp and len(clean_text_fp) >= 10:
                if clean_text_fp in text_fps:
                    old_k = text_fps[clean_text_fp]
                    fused_q = fuse_duplicate_question_solutions(merged_map[old_k], q)
                    del merged_map[old_k]
                    merged_map[key] = fused_q
                    text_fps[clean_text_fp] = key
                    continue
                text_fps[clean_text_fp] = key

            merged_map[key] = q

        merged_list = list(merged_map.values())

        # 🚨 1. 執行題型扶正與題組 shared_context 歸一化
        merged_list = heal_question_types(merged_list)
        merged_list = harmonize_group_shared_context(merged_list)

        # 🚨 2. 全面統一 exam_source、academic_year，並強制寫入 exam_type 與 exam_category
        for q in merged_list:
            q["academic_year"] = meta["academic_year"]
            q["exam_source"] = meta["exam_source"]
            # 🚨 注入精確大考分類，解決網頁後台全部變成 MOCK 的標籤遺失問題！
            q["exam_type"] = meta["exam_type"]
            q["exam_category"] = meta["exam_category"]
            main_cat = sanitize_knowledge_point(q.get("topic_category", ""))
            cats = [sanitize_knowledge_point(c) for c in q.get("topic_categories", []) if c]
            q["topic_categories"] = list(dict.fromkeys(cats)) if cats else [main_cat]
            q["topic_category"] = q["topic_categories"][0]
            for k in ["question_text", "detailed_solution", "question_analysis", "solving_strategy", "scoring_rubric", "concept_review", "traps_and_warnings", "shared_context"]:
                if k in q: q[k] = clean_latex_corruptions(q[k])

        def safe_sort(x):
            try: p = int(x.get("page_number", 1))
            except: p = 1
            return (p, natural_sort_key(x.get("question_number", "0")))
        merged_list.sort(key=safe_sort)

        current_count = len(merged_list)
        r_min, r_max = EXAM_REASONABLE_RANGES.get(clean_sub, (12, 50))
        
        # 🚨 剛性題數門檻：若有實體 PDF，必須達到實體題數的 90% 以上（且至少達到 r_min），杜絕 20 題只做 15 題就假完工！
        if max_pdf_q > 0:
            target_threshold = max(r_min, int(max_pdf_q * 0.9))
            is_complete = (current_count >= target_threshold)
            display_threshold = max_pdf_q
        else:
            is_complete = (current_count >= r_min)
            display_threshold = r_min

        # 寫入正規命名的目標檔案
        if is_complete:
            with open(canonical_db_path, "w", encoding="utf-8") as f:
                json.dump(merged_list, f, ensure_ascii=False, indent=4)
            if os.path.exists(canonical_partial_path):
                os.remove(canonical_partial_path)
                register_cloud_purge(canonical_partial_path)
            status_str = f"✅ 100% 完工存檔 (實體收錄 {current_count}/{display_threshold} 題，已統一來源為「{meta['exam_source']}」)"
        else:
            with open(canonical_partial_path, "w", encoding="utf-8") as f:
                json.dump(merged_list, f, ensure_ascii=False, indent=4)
            if os.path.exists(canonical_db_path):
                try: os.remove(canonical_db_path); register_cloud_purge(canonical_db_path)
                except: pass
            status_str = f"⏳ 題數不足 ({current_count}/{display_threshold} 題)，已降級 partial，待補齊真題"

        # 🚨 關鍵清理：將組內所有「非正規檔名」的舊版本徹底從本機刪除，並登記雲端銷毀！
        for old_fp in file_list:
            norm_old = os.path.abspath(old_fp)
            if norm_old not in [os.path.abspath(canonical_db_path), os.path.abspath(canonical_partial_path)]:
                try:
                    os.remove(old_fp)
                    register_cloud_purge(old_fp)
                    logging.info(f"🗑️ [清理舊版異名檔案] 已刪除並登記雲端銷毀：{os.path.basename(old_fp)}")
                except: pass

        total_merged_papers += 1
        print(f"📦 [{meta['spec_name']}] (整合了 {len(file_list)} 個來源檔) -> {status_str}")

    print("=" * 75)
    print(f"🎉 處理完畢！共智慧合併 {total_merged_papers} 份大考題庫，斬除 {total_ghosts_purged} 筆多餘幽靈題。")
    print("=" * 75)

def manual_purge_by_keyword(keyword: str):
    """手動透過指令列或關鍵字，強制銷毀所有符合的題庫檔案與圖片目錄"""
    import shutil
    print(f"🗑️ [手動銷毀] 正在搜尋並銷毀包含關鍵字『{keyword}』的所有檔案與圖檔目錄...")
    purged_cnt = 0
    clean_kw = keyword.replace(" ", "")

    for root, dirs, files in os.walk(DATABASE_DIR):
        for f in files:
            if clean_kw in f.replace(" ", "") and f.endswith((".json", ".txt")):
                fp = os.path.join(root, f)
                try:
                    os.remove(fp)
                    register_cloud_purge(fp)
                    print(f"  -> 已刪除檔案: {f}")
                    purged_cnt += 1
                except Exception as e:
                    print(f"  -> 無法刪除檔案 {f}: {e}")

        for d in list(dirs):
            if clean_kw in d.replace(" ", ""):
                dp = os.path.join(root, d)
                try:
                    shutil.rmtree(dp, ignore_errors=True)
                    register_cloud_purge(dp)
                    print(f"  -> 🖼️ 已刪除圖片目錄: {d}")
                    purged_cnt += 1
                except Exception as e:
                    print(f"  -> 無法刪除目錄 {d}: {e}")

    print(f"✅ 銷毀完畢！共清理 {purged_cnt} 個項目，並已同步寫入 pending_cloud_purges.txt 登記雲端銷毀！")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--purge":
        target_kw = sys.argv[2] if len(sys.argv) > 2 else ""
        if target_kw:
            manual_purge_by_keyword(target_kw)
        else:
            print("❌ 請指定要銷毀的關鍵字，例如: python repair_database.py --purge 107學年度_指考模擬_理組")
    else:
        run_database_repair_suite()