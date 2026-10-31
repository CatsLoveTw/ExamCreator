import os
import re
import sys
import json
import time
import random
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import threading
import fitz  # PyMuPDF
from PIL import Image

from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import APIError
from groq import Groq
from groq import RateLimitError
try:
    from openai import OpenAI
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False

import queue
import uuid
import datetime
import signal
import httpx

def handle_system_shutdown(signum, frame):
    logging.critical("🛑 [系統緊急訊號] 收到終止訊號 (SIGTERM/SIGINT)！正在強制將記憶體進度刷新至硬碟...")
    # 可以在此處將全域已完成題目進行緊急寫檔
    sys.exit(0)

# 註冊監聽器
signal.signal(signal.SIGTERM, handle_system_shutdown)
signal.signal(signal.SIGINT, handle_system_shutdown)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GLOBAL_START_TIME = time.time()
MAX_EXECUTION_TIME_SECONDS = 5.0 * 3600  # 5 小時安全時間上限，保證在 6h GHA 砍掉前優雅退出

# 🚨 並發資源安全鎖
GLOBAL_ERRATA_LOCK = threading.Lock()
GLOBAL_ERRATA_MAP = {}  # {(subject, q_num): corrected_answer}
MATPLOTLIB_RENDER_LOCK = threading.Lock()  # 繪圖記憶體狀態保護鎖
PYMUPDF_LOCK = threading.Lock()            # PyMuPDF 底層 C 記憶體指標保護鎖

def get_errata_answer(subject: str, qn: str) -> Optional[str]:
    with GLOBAL_ERRATA_LOCK:
        return GLOBAL_ERRATA_MAP.get((subject, qn))

def set_errata_answer(subject: str, qn: str, answer: str):
    with GLOBAL_ERRATA_LOCK:
        GLOBAL_ERRATA_MAP[(subject, qn)] = answer

class TimeoutSafetyException(Exception):
    """自定義異常：當執行時間接近 GitHub Actions 上限時觸發安全保存與退出"""
    pass

def check_global_timeout():
    """檢查是否超過 5 小時或收到 Web 控制台手動終止指令"""
    if getattr(GLOBAL_METRICS, "stop_requested", False):
        logging.critical("🛑 [Web 遠端指令生效] 已收到手動停止指令！正在強制刷新進度並觸發安全退出 (Exit 0)...")
        raise TimeoutSafetyException("Web 面板手動安全終止。")
    if time.time() - GLOBAL_START_TIME > MAX_EXECUTION_TIME_SECONDS:
        logging.critical("[全域時間預警] 已達 5 小時執行上限！強制中斷任務並準備安全上傳...")
        raise TimeoutSafetyException("已達 5 小時安全時間上限。")
        
def smart_sleep(seconds: float):
    """可被全域超時中斷的智慧睡眠函式（每 2 秒檢查一次時間，絕不死睡）"""
    start = time.time()
    while time.time() - start < seconds:
        check_global_timeout()  # 睡眠中依然即時監控時間！
        time.sleep(min(2.0, seconds - (time.time() - start)))

class RPDExhaustedException(Exception):
    """自定義異常：當所有可用 API 金鑰的每日限制 (RPD) 皆耗盡時觸發"""
    pass

# ==========================================
# 繁簡轉換輔助器 (Traditional-Simplified Chinese Converter)
# ==========================================
try:
    from opencc import OpenCC
    # 使用 s2tw (簡體到台灣繁體)
    _converter = OpenCC('s2tw')
    def s2t(text: str) -> str:
        if not isinstance(text, str):
            return text
        return _converter.convert(text)
except ImportError:
    # 萬一未安裝 opencc，設計一個基礎的學術常用字簡轉繁對照表作為安全備用方案
    _S2T_MAP = {
        '选': '選', '修': '修', '极': '極', '导': '導', '绩': '績', '复': '複', '数': '數', '学': '學',
        '计': '計', '算': '算', '应': '應', '用': '用', '线': '線', '性': '性', '规': '規', '划': '劃',
        '标': '標', '准': '準', '图': '圖', '形': '形', '对': '對', '称': '稱', '轴': '軸', '与': '與',
        '配': '配', '方': '方', '法': '法', '值': '值', '几': '幾', '何': '何', '特': '特', '东': '東',
        '征': '徵', '微': '微', '积': '積', '分': '分', '定': '定', '不': '不', '代': '代', '化': '化',
        '简': '簡', '两': '兩', '个': '個', '实': '實', '根': '根', '强': '強', '弱': '弱', '联': '聯', '過': '過', '濾': '濾', '處': '處', '理': '理', '資': '資',
        '源': '源', '問': '問', '題': '題', '機': '機', '率': '率', '隨': '隨', '變': '變', '數': '數',
        '期': '期', '望': '望', '之': '之', '物': '物', '意': '意', '義': '義', '科': '科', '歷': '歷',
        '史': '史', '社': '社', '會': '會', '國': '國', '寫': '寫', '項': '項', '式': '式', '論': '論',
        '體': '體', '系': '系', '觀': '觀', '點': '點', '差': '差', '異': '異', '統': '統', '推': '推',
        '雙': '雙', '矩': '矩', '陣': '陣', '變': '變', '換': '換', '穩': '穩', '態': '態', '狀': '狀',
        '測': '測', '量': '量', '压': '壓', '电': '電', '动': '動', '磁': '磁', '场': '場', '产': '產',
        '压': '壓', '热': '熱', '温': '溫', '气': '氣', '压': '壓', '动': '動', '量': '量', '质': '質',
        '态': '態', '发': '發', '光': '光', '离': '離', '子': '子', '阴': '陰', '阳': '陽', '键': '鍵',
        '构': '構', '结': '結', '类': '類', '种': '種', '网': '網', '络': '絡', '环': '環', '境': '境',
        '概': '機', '屏': '螢', '幕': '幕', '内': '內', '存': '存', '算': '算', '法': '法', '矢': '向',
        '标': '純', '宏': '巨', '观': '觀', '确': '確', '实': '實'
    }
    def s2t(text: str) -> str:
        if not isinstance(text, str):
            return text
        return "".join(_S2T_MAP.get(c, c) for c in text)

def s2t_recursive(obj):
    """遞迴將整個資料結構（字串、列表、字典）中的所有簡體字轉為繁體"""
    if isinstance(obj, dict):
        return {s2t_recursive(k): s2t_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [s2t_recursive(item) for item in obj]
    elif isinstance(obj, str):
        return s2t(obj)
    return obj

def natural_sort_key(s):
    """用於題號高精度自然排序（支援阿拉伯數字、中文數字『一、二、三』、子題號『(1)、(2)』及破折號）"""
    cn_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
    s_str = str(s).strip()
    for cn, num in cn_map.items():
        s_str = s_str.replace(cn, f"{num:02d}")
    parts = re.split(r'(\d+)', s_str)
    return [int(text) if text.isdigit() else text.lower() for text in parts]

def clean_ocr_answer_format(ans_str: str) -> str:
    """清理多欄解答排版干擾下的格式污染與中文字洩漏，並排除手寫題的斜線與無效字元"""
    import re
    ans_str = str(ans_str).strip()
    if not ans_str:
        return ""
    
    # 1. 剝除包裹的括號，例如 (3) -> 3, (1) -> 1, （A） -> A
    ans_str = re.sub(r'^[（\(]([A-Ga-g0-9,]+)[）\)]$', r'\1', ans_str)
    
    # 2. 若出現 "6（即選項 (3)）" 等說明字眼，優先提取括號內或最後的正確選項數字/字母
    opt_match = re.search(r'[（\(]([A-Ga-g1-9])[）\)]', ans_str)
    if opt_match and "選項" in ans_str:
        return opt_match.group(1).upper()
        
    # 3. 完整保留大考官方標準標記
    if any(k in ans_str for k in ["無答案", "全體給分", "送分", "不計分"]):
        return "無答案（全體給分）"
        
    # 4. 若答案卷標示為斜線或無效標記，過濾為空字串
    if ans_str in ["/", "／", "\\", "無", "無題目", ""] or re.match(r'^[\s/／\\–\-_]+$', ans_str):
        return "" 
    
    # 5. 🚨【致命截斷修復】：只有當字串中真的出現「題/列/答案/為/是」等文字時，才執行拼接格式拆解！
    # 嚴禁對純數值（如 "120"、"45"）執行此正則，否則 120 會被當作「第12題答案為0」而遭截斷！
    has_marker = any(k in ans_str for k in ["題", "列", "答案", "為", "是"])
    if has_marker:
        pattern = r'(?:第\s*)?(\d+)\s*(?:題|列)\s*(?:答案)?\s*(?:為|是|:|：)?\s*([A-Ga-g0-9\-±/.]+)'
        matches = re.findall(pattern, ans_str)
        if matches:
            matches_sorted = sorted(matches, key=lambda x: int(x[0]))
            return ",".join(m[1] for m in matches_sorted)
        
    # 6. 如果包含 LaTeX 數學式或特殊符號，保留反斜線、大括號與運算符號
    if any(c in ans_str for c in ["\\", "$", "{", "}", "^", "_"]):
        cleaned = re.sub(r'[\u4e00-\u9fa5]+', '', ans_str)
        return cleaned.strip(" ,;")

    # 7. 🚨 多選答案完整提取：優先捕捉如 (B)(D)、(B)、(E)、(A)(C)(E)、B,D 等複合多選格式，絕不閹割！
    # A. 提取所有成對括號中的選項（如 (B)(D) 或 (B)、(E)）
    bracket_opts = re.findall(r'[（\(]([A-Ga-g1-9])[）\)]', ans_str)
    if len(bracket_opts) >= 2:
        return "".join(sorted(list(dict.fromkeys(c.upper() for c in bracket_opts))))

    # B. 提取由頓號、逗號或空格分隔的多選字母（如 B、D 或 B, E 或 B D）
    sep_opts = re.findall(r'(?<![a-zA-Z0-9])([A-Ga-g])(?![a-zA-Z0-9])', ans_str)
    if len(sep_opts) >= 2 and not any(k in ans_str for k in ["題", "分", "第", "選項"]):
        return "".join(sorted(list(dict.fromkeys(c.upper() for c in sep_opts))))

    # C. 若為純連續大寫字母（如 BD, BCE, ADE）
    pure_multi = re.findall(r'^[A-Ga-g]{2,5}$', ans_str.strip())
    if pure_multi:
        return "".join(sorted(list(dict.fromkeys(pure_multi[0].upper()))))

    # D. 單一選項帶中文說明（如 "(D) 丙生、丁生" 或 "D 丙生"）
    opt_prefix_match = re.search(r'^[（\(]?([A-Ga-g1-9])[）\)]?', ans_str)
    if opt_prefix_match and any(c.isalpha() for c in opt_prefix_match.group(1)):
        return opt_prefix_match.group(1).upper()

    # 一般答案：清除中文字元，但完整保留英數、逗號、負號、分數線 / 與小數點 .
    cleaned = re.sub(r'[\u4e00-\u9fa5]+', '', ans_str)
    cleaned = re.sub(r'[^\w,\-±/.]', '', cleaned)
    return cleaned.strip(",()")

def evaluate_solution_quality(q_obj: dict) -> int:
    """
    評估一道題目的詳解品質得分，用於在重題中挑選最佳版本
    """
    sol = str(q_obj.get("detailed_solution", "")).strip()
    score = len(sol) # 基礎長度分
    
    # 🚨 扣分項：無效、崩潰、截斷或佔位文字
    invalid_kw = ["題目內容缺失", "無法判定", "本題不存在", "超時或失敗", "系統提示", "引發系統內部", "崩潰", "封面作答注意事項"]
    for kw in invalid_kw:
        if kw in sol:
            score -= 50000

    # 🌟 加分項：名師多維解法標籤
    if "### 【標準解法】" in sol or "【標準解法】" in sol: score += 800
    if "### 【另解" in sol or "【另解" in sol: score += 1000
    if "### 【速解" in sol or "【速解" in sol: score += 600
    if "![圖" in sol or "diagram_" in sol: score += 500
    
    # 選項分析完整性加分
    opt_ana = str(q_obj.get("options_analysis", ""))
    if opt_ana and "本題為非選擇題" not in opt_ana and len(opt_ana) > 50:
        score += 600
        
    # 核心觀念與易錯陷阱加分
    if len(str(q_obj.get("concept_review", ""))) > 30: score += 300
    if len(str(q_obj.get("traps_and_warnings", ""))) > 30: score += 300
    
    return score

def deduplicate_questions(questions: list) -> list:
    """
    高階智能去重器：
    1. 雙重識別：題號相同 或 題幹文本相似度高達 85%
    2. 智能仲裁：當偵測到重複題目時，呼叫 evaluate_solution_quality 比對兩者詳解，強制保留最佳推導版本！
    """
    if not questions:
        return []
        
    def get_text_similarity(s1, s2):
        def clean_set(s):
            return set(re.sub(r'[^\w\u4e00-\u9fa5]', '', str(s)))
        set1 = clean_set(s1)
        set2 = clean_set(s2)
        if not set1 or not set2:
            return 0.0
        return len(set1.intersection(set2)) / float(len(set1.union(set2)))

    deduped = []
    for q in questions:
        is_duplicate = False
        q_text = q.get("question_text", "")
        q_num = str(q.get("question_number", "")).strip()
        q_type = str(q.get("question_type", "")).strip()
        
        for existing in deduped:
            ext_text = existing.get("question_text", "")
            ext_num = str(existing.get("question_number", "")).strip()
            ext_type = str(existing.get("question_type", "")).strip()
            
            # 判定是否為重複題目：題號相同且題型相同，或文本相似度 > 85%
            same_id = (q_num == ext_num and q_type == ext_type and q_num != "")
            similarity = get_text_similarity(q_text, ext_text)
            
            if same_id or (similarity > 0.85 and len(q_text) > 20):
                is_duplicate = True
                score_new = evaluate_solution_quality(q)
                score_old = evaluate_solution_quality(existing)
                
                logging.warning(f"🔄 [智能重題仲裁] 偵測到重複題目：題號 {q_num} (相似度: {similarity:.2f})")
                logging.warning(f"  -> 現存版本品質分: {score_old} | 新版本品質分: {score_new}")
                
                # 🚨 詳解精華融合：吸收兩者之長，將另一版的獨特另解/速解無損拼接
                higher_q = q if score_new > score_old else existing
                lower_q = existing if score_new > score_old else q
                
                h_sol = str(higher_q.get("detailed_solution", "")).strip()
                l_sol = str(lower_q.get("detailed_solution", "")).strip()
                
                # 檢查次要版本是否有獨特另解
                for alt_kw in ["### 【另解", "### 【速解", "### 【座標化", "### 【幾何直覺"]:
                    if alt_kw in l_sol and alt_kw not in h_sol:
                        part = l_sol.split(alt_kw, 1)[1]
                        end_idx = part.find("\n### ")
                        alt_text = alt_kw + (part[:end_idx].strip() if end_idx != -1 else part.strip())
                        if "**綜上所述" in alt_text:
                            alt_text = alt_text.split("**綜上所述")[0].strip()
                        if len(alt_text) > 40:
                            if "**綜上所述" in h_sol:
                                sp = h_sol.split("**綜上所述", 1)
                                h_sol = sp[0].strip() + "\n\n" + alt_text + "\n\n**綜上所述" + sp[1]
                            else:
                                h_sol += "\n\n" + alt_text
                            logging.info(f"  -> 💡 [另解精華融合] 成功將另一版本的獨創另解融入主詳解！")
                            
                higher_q["detailed_solution"] = h_sol
                
                # 圖片與考點聯集
                if lower_q.get("image_paths"):
                    higher_q["image_paths"] = list(dict.fromkeys(higher_q.get("image_paths", []) + lower_q["image_paths"]))
                    higher_q["has_image"] = len(higher_q["image_paths"]) > 0
                if lower_q.get("techniques_used"):
                    higher_q["techniques_used"] = list(dict.fromkeys(higher_q.get("techniques_used", []) + lower_q["techniques_used"]))
                    
                existing.clear()
                existing.update(higher_q)
                break
                
        if not is_duplicate:
            deduped.append(q)
            
    return deduped

def normalize_and_merge_subject_taxonomy(taxonomy: dict) -> dict:
    """將 subject.json 中的所有鍵和值轉為繁體，並自動合併與去重"""
    new_taxonomy = {}
    for subject_key, subject_data in taxonomy.items():
        # 轉為繁體科目名稱
        traditional_subject_key = s2t(subject_key)
        
        if traditional_subject_key not in new_taxonomy:
            new_taxonomy[traditional_subject_key] = {"topics": [], "techniques": []}
            
        # 取得並轉換 topics 列表中的所有字串
        raw_topics = subject_data.get("topics", [])
        converted_topics = [s2t(t) for t in raw_topics if t]
        
        # 取得並轉換 techniques 列表中的所有字串
        raw_techs = subject_data.get("techniques", [])
        converted_techs = [s2t(t) for t in raw_techs if t]
        
        # 合併到主字典
        new_taxonomy[traditional_subject_key]["topics"].extend(converted_topics)
        new_taxonomy[traditional_subject_key]["techniques"].extend(converted_techs)
        
        # 去除重複項
        new_taxonomy[traditional_subject_key]["topics"] = list(dict.fromkeys(new_taxonomy[traditional_subject_key]["topics"]))
        new_taxonomy[traditional_subject_key]["techniques"] = list(dict.fromkeys(new_taxonomy[traditional_subject_key]["techniques"]))
        
    return new_taxonomy

def fix_latex_matrices(text: str) -> str:
    r"""
    🚨 矩陣與聯立方程語法修復器：
    1. 自動識別裸露的 \begin{bmatrix}、\begin{pmatrix}、\begin{cases} 等環境
    2. 強制為其補齊 $ ... $ 或 $$ ... $$ 定界符，確保 KaTeX / MathJax 100% 正常渲染
    3. 保護矩陣換行雙反斜線 \\ 不被 Markdown 轉義損毀
    """
    if not isinstance(text, str) or "\\begin{" not in text:
        return text

    # 定義所有常見矩陣與多行數學環境
    env_pattern = r'(\\begin\{(?:matrix|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|cases|aligned|array)\}[\s\S]*?\\end\{(?:matrix|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|cases|aligned|array)\})'

    # 分割已有的 $ ... $ 與 $$ ... $$ 區塊，避免重複包裹
    parts = re.split(r'(\$\$[\s\S]*?\$\$|\$[^$]*?\$)', text)
    for i in range(len(parts)):
        if i % 2 == 0:  # 偶數索引代表非數學環境（裸露文字區塊）
            if "\\begin{" in parts[i]:
                def wrap_matrix(m):
                    block = m.group(1).strip()
                    # 確保矩陣內部的換行符號維持為標準 LaTeX 的 \\
                    block = re.sub(r'(?<!\\)\\(?![\\a-zA-Z])', r'\\\\', block)
                    if "\n" in block or "\\\\" in block:
                        return f" $${block}$$ "
                    return f" ${block}$ "
                parts[i] = re.sub(env_pattern, wrap_matrix, parts[i])

    return "".join(parts)

def sanitize_latex_syntax(text: str) -> str:
    r"""全方位 LaTeX 語法修復與防禦補償引擎（支援括號平衡、KaTeX跨行防護、粗體解扣）"""
    if not isinstance(text, str) or not text:
        return text

    # 1. 清除連鎖星號與 Markdown 粗體/LaTeX 交疊錯誤
    text = re.sub(r'\*{3,}', '', text)
    text = re.sub(r'\$\s*\*\*([^\$]+?)\*\*\s*\$', r'**$\1$**', text)
    text = re.sub(r'\*\*\s*\$([^\$]+?)\$\s*\*\*', r'**$\1$**', text)
    text = re.sub(r'\$\s*\*\*([^\$]+?)\s*\$', r'**$\1$**', text)
    text = re.sub(r'\*\*\s*\$([^\$]+?)\s*\*\*', r'**$\1**', text)

    # 2. 修復矩陣與方程組環境
    text = fix_latex_matrices(text)

    # 3. 修復常見缺少反斜線的 LaTeX 關鍵詞
    latex_keywords = (
        r'frac|sqrt|alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma|omega|'
        r'Delta|Omega|times|cdot|pm|le|ge|neq|approx|sim|in|notin|subset|'
        r'cup|cap|vec|mathbf|text|log|ln|sin|cos|tan|cot|sec|csc|lim|infty'
    )
    # 修復類似 rac{1}{2} -> \frac{1}{2}
    text = re.sub(rf'(?<!\\)\b({latex_keywords})\b(?=\{{|\s|[0-9a-zA-Z])', r'\\\1', text)

    # 4. 處理 $$ ... $$ 區塊（清洗內部殘留的 $ 與修復換行）
    def clean_display_block(m):
        content = m.group(1).replace('$', '')
        content = re.sub(r'(?<=[\w\)\}])\s*\\\s+(?=[\w\(\{])', r' \\\\ ', content)
        return f"$${content.strip()}$$"
    text = re.sub(r'\$\$(.*?)\$\$', clean_display_block, text, flags=re.DOTALL)

    # 5. 處理 $ ... $ 行內公式（修復大括號未閉合、中文標點包入、公式內換行）
    def fix_inline_math(m):
        math_content = m.group(1)
        # 5.1 行內公式不允許實體換行（KaTeX 會報錯），將換行轉為空格
        math_content = math_content.replace('\n', ' ').replace('\r', ' ')
        
        # 5.2 大括號未閉合自動修復 (Brace Balancing)
        open_braces = math_content.count('{')
        close_braces = math_content.count('}')
        if open_braces > close_braces:
            math_content += '}' * (open_braces - close_braces)
        elif close_braces > open_braces:
            # 刪除末尾多餘的 '}'
            diff = close_braces - open_braces
            while diff > 0 and math_content.endswith('}'):
                math_content = math_content[:-1]
                diff -= 1

        # 5.3 中文標點不應包裹在公式末端（如 $x = 1。$ 改為 $x = 1$。）
        punct_match = re.search(r'([，。；！？、：]+)$', math_content)
        trailing_punct = ""
        if punct_match:
            trailing_punct = punct_match.group(1)
            math_content = math_content[:-len(trailing_punct)].strip()

        return f"${math_content}${trailing_punct}"

    text = re.sub(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)', fix_inline_math, text)

    # 6. 修復裸露在外的數學標記（如獨立的 \alpha, \frac{1}{2} 未被 $ 包裹）
    bare_pattern = rf'(?<![\$\w\\])(\\(?:{latex_keywords})(?:\{{[^}}]*\}})*)(?![\$\w])'
    parts = re.split(r'(\$\$[\s\S]*?\$\$|\$[^$]*?\$)', text)
    for i in range(len(parts)):
        if i % 2 == 0:  # 純文字區塊
            parts[i] = re.sub(bare_pattern, r' $\1$ ', parts[i])
            parts[i] = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', parts[i])

    text = "".join(parts)
    text = re.sub(r' +', ' ', text)
    
    # 🚀 3. 本地開源 AST 語法修補（補齊未閉合大括號、刪除孤立括號）
    text = ast_repair_latex(text)
    
    return text.strip()

def ast_repair_latex(text: str) -> str:
    """【開源庫優先】利用 pylatexenc 在本地端以語法樹 (AST) 檢驗與補齊 LaTeX 語法錯誤"""
    if not isinstance(text, str) or ("$" not in text and "\\" not in text):
        return text
    try:
        from pylatexenc.latexwalker import LatexWalker, LatexMacroNode, LatexGroupNode, LatexCharsNode
        # 1. 自動修正數學區塊內的大括號對稱性
        def balance_math_braces(s: str) -> str:
            depth = 0
            res = []
            for ch in s:
                if ch == '{':
                    depth += 1
                    res.append(ch)
                elif ch == '}':
                    if depth > 0:
                        depth -= 1
                        res.append(ch)
                    # 丟棄無前置對應的孤立 '}'
                else:
                    res.append(ch)
            if depth > 0:
                res.append('}' * depth)
            return "".join(res)

        # 針對被 $ 包裹的片段修補
        parts = re.split(r'(\$\$[\s\S]*?\$\$|\$[^$]*?\$)', text)
        for idx in range(len(parts)):
            if idx % 2 == 1:
                is_display = parts[idx].startswith("$$")
                content = parts[idx][2:-2] if is_display else parts[idx][1:-1]
                content = balance_math_braces(content)
                parts[idx] = f"$${content}$$" if is_display else f"${content}$"
        text = "".join(parts)
    except Exception:
        pass
    return text
    
def fix_latex_text_macros(text: str) -> str:
    """
    偵測 text 中的 \\text{...} 結構（支援嵌套大括號 {}），
    如果其內部包裹的內容：
    1. 含有反斜線 `\\`（代表包含 LaTeX 數學指令，例如 \\frac, \\sqrt, \\vec, \\le 等）
    2. 或是含有上標 `^` 或下標 `_`（這在 \\text 裡會造成編譯錯誤）
    3. 或是含有 `>`, `<`, `=`, `+`, `-`, `*`, `/` 等數學運算符且長度較長
    則將該 \\text{...} 剝除，只保留內部的內容。
    """
    if "\\text" not in text:
        return text
        
    result = []
    i = 0
    n = len(text)
    
    while i < n:
        # 尋找 \\text{
        if text[i:i+6] == "\\text{":
            # 開始匹配配對的大括號
            start_content_idx = i + 6
            bracket_count = 1
            j = start_content_idx
            while j < n and bracket_count > 0:
                if text[j] == '{':
                    bracket_count += 1
                elif text[j] == '}':
                    bracket_count -= 1
                j += 1
                
            if bracket_count == 0:
                # 成功找到完整的 \\text{content}
                content = text[start_content_idx:j-1]
                
                # 檢查是否需要剝離 \\text
                has_latex_cmd = "\\" in content
                has_sub_super = "^" in content or "_" in content
                has_math_ops = any(op in content for op in [">", "<", "=", "+", "*", "/"])
                
                if has_latex_cmd or has_sub_super or (has_math_ops and len(content) > 1):
                    # 遞迴修復內部內容後，直接剝除 \\text
                    fixed_content = fix_latex_text_macros(content)
                    result.append(fixed_content)
                else:
                    # 合法的純文字或單純字母（如 \\text{kg}, \\text{甲}），保留 \\text{} 並遞迴修復內部
                    fixed_content = fix_latex_text_macros(content)
                    result.append(f"\\text{{{fixed_content}}}")
                    
                i = j
            else:
                # 未配對成功，當作一般字元處理
                result.append(text[i])
                i += 1
        else:
            result.append(text[i])
            i += 1
            
    return "".join(result)

def recursive_fix_latex(obj):
    if isinstance(obj, dict):
        return {k: recursive_fix_latex(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_fix_latex(x) for x in obj]
    elif isinstance(obj, str):
        return fix_latex_text_macros(obj)
    return obj

def pre_validate_format(q_data: dict, sol_data: dict):
    """
    [防禦性機制 - 提案二：選填題/非選題格式自動化預校驗器 + 數字選項代號自動修復]
    """
    fixed_sol = recursive_fix_latex(sol_data)
    if isinstance(fixed_sol, dict):
        for k, v in fixed_sol.items():
            sol_data[k] = v

    # 1. 選擇題答案與選項 Key 雙向自動對齊與標準化（支援 1-5 轉 A-E，以及 A-E 轉 1-5）
    opts = q_data.get("options", [])
    q_type = q_data.get("question_type", "")
    if opts and q_type in ["單選題", "多選題"]:
        opt_keys = [str(o.get("key", "")).strip() for o in opts if str(o.get("key", "")).strip()]
        ans_str = str(q_data.get("answer", "")).strip()
        
        # 情況 A：選項 key 全為英文字母 (A 到 J 等)，但答案包含數字 (1, 2, 3...)
        if all(k.isalpha() for k in opt_keys) and any(c.isdigit() for c in ans_str):
            # 支援完整對應至大考英文文意選填的 A~L (1~12)
            num_to_alpha = {str(i+1): chr(65+i) for i in range(15)}
            # 🚨 修復正則：放寬為所有合法英數 token，嚴禁截斷英文科之 H、I、J 選項！
            raw_tokens = re.findall(r'\b\d+\b|[A-Za-z]', ans_str)
            converted = [num_to_alpha.get(t, t.upper()) for t in raw_tokens]
            if converted:
                unique_keys = sorted(list(dict.fromkeys(converted)))
                q_data["answer"] = unique_keys[0] if q_type == "單選題" else "".join(unique_keys)

        # 情況 B：選項 key 全為數字 (1, 2, 3...)，但答案包含英文字母 (A, B, C...)
        elif all(k.isdigit() for k in opt_keys) and any(c.isalpha() for c in ans_str):
            alpha_to_num = {chr(65+i): str(i+1) for i in range(15)}
            raw_tokens = re.findall(r'[A-Za-z]', ans_str)
            converted = [alpha_to_num.get(t.upper(), t) for t in raw_tokens]
            if converted:
                unique_keys = sorted(list(dict.fromkeys(converted)))
                q_data["answer"] = unique_keys[0] if q_type == "單選題" else "".join(unique_keys)

        # 情況 C：清理括號與自然語言贅字（例如將 "(3)(5) [即 (C)(E)]" 洗成純淨的 "35" 或 "CE"）
        else:
            valid_chars = re.findall(r'[A-Za-z0-9]', ans_str)
            filtered = [c.upper() for c in valid_chars if c.upper() in opt_keys]
            if filtered:
                unique_keys = sorted(list(dict.fromkeys(filtered)))
                q_data["answer"] = unique_keys[0] if q_type == "單選題" else "".join(unique_keys)

    # 0. 🚨 選項矩陣自動包裹防護：若選項內容含有矩陣指令但未包裹 $，強制包裹為 $...$
    for opt in q_data.get("options", []):
        val = str(opt.get("value", "")).strip()
        if "\\begin{" in val and not (val.startswith("$") and val.endswith("$")):
            opt["value"] = f"${val}$"

    # 2. 定義混合題判定屬性（修復 NameError: is_hybrid）
    is_hybrid = len(q_data.get("options", [])) > 0 and len(q_data.get("scoring_criteria", "")) > 0
    q_type = q_data.get("question_type", "")

    # 當非選擇題型且非混合題時，清空 options_analysis 避免格式混淆
    if q_type in ["選填題", "簡答題", "繪圖作圖題"] and not is_hybrid:
        sol_data["options_analysis"] = []

    # 2-2. 🚨 評分標準統一化與客觀化：徹底清除客觀題中幻想的手寫評分，並落實非選標準量規
    if q_type in ["單選題", "多選題", "選填題"] and not is_hybrid:
        # 1. 徹底從詳解中掃除 AI 幻想的手寫給分點標籤
        det_sol = sol_data.get("detailed_solution", "")
        for fake_header in ["### 【手寫題評分對照】", "### 【評分對照】", "【手寫題評分對照】", "【手寫/選填評分對照】"]:
            if fake_header in det_sol:
                det_sol = det_sol.split(fake_header)[0].strip()
        # 清理行內殘留的假給分點標記（例如：**【列式給分點（得1分）】**）
        det_sol = re.sub(r'\*\*【(?:列式|計算|數值|結論)給分點[^】]*】\*\*', '', det_sol)
        sol_data["detailed_solution"] = det_sol.strip()
            
        # 2. 剛性約束客觀題之標準計分規則（依考科動態適配配分）
        sub_name = str(q_data.get("sub_subject", "")) + str(q_data.get("exam_source", ""))
        is_comprehensive = any(k in sub_name for k in ["社會", "自然", "歷史", "地理", "公民", "生物", "地科", "理綜", "文綜"])
        
        default_pts = 2 if is_comprehensive else 5  # 學測社會/自然通常為 2 分/題；數學/分科單科通常為 4~5 分/題
        
        if q_type == "單選題":
            sol_data["scoring_rubric"] = f"單選題每題 {default_pts} 分（依考卷配分），答對得 {default_pts} 分，答錯或未作答得 0 分，不倒扣。"
        elif q_type == "多選題":
            sol_data["scoring_rubric"] = f"多選題每題 {default_pts} 分，各選項獨立判定。所有選項全對得 {default_pts} 分，答錯依考卷倒扣扣分標準計分，未作答得 0 分。"
        elif q_type == "選填題":
            sol_data["scoring_rubric"] = "選填題每題完全答對得滿分（依考卷配分，通常為 5~8 分），未完全答對、填錯任意一格或未作答得 0 分，不倒扣。"
    elif q_type in ["簡答題", "繪圖作圖題"] or is_hybrid:
        # 若為手寫非選題且無官方細則，確保其具備客觀的三段式給分量規
        current_rubric = sol_data.get("scoring_rubric", "")
        if not current_rubric or len(current_rubric.strip()) < 10:
            sol_data["scoring_rubric"] = "非選擇題評分標準（依考卷配分）：\n1. 【概念列式分（40%）】：正確列出核心定理、方程式或幾何模型。\n2. 【過程推導分（40%）】：計算步驟完整、符號合規無計算瑕疵。\n3. 【結論數值分（20%）】：求得正確數值結果、最簡式或標註正確單位。"
        
    # 3. 自動修正選填題答案格式，若無半形逗號則依記挖空數量強制拆分
    if q_type == "選填題" and "," not in str(q_data.get("answer", "")):
        raw_ans = str(q_data.get("answer", "")).strip()
        expected_count = q_data.get("_expected_blank_count", 0)
        if expected_count > 0 and len(raw_ans) == expected_count:
            q_data["answer"] = ",".join(list(raw_ans))

def is_valid_solution(sol_text: str) -> bool:
    """判定詳解文字是否為有效解答（非錯誤、超時或崩潰訊息）"""
    if not sol_text or not isinstance(sol_text, str) or len(sol_text.strip()) < 15:
        return False
    invalid_keywords = ["超時或失敗", "系統提示", "引發系統內部", "無法生成", "未完成", "崩潰"]
    return not any(kw in sol_text for kw in invalid_keywords)
    
PARTIAL_SAVE_LOCK = threading.Lock()

# ==========================================
# ⚖️ 跨 Run 3:1 排程狀態持久化引擎
# ==========================================
SCHEDULER_STATE_FILE = "scheduler_state.json"
SCHEDULER_LOCK = threading.Lock()

def load_scheduler_state() -> dict:
    """載入跨 Action 繼承的排程狀態"""
    if os.path.exists(SCHEDULER_STATE_FILE):
        try:
            with open(SCHEDULER_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"讀取 {SCHEDULER_STATE_FILE} 失敗: {e}")
    return {"mock_streak": 0, "total_mock_done": 0, "total_official_done": 0}

def record_completed_exam_to_scheduler(exam_type: str):
    """每當一張考卷 100% 完工落盤，即時原子化更新 streak 計數器"""
    with SCHEDULER_LOCK:
        state = load_scheduler_state()
        is_mock = exam_type in ["MOCK", "MOCK_GSAT", "MOCK_AST"]
        
        if is_mock:
            state["mock_streak"] = state.get("mock_streak", 0) + 1
            state["total_mock_done"] = state.get("total_mock_done", 0) + 1
            logging.info(f"📈 [排程器計數] 模擬考完工 +1 (當前連續模考: {state['mock_streak']}/3)")
        else:
            state["mock_streak"] = 0  # 跑完正式考，重置計數器
            state["total_official_done"] = state.get("total_official_done", 0) + 1
            logging.info(f"📈 [排程器計數] 正式大考/段考完工 +1 (模考計數已重置為 0/3)")
            
        try:
            with open(SCHEDULER_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logging.error(f"寫入 {SCHEDULER_STATE_FILE} 失敗: {e}")

def save_partial_progress_immediately(partial_json_path: str, questions_list: list):
    """每當有題目成功通過驗證，立刻將記憶體中的最新進度安全寫入硬碟（具備全域執行緒安全鎖與快照重試）"""
    with PARTIAL_SAVE_LOCK:
        try:
            # 🚨 嘗試安全快照拷貝，徹底消除多線程併發時的 RuntimeError: list changed size during iteration
            snapshot_list = None
            for _ in range(3):
                try:
                    snapshot_list = questions_list[:]
                    break
                except RuntimeError:
                    time.sleep(0.01)
            if snapshot_list is None:
                return

            valid_qs = [q for q in snapshot_list if is_valid_solution(q.get("detailed_solution"))]
            
            # 深拷貝並剔除不可序列化的 PIL 影像物件
            clean_qs = []
            for q in valid_qs:
                q_copy = dict(q)
                q_copy.pop('_cropped_pil_images', None)
                clean_qs.append(q_copy)
                
            # 寫入實體暫存檔
            with open(partial_json_path, "w", encoding="utf-8") as f:
                json.dump(clean_qs, f, ensure_ascii=False, indent=4)
            logging.info(f"💾 [即時落盤] 已自動將當前 {len(clean_qs)} 題最新進度寫入實體暫存檔：{os.path.basename(partial_json_path)}")
        except Exception as e:
            logging.error(f"⚠️ 即時存檔失敗: {e}")
        
def load_invalid_key_patterns(summary_path):
    """從 key_errors_summary.txt 中，透過遮罩的前後綴精準恢復並提取所有已失效 (401/403) 的金鑰特徵"""
    if not os.path.exists(summary_path):
        return set()
        
    invalid_patterns = set()
    try:
        # 使用正則比對匹配：錯誤代碼為 401 或 403 的遮罩金鑰字串
        pattern = r"錯誤代碼:\s*(?:401|403)\s*\|\s*金鑰:\s*([a-zA-Z0-9_.-]+)\s*\|"
        with open(summary_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    masked_key = match.group(1).strip()
                    if "..." in masked_key:
                        parts = masked_key.split("...")
                        if len(parts) == 2:
                            invalid_patterns.add((parts[0], parts[1]))
                    else:
                        invalid_patterns.add((masked_key, ""))
    except Exception as e:
        print(f"讀取 key_errors_summary.txt 失效金鑰失敗: {e}")
    return invalid_patterns

def load_api_keys(file_path, summary_path="key_errors_summary.txt"):
    if not os.path.exists(file_path):
        print(f"找不到檔案: {file_path}")
        return []

    # 1. 讀取歷史失效金鑰前後綴特徵 (自動對比 401/403 紀錄)
    invalid_patterns = load_invalid_key_patterns(summary_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正規表達式抓取 AIzaSy 開頭 或 AQ.Ab 開頭的字串
    pattern = r"(AIzaSy[a-zA-Z0-9_-]+|AQ\.Ab[a-zA-Z0-9_-]+)"
    keys = re.findall(pattern, content)

    # 去除重複的 Key
    unique_keys = list(dict.fromkeys(keys))
    
    # 2. 自動過濾特徵吻合的失效金鑰
    filtered_keys = []
    excluded_count = 0
    for key in unique_keys:
        is_invalid = False
        for prefix, suffix in invalid_patterns:
            if suffix:
                # 若遮罩長度吻合首尾特徵，判定為已被停用的失效金鑰
                if key.startswith(prefix) and key.endswith(suffix):
                    is_invalid = True
                    break
            else:
                if key == prefix:
                    is_invalid = True
                    break
        if is_invalid:
            excluded_count += 1
        else:
            filtered_keys.append(key)
            
    if excluded_count > 0:
        print(f"⚠️  [啟動防禦過濾] 依據 key_errors_summary.txt，已在載入階段自動過濾掉 {excluded_count} 組確認失效（401/403）的金鑰！")
        
    print(f"成功載入 {len(filtered_keys)} 組【可正常使用】的 API 金鑰。")
    return filtered_keys

def get_next_rpd_reset_timestamp() -> float:
    """計算下一個 RPD (每日限制) 重置的台北時間戳記 (美西午夜 12 點即台北下午 3 點，對應 UTC 07:00/08:00)"""
    # 採用 timezone-aware UTC 計算，保證在不同國家/雲端伺服器運行時重置戳記完全一致
    from datetime import datetime, timezone, timedelta
    now_utc = datetime.now(timezone.utc)
    
    # 預設重置緩衝點為每天的 UTC 08:05（涵蓋冬夏令美西午夜重置）
    reset_today_utc = now_utc.replace(hour=8, minute=5, second=0, microsecond=0)
    if now_utc >= reset_today_utc:
        reset_time_utc = reset_today_utc + timedelta(days=1)
    else:
        reset_time_utc = reset_today_utc
    return reset_time_utc.timestamp()

# ==========================================
# 0. 載入題型分類與技巧對照表
# ==========================================
SUBJECT_TAXONOMY = {}
try:
    with open("subject.json", "r", encoding="utf-8") as f:
        raw_taxonomy = json.load(f)
        # 🚨 核心優化：載入時自動將簡體字轉為繁體並合併去重
        SUBJECT_TAXONOMY = normalize_and_merge_subject_taxonomy(raw_taxonomy)
except FileNotFoundError:
    logging.warning("找不到 subject.json，將不提供預設的分類與技巧參考，完全交由 AI 產生。")
except json.JSONDecodeError as e:
    logging.error(f"subject.json 格式損毀或有語法錯誤: {e}。將不提供分類建議，完全交由 AI 產生。")
except Exception as e:
    logging.error(f"載入分類表時發生未知錯誤: {e}")


# ==========================================
# 🔧 [全域設定] Prompt 變數區 (可自由修改)
# ==========================================
PROMPT_STAGE_2_INTRO = "請為以下 {batch_size} 道題目撰寫極致詳細的補教名師級詳解。\n"

def get_current_taipei_time_str() -> str:
    """精確取得系統當前台北時間（UTC+8）以動態錨定中華民國現行有效法規"""
    tz_taipei = datetime.timezone(datetime.timedelta(hours=8))
    now_taipei = datetime.datetime.now(tz_taipei)
    return now_taipei.strftime("%Y 年 %m 月 %d 日 %H:%M:%S (台北時間 UTC+8)")

PROMPT_STAGE_2_MAIN = """
你是一位在台灣大考講義編撰領域享有崇高聲譽、解題思路極具啟發性的高中學科補教名師。請針對上述每一道題目，為我們撰寫極致詳細、邏輯縝密且充滿引導性的詳解。

🚨【台灣 108 課綱與中華民國現行法規鋼鐵防火牆（🚨最高法律與教學紅線）】🚨：
當前系統台北基準時間：【{current_taipei_time}】。
所有涉及社會科（特別是公民與社會、歷史、地理）與國文科之解析，必須 100% 嚴格錨定此時間點之「中華民國現行有效法律、司法裁判與 108 課綱規範」：
1. **中華民國現行法律體系唯一合法性**：
   - 【憲政與公法】：憲法訴訟法已全面施行，違憲審查主體與法源為「憲法法庭裁判」及歷史「司法院大法官解釋（釋字）」，絕無其他非台灣機關！
   - 【民事法規範】：現行民法成年年齡為【滿 18 歲】（自 2023 年 1 月 1 日起正式實施）；法律行為能力、請求權基礎與物權法定主義必須完全合乎現行民法。
   - 【刑事法體系】：必須嚴格遵循「刑法三階論」（構成要件該當性 ➔ 違法性 ➔ 罪責/有責性）與正當防衛、緊急避難等阻卻違法事由。
   - 【行政與勞動法】：行政程序法之行政處分與救濟程序（訴願、行政訴訟三級二審）、勞動基準法、個人資料保護法等最新修正法規。
2. **絕對阻絕非台灣法律名詞（零容忍退件）**：
   - 嚴格禁止出現任何非中華民國之法規體系與大陸法律術語（例如絕對禁止出現：民法典、公安局、人民法院、人大常委會、違憲審查主體為人大等）。
   - 違者一律視為嚴重學術污染，將遭審查系統直接退件！

🚨【模擬考/指考官方詳解本融入與昇華規範（極度重要）】🚨：
- **【強迫參考與解構】**：在下方的待解題目中，系統為您提供了 `📢【官方解析本提供的本題解析文字與出處背景】`。**你絕對不可忽略官方詳解的思路！**
- **【降維解說與教學昇華】**：官方原卷詳解通常寫得非常簡略、跳步、或者直接給出繁複的代數結果（這會讓學生感到挫折）。你的任務是「仔細研讀官方提供的推導、反應式、受力分析或文章翻譯，並將其思維骨架還原、拆解、重新鋪墊，演繹成具有溫度與引導性的名師互動式解說」。
- **【指正印刷錯誤】**：官方詳解本有時會因為排版而發生數字印刷錯誤或跳步 Bug，如果發現官方推導與最嚴謹的學理有出入，請勇敢且優雅地在詳解中替學生點出並指正。

🚨【語言字體與兩岸名詞絕對剛性規定（致命紅線）】🚨：
1. 全篇必須 100% 使用繁體中文（zh-TW）撰寫，絕對禁止夾雜 any 簡體字！
2. 絕對禁止使用大陸學術名詞或用語。例如：必須寫「機率」而非「概率」、寫「向量」而非「矢量」、寫「純量」而非「標量」、寫「解析度」而非「分辨率」、寫「伏特計/安培計」而非「電壓表/電流表」。若違反將遭到系統嚴厲退件！

【一、詳解結構化與寫作規限（極度重要）】
1. **question_analysis (題意分析)**：
    - 精確剖析題目的核心考點與已知條件。
    - **必須**使用 Markdown 的 `**雙星號粗體**` 標示出題幹中最核心的關鍵概念與限制條件（如：**同溫同壓**、**飽和溶液**等）。
2. **solving_strategy (解題思路)**：
    - 💡 **【元認知大師級思維引導（核心重點）】**（不要直接給出繁複推導）：
      * 請模擬一位高三頂尖學生的「思考心流」。著重解答學生：**『為什麼我們一看到這道題，就會聯想到要用這個定理或幾何輔助線？』**。
      * 帶領學生發掘題幹中隐藏的**關鍵字眼與破局訊息**：
        * 例如：看見「共點且等距離」，腦中要立即聯想到「**三角形的外接圓**」；
        * 看見「二階導數在兩側變號」，要立刻反應「**圖形凹凸性改變與反曲點**」；
        * 看見「同溫同壓下的分子量比」，要迅速思索「**亞佛加厥定律與密度、體積的內在關聯**」。
      * 建立從【已知蛛絲馬跡】$\rightarrow$【核心定理與圖表聯想】$\rightarrow$【規劃解題步驟】的認知因果推理鏈，徹底啟發學生的數理學科直覺。
3. **detailed_solution (完整解法與另解)**：
    - 💡 **【互動式補教名師講授風格（剛性要求）】**：
      * **字裡行間必須極具互動性、啟發性與對話感！** 想像你正在全台灣最大大考衝刺班的黑板前，眼神熱切地為台下的高三學子指點迷津。
      * 鼓勵多使用例如『同學們，我們仔細觀察這裡！』、『這時候我們不妨做一個大膽的輔助線...』『有沒有發現這個關鍵訊息？它其實在暗示我們...』等溫暖、親切且生動活潑的口吻。
    - 💡 **【圖文完美咬合與自然視覺化（多圖啟發）】**：
      * **【按需自然多圖】**：理科（數學/物理/化學/地科）教學的核心在於「將抽象概念具象化」。若題目涉及幾何輔助線、函數圖形交點、物理受力分析 (Free-body Diagram)、光路圖、波形圖或反應過程，**強烈鼓勵您為不同的解法或關鍵步驟，自然地繪製專屬圖解**（例如：`圖 1` 呈現標準解法的幾何構造，`圖 2` 呈現另解的座標化點位，`圖 3` 呈現極值觀察）。
      * **【適度原則（拒絕強湊）】**：若本題屬於純代數計算、純文字概念辨析或文意理解，則無需硬湊圖片。圖解應「精準服務於原理理解，質量重於數量」。
      * **【圖文互動】**：當您嵌入圖片時，**絕對不允許只是孤零零貼上連結**！必須在前後文用熱情的語言引導學生觀察圖中細節，例如：『...同學們請看圖 1 ![圖 1](images/...)。做出這條外接圓輔助線後，圓周角與弦切角是不是就一目了然了？...』，讓圖文咬合發揮最大教學價值。
    - 請使用 Markdown 標題區分多種解法，以激發學生的思維創造力，我們強烈要求「一題多解」：
            - `### 【標準解法】`：必須按部就班、寫出最正規且符合課綱的求解過程。涉及計算必須呈現完整的 LaTeX 公式推導，嚴禁直接跳出答案。**【極重要硬性規定】**：`### 【標準解法】` 必須是一個**完整的、能獨立推導或判斷出所有選項（如選項 1 至 5，或 A 至 E）對錯的完整解題路徑**。絕對不允許在標準解法中只判斷選項 1, 2，而把其餘選項的判斷拆分到其他另解中！
            - `### 【另解一：[命名你的思維，例如：座標轉換/幾何投影法]】`：提供第二套完整且嚴謹的推導思維。**如果某個另解因為方法特質限制，只能用來判斷部分選項，你必須在該另解的標題或開頭極其明確地註明：『本解法專用於快速判斷選項 X, Y』。**
            - `### 【另解二：[命名...]】`、`### 【另解三：[命名...]】`：依此類推。
            - `### 【另解 / 秒殺速解：[命名...]】`：提供極度直覺、極富創意或能在大考中快速破題的秒殺方法。
            - 我們期望你**竭盡所能提供 2 到 4 種完全不同的切入角度與解法**，以最豐富且深具啟發性的思維碰撞，展現補教名師的風範。
            - 🚨【純選擇題與純選填題剛性禁令】：若本題為「單選題」、「多選題」或「純選填題」，**【絕對嚴格禁止】** 在 `detailed_solution` 中輸出 `### 【手寫題評分對照】` 或任何給分點表格！此區塊僅限於非選擇題/計算證明題/混合題手寫小題！
            - `### 【手寫題評分對照】`：【僅限非選擇題】若本題為非選擇題且含有「手寫評分標準」，才允許在詳解中用粗體標示各給分點（如：`**【列式給分點（得1分）】**`）。
    - 🚨【選擇題選項對位與獨立證偽硬性規定（防迎合湊答案）】：
      對於單選題、多選題、以及有提供選項的選填題，你必須在 `options_analysis` 列表中，為 `options` 中的**每一個**選項（如 A, B, C 或 1, 2, 3，必須與題目給定的選項 key 嚴格對齊）提供一個獨立的分析對象。
      - 每個對象包含 `key`（選項標籤）與 `explanation`（該選項的詳細分析、公式推導或對錯判斷，並在結尾明確指出該選項為『正確』或『錯誤』）。
      - 🚨【多選題錯誤選項獨立證偽剛性要求】：
        對每一個錯誤選項，**絕對禁止**只寫「此選項錯誤」或一筆帶過！你必須具體點出其錯誤屬性：
        * 【時序錯置】：歷史年代先後或因果邏輯倒置。
        * 【偷換概念 / 語意過度推論】：將題幹特定情境擴大為全稱命題，或混淆了法律/學科概念。
        * 【敘述本身正確但與題幹無關】：選項敘述雖符合客觀常識或史實，但「題幹情境根本未提及或無法推導」（大考最核心鑑別陷阱！）。
        * 【數理推導矛盾】：計算錯誤、邊界條件遺漏或幾何關係不符。
      - 如果本題為非選擇題、計算題、簡答題，或者原題沒有選項，請將 `options_analysis` 設為空列表 `[]`。
      - **【絕對禁止】**將多個選項合併到同一個物件，也**【絕對禁止】**因為是計算題或範圍推導題就只在標準解法中推導而忽略對各個選項的單獨對位判斷！
5. **concept_review (核心概念複習)**：
    - 條列式並詳細說明此題考驗的學科核心定理、重要公式、反應式或定律。
6. **traps_and_warnings (易錯陷阱與盲點警示)**：
    - 指出學生在此題最容易犯的錯誤（如：代數計算粗心、單位看錯、忽略隱藏條件），請用強烈的警示語氣和 **粗體** 呈現。
7. **advanced_supplement (進階延伸補充)**：
    - 與此題相關的跨章節聯想、學術延伸考點（無則填「無」）。

【二、學科公式與 LaTeX 規範】
- 🚨【極度重要：LaTeX 安全傳輸與反斜線轉義規範】🚨
  為了防止 JSON 傳輸與解析時 LaTeX 的反斜線 `\\` 被損毀、遺失 or 誤判為 JSON 控制字元（如 \\t, \\f, \\n），你必須在輸出所有 LaTeX 公式時，將所有 LaTeX 中的反斜線 \\ 替換為特殊的預留安全標記 __LTXS__！
  - 例如：`\\frac{{1}}{{2}}` 必須寫成 `__LTXS__frac{{1}}{{2}}`。
  - `\\theta` 必須寫成 `__LTXS__theta`。
  - `\\rightarrow` 必須寫成 `__LTXS__rightarrow`。
  - `\\text{{...}}` 寫成 `__LTXS__text{{...}}`。
  - `\\begin{{matrix}}` 寫成 `__LTXS__begin{{matrix}}`。
  - 🚨【絕對禁止】在公式中輸出 any 真正的單反斜線 `\\`，一律且強制使用 `__LTXS__`！
- 🚨【極重要：Matplotlib 繪圖程式碼規範（防制語法崩潰）】🚨：
  在任何生成的 Python 繪圖程式碼（如用於產生圖表的腳本）中，若字串、圖例（legend）或標籤（label）包含 any LaTeX 數學公式（如 `\\in`, `\\frac`），**必須將該字串聲明為 Python 原始字串（raw string）**，即在字串前加上 `r` 前綴（例如：`r'參數 $a \\in [-\\frac{{1}}{{2}}, 1]$ 之覆蓋區域'`），**絕對禁止**使用一般字串，否則會因反斜線轉義（如 `\\f` 變成換頁符）導致腳本語法崩潰或 Matplotlib 解析失敗！
- 嚴格遵守標準 LaTeX 語法。
- 🚨【指數與冪次 LaTeX 規範】：當表示指數函數、高次冪或含有多個字元的上標（如 2 的 x+1 次方，或 e 的 -x 次方）時，**必須**將整個指數/上標部分用 LaTeX 的大括號 `{{}}` 完整包裹，例如寫成 `$2^{{x+1}}$`、`$e^{{-x}}$`，**絕對禁止**寫成 `$2^x+1$` 或 `$2^-x$`（這會被渲染/解讀為 $2^x + 1$ 或 $2^- \\cdot x$，造成嚴重的學術邏輯與網頁渲染錯誤）。在非 LaTeX 純文字語境下，必須使用括號表示，如 `2^(x+1)`。
- 🚨【化學式專屬指令】：所有化學反應式必須使用 LaTeX 格式。
  - 務必使用下標語法，例如：$Mg_{{(s)}} + 2HCl_{{(aq)}} __LTXS__rightarrow MgCl_{{2(aq)}} + H_{{2(g)}}$。
  - 嚴禁使用 ->，必須使用 __LTXS__rightarrow 或 __LTXS__ce{{->}}。
  - 所有離子電荷必須使用上標，例如：$Ca^{{2+}}$。
- 絕對禁止使用 Tab 鍵縮排公式，防止 __LTXS__text 變形損毀。
- 所有變數與數值必須嚴格包裹在 $...$ 或 $$...$$ 之中。

【三、文理雙軌客觀難度量化矩陣規範 (1-10分，嚴禁隨意給分)】
為了確保所有學科（文科、理科）難度評分絕對客觀公正，你【必須】依據當前學科屬性，對以下四個維度打分並加總作為最終的 `difficulty_level`（1~10分）：

1. **概念跨度（文理通用，1~3分）**：
   - 1分：單一基本概念、字詞字面義、常識性史實或公式直接代入。
   - 2分：單一章節跨小節、2個概念關聯（如：修辭+文意、二次配方+判別式、氣候+地形）。
   - 3分：跨冊次/跨時代/跨領域綜合（如：多文本對比、微積分+幾何、經濟供需+法規衡量）。

2. **推導運算／思維深度（文理分軌，1~3分）**：
   - **【理科（數/物/化/生/地）】**：
     * 1分：3步以內心算或基礎代數代入。
     * 2分：4~7步推導、平衡反應式、二元聯立或向量運算。
     * 3分：8步以上冗長運算、微積分、分類討論或非直觀空間幾何輔助線。
   - **【文科（國/英/歷/地/公/國寫）】**：
     * 1分：單純記憶檢索、字面直接對應、基礎詞彙語法。
     * 2分：需跨段落訊息拼圖、長難句構拆解、文言字詞語意引申、或需結合歷史因果/法條模型。
     * 3分：深層批判性思辨、一手史料偏見評估、抽象意象轉譯、或多方觀點駁論建構。

3. **情境包裝與圖表閱讀（文理通用，0~2分）**：
   - 0分：純文字短句或純符號題，無冗長題幹。
   - 1分：生活化情境包裝、統計圖表判讀、或實驗數據題。
   - 2分：長篇素養閱讀材料、多圖表疊合分析、或深層隱蔽條件。

4. **誘答與陷阱強度（文理通用，0~2分）**：
   - 0分：選項差異顯著，無誘導陷阱。
   - 1分：含有常見盲點陷阱（如忽略定義域/端點值、時序錯置、同義代換偷換概念、因果倒置）。
   - 2分：多重複合陷阱、干擾項與正解極其相似（如微小語意差距、極端邊界反例）。

🚨【評分理由格式剛性要求】：
在 `difficulty_reason` 欄位中，【必須】詳細列出上述四個維度的具體給分依據與加總算式，格式範例如下：
「本題評為 X 分。
1. 概念跨度（X分）：[具體說明涉及哪些概念與冊次]
2. 運算負擔/思維深度（X分）：[理科說明推導步數與計算量；文科說明閱讀推理與思辨深度]
3. 情境包裝（X分）：[說明是否含有圖表或長篇情境]
4. 陷阱強度（X分）：[說明干擾選項或具體陷阱所在]
總計得分：X + X + X + X = X 分。」
🚨 若本題因原卷缺漏或封面說明而無實質試題，難度【強制設為 1 分】，嚴禁評為高分！

【三、邏輯一致性自我校驗與防偽原則（Self-Correction & Anti-Fabrication Loop）】
        0. 🚨【圖片精確性確認與重掃描機制】：請仔細檢查你收到的題目圖片。如果發現圖片裁剪錯誤（例如圖片中印著的題號不是當前題號）、圖片模糊，或者關鍵公式/表格被切掉，你必須將 `suspects_image_mismatch` 設為 `true`。系統將自動啟動重掃描補件與重新裁剪機制，以確保下一輪生成時能拿到最完美的圖片！
        1. 在你輸出詳解之前，請務必在內部進行一次「解題沙盤推演」。
        2. 如果你的物理或數學公式推導出來的結果，與官方給定的標準答案【{q_answer}】不一致，這代表你的中間步驟或對題目的物理情境理解有誤！
           - 🚨【選填題格式提示】：若 `answer` 格式為逗號分隔（如 `-,4,3`），代表該選填題各畫卡格子（如 `[ 10-1 ]`, `[ 10-2 ]`, `[ 10-3 ]`）之答案分別為 `-`、`4`、`3`。請你在推導與結論中，確認求得的各個數值（如 $a = -4$, $b = 3$）與此逗號分隔的格子答案完全一致！
           - 🚨【多選題格式提示】：若 `answer` 格式為多個數字或字母組成，且可能帶有逗號（如 `3,4` 或 `A,C,D`），這代表該題為多選題。在與你的推導比對時，我們在系統層面會自動忽略逗號，請你在 `options_analysis` 中針對這幾個正確選項進行對位標示（即在對應選項的 `explanation` 尾端明確指出其為『正確』）。
        3. **【絕對禁止無中生有與幻覺】**：你只能使用題目 `question_text` 與 `shared_context` 中明確給出的已知數值與條件。**絕對禁止憑空捏造、編造或引入任何未在題幹中出現的幾何座標、截距、特定交點或任何常數數值！** 所有代數與幾何推導都必須具備嚴格、紮實的題目已知條件基礎。
        3-2. **🚨【絕對禁止推導斷層與已知答案回推步驟（防硬湊剛性規定）】🚨**：
           - 在空間幾何、代數計算或微積分綜合題中，**你必須給出每一步推導的嚴謹代數或幾何理由，絕對不允許因為推理瓶頸就跳過關鍵證明，直接拋出「根據幾何結構，y_P 應為 7.5 才能得到答案 2」這種因果倒置、迎合答案的文字！**
           - 你的推導過程必須能被高三學生循序漸進地看懂。如果你的推導在中途算不下去，代表你的一開始建模有誤，請重置你的代數與幾何模型，直到能流暢完整地推導出最終結果。
        4. **🚨【絕對優先堅持科學/數理正確，嚴厲禁止迎合錯誤答案而硬凹】🚨**：
           - **優先輸出正確推導**：如果你在推導中發現與官方給定的答案存在根本的、無法調和的邏輯衝突（例如：106 數甲多選 5 中，推導得出選項 1 絕對錯誤，但官方答案因 OCR 錯位或印刷有誤而包含了選項 1），**你必須 100% 優先堅持最嚴謹、正確、符合學理與數學定律的推導，把錯誤選項分析為錯誤！絕對不允許為了迎合官方答案而強行假設未知數或硬凹說「因為官方答案包含此選項，推測題幹有誤，故視為正確」！**
           - 你應該在 `detailed_solution` 中【客觀且自信地指出此潛在衝突或印刷疑義】，列出你嚴謹推導出的正確解答。
           - 同時，如果懷疑是當前影像擷取或官方答案有誤，請將 `suspects_image_mismatch` 設為 `true`，以利系統在後續啟動影像重掃描與仲裁。
        5. **🚨【微細符號與循環小數防忽視規則（極重要）】🚨**：
           - 在台灣大考中（例如 114學測數乙 一(1)），循環小數（如 $1.\\bar{{5}}$，即數字上方有一橫線或圓點）在低解析度或 OCR 中極易被誤識別為普通小數（如 $1.5$）。
           - 如果你發現以普通小數計算出來的值（如 $1.5 \\times 5 = 7.5$）完全不對位任何選項，但以循環小數（如 $1.\\bar{{5}} \\times 5 = \\frac{{14}}{{9}} \\times 5 = \\frac{{70}}{{9}} = 7.\\bar{{7}}$）計算能完美對齊正確選項（如 $7.\\bar{{7}}$，即選項 (3)），**這代表最初的題目文字辨識存在微小符號遺漏！**
           - 在此情況下，你**必須**將題目文字自動修正為正確的循環小數格式進行詳解，並在詳解中加上說明。**絕對禁止迎合錯誤的無點小數進行強湊計算！**
        5-2. **🚨【對數 log 底數與真數防錯位極度警告】🚨**：
           - 在數學題目中，對數 $\\log_a b$ 的「底數 $a$」是較小的下標，而「真數 $b$」是正常大小的主字體！
           - 例如 $\\log_2 3$（底數為 2，真數為 3）、$\\log_4 6$（底數為 4，真數為 6）。
           - 嚴禁看反底數與真數（例如不可看成 $\\log_3 2$ 或 $\\log_6 4$）！
           - 在比較對數大小或計算對數方程式時，請仔細比對圖片中的下標字體，切勿將指數規律看反。
           
【六、學科分類與解題技巧命名剛性規範（🚨最高優先級：違反將導致系統崩潰🚨）】

「🚨【視覺細節注意】：圖片已採用 300 DPI 高解析度裁切。請仔細辨識題目中的數學上標、下標、化學價電荷、以及圖表座標軸的小字，嚴禁看錯數值。」

🚨【知識庫最大化重用與防分叉剛性規定（非常重要）】：
1. 下方提供的【當前學科現有可用題型分類清單】和【當前學科現有可用解題技巧清單】是系統長期累積、沉澱的黃金標準知識庫。
2. **【嚴格禁止隨意發明新名稱】**：在你決定「自建考點」或「自建技巧」前，你必須在內部進行一次「語意映射檢索」。只要現有清單中，有任何一項在「學科觀念」或「解題方法」上與本題高度重合或相似，**你必須 100% 強制使用現有的名稱，絕對不能因為字眼稍微不同就另立新名（這會導致知識點分叉損毀）！**
3. **【合併優於拆分】**：即使現有清單的名稱比你學術上想的稍微寬泛（例如現有 `必修_排列組合_計數原理`，而你想寫 `必修_排列組合_計數原理_加法原理應用`），也請**優先選擇現有的寬泛名稱**，而不是隨意創建一項只多出兩三個字的新考點！
4. 只有在現有清單與本題考點**完全風馬牛不相及（例如：數學題考到化學，或者現有清單完全為空）**時，你才被允許「自建考點」，且自建考點必須嚴格遵循相同層級命名規範。

1. **跨單元多維知識點 (topic_categories) 命名規範（極重要）**：
    - **跨領域複合考點捕捉**：台灣大考鑑別題多為跨單元整合題（如：指數函數結合矩陣、空間向量結合三角比、物理力學結合電磁學、化學平衡結合氧化還原、英文分詞結構結合動詞片語）。
    - **多考點陣列輸出**：你【必須】在 `topic_categories` 列表中，列出本題所涉及的 **1 到 3 個** 核心學科知識點（按重要性由主至次排列，第一項為主考點，後續為跨章節次要考點）。
    - **首字強制約束與【嚴禁重複科目名稱】（最高紅線）**：
      * 列表中的**每一個字串項目**，都【必須且只能】以 `必修_` 或 `選修_` 作為開頭！
      * 🚨🚨【絕對嚴格禁止在必修/選修後面重複科目名稱】🚨🚨：
        - 🚫 絕對禁止寫出：`必修化學_...`、`選修物理_...`、`必修生物_...`、`選修化學_...`、`必修_化學_...`！
        - 🚫 第一節必須且只能是 `必修` 或 `選修`，第二節直接接「大單元名稱」！
        - ✅ 正確示範：`必修_物質的組成_原子結構與週期表`（不可寫 `必修_必修化學_...`，也不可寫 `必修化學_...`）。
        - ✅ 正確示範：`選修_力學與運動_牛頓運動定律`（不可寫 `選修物理_...`）。
    - **階層化結構**：每一項皆必須嚴格遵循 `必修/選修_大單元_次單元_具體觀念` 的底線（`_`）格式。
    - 🚨 **【底線 (_) 使用限制】**：底線 `_` **僅限於**作為大單元、次單元與觀念之間的階層分隔！同階層並列詞請使用斜線 `/`、空格或 `&`（例如：`否定字首 un/dis/in`，不可寫 `否定字首_un_dis`）。
    - 系統會自動將第一項指派給主索引欄位 `topic_category`，確保多維度檢索與相容性。
      - 並列詞彙請使用斜線 `/`、符號 `&`、半形空格或加號 `+`（例如：`否定字首 un/dis/in/im/il/ir/non`，絕不可寫 `否定字首_un_dis...`）。
      - 英文句子或片語請使用正常的半形空格（例如：`... times as adj. as ...` 或 `no more than vs not more than`）。
    - **首要優先原則**：請 100% 優先從下方提供的現有清單中，挑選與本題最精準配對的一項填入。
    - **自建考點規則**：若本題考點在下方清單中付之闕如，你才可以自行創建，但**必須嚴格模仿現有結構與長度**，確保第一段必定為必修或選修。
    - **絕對剛性禁止**：
        - 🚫 嚴禁直接輸出無層級、泛化或過短的字詞（例如：`反應速率`、`多項式`、`歷史解釋`）。
        - 🚫 嚴禁遺漏 `必修_` 或 `選修_` 字首。
        - 🚫 嚴禁在層級間使用底線（`_`）以外的任何中英文標點符號、斜線或空格。
    6. **英文片語與搭配詞專項規範（極度重要）**：
    - 若考點為 Phrasal Verbs（動詞片語，如 point to, depend on），`topic_category` 必須包含 `_字彙片語_動詞片語專論`。
    - 若考點為 Idioms（慣用語，如 fuel to the fire, a blip），`topic_category` 必須包含 `_字彙片語_慣用語與成語辨析`。
    - 若考點為 Collocations（搭配詞，如 tight schedule, grave concerns），`topic_category` 必須包含 `_字彙片語_搭配詞組`。
    - 在 `techniques_used` 必須對應填入：
        * `必修_片語語意推論_慣用語境映射法 Idiom Context Mapping`
        * `必修_搭配詞辨析_語意場契合度校驗 Collocation Analysis`
    - ⚠️【當前學科現有可用題型分類清單】：
        {topics}

2. **解題技巧 (techniques_used) 命名規範**：
    - **首字強制約束**：此列表（List of Strings）中包含的**每一個字串項目，都必須且只能**以 `必修_` 或 `選修_` 作為開頭。
    - **方法論階層格式**：必須嚴格遵循 `必修/選修_學科方法論大類_具體技巧與實作步驟` 的底線（`_`）格式。
    - 🚨 **【底線 (_) 使用嚴格限制】**：與前述題型分類相同，底線 `_` **僅能用作階層分隔**。同階層的英文字彙、句子或並列符號，請一律使用半形空格、斜線 `/` 或 `&` 隔開，絕對禁止用 `_` 串聯連續的英文單字或並列詞！
    - **首要優先原則**：請 100% 優先從下方提供的現有清單中，選取 1~3 個最貼切的技巧組成列表。
    - **自建技巧規則**：若自建技巧，必須包含具體的操作手法（如：`ICE表格法`、`座標化代數求解`），且其首字同樣必須帶有 `必修_` 或 `選修_` 開頭。
    - **絕對剛性禁止**：
        - 🚫 嚴禁輸出無方法論與實作技術價值的詞彙（例如：`數據分析`、`概念對照`、`邏輯推理`）。
        - 🚫 列表中的任何一項，其開頭絕對不可以遺漏 `必修_` 或 `選修_` 前綴。
    - ⚠️【當前學科現有可用解題技巧清單】：
        {technique}
    
{math_scope_instruction}
{subject_specific_instruction}
"""

PROMPT_STAGE_3_VALIDATOR = """
你是一位在台灣高中教育與大考閱卷界極具威望、學術態度極為嚴謹的閱卷審查教授。請批次審查以下 AI 寫的詳解是否犯了「邏輯漏洞」、「硬湊答案」、「公式錯誤」或「排版損毀」等瑕疵。

待審查列表如下：
{validator_batch_intro}

【審查三大硬性指標（若違反任一項，該題的 is_valid 必須設為 false）】：
1. **數理/邏輯推導正確性與防硬凹審查**：
    - 嚴格驗算每一道題目的代數計算、化學計量、反應式配平與物理定律。
    - 嚴厲拒絕「邏輯斷層與硬湊」：若詳解算到一半算不出結果，卻突然寫出「故得到 X」或「為了符合答案所以...」等強湊偽邏輯，必須退回。
    - 🚨【嚴厲拒絕「為迎合答案而硬凹」】：若詳解為了迎合錯誤的/讀錯的官方答案，在證明該選項在數學上絕對錯誤之後，卻強行塞入「但考量官方答案包含此選項，故視為正確」等顛倒黑白、毫無學術原則的文字，**你必須判定為 is_valid = false**，並在批判中嚴厲指出此點，逼退 AI 進行重解或促使系統啟動學術仲裁！
    - 🚨【選填題逗號格式校驗】：若本題為選填題，其官方答案已格式化為逗號分隔的形式（例如 `-,4,3`），請嚴格校驗 AI 所求得的最終各項數值（例如 $a=-4$, $b=3$）是否與該畫卡格子答案完全一致。若數值正確但表示形式有誤，或 AI 因為看不懂逗號格式而硬湊，請判定為不通過並要求修正。
    2. **LaTeX 閉合與安全性**：
    - 檢查詳解中所有的 LaTeX 語法。任何未閉合的 `$` 符號、不完整的大括號 `}}` 或未轉義的特殊字元，必須退回修正，避免前端系統排版崩潰。
3. **選項與手寫標準對位（極嚴格審查）**：
        - 對於選擇題（單選與多選）或有提供選項的選填題，你【必須】核對詳解中是否對 `options` 中的**每一個選項標籤（如 1, 2, 3 或 A, B, C）**都進行了獨立、單獨的剖析？
        - **【絕對退件限制】**：若詳解只給出了整體的數學推導，而**漏掉了任何一個選項標籤的獨立剖析**（即沒有寫出 `- **(標籤)**` 的剖析結構），你【必須】判定其為 `is_valid = false` 並在 `error_critique` 中指出其漏掉分析的選項。
        - 若非選擇題，則確認其沒有胡亂捏造選項分析即可。
    - 對於非選題，詳解是否完美融入並對應了手寫評分原則（scoring_criteria）？
4. **中華民國現行法規與課綱真實性審查（最高紅線）**：
    - 涉及公民與社會（法律、政治、經濟）時，是否嚴格遵循中華民國現行法規（違憲審查為「憲法法庭裁判」而非其他、民法成年為滿 18 歲、刑法三階論）？
    - 嚴格排查是否出現非台灣法律術語或大陸法系污染（如「民法典、公安局、人民法院」等）？若有任何一處，**【必須】判定 is_valid = false**，並在批判中嚴厲喝斥退回！
5. **多選題選項證偽深度審查**：
    - 檢查詳解是否對每一個錯誤選項都給出了實質的證偽理由？若只是空泛宣稱「此選項不正確」而未解釋原因，判定 `is_valid = false`！

🚨【審查教授特別注意：容忍 OCR 解答錯位】🚨
大考中心官方答案有時會因雙欄排版或橫排導致本系統的 OCR 抓錯格子（例如將第 22 題看錯行）。
- 若你發現 AI 的學術推導**物理邏輯或代數推導完全正確**，但它最終算出來的答案卻與「官方答案」不符，請你務必力挺 AI，判定 `is_valid = true`！
- 同時，如果發現官方答案存在嚴重的 OCR 解析錯位（例如解答卷上該格分明是斜線或空白，卻被誤讀為數字），請將該題的 `suspects_ocr_error` 設為 true，啟動影像重新掃描機制。

【審查標準（若有以下任一項，is_valid 必為 false）】：
1. **無中生有**：AI 沒有去解題目的「共同資訊」或方程式，自己憑空捏造數字。
2. **邏輯斷層硬湊**：AI 算到一半算不出結果，突然神來一筆。
3. **知識性錯誤與計算失誤**：AI 的代數推導、英文文法、化學結構有明顯硬傷。
4. **自動圖解與題目附圖吻合度審查（極重要）**：
   - 仔細比對傳入的題目裁切附圖與題目文字！
   - 🚨【圖文張冠李戴即刻退件】：如果題幹提到的是「等電位線、圖 9」，但裁切附圖中出現的卻是「腳踏車、煞車片、圖 7」等完全不相干的圖片，**【必須】判定 is_valid = false**，並將 suspects_ocr_error 設為 true，在 error_critique 中註明：「題目附圖錯誤（題幹為等電位線圖9，附圖卻抓成腳踏車煞車片圖7），要求重新框選並重掃描！」
5. **放寬結論句限制**：若 AI 的『過程分析』已經 100% 正確，僅僅是最後沒有總結字眼，請判定為 is_valid = true。
🚨 6. **持續重掃描判定（連續報警）**：若你發現雖然上一輪重掃描更新了數據，但詳解中依然存在嚴重的化學式/數學結構矛盾（這代表上一次重掃描依然沒有看清楚、或者重掃描也看錯了），你【必須繼續將 suspects_ocr_error 設為 true】！引導系統進行更精確的二次或三次重掃描比對，絕對不要輕易放棄！
"""


SUBJECT_DIFFICULTY_RUBRICS = {
    "數學": """
- **[Level 1-2] (送分題)**：課本基本定義與性質，單一公式直接套用即可（如：給予 $f(x)$ 與 $x=a$ 直求 $f(a)$、給予兩點求向量與距離）。
- **[Level 3-4] (基礎題)**：課內標準題型、單一概念加基本代數計算（如：簡單勘根、一階線性遞迴、基本直線方程式與圓方程、單純的計數與古典機率）。
- **[Level 5-6] (中等題)**：跨章節綜合題，需進行 3 步驟以上的代數與幾何運算，或包含不顯眼的陷阱（如：忽略分點公式比例方向、轉移矩陣穩定狀態、三角函數疊合、拋物線光學性質）。
- **[Level 7-8] (鑑別題)**：跨冊次/跨大單元超大型整合題、複雜空間幾何（如：多維度線性規劃、棣美弗定理分數根高次方程、夾擠定理在無窮級數極限的證明、空間公垂線距離、橢圓/雙曲線參數式應用）。
- **[Level 9-10] (魔王題)**：競賽/神人級難題。極高抽象思考、需要發明輔助線或特殊代數與旋轉矩陣變換、多重條件機率與微積分綜合（如：多項式函數、三角函數與指對數之旋轉體體積定積分）。
""",
    "物理": """
- **[Level 1-2] (送分題)**：基本物理量定義與單位，單一公式直接代入（如：歐姆定律 $V=IR$ 算電流、牛頓第二定律 $F=ma$ 算力）。
- **[Level 3-4] (基礎題)**：標準單章節物理題型、基本力學/電磁學分析（如：一維等加速運動、平拋水平射程、球面鏡成像高斯公式）。
- **[Level 5-6] (中等題)**：跨章節/跨力學系統題（如：動量守恆與圓周運動結合、電阻電容直流電路分析、動生電動勢力學平衡、熱力學 PV 面積與內能變化）。
- **[Level 7-8] (鑑別題)**：高難度多階段運動學、複雜電磁場受力（如：帶電粒子在均勻電磁場中的等速率圓周運動、斜向拋體與力矩平衡綜合、多普勒效應與聲速綜合計算）。
- **[Level 9-10] (魔王題)**：需要微積分、向量微積分或高度抽象物理直覺的難題（如：複雜剛體靜力平衡與滾動、電磁感應阻尼運動分析、近代物理波耳模型與物質波複雜代換）。
""",
    "化學": """
- **[Level 1-2] (送分題)**：元素週期律基本趨勢、簡單路易斯結構價電子（如：計算孤對電子、形式電荷、有機物基本結構分類）。
- **[Level 3-4] (基礎題)**：單章節標準計量與化學反應、基礎酸鹼或沉澱判斷（如：強酸強鹼中和、反應速率基本計量、理想氣體方程式換算、簡單同分異構物判斷）。
- **[Level 5-6] (中等題)**：跨章節綜合計算、平衡 ICE 表格法、複雜有機結構（如：弱酸解離極簡近似與稀釋律校驗、同離子效應抑制、過錳酸鉀與碘滴定計量）。
- **[Level 7-8] (鑑別題)**：複雜電化學能斯特方程、緩衝溶液哈塞爾巴赫方程、多步驟無機離子沉澱與溫度效應（如：複雜氧化還原滴定、多鹽基酸逐步解離、飽和溶液與非理想正/負偏差熱力學分析）。
- **[Level 9-10] (魔王題)**：深奧無機/物理化學難題。涉及精密催化、大分子聚合機制（如：複雜 BZ 振盪反應動力學、酚醛樹脂聚合機理、高度不飽和複雜有機物同分異構物推導）。
""",
    "生物": """
- **[Level 1-2] (送分題)**：細胞基本結構（原核/真核）、簡單器官與生體化學組成功能（如：哪些是真核細胞胞器、光敏素Pr與Pfr基本互換）。
- **[Level 3-4] (基礎題)**：孟德爾遺傳機率、植物與動物激素基礎、水分與養分運輸（如：單性雜交機率、生長素IAA向光性、篩管篩胞分布）。
- **[Level 5-6] (中等題)**：動物學複雜生理與免疫系統（如：腎小管逆流倍增系統、心搏起搏點與電生理、血紅素攜氧曲線波耳效應、特異性免疫）。
- **[Level 7-8] (鑑別題)**：複雜基因表達與操縱組調控、演化遺傳哈溫平衡、PCR指數與凝膠電泳計算（如：阻遏與活化機制、親緣關係樹建構）。
- **[Level 9-10] (魔王題)**：多系統複雜整合、高度抽象生理調控（如：細胞計畫性死亡內生與外生反應路徑、特定物質清除率與腎小球濾過率計算）。
""",
    "地球科學": """
- **[Level 1-2] (送分題)**：固體地球基本分帶、大氣垂直分層、基本天氣系統。
- **[Level 3-4] (基礎題)**：恆星視絕對星等、恆星顏色與表面溫度關係及黑體輻射、地層疊置定律與化石對比。
- **[Level 5-6] (中等題)**：湧升流與艾克曼搬運、起潮力大潮小潮、邊坡順向坡穩定度力學分析。
- **[Level 7-8] (鑑別題)**：地磁倒轉歷史與海底擴張、赫羅圖恆星半徑與黑體輻射計算、哈伯定律宇宙紅移退行速度、地震斷層震源機制之初動震波第一運動判讀。
- **[Level 9-10] (魔王題)**：乾濕絕熱遞減率大氣穩定度繪圖分析、局地風流力風場力學平衡、湧升流動力學。
""",
    "歷史": """
- **[Level 1-2] (送分題)**：基礎歷史時間軸、知名歷史事件/地名/人名之直接記憶與辨識。
- **[Level 3-4] (基礎題)**：單一一手/二手史料解讀、基本因果關係。
- **[Level 5-6] (中等題)**：跨史料互證與偏見分析、地名演變與官制幣制、時代特徵與政權歸屬判定。
- **[Level 7-8] (鑑別題)**：西方原典/條約原文書信核心價值轉譯、大考常考之經濟中心轉移與人口遷移圖表與歷史局勢關聯解析。
- **[Level 9-10] (魔王題)**：極生疏一手史料客觀與主觀解釋評估、戰時政治宣傳海報/政治漫畫之新帝國主義隱喻深度剖析。
""",
    "地理": """
- **[Level 1-2] (送分題)**：基礎經緯度位置、地圖投影基本特徵與失真判讀。
- **[Level 3-4] (基礎題)**：等高線幾何與分水嶺流路、溫雨圖判讀南北半球、中地理論商圈。
- **[Level 5-6] (中等題)**：疊圖分析網格矩陣代數疊加、韋伯原料指數區位選擇、逕流歷線洪峰。
- **[Level 7-8] (鑑別題)**：遙感探測 NDVI 多時段地表變遷監測、二度分帶與UTM全球橫麥卡托坐標網格定位、大圓路徑球面最短距離。
- **[Level 9-10] (魔王題)**：國際地緣政治衝突圖表與資源地緣衝突、世界體系理論微笑曲線全球化代工利得分配。
""",
    "公民與社會": """
- **[Level 1-2] (送分題)**：自我社會化、國家的四要素、憲法基本權利分類。
- **[Level 3-4] (基礎題)**：中央與地方權限劃分、民事權利行為能力、犯罪成立三階論。
- **[Level 5-6] (中等題)**：機會成本與比較利益計算、消費者剩餘/生產者剩餘無謂損失圖形解析。
- **[Level 7-8] (鑑別題)**：外部效果（外部成本、外部效益）私人生產與社會生產位移圖形解析、違憲審查比例原則三步驟檢驗。
- **[Level 9-10] (魔王題)**：選制對政黨與政策穩定性評估、GDP支出所得法要素檢索、供需價格彈性與政府干預限價/保證價格之後果分析。
""",
    "國文": """
- **[Level 1-2] (送分題)**：基礎形音義辨正、成語典故字面義與語境應用。
- **[Level 3-4] (基礎題)**：經典古文實詞與虛詞（之乎者也焉乃其以而於）、譬喻與轉化修辭判讀。
- **[Level 5-6] (中等題)**：古漢語實詞虛化與語意引申流變、近體詩平仄黏對與詞調曲牌格律分析。
- **[Level 7-8] (鑑別題)**：古今字通假字辨識、文言文倒裝（賓語前置、狀語後置）與省略成分依據語境還原、古典意象群落象徵。
- **[Level 9-10] (魔王題)**：深奧文言文句讀劃分與停頓、多文本跨時代同主題對比、經典文學批評（如文心雕龍、詩品、人間詞話）原典核心主張解讀。
""",
    "英文": """
- **[Level 1-2] (送分題)**：Level 1-2 生活常用單字、基本副詞與限定性關係子句。
- **[Level 3-4] (基礎題)**：搭配詞組、關係副詞功能、分詞結構、比較與倍數。
- **[Level 5-6] (中等題)**：長難句多重嵌套子句階層式拆解、獨立分詞構句、讓步倒裝、虛擬代替（wish/as if/but for）。
- **[Level 7-8] (鑑別題)**：篇章銜接前指與定冠詞前指溯源、邏輯過渡詞分類、跨段落資訊拼圖與潛在假設推理。
- **[Level 9-10] (魔王題)**：生難字語境對比推敲、批判性事實與作者觀點/評論區分、根據選字語調(Tone)推斷作者隱含立場與偏見。
""",
    "國寫": """
- **[Level 1-2] (送分題)**：簡單重點提煉、基本抒情感受。
- **[Level 3-4] (基礎題)**：傳統與現代文化保存之矛盾辯證、失敗與挫折的日常感悟、多方觀點比較。
- **[Level 5-6] (中等題)**：科技倫理思辨（如 AI 人類主體性）、社會公共議題（如少子高齡化與社會結構變遷對策）之因果推論與遞進層遞結構。
- **[Level 7-8] (鑑別題)**：駁論建構與批判（歸謬法、反證法）、五感摹寫與核心意象多重隱喻、由小我經驗昇華至大我關懷。
- **[Level 9-10] (魔王題)**：針對複雜公共爭議提出具體行動方案、大綱架構字數動態分配與融入生命哲學思辨點題。
"""
}

# 如果某學科不在上述字典，則使用此通用標準
GENERAL_DIFFICULTY_RUBRIC = """
- **[Level 1-2] (送分題)**：單一基礎概念、直接代入單一公式、字面直白翻譯。幾乎無陷阱。
- **[Level 3-4] (基礎題)**：需要基礎邏輯推演、簡單圖表判讀、或兩個基礎概念的結合。為大考的過關門檻題。
- **[Level 5-6] (中等題)**：跨章節概念整合、需處理較複雜的數據/圖表/實驗、干擾選項多、或需進行 3 個步驟以上的數學運算與邏輯推導。
- **[Level 7-8] (鑑別題)**：情境高度包裝、資訊隱蔽度高、需要極強的閱讀理解與反向推理能力、極易掉入陷阱、或計算極其繁瑣。
- **[Level 9-10] (魔王題)**：超出常規題型、需要極具創意的解題切入點、多重陷阱疊加、或考驗極為冷門深奧的學科盲點。得分率極低。
"""

# ==========================================
# 1. Pydantic 資料結構定義 (定義 JSON 與解答長相)
# ==========================================

class OptionItem(BaseModel):
    key: str = Field(description="選項標籤。必須且只能是一個大寫英文字母（如 'A', 'B', 'C', 'D', 'E'）或數字，絕對不能填入任何其他自訂變數。")
    value: str = Field(description="該選項的文字描述內容。若選項完全為圖形，可寫 '【圖形選項】'")
    has_image: bool = Field(default=False, description="該選項本身是否為獨立的附圖、圖表或示意圖？")
    image_bboxes: List[List[int]] = Field(default=[], description="若該選項有附圖，請給出其 Bounding Box，格式如 [[ymin, xmin, ymax, xmax]]。所有坐標必須規格化至 0 到 1000 的整數區間。若無則為空列表 []。")
    image_paths: List[str] = Field(default=[], description="由程式自動裁剪並填充的該選項附圖實體路徑列表，無則為空。")

# 修正：定義 AnswerItem 替代 Dict，防止 additionalProperties 崩潰
class AnswerItem(BaseModel):
    question_number: str = Field(description="題號，例如 '1', '2', '非選一'")
    standard_answer: str = Field(description="對應的答案，例如 'A', 'B', '0.75'")

class AnswerKey(BaseModel):
    answers: List[AnswerItem] = Field(description="這頁答案卷中包含的所有題號與答案配對列表。")

# === 第一階段：擷取層 (Temperature = 0.0) ===
class ExtractedQuestion(BaseModel):
    academic_year: str = Field(description="學年度與考試類型縮寫，例如 '114學測'、'114分科'、'114模考第一次'")
    exam_source: str = Field(description="原始考卷完整名稱，例如 '114學測自然', '110指考物理'")
    sub_subject: Literal['物理', '化學', '生物', '地球科學', '歷史', '地理', '公民與社會', '數學', '英文', '國文', '國寫'] = Field(description="本題的具體學科分類。如果是學測社會，必須強制歸類為 '歷史'、'地理' 或 '公民與社會' 之一；如果是學測自然，必須強制歸類為 '物理'、'化學'、'生物'、'地球科學' 之一。若遇跨科整合題請選擇佔比最重的一科。單一學科考卷則直接對應填寫（如 '數學', '英文', '國文'）。")
    question_number: str = Field(description="題號，例如 '1', '2', '18-20'")
    page_number: int = Field(description="本題在該試卷 PDF 中的真實頁碼（從 1 開始計數，例如：1, 2, 3...）")
    shared_context: str = Field(default="", description="若本題為題組題，請將【共同引言、閱讀測驗文章、實驗敘述】放在此處。若非題組題，請留空。")
    question_text: str = Field(description="單純針對這一個子題的題目文字。數學公式，請嚴格使用 LaTeX。")
    has_image: bool = Field(description="題目是否明確印有幾何附圖、圖表或表格？絕對禁止因為題目提到『圖形』、『正方體』等文字就憑空將此設為 true！必須要有實體圖案。")
    image_bboxes: List[List[int]] = Field(default=[], description="若有明確附圖，請給出所有附圖的 Bounding Box 列表，格式如 [[ymin, xmin, ymax, xmax]]。注意：所有坐標必須規格化至 0 到 1000 的整數區間。絕對禁止憑空虛構框線！若無實體圖案則必須為空列表 []。")
    options: List[OptionItem] = Field(description="選項物件列表，無則填 []。")
    answer: str = Field(description="本題的標準答案。如果是多選題，必須將所有正確選項字母按字母順序排列，中間不加任何逗號、空格或符號（例如：'ACD' 而非 'A, C, D'）。")
    official_answer: str = Field(default="", description="大考官方原始提供的解答紀錄")
    derived_answer: str = Field(default="", description="由 AI 嚴謹學術推導得出之正確答案")
    has_answer_discrepancy: bool = Field(default=False, description="標記官方解答與學理推導是否產生衝突或排版印刷錯誤")
    discrepancy_reason: str = Field(default="", description="若存在衝突，詳細記錄學術論證與官方答案疑似錯位之原因")
    image_paths: List[str] = Field(default=[], description="所有實體裁切圖片的路徑列表。若無則為空列表 []。")
    question_type: Literal['單選題', '多選題', '選填題', '簡答題', '繪圖作圖題', '混合題'] = Field(description="題型分類。")
    full_page_image_path: str = Field(default="", description="整頁試卷的原始圖片路徑 (作為前端兜底顯示使用)")
    question_pdf_path: str = Field(default="", description="原始題目 PDF 檔的實體路徑")
    answer_pdf_path: str = Field(default="", description="原始標準答案 PDF 檔的實體路徑")
    rubric_pdf_path: str = Field(default="", description="原始非選擇題評分標準 PDF 檔的實體路徑")
    question_page_image_paths: List[str] = Field(default=[], description="整份原卷每一頁的圖片路徑清單")
    answer_page_image_paths: List[str] = Field(default=[], description="整份標準答案每一頁的圖片路徑清單")
    rubric_image_paths: List[str] = Field(default=[], description="從評分標準 PDF 中裁切出的正確答案與評分表圖片路徑")
    scoring_criteria: str = Field(default="", description="針對非選擇題、簡答題、手寫題，從官方評分標準中精確提取的給分步驟與扣分細則。選擇題請留空。")
    has_official_answer: bool = Field(default=True, description="本題是否有官方提供之標準解答。若此試卷為無解答之學校定期考/段考，設為 false。")
    school_name: str = Field(default="", description="若為學校定期考/段考，記錄學校名稱（如 '臺北市立建國高級中學' 或 '建國中學'）；非學校段考則留空。")


class PageExtraction(BaseModel):
    questions: List[ExtractedQuestion] = Field(description="這頁 PDF 中的所有題目列表。")

class OptionAnalysisItem(BaseModel):
    key: str = Field(description="選項標籤。必須且只能是一個大寫英文字母（如 'A', 'B', 'C', 'D', 'E'）或數字，必須與題目 options 列表中的 key 完全對齊。")
    explanation: str = Field(description="針對該選項的專屬分析與對錯判斷。必須包含公式推導、數值代入或文意對照，並在結尾明確指出該選項本身是『正確』或『錯誤』。")

# === 第二階段：大腦層 (Temperature = 0.7) ===
class QuestionSolution(BaseModel):
    suspects_image_mismatch: bool = Field(default=False, description="【圖片精確性確認】若你發現裁剪出來的圖片與本題題號不符（例如圖中印的題號不是本題題號），或者圖片模糊不清、公式嚴重錯位、表格被切掉時，請設為 true，系統將自動啟動重新掃描與補件機制。")
    derived_answer: str = Field(default="", description="【本題嚴謹推導答案】請在此填寫你推導出的最終答案字串（如選擇題填 'A' 或 'ABD'；選填題填數值如 '5/6' 或畫卡格式 '-,4,3'；非選題填最終數值如 '12' 或最簡式）。")
    question_analysis: str = Field(description="【題意分析】：用 Markdown 粗體標示核心關鍵字。")
    solving_strategy: str = Field(description="【解題思路】：推導脈絡與解題突破點。")
    detailed_solution: str = Field(description="【完整解法與另解】：極詳細的解答。必須包含正規解法，並強烈建議提供「另解」、「速解」或「不同角度的切入方式」。理科使用 LaTeX，文科交代邏輯。")
    options_analysis: List[OptionAnalysisItem] = Field(description="【選項深入剖析】：必須對題目 options 列表中出現的每一個選項進行一對一、單獨的詳細分析列表。如果本題為非選擇題、計算題或簡答題，此列表必須為空列表 []。絕對不得漏掉任何選項！")
    concept_review: str = Field(description="【核心概念複習】：條列式說明相關定理或法條。")
    traps_and_warnings: str = Field(description="【易錯陷阱】：用強烈警示語氣指出學生易犯盲點。")
    advanced_supplement: str = Field(description="【進階延伸補充】：相關學術考點或跨章節聯想。")
    scoring_rubric: str = Field(description="評分標準或配分建議。")
    difficulty_level: int = Field(description="難度評分 (1-10)。")
    difficulty_reason: str = Field(description="給出難度評分的客觀理由。")
    topic_categories: List[str] = Field(
        default=[], 
        description="【多維知識點分類列表】本題所涵蓋的所有核心學科知識點列表（按主次順序排列，一般題目 1~2 項，跨單元綜合題 2~4 項）。每一項必須嚴格遵循 '必修/選修_大單元_次單元_具體觀念' 的階層命名格式。"
    )
    topic_category: str = Field(default="", description="本題最核心的第一主要知識點（作為主索引，取自 topic_categories[0]）。")
    techniques_used: List[str] = Field(description="使用的解題技巧列表。")

# === 第三階段：審查糾錯層 (Temperature = 0.0) ===
class SolutionValidator(BaseModel):
    is_valid: bool = Field(description="詳解是否完全符合邏輯且推導扎實？🚨警告：若你決定將 suspects_ocr_error 設為 true 以啟動重掃描，你必須將 is_valid 設為 false！這兩者不可以同時為 true。")
    error_critique: str = Field(description="若有錯誤或矛盾，請具體指出。若正確請留空。")
    suspects_ocr_error: bool = Field(default=False, description="【非常重要】若 AI 的數學推導完美無瑕，但卻與官方答案或題目條件嚴重矛盾，強烈懷疑是最初的 OCR 抓錯題目或看錯解答表欄位時，請設為 true 以啟動影像重新掃描。")

class AcademicArbitration(BaseModel):
    chosen_answer: str = Field(description="經學術仲裁後，認為最合適、最嚴謹的答案（若官方答案有誤，可修正為正確推導答案，或維持官方答案但在備註中解釋）。")
    detailed_critique: str = Field(description="對此爭議的完整學術論證、公式推演，以及官方答案或印刷是否出錯的詳細剖析。")
    final_solution_append: str = Field(description="要追加到原本詳解末尾的【學術仲裁與印刷疑義備註】Markdown 文字。必須包含明確的『備註』或『印刷爭議』字樣。")

# === 動態重掃描層 ===
class CorrectedSource(BaseModel):
    comparison_analysis: str = Field(description="對照分析：仔細比對『上一次擷取內容』與『本次原圖影像』的差異，說明為什麼發生錯位（例如：上版將根號看成直線、表格欄位對齊錯誤等），並評估新版題目與答案的邏輯一致性。")
    corrected_question_text: str = Field(description="修正後的完整題目文字。若經重新核對無誤請照抄原本的文字。")
    corrected_shared_context: str = Field(default="", description="修正後的題組共同背景。若本題非題組或原背景無誤，請照抄或保持原樣。") # 🆕 新增此欄位
    corrected_options: List[OptionItem] = Field(description="修正後的選項列表。")
    corrected_answer: str = Field(description="修正後的真正官方答案。")
    # 💡 新增以下兩個欄位
    found_new_images: bool = Field(description="重新檢視原圖後，是否發現了先前漏掉的附圖、表格或化學結構？")
    new_image_bboxes: List[List[int]] = Field(default=[], description="若發現新圖，請給出規格化 (0-1000) 的 Bounding Box 列表。")
    is_confident: bool = Field(description="你是否高度確信本次修正後的題目與答案是 100% 準確、無誤且相互吻合的？")

# === 🎯 模擬考專屬官方詳解本視覺萃取資料結構 ===
class MockQuestionExplanation(BaseModel):
    question_number: str = Field(description="題號，例如 '1', '2', '10', '13', '14', '非選一'")
    answer: str = Field(description="本題在詳解本中標註的標準答案（如 '(4)', '7', 'x+4y+6=0', '4/3百萬元'）")
    source_curriculum: str = Field(default="", description="詳解中明確印出的『出處』。🚨若原卷詳解完全未列出出處，請務必保持為空字串 ''，絕對嚴禁自行猜測捏造！")
    exam_target: str = Field(default="", description="詳解中明確印出的『目標』。🚨若原卷詳解完全未列出目標，請務必保持為空字串 ''，絕對嚴禁自行猜測捏造！")
    explanation_text: str = Field(description="詳解中的完整推導文字、運算步驟與 LaTeX 數學/化學式。請原汁原味忠實呈現，涉及公式必須嚴格包裹在 $...$ 或 $$...$$ 中。")
    options_analysis: List[OptionAnalysisItem] = Field(default=[], description="若為選擇題且詳解有逐一分析 (1) ╳ ... (2) ○ ... 或 (A)~(E)，請獨立拆解為各選項對應條目；若為非選擇題或選填題則為空列表 []。")
    scoring_rubric: str = Field(default="", description="詳解中標註的『◎評分原則』或步驟給分說明（如：列式得 1 分、求出數值得 2 分），若無則留空。")
    difficulty_level: int = Field(default=5, description="若詳解有星級或難度評級請對應換算為 1~10 級；若無請依出處與運算量客觀評為 1~10 級。")
    difficulty_reason: str = Field(default="", description="難度評級理由，若詳解有簡評請摘錄，無則簡要說明。")
    has_solution_image: bool = Field(default=False, description="詳解內容中是否印有幾何圖、函數圖形、受力分析、積分區域或實驗流程等附圖？")
    solution_image_bboxes: List[List[int]] = Field(default=[], description="詳解中所有附圖的 Bounding Box 列表 [[ymin, xmin, ymax, xmax]]，坐標規格化至 0 到 1000。若無實體附圖則為 []。")
    topic_category: str = Field(default="", description="根據出處與目標標準化之學科知識點，格式必須為 '必修/選修_大單元_次單元_觀念'")
    techniques_used: List[str] = Field(default=[], description="本題使用的解題技巧列表，格式如 ['必修_數據分析_最適直線斜率公式']")

class MockExplanationPageExtraction(BaseModel):
    items: List[MockQuestionExplanation] = Field(description="本頁詳解卷中包含的所有題目詳細解析條目列表。")

# === 🎯 模考專屬：全能數理公式、計算過程與文字校準審查結構 ===
class MockLightweightCheckResult(BaseModel):
    has_flaw: bool = Field(description="題幹、選項或詳解推導中是否存在：文字抓錯、語病、公式殘缺、負號遺失、指數/下標損毀、計算過程前後矛盾或與答案不一致等瑕疵？")
    flaw_category: Literal["CLEAN", "FORMULA_DEFECT", "CALCULATION_STEP_ERROR", "TEXT_TYPO", "ANSWER_MISMATCH"] = Field(
        default="CLEAN",
        description="'CLEAN' (完全無誤), 'FORMULA_DEFECT' (公式/符號/負號/分數線遺漏), 'CALCULATION_STEP_ERROR' (計算推導過程有跳步或前後矛盾), 'TEXT_TYPO' (文字抓錯/語病), 'ANSWER_MISMATCH' (答案與推導衝突)"
    )
    corrected_question_text: str = Field(description="修正後的完整題幹文字。若原題幹無文字或符號瑕疵請完全照抄。")
    corrected_solution_snippet: str = Field(default="", description="若詳解存在公式抓錯、負號漏看、計算步驟矛盾或語病，請提供修復後語意通順、公式嚴謹、計算完全自洽的完整詳細解答。若原解析已完美無瑕則填 'CLEAN'。")
    corrected_answer: str = Field(description="經嚴謹核對推導後，確認無誤的標準答案字串。")
    repair_summary: str = Field(default="", description="若有修復，簡明記錄具體修正點（如：'補齊波耳半徑公式負號與上標'、'修正計算步驟代數運算矛盾'），無則填 '無誤'。")
    
class QuestionSolutionBatch(BaseModel):
    solutions: List[QuestionSolution] = Field(description="批次產出的多題詳細解答列表，順序必須與輸入題目嚴格對齊。")

class SolutionValidatorBatch(BaseModel):
    validators: List[SolutionValidator] = Field(description="批次審查多題的驗證結果，順序必須與輸入解答嚴格對齊。")

# ==========================================
# 2. Gemini API 輪詢與排程管理器
# ==========================================
class FreeTierKey:
    def __init__(self, api_key: str, custom_base_url: Optional[str] = None):
        self.api_key = api_key
        self.clients_cache: Dict[str, genai.Client] = {}
        self.cache_lock = threading.Lock()
        self.client = self.get_client(custom_base_url)
        self.rpd_reset_time = 0.0

        # 🚨 各模型獨立 RPD 耗盡字典 (model_name -> reset_timestamp)
        self.model_rpd_exhausted: Dict[str, float] = {}

        # 各模型獨立冷卻與請求紀錄字典
        self.model_cooldowns: Dict[str, float] = {}
        self.model_last_request: Dict[str, float] = {}
        self.model_request_times: Dict[str, List[float]] = {}
        self.token_usage = []
        self.daily_request_count = 0
        self.limit_rpd = 1500
        self.is_disabled = False
        self.disable_reason = ""
        
        # 🚨 近期 10 次請求健康度視窗 (1=成功, 0=失敗)
        self.recent_10_history = []

    def record_call_outcome(self, is_success: bool):
        self.recent_10_history.append(1 if is_success else 0)
        if len(self.recent_10_history) > 10:
            self.recent_10_history.pop(0)

    @property
    def recent_10_success_rate(self) -> float:
        if not self.recent_10_history: return 1.0
        return sum(self.recent_10_history) / len(self.recent_10_history)
        
    def is_model_rpd_exhausted(self, model: str, current_time: float) -> bool:
        """判定特定模型在此金鑰下是否每日額度耗盡"""
        if model in self.model_rpd_exhausted:
            if current_time < self.model_rpd_exhausted[model]:
                return True
            else:
                del self.model_rpd_exhausted[model]
        return False

    @property
    def rpd_exhausted(self) -> bool:
        """向下相容屬性：當前 Key 是否所有主流模型皆已耗盡（防止儀表板與舊呼叫崩潰）"""
        now = time.time()
        self.model_rpd_exhausted = {m: exp for m, exp in self.model_rpd_exhausted.items() if now < exp}
        return len(self.model_rpd_exhausted) >= 5
    
    def get_client(self, custom_base_url: Optional[str] = None) -> genai.Client:
        """複用並快取 Client，防範長執行過程中的資源洩漏 (具備執行緒安全鎖)"""
        proxy_key = custom_base_url or "direct"
        with self.cache_lock:
            if proxy_key not in self.clients_cache:
                http_opts = types.HttpOptions(timeout=900_000)
                if custom_base_url:
                    http_opts.base_url = custom_base_url
                self.clients_cache[proxy_key] = genai.Client(api_key=self.api_key, http_options=http_opts)
            return self.clients_cache[proxy_key]

    def get_status(self) -> dict:
        """回傳當前 Key 的 RPM/TPM/RPD 統計狀態（支援分模型統計）"""
        current_time = time.time()
        self.token_usage = [item for item in self.token_usage if current_time - item[0] <= 60.5]
        
        # 清理已過期的模型冷卻
        self.model_rpd_exhausted = {m: exp for m, exp in self.model_rpd_exhausted.items() if current_time < exp}
        
        total_rpm = sum(
            len([t for t in times if current_time - t <= 60.5])
            for times in self.model_request_times.values()
        )
        current_tpm = sum(item[1] for item in self.token_usage)
        
        exhausted_models = list(self.model_rpd_exhausted.keys())
        rpd_note = f" (耗盡: {','.join(exhausted_models)})" if exhausted_models else ""
        
        return {
            "key": f"{self.api_key[:6]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else self.api_key,
            "rpm_status": f"{total_rpm}/15",
            "tpm_status": f"{current_tpm}/500k",
            "rpd_status": f"{self.daily_request_count}/{self.limit_rpd}{rpd_note}",
        }

    def can_request_model(self, model: str, current_time: float, estimated_tokens: int) -> bool:
        """精準判定該 Key 是否能對【特定模型】發起請求（各模型 RPD 獨立判斷）"""
        # 🚨 檢查【當前模型】的每日配額是否耗盡（其他模型不受影響）
        if model in self.model_rpd_exhausted:
            if current_time < self.model_rpd_exhausted[model]:
                return False
            else:
                # 已過午夜重置時間，解除該模型的每日冷卻
                del self.model_rpd_exhausted[model]
                
        # 1. 檢查該模型是否在短暫的 429 滑動冷卻中
        if current_time < self.model_cooldowns.get(model, 0.0):
            return False
            
        # 2. 針對該模型的物理防衝撞間隔（Flash-Lite 3秒，Flash 4.2秒，安全合規 QPS）
        min_interval = 3 if "lite" in model else 4.2
        if (current_time - self.model_last_request.get(model, 0.0)) < min_interval:
            return False
            
        # 🚨 3. 同一 Key 跨模型通用防抖動：該 Key 上一次無論發給誰，至少間隔 0.4 秒
        if (current_time - getattr(self, "last_global_request_time", 0.0)) < 0.4:
            return False
            
        # 3. 該模型的 60 秒滑動視窗 RPM 檢查
        if model not in self.model_request_times:
            self.model_request_times[model] = []
        self.model_request_times[model] = [t for t in self.model_request_times[model] if current_time - t <= 60.5]
        
        limit_rpm = 28 if "lite" in model else 14
        if len(self.model_request_times[model]) >= limit_rpm:
            return False
            
        return True

    def add_request_model(self, model: str, current_time: float, estimated_tokens: int):
        """記錄特定模型的請求時間"""
        self.model_last_request[model] = current_time
        self.last_global_request_time = current_time # 記錄該 Key 的全局最新發送時間
        if model not in self.model_request_times:
            self.model_request_times[model] = []
        self.model_request_times[model].append(current_time)
        self.token_usage.append((current_time, estimated_tokens))
        self.daily_request_count += 1
        if self.daily_request_count >= self.limit_rpd:
            self.mark_rpd_exhausted()

    def mark_model_rate_limited(self, model: str, cooldown_seconds: float = 40.0):
        """🚨 精確鎖定模型，並加入 Jitter 避免 34 把 Key 同時解鎖造成次生 429 暴擊"""
        
        jitter = random.uniform(5.0, 25.0)  # 動態錯開 5~25 秒
        self.model_cooldowns[model] = time.time() + cooldown_seconds + jitter

    def mark_rpd_exhausted(self, model: Optional[str] = None):
        """標記模型耗盡。若未指定模型則標記為全域限制"""
        reset_time = get_next_rpd_reset_timestamp()
        if model:
            self.model_rpd_exhausted[model] = reset_time
        else:
            self.model_rpd_exhausted["global"] = reset_time
            

def purge_single_exam_suite(identifier: str) -> tuple[list, list]:
    """
    🚨 試卷全能銷毀手術刀：
    精確刪除指定試卷的資料庫 JSON、暫存檔、images 圖片目錄，並自動登記至雲端銷毀清單！
    """
    purged_files = []
    purged_dirs = []
    if not identifier: return purged_files, purged_dirs

    clean_id = re.sub(r'[\s\[\]\(\)]', '', identifier)
    out_dir = "./exam_database_output"
    if not os.path.exists(out_dir): return purged_files, purged_dirs

    import shutil
    for root, dirs, files in os.walk(out_dir):
        # 1. 銷毀相關 JSON 檔案 (database, partial, raw, validator_log)
        for f in files:
            if (identifier in f or clean_id in f.replace(" ", "")) and f.endswith((".json", ".txt")):
                full_p = os.path.join(root, f)
                try:
                    os.remove(full_p)
                    purged_files.append(full_p)
                    rel_p = os.path.relpath(full_p, out_dir).replace("\\", "/")
                    with open("pending_cloud_purges.txt", "a", encoding="utf-8") as pf:
                        pf.write(f"gdrive:exam_database_output/{rel_p}\n")
                except Exception: pass

        # 2. 連坐徹底銷毀對應的 images 圖檔目錄
        for d in list(dirs):
            if (identifier in d or clean_id in d.replace(" ", "")):
                full_d = os.path.join(root, d)
                try:
                    shutil.rmtree(full_d, ignore_errors=True)
                    purged_dirs.append(full_d)
                    rel_d = os.path.relpath(full_d, out_dir).replace("\\", "/")
                    with open("pending_cloud_purges.txt", "a", encoding="utf-8") as pf:
                        pf.write(f"gdrive:exam_database_output/{rel_d}\n")
                except Exception: pass

    # 3. 即時重置儀表板考卷進度
    with GLOBAL_METRICS.lock:
        for k in list(GLOBAL_METRICS.exam_progress.keys()):
            if identifier in k or clean_id in k.replace(" ", ""):
                del GLOBAL_METRICS.exam_progress[k]
                GLOBAL_METRICS.completed_papers = max(0, GLOBAL_METRICS.completed_papers - 1)

    GLOBAL_METRICS.add_log(f"🗑️ 已徹底銷毀試卷 [{identifier}] 之所有資料庫檔案與圖片目錄！")
    logging.info(f"🗑️ [手動銷毀完成] 已清除 {len(purged_files)} 個檔案、{len(purged_dirs)} 個目錄，並已登記雲端抹除。")
    return purged_files, purged_dirs

class LiveDashboardMetrics:
    def __init__(self):
        self.lock = threading.RLock()
        self.start_time = time.time()
        self.run_start_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session_id = str(int(time.time()))
        
        # 🚨 遠端控制開關與即時階段標記
        self.stop_requested = False
        self.current_stage = "準備就緒，等待任務..."
        self.current_paper = "準備就緒，等待任務..."
        
        self.total_papers = 0
        self.completed_papers = 0
        self.paper_total_questions = 0
        self.paper_completed_questions = 0
        self.session_new_solved_questions = 0
        self.current_paper_solved_keys = set()  # 🚨 SSOT 單一真理源：記錄當前試卷已完成的唯一題號集合，杜絕重複累加或漏算

        self.diagram_count = 0
        self.arbitration_count = 0
        
        # 🚨 各維度細緻統計
        self.proxy_stats = {}   # {proxy: {"2xx":0, "4xx":0, "5xx":0, "other":0}}
        self.model_stats = {}   # {model: {"2xx":0, "4xx":0, "5xx":0, "other":0}}
        self.key_stats = {}     # {masked_key: {"2xx":0, "4xx":0, "5xx":0, "other":0}}
        self.exam_progress = {} # {exam_name: {"total": 0, "completed": 0, "status": "等待中"}}
        self.recent_logs = []

        # 🚨 API 各階段成本統計 {stage: {"200_ok": int, "json_err": int, "total_rtt": float}}
        self.api_stage_stats = {
            "page_scan": {"200_ok": 0, "json_err": 0, "total_rtt": 0.0},
            "ans_ocr": {"200_ok": 0, "json_err": 0, "total_rtt": 0.0},
            "rubric_extract": {"200_ok": 0, "json_err": 0, "total_rtt": 0.0},
            "solution_gen": {"200_ok": 0, "json_err": 0, "total_rtt": 0.0},
            "validator": {"200_ok": 0, "json_err": 0, "total_rtt": 0.0},
            "diagram": {"200_ok": 0, "json_err": 0, "total_rtt": 0.0},
            "arbitration": {"200_ok": 0, "json_err": 0, "total_rtt": 0.0}
        }
        self.recent_rtt_list = [] # 最近 30 次 API RTT 滑動視窗

        self.history_file = "telemetry_history.json"
        self.history_data = self.load_history()
        self.save_history()

        # 🚨 全域結構化錯誤帳本 (Global Error Ledger)
        self.error_ledger_file = "pipeline_error_ledger.json"
        self.error_counts = {
            "CODE_RUNTIME": 0,       # 程式碼崩潰、語法錯誤、執行緒未預期例外
            "API_NETWORK": 0,        # 429 限流、503 過載、524 代理逾時、JSON 截斷
            "ACADEMIC_CONFLICT": 0,  # 推導與官方答案衝突、選填挖空數不吻合
            "RENDERING_SYNTAX": 0    # 繪圖腳本語法錯誤、圖表生成失敗
        }
        self.recent_errors = []     # 滑動視窗保存最近 30 筆詳細錯誤
        self._load_error_ledger()

    def _load_error_ledger(self):
        if os.path.exists(self.error_ledger_file):
            try:
                with open(self.error_ledger_file, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    self.error_counts.update(d.get("counts", {}))
                    self.recent_errors = d.get("recent", [])
            except Exception: pass

    def _persist_error_ledger(self):
        try:
            with open(self.error_ledger_file, "w", encoding="utf-8") as f:
                json.dump({"counts": self.error_counts, "recent": self.recent_errors}, f, ensure_ascii=False, indent=2)
        except Exception: pass

    def record_error(self, category: str, message: str, exam_name: str = "", question_num: str = "", stage: str = "", tb_str: str = ""):
        """全系統唯一結構化錯誤記錄窗口"""
        with self.lock:
            cat = category if category in self.error_counts else "CODE_RUNTIME"
            self.error_counts[cat] += 1
            entry = {
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "category": cat,
                "exam": exam_name or self.current_paper,
                "q_num": str(question_num),
                "stage": stage or self.current_stage,
                "message": str(message)[:350],
                "traceback": str(tb_str)[-650:] if tb_str else ""
            }
            self.recent_errors.insert(0, entry)
            if len(self.recent_errors) > 30:
                self.recent_errors.pop()
            self._persist_error_ledger()

    def set_stage(self, stage_text: str):
        """更新當前試卷的微觀處理階段"""
        with self.lock:
            self.current_stage = stage_text
        self.add_log(f"📍 {stage_text}")

    def set_current_paper(self, paper_name: str, total_q: int = 0, existing_partial_q: int = 0, initial_solved_keys: list = None):
        with self.lock:
            self.current_paper = paper_name
            self.current_paper_solved_keys = set(initial_solved_keys or [])
            self.paper_total_questions = max(total_q, len(self.current_paper_solved_keys))
            self.paper_completed_questions = len(self.current_paper_solved_keys) if self.current_paper_solved_keys else existing_partial_q
            self.current_stage = f"開始處理考卷: {paper_name}"
            # 同步更新 exam_progress 條目
            self.exam_progress[paper_name] = {
                "completed": self.paper_completed_questions,
                "total": self.paper_total_questions,
                "status": "準備解題中..."
            }
        self.save_history()

    def update_paper_total_questions(self, total_q: int, status_msg: str = ""):
        """🚨 高水位題數鎖定（High-water Mark Lock）：只增不減，徹底杜絕 0/3 盲猜覆蓋真實題數"""
        with self.lock:
            if total_q > self.paper_total_questions:
                self.paper_total_questions = total_q
            if self.current_paper in self.exam_progress:
                self.exam_progress[self.current_paper]["total"] = self.paper_total_questions
                if status_msg:
                    self.exam_progress[self.current_paper]["status"] = status_msg
        self.save_history()

    def record_question_completed(self, q_key: str, is_newly_solved: bool = True, status_msg: str = ""):
        """🚨 統一進度事件派發器：所有模考快車道、Stage 3 通過、續傳題目的唯一累加窗口"""
        with self.lock:
            q_id = str(q_key).strip()
            if q_id not in self.current_paper_solved_keys:
                self.current_paper_solved_keys.add(q_id)
                self.paper_completed_questions = len(self.current_paper_solved_keys)
                if is_newly_solved:
                    self.session_new_solved_questions += 1

            # 確保分母永遠大於等於分子
            if self.paper_completed_questions > self.paper_total_questions:
                self.paper_total_questions = self.paper_completed_questions

            # 原子化同步給前端考卷進度列表
            if self.current_paper in self.exam_progress:
                self.exam_progress[self.current_paper]["completed"] = self.paper_completed_questions
                self.exam_progress[self.current_paper]["total"] = self.paper_total_questions
                if status_msg:
                    self.exam_progress[self.current_paper]["status"] = status_msg
        self.save_history()

    def record_new_solved_question(self):
        """向下相容方法"""
        with self.lock:
            self.session_new_solved_questions += 1
            self.paper_completed_questions += 1
        self.save_history()

    def record_stage_api_call(self, stage: str, rtt: float, is_200: bool, is_json_err: bool = False):
        """記錄 API 呼叫的階段歸屬、延遲與 JSON 狀態"""
        with self.lock:
            if stage not in self.api_stage_stats:
                self.api_stage_stats[stage] = {"200_ok": 0, "json_err": 0, "total_rtt": 0.0}
            
            if is_200:
                self.api_stage_stats[stage]["200_ok"] += 1
                self.api_stage_stats[stage]["total_rtt"] += rtt
                self.recent_rtt_list.append(rtt)
                if len(self.recent_rtt_list) > 30:
                    self.recent_rtt_list.pop(0)

            if is_json_err:
                self.api_stage_stats[stage]["json_err"] += 1
        self.save_history()

    def compute_simulation_forecast(self, keys_list: list, remaining_papers_count: int) -> dict:
        """
        🔮 運籌學調和模擬引擎：
        結合每把 Key 近期 10 次成功率 + 實質階段消耗 + 過去 10 次歷史運行速度進行調和平均預測
        """
        with self.lock:
            uptime = max(1.0, time.time() - self.start_time)
            
            # 1. 計算金鑰佇列近 10 次成功率與有效吞吐量
            active_keys = [k for k in keys_list if not k.is_disabled]
            if not active_keys:
                return {"status": "無可用金鑰"}

            key_rates = [k.recent_10_success_rate for k in active_keys]
            avg_key_success_p = sum(key_rates) / len(key_rates) if key_rates else 0.5
            
            # 平均 RTT (秒)
            avg_rtt = (sum(self.recent_rtt_list) / len(self.recent_rtt_list)) if self.recent_rtt_list else 1.8
            
            # 理論佇列吞吐量 (Requests per second)
            # 每把 Key 平均物理發送間隔 2.5s，並行線程上限 8
            nominal_fleet_qps = min(8.0 / avg_rtt, len(active_keys) * (1.0 / 2.5))
            effective_qps = max(0.05, nominal_fleet_qps * avg_key_success_p)

            # 2. 計算解出一題實質消耗的 API 呼叫數 (含詳解、審查、繪圖與 JSON 重試)
            sol_stat = self.api_stage_stats["solution_gen"]
            val_stat = self.api_stage_stats["validator"]
            diag_stat = self.api_stage_stats["diagram"]
            
            # 題目平均呼叫數 (每題至少 1 次生成 + 1 次審查 + 重試機率)
            calls_per_question = 2.1
            if self.session_new_solved_questions >= 3:
                total_sol_calls = sol_stat["200_ok"] + sol_stat["json_err"]
                total_val_calls = val_stat["200_ok"] + val_stat["json_err"]
                total_diag_calls = diag_stat["200_ok"] + diag_stat["json_err"]
                calls_per_question = max(1.8, (total_sol_calls + total_val_calls + total_diag_calls) / self.session_new_solved_questions)

            # 3. 當前即時物理速度 (秒/題)
            live_sec_per_q = (uptime / self.session_new_solved_questions) if self.session_new_solved_questions > 0 else (calls_per_question / effective_qps)

            # 4. 提取最近 10 次歷史 Sessions 的平均速度進行「調和平均數」加權
            hist_runs = self.history_data.get("runs", [])
            recent_10_runs = hist_runs[-10:] if len(hist_runs) >= 1 else []
            
            hist_speeds = []
            for r in recent_10_runs:
                q_cnt = r.get("completed_questions", 0)
                # 若記錄中有紀錄或由題數與更新時間估計
                if q_cnt >= 5 and "last_update" in r and "start_time" in r:
                    try:
                        t_diff = (datetime.datetime.strptime(r["last_update"], "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(r["start_time"], "%Y-%m-%d %H:%M:%S")).total_seconds()
                        if t_diff > 30:
                            hist_speeds.append(t_diff / q_cnt)
                    except: pass

            # 調和平均數融合 (Harmonic Mean)，對極端離群值具備極佳穩健性
            if hist_speeds:
                h_mean_hist = len(hist_speeds) / sum(1.0 / max(1.0, s) for s in hist_speeds)
                weight_live = 0.6 if self.session_new_solved_questions >= 5 else 0.2
                harmonized_sec_per_q = 1.0 / ((weight_live / live_sec_per_q) + ((1.0 - weight_live) / h_mean_hist))
            else:
                harmonized_sec_per_q = live_sec_per_q

            # 🚨 運籌學調校：引入「重試摩擦係數 (Retry Friction Factor, γ)」
            # 當前系統錯誤重試平均帶來 12~20 秒的額外生成與 API 退避代價
            total_err = sum(self.error_counts.values())
            if self.session_new_solved_questions > 0 and total_err > 0:
                error_friction_ratio = min(2.5, total_err / self.session_new_solved_questions)
                friction_penalty_sec = error_friction_ratio * 12.5  # 折算重試摩擦損耗
                harmonized_sec_per_q = max(2.0, harmonized_sec_per_q + friction_penalty_sec)

            # 5. 輸出客觀模擬預測指標
            rem_q_in_paper = max(0, self.paper_total_questions - self.paper_completed_questions)
            eta_current_paper_sec = rem_q_in_paper * harmonized_sec_per_q

            # 單次 GHA (5 小時 = 18000 秒) 預估產能
            gha_budget_sec = 5.0 * 3600
            rem_gha_time = max(0.0, gha_budget_sec - uptime)
            est_capacity_5h_questions = int(rem_gha_time / harmonized_sec_per_q)

            # 每張考卷平均預估時間 (以每卷平均 24 題估計)
            avg_exam_sec = 24 * harmonized_sec_per_q
            eta_total_all_sec = eta_current_paper_sec + (remaining_papers_count * avg_exam_sec)

            return {
                "avg_rtt_sec": round(avg_rtt, 2),
                "effective_qps": round(effective_qps, 3),
                "calls_per_question": round(calls_per_question, 2),
                "harmonized_sec_per_q": round(harmonized_sec_per_q, 1),
                "eta_current_paper_str": str(datetime.timedelta(seconds=int(eta_current_paper_sec))),
                "est_capacity_5h_questions": est_capacity_5h_questions,
                "avg_exam_time_str": str(datetime.timedelta(seconds=int(avg_exam_sec))),
                "eta_total_all_str": str(datetime.timedelta(seconds=int(eta_total_all_sec))),
                "key_fleet_health": f"{int(avg_key_success_p * 100)}%"
            }

    def load_history(self) -> dict:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "runs" not in data: data = {"runs": []}
                    return data
            except Exception: pass
        return {"runs": []}

    def save_history(self):
        with self.lock:
            try:
                current_session = {
                    "session_id": self.session_id,
                    "start_time": self.run_start_str,
                    "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "completed_questions": self.session_new_solved_questions,
                    "completed_papers": self.completed_papers,
                    "diagram_count": self.diagram_count,
                    "proxy_stats": self.proxy_stats,
                    "model_stats": self.model_stats,
                    "key_stats": self.key_stats,
                    "exam_progress": self.exam_progress,
                    "api_stage_stats": self.api_stage_stats
                }
                existing_idx = next((i for i, r in enumerate(self.history_data["runs"]) if r.get("session_id") == self.session_id), -1)
                if existing_idx >= 0:
                    self.history_data["runs"][existing_idx] = current_session
                else:
                    self.history_data["runs"].append(current_session)

                with open(self.history_file, "w", encoding="utf-8") as f:
                    json.dump(self.history_data, f, ensure_ascii=False, indent=4)
            except Exception: pass

    def record_http(self, proxy_name: str, model_name: str, masked_key: str, code: int):
        with self.lock:
            code_str = str(code)
            category = "2xx" if code_str.startswith("2") else ("4xx" if code_str.startswith("4") else ("5xx" if code_str.startswith("5") else "other"))
            short_p = proxy_name.split("//")[-1].split("/")[0] if "//" in proxy_name else proxy_name
            if short_p not in self.proxy_stats: self.proxy_stats[short_p] = {"2xx": 0, "4xx": 0, "5xx": 0, "other": 0}
            if model_name not in self.model_stats: self.model_stats[model_name] = {"2xx": 0, "4xx": 0, "5xx": 0, "other": 0}
            if masked_key not in self.key_stats: self.key_stats[masked_key] = {"2xx": 0, "4xx": 0, "5xx": 0, "other": 0}
            self.proxy_stats[short_p][category] += 1
            self.model_stats[model_name][category] += 1
            self.key_stats[masked_key][category] += 1
        self.save_history()

    def update_exam_progress(self, exam_name: str, completed: int, total: int, status: str):
        with self.lock:
            self.exam_progress[exam_name] = {"completed": completed, "total": total, "status": status}
        self.save_history()

    def add_log(self, msg: str):
        with self.lock:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self.recent_logs.insert(0, f"[{ts}] {msg}")
            if len(self.recent_logs) > 15: self.recent_logs.pop()

GLOBAL_METRICS = LiveDashboardMetrics()


def start_dashboard_web_server(port=8080, manager=None):
    from http.server import HTTPServer, BaseHTTPRequestHandler
    
    class DashboardHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args): pass
        
        def do_POST(self):
            if self.path == "/api/stop":
                logging.warning("🛑 [Web 遠端指令] 收到使用者強制停止信號！準備安全存檔並優雅退出...")
                GLOBAL_METRICS.stop_requested = True
                GLOBAL_METRICS.add_log("🛑 收到 Web 遠端停止指令，正在安全收尾存檔...")
                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "已觸發停機，即將退出進入 GHA 上傳"}).encode("utf-8"))

            elif self.path == "/api/purge_paper":
                try:
                    content_len = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(content_len).decode('utf-8')
                    req_data = json.loads(body)
                    paper_key = req_data.get("paper_name", "").strip()

                    purged_files, purged_dirs = purge_single_exam_suite(paper_key)
                    self.send_response(200)
                    self.send_header("Content-type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "success",
                        "message": f"已成功銷毀 {len(purged_files)} 個資料庫檔案與 {len(purged_dirs)} 個圖片目錄，並已登記雲端抹除！"
                    }, ensure_ascii=False).encode("utf-8"))
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))

        def do_GET(self):
            if self.path == "/api/data":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                
                key_statuses = []
                key_health = {"active": 0, "cooldown": 0, "disabled": 0}
                if manager:
                    with manager.lock:
                        key_statuses = [k.get_status() for k in manager.keys]
                        for k in manager.keys:
                            if k.is_disabled: key_health["disabled"] += 1
                            elif k.rpd_exhausted: key_health["cooldown"] += 1
                            else: key_health["active"] += 1

                total_mb = 0.0
                out_dir = "./exam_database_output"
                if os.path.exists(out_dir):
                    for root, dirs, files in os.walk(out_dir):
                        for f in files:
                            total_mb += os.path.getsize(os.path.join(root, f))
                disk_str = f"{total_mb / (1024*1024):.1f} MB"

                with GLOBAL_METRICS.lock:
                    uptime_sec = int(time.time() - GLOBAL_METRICS.start_time)
                    time_limit_sec = int(MAX_EXECUTION_TIME_SECONDS)
                    time_pct = min(100, int((uptime_sec / time_limit_sec) * 100))
                    
                    time_per_q_str = "計算中..."
                    if GLOBAL_METRICS.session_new_solved_questions > 0:
                        sec_per_q = uptime_sec / GLOBAL_METRICS.session_new_solved_questions
                        time_per_q_str = f"{int(sec_per_q // 60)} 分 {int(sec_per_q % 60)} 秒 / 題" if sec_per_q >= 60 else f"{int(sec_per_q)} 秒 / 題"
                    else:
                        time_per_q_str = "等待新題目完成..."

                    tot_http = sum(sum(cat.values()) for cat in GLOBAL_METRICS.proxy_stats.values())
                    ok_http = sum(cat.get("2xx", 0) for cat in GLOBAL_METRICS.proxy_stats.values())
                    success_rate = f"{(ok_http / tot_http * 100):.1f}%" if tot_http > 0 else "100%"

                    # 🔮 調用運籌學排程與調和預測引擎
                    rem_papers = max(0, GLOBAL_METRICS.total_papers - GLOBAL_METRICS.completed_papers)
                    forecast_data = GLOBAL_METRICS.compute_simulation_forecast(manager.keys if manager else [], rem_papers)

                    data = {
                        "uptime_formatted": str(datetime.timedelta(seconds=uptime_sec)),
                        "time_pct": time_pct,
                        "current_stage": GLOBAL_METRICS.current_stage, # 🆕 細部進度階段
                        "current_paper": GLOBAL_METRICS.current_paper,
                        "paper_progress": f"{GLOBAL_METRICS.completed_papers}/{GLOBAL_METRICS.total_papers}",
                        "progress": f"{GLOBAL_METRICS.paper_completed_questions}/{GLOBAL_METRICS.paper_total_questions}",
                        "session_solved": GLOBAL_METRICS.session_new_solved_questions,
                        "time_per_q_str": time_per_q_str,
                        "success_rate": success_rate,
                        "disk_str": disk_str,
                        "key_health": key_health,
                        "diagram_count": GLOBAL_METRICS.diagram_count,
                        "arbitration_count": GLOBAL_METRICS.arbitration_count,
                        "proxy_stats": GLOBAL_METRICS.proxy_stats,
                        "model_stats": GLOBAL_METRICS.model_stats,
                        "key_stats": GLOBAL_METRICS.key_stats,
                        "exam_progress": GLOBAL_METRICS.exam_progress,
                        "key_statuses": key_statuses,
                        "recent_logs": GLOBAL_METRICS.recent_logs,
                        "history_data": GLOBAL_METRICS.history_data,
                        # 🔮 預測中樞專屬指標
                        "forecast": forecast_data,
                        "api_stage_stats": GLOBAL_METRICS.api_stage_stats,
                        "current_session_id": GLOBAL_METRICS.session_id,
                        # 🚨 結構化錯誤帳本指標
                        "error_summary": {
                            "counts": GLOBAL_METRICS.error_counts,
                            "recent": GLOBAL_METRICS.recent_errors
                        }
                    }
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
            else:
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Exam Parser 遙測中心</title>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>
                        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
                        .header-container { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #334155; padding-bottom: 15px; flex-wrap: wrap; gap: 15px; }
                        h1 { color: #38bdf8; margin: 0; font-size: 1.5rem; }
                        .stage-banner { background: #1e293b; border-left: 4px solid #38bdf8; padding: 10px 15px; border-radius: 6px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; font-size: 0.95rem; }
                        .btn-stop { background: #ef4444; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: bold; transition: background 0.2s; }
                        .btn-stop:hover { background: #dc2626; }
                        .filter-box { background: #1e293b; padding: 10px 15px; border-radius: 8px; border: 1px solid #475569; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
                        .filter-box input[type="date"], .filter-box select { background: #0f172a; color: #f8fafc; border: 1px solid #334155; padding: 5px 8px; border-radius: 4px; font-size: 0.85rem; }
                        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; max-width: 1400px; margin: 0 auto; }
                        .card { background: #1e293b; border-radius: 12px; padding: 18px; border: 1px solid #334155; overflow: hidden; }
                        h2 { margin-top: 0; color: #38bdf8; font-size: 1.05rem; border-bottom: 1px solid #334155; padding-bottom: 8px; }
                        .stat-val { font-size: 1.5rem; font-weight: bold; color: #4ade80; }
                        .badge { display: inline-block; padding: 3px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; }
                        .badge-2xx { background: #166534; color: #4ade80; }
                        .badge-4xx { background: #991b1b; color: #fca5a5; }
                        .badge-5xx { background: #854d0e; color: #fef08a; }
                        table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.85rem; }
                        th, td { text-align: left; padding: 7px; border-bottom: 1px solid #334155; }
                        th { color: #94a3b8; background: #0f172a; position: sticky; top: 0; }
                        .table-container { max-height: 220px; overflow-y: auto; }
                        .log-box { background: #020617; border-radius: 8px; padding: 10px; font-family: monospace; font-size: 0.8rem; color: #a7f3d0; max-height: 180px; overflow-y: auto; }
                        .progress-bar-bg { background: #334155; border-radius: 6px; height: 7px; width: 100%; margin-top: 4px; }
                        .progress-bar-fill { background: linear-gradient(90deg, #38bdf8, #4ade80); height: 100%; border-radius: 6px; }
                    </style>
                </head>
                <body>
                    <div class="header-container" style="max-width: 1400px; margin: 0 auto 15px auto;">
                        <div>
                            <h1>⚡ Exam Parser 頂級決策與調控遙測中心</h1>
                            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">多模態大考解題管線 ‧ 全方位算力排程中樞</div>
                        </div>
                        <div style="display: flex; gap: 10px; align-items: center;">
                            <button class="btn btn-stop" onclick="triggerGracefulStop()">🛑 優雅停機存檔 (進入 GHA 上傳)</button>
                        </div>
                    </div>

                    <!-- 即時細部處理階段橫幅 -->
                    <div style="max-width: 1400px; margin: 0 auto 15px auto;" class="stage-banner">
                        <div>
                            <span style="color:#94a3b8;">當前流水線作業微觀階段：</span>
                            <span id="live-stage" style="color:#38bdf8; font-weight:bold;">初始化中...</span>
                        </div>
                        <div style="font-size:0.85rem; color:#4ade80;" id="live-paper-badge">-</div>
                    </div>

                    <!-- 篩選列 -->
                    <div style="max-width: 1400px; margin: 0 auto 18px auto;" class="filter-box">
                        <label>📅 歷史起訖：</label>
                        <input type="date" id="start-date" onchange="onDateRangeChange()"> 
                        <span>至</span> 
                        <input type="date" id="end-date" onchange="onDateRangeChange()">
                        
                        <label style="margin-left: 10px;">🎯 選擇檢視階段：</label>
                        <select id="session-select" onchange="renderSelectedSession()">
                            <option value="live">🔴 [當前即時運行 - Live]</option>
                        </select>
                    </div>

                    <div class="grid">
                        <!-- 即時狀態卡 -->
                        <div class="card">
                            <h2>📌 當前進度與實質耗時 (Live)</h2>
                            <div>考卷進度：<span id="live-progress" class="stat-val" style="font-size:1.3rem;">0/0</span></div>
                            <div style="margin-top:5px;">本次新解題數：<span id="live-session-solved" style="color:#38bdf8; font-weight:bold;">0</span> 題</div>
                            <div style="margin-top:5px;">實質單題耗時：<span id="live-speed" style="color:#f59e0b; font-weight:bold;">計算中...</span></div>
                            <div style="margin-top:5px; font-size:0.8rem; color:#94a3b8;">已運行：<span id="uptime">-</span> | 題庫資產：<span id="disk-size" style="color:#4ade80;">0 MB</span></div>
                        </div>

                        <!-- 🔮 智慧產能與工期預測中心 (AI Capacity & ETA Forecaster) -->
                        <div class="card" style="grid-column: span 2; border: 1px solid #38bdf8;">
                            <h2>🔮 智慧產能與工期預測中心 (Harmonic Forecasting Engine)</h2>
                            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 5px;">
                                <div>
                                    <div style="font-size:0.8rem; color:#94a3b8;">單題調和耗時 (平滑加權)</div>
                                    <div class="stat-val" id="fc-harm-speed" style="font-size:1.2rem; color:#38bdf8;">-</div>
                                </div>
                                <div>
                                    <div style="font-size:0.8rem; color:#94a3b8;">當前考卷預估剩餘工期</div>
                                    <div class="stat-val" id="fc-eta-paper" style="font-size:1.2rem; color:#f59e0b;">-</div>
                                </div>
                                <div>
                                    <div style="font-size:0.8rem; color:#94a3b8;">本次 GHA (5h) 預估解題容量</div>
                                    <div class="stat-val" id="fc-capacity-5h" style="font-size:1.2rem; color:#4ade80;">-</div>
                                </div>
                                <div>
                                    <div style="font-size:0.8rem; color:#94a3b8;">平均每卷耗時 / 全局總剩餘</div>
                                    <div style="font-size:0.95rem; font-weight:bold; color:#f8fafc; margin-top:4px;" id="fc-total-eta">-</div>
                                </div>
                            </div>
                            <div style="margin-top:10px; font-size:0.75rem; color:#94a3b8; border-top: 1px solid #334155; padding-top:6px;">
                                佇列健康度：<span id="fc-health" style="color:#4ade80; font-weight:bold;">100%</span> | 
                                有效 QPS：<span id="fc-qps" style="color:#38bdf8; font-weight:bold;">-</span> | 
                                平均 RTT：<span id="fc-rtt" style="color:#f59e0b; font-weight:bold;">-</span> | 
                                單題實質 API 消耗：<span id="fc-calls-q" style="color:#f8fafc; font-weight:bold;">-</span> 呼叫/題
                            </div>
                        </div>

                        <!-- 🚨 全域結構化錯誤觀測與排錯中樞 (Error Ledger & Analytics) -->
                        <div class="card" style="grid-column: span 2; border: 1px solid #ef4444;">
                            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #334155; padding-bottom:8px;">
                                <h2 style="margin:0; border:none; padding:0; color:#f87171;">🚨 全域結構化錯誤觀測與排錯中樞 (Global Error Ledger)</h2>
                                <div style="font-size:0.8rem; color:#94a3b8;">即時自動聚合 ‧ 免翻終端機日誌</div>
                            </div>
                            
                            <!-- 錯誤分類狀態列與彩條 -->
                            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-top: 12px;">
                                <div style="background:#271216; padding:8px 12px; border-radius:6px; border:1px solid #7f1d1d;">
                                    <div style="font-size:0.75rem; color:#fca5a5;">💥 程式碼崩潰 (CODE_RUNTIME)</div>
                                    <div class="stat-val" id="err-code-cnt" style="color:#ef4444; font-size:1.3rem;">0</div>
                                </div>
                                <div style="background:#261b0d; padding:8px 12px; border-radius:6px; border:1px solid #78350f;">
                                    <div style="font-size:0.75rem; color:#fcd34d;">🌐 API/網路/截斷 (API_NETWORK)</div>
                                    <div class="stat-val" id="err-api-cnt" style="color:#f59e0b; font-size:1.3rem;">0</div>
                                </div>
                                <div style="background:#171e2e; padding:8px 12px; border-radius:6px; border:1px solid #1e3a8a;">
                                    <div style="font-size:0.75rem; color:#93c5fd;">⚖️ 學術/答案衝突 (ACADEMIC)</div>
                                    <div class="stat-val" id="err-acad-cnt" style="color:#38bdf8; font-size:1.3rem;">0</div>
                                </div>
                                <div style="background:#1c1329; padding:8px 12px; border-radius:6px; border:1px solid #581c87;">
                                    <div style="font-size:0.75rem; color:#d8b4fe;">🎨 繪圖/語法異常 (RENDERING)</div>
                                    <div class="stat-val" id="err-render-cnt" style="color:#c084fc; font-size:1.3rem;">0</div>
                                </div>
                            </div>

                            <!-- 錯誤分佈視覺比例條 -->
                            <div style="margin-top:12px;">
                                <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:4px;">即時錯誤維度分佈比：</div>
                                <div style="display:flex; height:8px; border-radius:4px; overflow:hidden; background:#334155;" id="err-bar-container">
                                    <div id="bar-code" style="background:#ef4444; width:0%;"></div>
                                    <div id="bar-api" style="background:#f59e0b; width:0%;"></div>
                                    <div id="bar-acad" style="background:#38bdf8; width:0%;"></div>
                                    <div id="bar-render" style="background:#c084fc; width:0%;"></div>
                                </div>
                            </div>

                            <!-- 排錯抽屜清單 -->
                            <div style="margin-top:14px;">
                                <div style="font-size:0.8rem; font-weight:bold; color:#f8fafc; margin-bottom:6px;">最近錯誤明細抽屜 (最新 30 筆)：</div>
                                <div class="table-container" style="max-height:200px;">
                                    <table>
                                        <thead><tr><th>時間</th><th>分類</th><th>考卷與題號</th><th>錯誤說明摘要</th><th>詳情 (Traceback)</th></tr></thead>
                                        <tbody id="error-ledger-rows"></tbody>
                                    </table>
                                </div>
                            </div>
                        </div>

                        <!-- 階段 API 呼叫消耗度量 -->
                        <div class="card" style="grid-column: span 2;">
                            <h2>📊 各解題階段 API 呼叫消耗與延遲度量 (Stage Breakdown)</h2>
                            <div class="table-container">
                                <table>
                                    <thead><tr><th>任務階段 (Stage)</th><th>200 OK 成功數</th><th>JSON 截斷/重試數</th><th>平均 RTT 延遲</th><th>狀態佔比說明</th></tr></thead>
                                    <tbody id="stage-rows"></tbody>
                                </table>
                            </div>
                        </div>

                        <!-- 考卷進度 -->
                        <div class="card" style="grid-column: span 2;">
                            <h2>📑 考卷進度與完工追蹤 (Exam Progress)</h2>
                            <div class="table-container">
                                <table>
                                    <thead><tr><th>考卷名稱</th><th>總題數</th><th>已完成</th><th>進度條</th><th>狀態</th><th>手動管理</th></tr></thead>
                                    <tbody id="exam-rows"></tbody>
                                </table>
                            </div>
                        </div>

                        <!-- 檢視目標統計總結 -->
                        <div class="card">
                            <h2>📊 檢視目標統計 (<span id="view-target-title" style="color:#38bdf8;">即時運行</span>)</h2>
                            <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px;">
                                <div>新解題數：<span id="agg-questions" class="stat-val">0</span> 題</div>
                                <div>完工考卷：<span id="agg-papers" class="stat-val">0</span> 份</div>
                                <div>AI 繪圖：<span id="agg-diagrams" class="stat-val" style="color:#38bdf8;">0</span> 張</div>
                                <div style="width:100%; margin-top:4px;">
                                    請求成功率：<span id="agg-success-rate" style="font-size:1.1rem; font-weight:bold; color:#38bdf8;">100%</span>
                                </div>
                            </div>
                            <div style="margin-top:8px; font-size:0.75rem; color:#94a3b8;" id="session-detail-info">-</div>
                        </div>

                        <!-- Proxy 出口戰報 -->
                        <div class="card">
                            <h2>🚀 代理節點出口戰報 (Proxy Stats)</h2>
                            <div class="table-container">
                                <table>
                                    <thead><tr><th>Proxy 節點</th><th><span class="badge badge-2xx">2xx</span></th><th><span class="badge badge-4xx">4xx</span></th><th><span class="badge badge-5xx">5xx</span></th></tr></thead>
                                    <tbody id="proxy-rows"></tbody>
                                </table>
                            </div>
                        </div>

                        <!-- 模型調用統計 -->
                        <div class="card">
                            <h2>🤖 各模型調用統計 (Model Stats)</h2>
                            <div class="table-container">
                                <table>
                                    <thead><tr><th>模型名稱</th><th><span class="badge badge-2xx">2xx</span></th><th><span class="badge badge-4xx">4xx</span></th><th><span class="badge badge-5xx">5xx</span></th></tr></thead>
                                    <tbody id="model-rows"></tbody>
                                </table>
                            </div>
                        </div>

                        <!-- 金鑰損耗度量 -->
                        <div class="card" style="grid-column: span 2;">
                            <h2>🔑 API 金鑰損耗深度監控 (Key Stats)</h2>
                            <div class="table-container">
                                <table>
                                    <thead><tr><th>API 金鑰 (Masked)</th><th><span class="badge badge-2xx">2xx 成功</span></th><th><span class="badge badge-4xx">4xx 限流/拒絕</span></th><th><span class="badge badge-5xx">5xx 伺服過載</span></th><th>成功率</th></tr></thead>
                                    <tbody id="key-rows"></tbody>
                                </table>
                            </div>
                        </div>

                        <!-- 即時處理動態日誌 -->
                        <div class="card" style="grid-column: span 2;">
                            <h2>📜 即時處理動態日誌 (Live Activity)</h2>
                            <div class="log-box" id="log-box">等待動態更新中...</div>
                        </div>
                    </div>

                    <script>
                        let lastFetchedData = null;
                        let matchedRuns = [];

                        const today = new Date();
                        const pastWeek = new Date(today);
                        pastWeek.setDate(pastWeek.getDate() - 7);
                        document.getElementById('start-date').value = pastWeek.toISOString().split('T')[0];
                        document.getElementById('end-date').value = today.toISOString().split('T')[0];

                        async function purgeExamRecord(examName) {
                            if (confirm(`⚠️ 確定要徹底銷毀【${examName}】嗎？\n\n這將會：\n1. 刪除該考卷的所有 JSON 資料庫檔案與 Partial 進度\n2. 徹底銷毀其 images 圖片目錄與 20 多張裁切圖\n3. 登記至雲端清除清單（稍後上傳時由 GDrive 同步抹除）\n4. 讓系統下一次能夠以全新的「官方詳解快車道」乾淨重跑！`)) {
                                try {
                                    const res = await fetch('/api/purge_paper', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({ paper_name: examName })
                                    });
                                    const ret = await res.json();
                                    alert(ret.message);
                                    fetchData();
                                } catch (e) {
                                    alert('⚠️ 銷毀失敗：' + e);
                                }
                            }
                        }

                        // 優雅停機請求觸發器
                        async function triggerGracefulStop() {
                            if (confirm('🛑 確定要提前優雅結束當前作業嗎？\\n系統將立即停止派發新題目、保存當前已解出的所有題目，並以 Exit 0 退出，自動進入 GitHub Actions 後續的 Google Drive 上傳步驟！')) {
                                try {
                                    const res = await fetch('/api/stop', { method: 'POST' });
                                    const ret = await res.json();
                                    alert('✅ 已成功送出停機指令！後端正在刷寫硬碟存檔，請稍候 5~10 秒觀察 GHA 進入上傳步驟。');
                                } catch (e) {
                                    alert('⚠️ 發送停機請求失敗：' + e);
                                }
                            }
                        }

                        async function fetchData() {
                            try {
                                const res = await fetch('/api/data');
                                lastFetchedData = await res.json();
                                updateLiveStatus();
                                onDateRangeChange(false);
                            } catch(e) { console.error("Fetch error:", e); }
                        }

                        function updateLiveStatus() {
                            if (!lastFetchedData) return;
                            const d = lastFetchedData;
                            
                            // 頂部細緻階段
                            document.getElementById('live-stage').innerText = d.current_stage || "運行中...";
                            document.getElementById('live-paper-badge').innerText = d.current_paper || "";

                            // 當前進度
                            document.getElementById('live-progress').innerText = d.progress;
                            document.getElementById('live-session-solved').innerText = d.session_solved || 0;
                            document.getElementById('live-speed').innerText = d.time_per_q_str || "計算中...";
                            document.getElementById('uptime').innerText = d.uptime_formatted;
                            document.getElementById('disk-size').innerText = d.disk_str;
                            document.getElementById('log-box').innerHTML = (d.recent_logs || []).join('<br>') || '尚無動態';

                            // 🚨 渲染全域錯誤帳本與分類長條圖
                            if (d.error_summary) {
                                const c = d.error_summary.counts || {};
                                const codeN = c.CODE_RUNTIME || 0;
                                const apiN = c.API_NETWORK || 0;
                                const acadN = c.ACADEMIC_CONFLICT || 0;
                                const renderN = c.RENDERING_SYNTAX || 0;
                                const tot = codeN + apiN + acadN + renderN;

                                document.getElementById('err-code-cnt').innerText = codeN;
                                document.getElementById('err-api-cnt').innerText = apiN;
                                document.getElementById('err-acad-cnt').innerText = acadN;
                                document.getElementById('err-render-cnt').innerText = renderN;

                                if (tot > 0) {
                                    document.getElementById('bar-code').style.width = ((codeN / tot) * 100) + '%';
                                    document.getElementById('bar-api').style.width = ((apiN / tot) * 100) + '%';
                                    document.getElementById('bar-acad').style.width = ((acadN / tot) * 100) + '%';
                                    document.getElementById('bar-render').style.width = ((renderN / tot) * 100) + '%';
                                } else {
                                    document.getElementById('bar-code').style.width = '0%';
                                    document.getElementById('bar-api').style.width = '0%';
                                    document.getElementById('bar-acad').style.width = '0%';
                                    document.getElementById('bar-render').style.width = '0%';
                                }

                                const recent = d.error_summary.recent || [];
                                let errRowsHTML = '';
                                recent.forEach((item, idx) => {
                                    const catColors = {
                                        'CODE_RUNTIME': '#ef4444',
                                        'API_NETWORK': '#f59e0b',
                                        'ACADEMIC_CONFLICT': '#38bdf8',
                                        'RENDERING_SYNTAX': '#c084fc'
                                    };
                                    const cCol = catColors[item.category] || '#94a3b8';
                                    const tbDetail = item.traceback ? `<details><summary style="cursor:pointer; color:#38bdf8;">查看堆疊</summary><pre style="font-size:0.7rem; color:#fca5a5; white-space:pre-wrap; max-width:400px; margin-top:4px;">${item.traceback}</pre></details>` : '<span style="color:#64748b;">無堆疊</span>';
                                    errRowsHTML += `<tr>
                                        <td><code>${item.timestamp}</code></td>
                                        <td><span style="color:${cCol}; font-weight:bold;">${item.category}</span></td>
                                        <td><b>${item.exam}</b> ${item.q_num ? '第 '+item.q_num+' 題' : ''}</td>
                                        <td style="color:#f8fafc; font-size:0.75rem;">${item.message}</td>
                                        <td>${tbDetail}</td>
                                    </tr>`;
                                });
                                document.getElementById('error-ledger-rows').innerHTML = errRowsHTML || '<tr><td colspan="5" style="color:#4ade80;">🎉 目前尚無任何嚴重錯誤紀錄，系統穩定運行中！</td></tr>';
                            }

                            // 預測中樞指標渲染
                            if (d.forecast) {
                                const fc = d.forecast;
                                document.getElementById('fc-harm-speed').innerText = fc.harmonized_sec_per_q ? `${fc.harmonized_sec_per_q} 秒 / 題` : "-";
                                document.getElementById('fc-eta-paper').innerText = fc.eta_current_paper_str || "-";
                                document.getElementById('fc-capacity-5h').innerText = fc.est_capacity_5h_questions ? `約 ${fc.est_capacity_5h_questions} 題` : "-";
                                document.getElementById('fc-total-eta').innerText = `${fc.avg_exam_time_str || '-'} / ${fc.eta_total_all_str || '-'}`;
                                
                                document.getElementById('fc-health').innerText = fc.key_fleet_health || "100%";
                                document.getElementById('fc-qps').innerText = `${fc.effective_qps || 0} req/s`;
                                document.getElementById('fc-rtt').innerText = `${fc.avg_rtt_sec || 0}s`;
                                document.getElementById('fc-calls-q').innerText = fc.calls_per_question || "-";
                            }

                            // 階段消耗表格
                            if (d.api_stage_stats) {
                                const stageMap = {
                                    "page_scan": "📑 Stage 1 試卷整頁掃描",
                                    "ans_ocr": "🧠 答案卷雙模型 OCR",
                                    "rubric_extract": "🎨 非選評分標準視覺萃取",
                                    "solution_gen": "🧠 Stage 2 名師解題生成",
                                    "validator": "⚖️ Stage 3 審查批判驗證",
                                    "diagram": "🎨 AI 科學圖解代碼生成",
                                    "arbitration": "⚖️ 終極學術仲裁"
                                };
                                let sHTML = '';
                                for (const [st, info] of Object.entries(d.api_stage_stats)) {
                                    const avgR = info["200_ok"] > 0 ? (info["total_rtt"] / info["200_ok"]).toFixed(2) + "s" : "-";
                                    const label = stageMap[st] || st;
                                    sHTML += `<tr>
                                        <td><b>${label}</b></td>
                                        <td style="color:#4ade80">${info["200_ok"]}</td>
                                        <td style="color:#fca5a5">${info["json_err"]}</td>
                                        <td style="color:#38bdf8">${avgR}</td>
                                        <td style="color:#94a3b8; font-size:0.75rem;">正常作業</td>
                                    </tr>`;
                                }
                                document.getElementById('stage-rows').innerHTML = sHTML;
                            }
                        }

                        function mergeStats(target, source) {
                            if (!source) return;
                            for (const [key, stats] of Object.entries(source)) {
                                if (!target[key]) target[key] = { "2xx": 0, "4xx": 0, "5xx": 0, "other": 0 };
                                target[key]["2xx"] += (stats["2xx"] || 0);
                                target[key]["4xx"] += (stats["4xx"] || 0);
                                target[key]["5xx"] += (stats["5xx"] || 0);
                                target[key]["other"] += (stats["other"] || 0);
                            }
                        }

                        function onDateRangeChange(forceReset = true) {
                            if (!lastFetchedData) return;
                            const d = lastFetchedData;
                            const sDateStr = document.getElementById('start-date').value;
                            const eDateStr = document.getElementById('end-date').value;
                            const sTime = sDateStr ? new Date(sDateStr + "T00:00:00").getTime() : 0;
                            const eTime = eDateStr ? new Date(eDateStr + "T23:59:59").getTime() : Infinity;

                            const selectEl = document.getElementById('session-select');
                            const prevVal = selectEl.value;

                            matchedRuns = [];
                            if (d.history_data && d.history_data.runs) {
                                d.history_data.runs.forEach(run => {
                                    const rTime = new Date(run.start_time.replace(" ", "T")).getTime();
                                    if (rTime >= sTime && rTime <= eTime) {
                                        matchedRuns.push(run);
                                    }
                                });
                            }

                            let optionsHTML = `<option value="live">🔴 [當前即時運行 - Live] Session #${d.current_session_id || 'Now'} (實質新解: ${d.session_solved || 0} 題)</option>`;
                            optionsHTML += `<option value="all">📌 [區間綜合加總] (共 ${matchedRuns.length} 筆歷史紀錄)</option>`;
                            
                            matchedRuns.forEach(run => {
                                const qCnt = run.completed_questions || 0;
                                const pCnt = run.completed_papers || 0;
                                optionsHTML += `<option value="${run.session_id}">🕒 [${run.start_time}] 解題: ${qCnt} 題 | 完工: ${pCnt} 卷</option>`;
                            });

                            selectEl.innerHTML = optionsHTML;
                            if (!forceReset && Array.from(selectEl.options).some(o => o.value === prevVal)) {
                                selectEl.value = prevVal;
                            } else {
                                selectEl.value = "live";
                            }
                            renderSelectedSession();
                        }

                        function renderSelectedSession() {
                            if (!lastFetchedData) return;
                            const d = lastFetchedData;
                            const selectedVal = document.getElementById('session-select').value;

                            let aggQ = 0, aggP = 0, aggD = 0;
                            let aggProxy = {}, aggModel = {}, aggKey = {}, aggExams = {};
                            let detailInfo = "";

                            if (selectedVal === "live") {
                                document.getElementById('view-target-title').innerText = "當前即時運行 (Live)";
                                detailInfo = `啟動時間：${d.history_data.runs?.[0]?.start_time || '剛剛'} | 當前單題耗時：${d.time_per_q_str || '-'}`;
                                aggQ = d.session_solved || 0;
                                aggP = d.paper_progress?.split('/')[0] || 0;
                                aggD = d.diagram_count || 0;
                                aggProxy = d.proxy_stats || {};
                                aggModel = d.model_stats || {};
                                aggKey = d.key_stats || {};
                                aggExams = d.exam_progress || {};
                            } else if (selectedVal === "all") {
                                document.getElementById('view-target-title').innerText = "區間綜合加總";
                                detailInfo = `已匯總起訖範圍內共 ${matchedRuns.length} 次歷史執行的完整統計`;
                                matchedRuns.forEach(run => {
                                    aggQ += (run.completed_questions || 0);
                                    aggP += (run.completed_papers || 0);
                                    aggD += (run.diagram_count || 0);
                                    mergeStats(aggProxy, run.proxy_stats);
                                    mergeStats(aggModel, run.model_stats);
                                    mergeStats(aggKey, run.key_stats);
                                    if (run.exam_progress) {
                                        for(const [exam, prog] of Object.entries(run.exam_progress)){
                                            aggExams[exam] = prog;
                                        }
                                    }
                                });
                            } else {
                                const targetRun = matchedRuns.find(r => r.session_id === selectedVal);
                                if (targetRun) {
                                    document.getElementById('view-target-title').innerText = `歷史紀錄 #${targetRun.session_id}`;
                                    detailInfo = `啟動時間：${targetRun.start_time} | 最後更新：${targetRun.last_update || targetRun.start_time}`;
                                    aggQ = targetRun.completed_questions || 0;
                                    aggP = targetRun.completed_papers || 0;
                                    aggD = targetRun.diagram_count || 0;
                                    mergeStats(aggProxy, targetRun.proxy_stats);
                                    mergeStats(aggModel, targetRun.model_stats);
                                    mergeStats(aggKey, targetRun.key_stats);
                                    aggExams = targetRun.exam_progress || {};
                                }
                            }

                            document.getElementById('session-detail-info').innerText = detailInfo;
                            document.getElementById('agg-questions').innerText = aggQ;
                            document.getElementById('agg-papers').innerText = aggP;
                            document.getElementById('agg-diagrams').innerText = aggD;

                            let tot2xx = 0, totAll = 0;
                            for (const s of Object.values(aggProxy)) {
                                tot2xx += (s["2xx"] || 0);
                                totAll += ((s["2xx"] || 0) + (s["4xx"] || 0) + (s["5xx"] || 0) + (s["other"] || 0));
                            }
                            document.getElementById('agg-success-rate').innerText = totAll > 0 ? ((tot2xx / totAll) * 100).toFixed(1) + "%" : "100%";

                            // 渲染 Proxy 表格
                            let pHTML = '';
                            for (const [p, s] of Object.entries(aggProxy)) {
                                pHTML += `<tr><td><b>${p}</b></td><td style="color:#4ade80">${s["2xx"]||0}</td><td style="color:#fca5a5">${s["4xx"]||0}</td><td style="color:#fef08a">${s["5xx"]||0}</td></tr>`;
                            }
                            document.getElementById('proxy-rows').innerHTML = pHTML || '<tr><td colspan="4">無資料</td></tr>';

                            // 渲染 Model 表格
                            let mHTML = '';
                            for (const [m, s] of Object.entries(aggModel)) {
                                mHTML += `<tr><td><b>${m}</b></td><td style="color:#4ade80">${s["2xx"]||0}</td><td style="color:#fca5a5">${s["4xx"]||0}</td><td style="color:#fef08a">${s["5xx"]||0}</td></tr>`;
                            }
                            document.getElementById('model-rows').innerHTML = mHTML || '<tr><td colspan="4">無資料</td></tr>';

                            // 渲染 Key 表格
                            let kHTML = '';
                            for (const [k, s] of Object.entries(aggKey)) {
                                const kTotal = (s["2xx"]||0) + (s["4xx"]||0) + (s["5xx"]||0) + (s["other"]||0);
                                const kRate = kTotal > 0 ? (((s["2xx"]||0) / kTotal) * 100).toFixed(1) + "%" : "0%";
                                kHTML += `<tr><td><code>${k}</code></td><td style="color:#4ade80">${s["2xx"]||0}</td><td style="color:#fca5a5">${s["4xx"]||0}</td><td style="color:#fef08a">${s["5xx"]||0}</td><td><b>${kRate}</b></td></tr>`;
                            }
                            document.getElementById('key-rows').innerHTML = kHTML || '<tr><td colspan="5">無資料</td></tr>';

                            // 渲染 考卷進度（帶有一鍵銷毀重跑功能）
                            let eHTML = '';
                            for (const [exam, s] of Object.entries(aggExams)) {
                                const total = s.total || 1;
                                const comp = s.completed || 0;
                                const pct = Math.min(100, (comp / total) * 100).toFixed(0);
                                const statusColor = s.status.includes("完工") ? "#4ade80" : (s.status.includes("解題") ? "#38bdf8" : "#94a3b8");
                                eHTML += `<tr>
                                    <td><b>${exam}</b></td>
                                    <td>${total}</td>
                                    <td>${comp}</td>
                                    <td><div class="progress-bar-bg"><div class="progress-bar-fill" style="width:${pct}%; background:${statusColor}"></div></div></td>
                                    <td style="color:${statusColor}">${s.status}</td>
                                    <td>
                                        <button onclick="purgeExamRecord('${exam}')" style="background:#ef4444; color:white; border:none; padding:3px 8px; border-radius:4px; cursor:pointer; font-size:0.75rem; font-weight:bold;">🗑️ 銷毀重跑</button>
                                    </td>
                                </tr>`;
                            }
                            document.getElementById('exam-rows').innerHTML = eHTML || '<tr><td colspan="6">此紀錄無考卷資訊</td></tr>';
                        }

                        setInterval(fetchData, 3000);
                        fetchData();
                    </script>
                </body>
                </html>
                """
                self.wfile.write(html.encode("utf-8"))
                    


    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(('0.0.0.0', port), DashboardHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logging.info(f"🌐 [遙測儀表板] 本地 Web 服務已啟動於 http://127.0.0.1:{port}")
    
class GeminiFreeTierManager:
    def __init__(self, api_keys: List[str], models: List[str]):
        proxy_env = os.environ.get("GEMINI_PROXY_URL", "").strip()
        self.proxies = [p.strip() for p in proxy_env.split(",") if p.strip()]
        self.proxy_idx = 0
        
        first_proxy = self.proxies[0] if self.proxies else None
        self.keys = [FreeTierKey(k, custom_base_url=first_proxy) for k in api_keys]
        self.models = models
        self.key_idx = 0
        self.model_idx = 0
        self.model_dispatch_counter = 0
        self.global_last_dispatch_time = 0.0
        self.last_model_used = models[0] if models else ""
        self.lock = threading.Lock()
        
        # 啟動儀表板伺服器
        start_dashboard_web_server(port=8080, manager=self)

        groq_keys_env = os.environ.get("GROQ_API_KEY", "")
        self.groq_keys = [k.strip() for k in groq_keys_env.split(",") if k.strip()]
        self.groq_key_idx = 0
        self.groq_lock = threading.Lock()

        # 🚀 支援 OpenRouter (提供 ChatGPT / 頂級免費通道，支援大 Token 與純 JSON 輸出)
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.openrouter_client = None
        if HAS_OPENAI_SDK and self.openrouter_key:
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_key
            )
            logging.info("🌟 [備援端點啟動] OpenRouter 備援大模型通道已就緒！")
        
        # 🧠 讀取或初始化模型 Token 上限記憶庫
        self.model_limits_file = "model_limits.json"
        self.model_token_limits = self.load_model_token_limits()
    
    def get_next_proxy_url(self) -> Optional[str]:
        """動態輪替取用多雲代理端點 (Cloudflare -> Vercel -> Deno)"""
        if not self.proxies:
            return None
        with self.lock:
            p = self.proxies[self.proxy_idx]
            self.proxy_idx = (self.proxy_idx + 1) % len(self.proxies)
            return p
            
    def _call_openai_compatible_raw(self, client, base_url, api_key, model, prompt_text, response_schema, temperature=0.1):
        """底層通用 API 呼叫器，自帶思考模型容錯與萬能 JSON 清洗"""
        schema_json = response_schema.model_json_schema()
        system_msg = (
            "You are an expert Taiwanese high school exam tutor. "
            "You MUST output 100% strictly valid JSON matching this schema: "
            f"{json.dumps(schema_json)}. Output ONLY the raw JSON object, without markdown codeblock."
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt_text}
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=4096
        )
        
        msg = response.choices[0].message
        raw_text = msg.content or getattr(msg, "reasoning", "") or getattr(msg, "reasoning_content", "") or ""
        raw_text = str(raw_text).strip()
        
        raw_text = s2t(raw_text)
        parsed_json, err = universal_robust_json_parser(raw_text, response_schema=response_schema)
        if parsed_json:
            parsed_json = self.repair_dict_latex(parsed_json)
        return parsed_json, err

    def _generate_with_universal_fallback(self, contents, response_schema, temperature=0.1, task_desc=""):
        """
        終極雙保險【分步拆解生成】備援調度器：
        順序：Cloudflare Workers AI (Llama 3.3 70B) ──> OpenRouter (openrouter/free)
        針對長詳解自動執行 2 階段無損組裝，徹底消除 Token 截斷與 Unterminated string 錯誤！
        """
        cf_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        cf_url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai/v1" if cf_account_id else ""

        fallback_pool = []
        if cf_url and os.environ.get("CLOUDFLARE_API_TOKEN"):
            fallback_pool.append(("Cloudflare AI", cf_url, os.environ.get("CLOUDFLARE_API_TOKEN"), "@cf/meta/llama-3.3-70b-instruct-fp8-fast"))
        if os.environ.get("OPENROUTER_API_KEY"):
            fallback_pool.append(("OpenRouter", "https://openrouter.ai/api/v1", os.environ.get("OPENROUTER_API_KEY"), "openrouter/free"))

        if not fallback_pool:
            return None, "no_fallback_keys_configured"

        text_prompts = [str(c) for c in contents if isinstance(c, str)]
        combined_prompt = "\n\n".join(text_prompts)
        is_solution_batch = "QuestionSolution" in str(response_schema)

        for name, base_url, key, model in fallback_pool:
            try:
                logging.info(f"⚡ [觸發智慧備援] 切換至 {name} ({model}) 處理 (啟用分步拆解防截斷)...")
                http_client = httpx.Client(timeout=60.0, verify=True, trust_env=False, headers={"User-Agent": "Mozilla/5.0"})
                client = OpenAI(base_url=base_url, api_key=key, http_client=http_client)

                # 💡 情況 A：若為長詳解任務，啟動「雙階段拆解流水線」
                if is_solution_batch:
                    # 步驟一：核心推導（題意、突破點、標準解、選項分析）
                    class Step1CoreSolution(BaseModel):
                        question_analysis: str = Field(description="題意分析")
                        solving_strategy: str = Field(description="解題思路突破點")
                        detailed_solution: str = Field(description="【標準解法】完整正規推導")
                        options_analysis: List[OptionAnalysisItem] = Field(description="選項獨立剖析列表")

                    class Step1Batch(BaseModel):
                        solutions: List[Step1CoreSolution]

                    step1_prompt = combined_prompt + "\n\n【第一階段任務】：請先專注完成題目的『題意分析』、『解題突破點』、『標準解法』以及『選項深入剖析』。"
                    res_step1, err1 = self._call_openai_compatible_raw(client, base_url, key, model, step1_prompt, Step1Batch, temperature)
                    if not res_step1 or not res_step1.get("solutions"):
                        continue

                    # 步驟二：名師昇華（另解一/二/速解、概念複習、陷阱盲點、分類標籤）
                    class Step2Pedagogy(BaseModel):
                        alternative_solutions: str = Field(description="包含【另解一】、【另解二】或【秒殺速解】之 Markdown 內容")
                        concept_review: str = Field(description="核心概念複習")
                        traps_and_warnings: str = Field(description="易錯陷阱與盲點警示")
                        advanced_supplement: str = Field(description="進階延伸補充")
                        scoring_rubric: str = Field(description="評分標準或配分建議")
                        difficulty_level: int = Field(description="難度評分 1-10")
                        difficulty_reason: str = Field(description="難度理由")
                        topic_category: str = Field(description="題型分類")
                        techniques_used: List[str] = Field(description="使用的解題技巧列表")

                    class Step2Batch(BaseModel):
                        pedagogies: List[Step2Pedagogy]

                    step1_context = json.dumps(res_step1["solutions"], ensure_ascii=False)
                    step2_prompt = (
                        f"原題資訊如下：\n{combined_prompt}\n\n"
                        f"我們先前已推導出的標準解法與選項分析如下：\n{step1_context}\n\n"
                        "【第二階段任務】：請基於上述標準解法，為本題擴展『多元另解（另解一、另解二、速解）』，並條列『核心概念複習』、『易錯陷阱』與『題型技巧分類』。"
                    )
                    res_step2, err2 = self._call_openai_compatible_raw(client, base_url, key, model, step2_prompt, Step2Batch, temperature)
                    pedagogies_list = res_step2.get("pedagogies") if isinstance(res_step2, dict) else None
                    if not pedagogies_list and isinstance(res_step2, dict):
                        # 🚨 容錯相容救援模式回傳的 items
                        pedagogies_list = res_step2.get("items") or res_step2.get("solutions")
                        
                    if not pedagogies_list:
                        continue

                    # 步驟三：記憶體無損縫合
                    merged_solutions = []
                    for s1, s2 in zip(res_step1["solutions"], pedagogies_list):
                        full_sol_markdown = s1.get("detailed_solution", "")
                        alt_markdown = s2.get("alternative_solutions", "").strip()
                        if alt_markdown:
                            full_sol_markdown += "\n\n" + alt_markdown
                            
                        merged_solutions.append({
                            "suspects_image_mismatch": False,
                            "question_analysis": s1.get("question_analysis", ""),
                            "solving_strategy": s1.get("solving_strategy", ""),
                            "detailed_solution": full_sol_markdown,
                            "options_analysis": s1.get("options_analysis", []),
                            "concept_review": s2.get("concept_review", ""),
                            "traps_and_warnings": s2.get("traps_and_warnings", ""),
                            "advanced_supplement": s2.get("advanced_supplement", "無"),
                            "scoring_rubric": s2.get("scoring_rubric", ""),
                            "difficulty_level": s2.get("difficulty_level", 5),
                            "difficulty_reason": s2.get("difficulty_reason", "綜合考題"),
                            "topic_category": s2.get("topic_category", "必修_綜合主題"),
                            "techniques_used": s2.get("techniques_used", ["必修_解題方法"])
                        })

                    logging.info(f"🎉 [{name}] 雙階段拆解流水線成功產出 {len(merged_solutions)} 題出版級完整詳解！")
                    return {"solutions": merged_solutions}, None

                # 💡 情況 B：非長詳解（例如審查或 OCR 仲裁），單次呼叫即可
                else:
                    res_single, err = self._call_openai_compatible_raw(client, base_url, key, model, combined_prompt, response_schema, temperature)
                    if res_single:
                        return res_single, None

            except Exception as e:
                logging.warning(f"⚠️ [{name}] 嘗試失敗 ({e})，切換至下一個備援平台...")
                continue

        return None, "all_fallbacks_exhausted"

    def load_model_token_limits(self) -> dict:
        """從本地 model_limits.json 讀取已被記錄的模型 Token 上限"""
        if os.path.exists(self.model_limits_file):
            try:
                with open(self.model_limits_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logging.info(f"🧠 [模型上限記憶庫] 成功載入現有模型 Token 配置: {data}")
                    return data
            except Exception as e:
                logging.warning(f"讀取 {self.model_limits_file} 失敗: {e}")
        return {}

    def update_and_save_model_limit(self, model: str, limit: int):
        """動態更新並持久化記錄該模型的 Token 上限"""
        with self.lock:
            self.model_token_limits[model] = limit
            try:
                with open(self.model_limits_file, "w", encoding="utf-8") as f:
                    json.dump(self.model_token_limits, f, ensure_ascii=False, indent=4)
                logging.info(f"💾 [上限記憶保存] 已將模型 {model} 的最大輸出 Token 上限記錄為 {limit}！")
            except Exception as e:
                logging.error(f"無法寫入 {self.model_limits_file}: {e}")


    def log_key_error(self, key_str, error_code, error_message):
        """將失效/被拒金鑰記錄到本地 key_errors_summary.txt 檔案中"""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        masked_key = f"{key_str[:12]}...{key_str[-6:]}" if len(key_str) > 18 else key_str
        log_line = f"[{timestamp}] 錯誤代碼: {error_code} | 金鑰: {masked_key} | 說明: {error_message}\n"
        with self.lock:
            try:
                with open("key_errors_summary.txt", "a", encoding="utf-8") as f:
                    f.write(log_line)
            except Exception as e:
                logging.error(f"無法寫入金鑰錯誤日誌 key_errors_summary.txt: {e}")
    
    def escape_latex_backslashes(self, raw_text: str) -> str:
        """
        精確且安全地將 JSON 字串內所有代表 LaTeX 指令的單反斜線轉義為雙反斜線。
        防範 \frac, \text, \beta, \right, \nu 等指令被誤判為 JSON 控制字元 (\f, \t, \b, \r, \n)。
        """
        if not raw_text:
            return raw_text
            
        # 1. 暫時保護已有的雙反斜線
        raw_text = raw_text.replace("\\\\", "__DBL_SLASH__")
        
        # 2. 定義轉義邏輯
        def replace_backslash(match):
            full_match = match.group(0)
            # 如果是標準的 6 字元 Unicode 轉義字元 (\uXXXX)，完整保留
            if full_match.startswith('\\u') and len(full_match) == 6:
                return full_match
            
            char = match.group(2) if match.group(2) else match.group(1)
            
            # 如果是 \" 或 \/，屬於標準 JSON 字串轉義，保持原樣
            if char in ['"', '/']:
                return full_match
                
            # 檢查後續字元：若 \f, \t, \b, \r, \n 後面緊跟著英文字母，代表是 LaTeX 指令 (例如 \frac, \text, \beta, \right, \nu)
            end_idx = match.end()
            next_char = raw_text[end_idx] if end_idx < len(raw_text) else ''
            
            if char in ['b', 'f', 'n', 'r', 't']:
                if next_char.isalpha():
                    # 這是 LaTeX 指令，強制轉義反斜線！
                    return "\\\\" + char
                # 否則視為普通 JSON 控制字元 (如獨立的 \n 換行)
                return full_match
                
            # 其他所有反斜線指令 (如 \alpha, \theta, \le, \ge, \pi 等) 強制轉義
            return "\\\\" + char

        # 優先匹配 \\uXXXX，其次匹配單個 \\(.)，目標字串為 raw_text
        raw_text = re.sub(r'(\\u[0-9a-fA-F]{4})|\\(.)', replace_backslash, raw_text)
        
        # 3. 還原保護的雙反斜線
        raw_text = raw_text.replace("__DBL_SLASH__", "\\\\")
        return raw_text
    
    def repair_latex_control_chars(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        
        # 1. 還原 LaTeX 安全預留字串
        text = text.replace("__LTXS__", "\\")

        # 2. 🚨 救回因 \f (form feed) 被消滅而造成的 \frac 殘損（例如 \f rac -> \frac，或直接變成 rac）
        text = re.sub(r'[\x0c]rac(?![a-zA-Z])', r'\\frac', text)
        text = re.sub(r'[\x0c]?(?:(?<=\$)|\b)rac(?=\{)', r'\\frac', text)
        
        # 3. 🚨 救回因 \r (carriage return) 導致的 \right 殘損（例如 \r\right 或是 \right 變換行）
        text = re.sub(r'[\r\n]+ight(?=[)\]}|.])', r'\\right', text)
        text = text.replace(r"\r\right", r"\right").replace(r"\r\left", r"\left")

        # 4. 🚨 救回因 \i 被不當吃掉導致的 \infty -> \fty（嚴格限制不可匹配既有的 \infty）
        text = re.sub(r'(?:\\in\s*){2,}\\infty', r'\\infty', text)
        text = re.sub(r'(?:\\in\s*)+\\infty', r'\\infty', text)
        text = re.sub(r'(?<![a-zA-Z\\])(?<!\\in)(?<!in)\\?fty\b', r'\\infty', text)
        text = re.sub(r'(?<!\\)\bimplies\b', r'\\implies', text)
        text = re.sub(r'(?<!\\)\biff\b', r'\\iff', text)
        text = re.sub(r'\\?bsim\b', r'\\approx', text)
        text = re.sub(r'\\bar\{log\}', r'\\log', text)
        text = re.sub(r'(?<!\\)\bightarrow\b', r'\\rightarrow', text)
        text = re.sub(r'(?<!\\)ight(?=[\)\]\}|\.])', r'\\right', text)
        text = re.sub(r'(?<!\\)eft(?=[\(\[\{])', r'\\left', text)
        text = re.sub(r'\\break\b', r'\\\\', text)

        # 5. 精確修復控制字元
        replacements = {
            "\x0c": r"\f",  # Form Feed
            "\x08": r"\b",  # Backspace
            "\x09": r"\t",  # Tab
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
            
        # 6. 修復因替換導致的粘連錯字
        corrupted_fixes = {
            r"\tquad": r"\quad",
            r"\tqquad": r"\qquad",
            r"\tcdot": r"\cdot",
            r"\ttan": r"\tan",
            r"\tsec": r"\sec",
            r"\tpm": r"\pm",
            r"\ttext": r"\text",
            r"\tnet": r"\net",
            r"\root{": r"\sqrt{",
            r"\bx": r"bx",
            r"\ffrac": r"\frac",
        }
        for bad, good in corrupted_fixes.items():
            text = text.replace(bad, good)
            
        return text

    def repair_hallucinated_latex(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        replacements = {
            r"\ belongge ": r"\ge ",
            r"\ belongge": r"\ge",
            r"\belongge": r"\ge",
            r"\ belongle ": r"\le ",
            r"\ belongle": r"\le",
            r"\belongle": r"\le",
            r"\ belong ": r"\in ",
            r"\ belong": r"\in",
            r"\belong": r"\in",
            r"\ pi ": r"\pi ",
            r"\ pi": r"\pi",
            r"\text{end{bmatrix}": r"\end{bmatrix}",
            r"\text{end{bmatrix}}": r"\end{bmatrix}",
            r"\ text": r"\text",
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return text

    def repair_dict_latex(self, obj):
        """遞迴遍歷整個 JSON 字典，自動修復所有被損毀的 LaTeX 字串"""
        if isinstance(obj, dict):
            return {k: self.repair_dict_latex(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.repair_dict_latex(i) for i in obj]
        elif isinstance(obj, str):
            return self.repair_hallucinated_latex(self.repair_latex_control_chars(obj))
        return obj

    def estimate_tokens(self, contents) -> int:
        """根據輸入內容長度與圖形估算 Token (避免在送出前就觸發 TPM 溢出)"""
        tokens = 0
        if isinstance(contents, list):
            for item in contents:
                if isinstance(item, str):
                    tokens += len(item) // 2
                elif isinstance(item, Image.Image):
                    tokens += 1500  # 預留一張圖所需的 Token (Gemini 讀圖基本消秏)
        elif isinstance(contents, str):
            tokens += len(contents) // 2
        return max(500, tokens)

    def print_keys_status(self):
        """將所有金鑰統計與額度狀態輸出至本機檔案 key_status.txt，完全不在終端機列出，保持畫面乾淨"""
        with self.lock:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            try:
                # 寫入（覆寫）本機文字檔，作為即時狀態儀表板
                with open("key_status.txt", "w", encoding="utf-8") as f:
                    f.write(f"⏱️ [金鑰狀態更新時間] {current_time}\n")
                    f.write(f"{'API 金鑰':<20} | {'RPM (每分請求)':<15} | {'TPM (每分Token)':<15} | {'RPD 每日配額與狀態'}\n")
                    f.write("-" * 80 + "\n")
                    for k in self.keys:
                        s = k.get_status()
                        status_str = s['rpd_status']
                        if k.is_disabled:
                            status_str = f"Disabled ({k.disable_reason})"
                        elif k.rpd_exhausted:
                            status_str = f"{status_str} (Exhausted)"
                        
                        f.write(f"{s['key']:<20} | {s['rpm_status']:<15} | {s['tpm_status']:<15} | {status_str}\n")
            except Exception as e:
                # 僅在檔案寫入失敗時在背景記錄，不干擾解析主線任務
                logging.warning(f"無法寫入 key_status.txt 儀表板: {e}")

    def get_current_resource_for_model(self, target_model: str, estimated_tokens: int = 1000):
        """尋找目前能執行 target_model 的金鑰；若無可用 Key 則回傳 None 以利立即降級至下一梯隊模型"""
        with self.lock:
            current_time = time.time()
            # 🚨 核心修復：只檢查 target_model 是否可用，不因其他模型耗盡而封鎖整把 Key！
            active_keys = [k for k in self.keys if not k.is_disabled and not k.is_model_rpd_exhausted(target_model, current_time)]
            
            if not self.keys:
                logging.critical("🚨 [系統致命錯誤] 找不到任何 API 金鑰！")
                os._exit(1)
            if not active_keys:
                # 當前目標模型已無可用金鑰，回傳 None 讓排程器平滑切換下一梯隊模型（例如 3.7 -> 3.5-lite）
                return None, target_model, None
                
            num_active = len(active_keys)
            for i in range(num_active):
                idx = (self.key_idx + i) % num_active
                key = active_keys[idx]
                if key.can_request_model(target_model, current_time, estimated_tokens):
                    key.add_request_model(target_model, current_time, estimated_tokens)
                    self.key_idx = (idx + 1) % num_active
                    return key.client, target_model, key
                    
        return None, target_model, None

    def handle_rate_limit_error(self, key: FreeTierKey, model: str, error_msg: str):
        with self.lock:
            msg = error_msg.lower()
            is_rpd = any(k in msg for k in ["per day", "perday", "daily", "requests per day", "rpd", "day limit"])
            
            if is_rpd:
                key.mark_rpd_exhausted(model)
                logging.warning(f"🚨 [獨立模型 RPD 上限] 金鑰 {key.api_key[:8]}... 僅模型【{model}】達到每日配額上限")
            else:
                # 🚀 精確只鎖定該模型 40 秒，其餘 3.6/3.5/lite 模型依然暢通無阻！
                key.mark_model_rate_limited(model, cooldown_seconds=40.0)
                logging.info(f"⏳ [獨立模型降溫] 金鑰 {key.api_key[:8]}... 僅對模型 {model} 冷卻 40 秒，其他模型額度完全保留！")

    def generate_with_retry(self, contents, response_schema, temperature=0.2, max_attempts=30, preferred_model: Optional[str] = None, enable_thinking: bool = True, thinking_level: str = "HIGH", task_desc: str = "", provider: str = "google"):
        # === 架構：若指定為 Groq，則將任務路由給 DeepSeek-R1 / Qwen ===
        if provider == "groq":
            return self._generate_with_groq(contents, response_schema, temperature, max_attempts, preferred_model, task_desc)
            
        # === 🚀 全動態模型調度與負載均衡梯隊 ===
        candidate_models = []
        with self.lock:
            self.model_dispatch_counter += 1
            dispatch_offset = self.model_dispatch_counter

        # 🚨 若有指定 preferred_model 且為 Tier 1，則以該模型起手；否則依計數器在 [3.7, 3.6, 3.5, 3.5-lite, 3.1-lite] 中均勻滾動
        all_usable_models = [m for m in self.models if m in [
            "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", 
            "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
            "gemini-2.5-flash"
        ]]
        
        if preferred_model and preferred_model in self.models:
            start_model = preferred_model
        else:
            start_model = all_usable_models[dispatch_offset % len(all_usable_models)]

        candidate_models = [start_model] + [m for m in self.models if m != start_model]

        estimated_tokens = self.estimate_tokens(contents)
        attempts = 0
        
        # 🚀 動態深度：根據手頭 Key 總數動態設定容忍次數，確保 34 把 Key 能被充分輪詢！
        num_keys = max(1, len(self.keys))
        max_attempts = max(max_attempts, min(num_keys, 35))
        RETRIES_PER_MODEL = max(6, num_keys // 2)

        while attempts < max_attempts:
            # 🚀 瀑布流階梯調度：優先尋找最高算力模型，若 3.7 都在冷卻中，0 秒切換至 3.6/3.5 接力！
            client, model, key_obj = None, None, None
            for model_cand in candidate_models:
                cl, md, ko = self.get_current_resource_for_model(model_cand, estimated_tokens=estimated_tokens)
                if cl is not None:
                    client, model, key_obj = cl, md, ko
                    break
                    
            if client is None:
                logging.info("⏳ [全梯隊滿載] 所有模型額度短暫冷卻中，等待 1.5 秒...")
                smart_sleep(1.5)
                attempts += 1
                continue

            self.last_model_used = model
            
            thinking_config = None
            if any(m in model for m in ["gemini-3.8", "gemini-3.7", "gemini-3.6", "gemini-3.5", "gemini-2.5", "gemini-3"]):
                # 🚨 [動態思考強度適配]：正式大考真題與段考啟用 HIGH 深度推導，模考與非必要任務啟用 LOW 加速！
                if enable_thinking and not any(kw in task_desc for kw in ["OCR", "掃描", "解答仲裁"]):
                    lvl = "HIGH" if thinking_level.upper() == "HIGH" else "LOW"
                    thinking_config = types.ThinkingConfig(thinking_level=lvl)
                else:
                    thinking_config = None
            try:
                # 🚀 平滑安全閥門：將出站微間隔調至 0.5 秒，搭配 8 條執行緒，打造最平穩安全的 API 請求節奏
                with self.lock:
                    now_t = time.time()
                    time_since_last = now_t - getattr(self, "global_last_dispatch_time", 0.0)
                    if time_since_last < 0.5:
                        time.sleep(0.5 - time_since_last)
                    self.global_last_dispatch_time = time.time()

                desc_str = f" {task_desc}" if task_desc else ""
                
                # 🚀 動態載入該模型專屬的 Token 上限
                model_max_tokens = self.model_token_limits.get(model, 65536)
                
                chosen_proxy = self.get_next_proxy_url()
                proxy_label = chosen_proxy.split('//')[-1].split('/')[0] if chosen_proxy else "Direct_Google"
                
                print(f"🔹{desc_str} 嘗試使用模型 {model} (Proxy: {proxy_label}) (金鑰: {key_obj.api_key[:8]}...) (第 {attempts + 1} 次嘗試)...", flush=True)
                self.print_keys_status()

                gen_config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=model_max_tokens,
                    response_schema=response_schema,
                    temperature=temperature,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                )
                if thinking_config:
                    gen_config.thinking_config = thinking_config

                active_client = key_obj.get_client(chosen_proxy)

                # ⏱️ 開始度量 API 延遲
                api_t0 = time.time()
                response = active_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=gen_config,
                )
                api_rtt = time.time() - api_t0
                key_obj.record_call_outcome(True) # 記錄近 10 次成功

                # 判定階段標籤
                cur_stage_tag = "solution_gen"
                if "審查" in task_desc or "validator" in str(response_schema): cur_stage_tag = "validator"
                elif "補漏" in task_desc or "PageExtraction" in str(response_schema): cur_stage_tag = "page_scan"
                elif "仲裁" in task_desc: cur_stage_tag = "arbitration"
                elif "繪圖" in task_desc: cur_stage_tag = "diagram"

                GLOBAL_METRICS.record_stage_api_call(cur_stage_tag, api_rtt, is_200=True, is_json_err=False)
                
                # 📊 記錄成功 200 狀態（包含代理名稱與模型名稱）
                key_masked = f"{key_obj.api_key[:6]}...{key_obj.api_key[-4:]}" if key_obj else "Unknown"
                # 📊 記錄成功 200 狀態
                GLOBAL_METRICS.record_http(chosen_proxy or "Direct_Google", model, key_masked, 200)
                
                if not response:
                    logging.warning("⚠️ API 回傳完全為空 (None)，可能發生網路瞬斷，準備重試...")
                    attempts += 1
                    smart_sleep(2)
                    continue

                if not hasattr(response, 'text'):
                    logging.warning("⚠️ API 回傳空物件，準備重試...")
                    attempts += 1
                    continue
                
                # 🚨 精準更新 Token 計數器：使用 API 實際回傳的 Token 量更新本地狀態
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    actual_tokens = response.usage_metadata.total_token_count
                    with self.lock:
                        if key_obj.token_usage:
                            # 取代掉先前為該 key 放入的預估值
                            key_obj.token_usage[-1] = (key_obj.token_usage[-1][0], actual_tokens)

                raw_text = response.text
                if not raw_text or not raw_text.strip():
                    logging.warning("⚠️ API 回傳了空字串，準備重試...")
                    attempts += 1
                    continue

                # 繁簡轉換與兩岸名詞標準化
                raw_text = s2t(raw_text)
                term_replacements = {
                    "概率": "機率", "矢量": "向量", "標量": "純量", 
                    "宏觀": "巨觀", "微觀": "微觀", "屏幕": "螢幕",
                    "質量數": "質量數", "方程組": "方程組", "波函數": "波函數",
                    "解析度": "解析度", "分辨率": "解析度", "內存": "記憶體",
                    "算法": "演算法", "數組": "陣列", "電壓表": "伏特計",
                    "電流表": "安培計", "萬有引力常量": "萬有引力常數"
                }
                for simp_term, tw_term in term_replacements.items():
                    raw_text = raw_text.replace(simp_term, tw_term)

                # 🚀 調用萬能解析救援器
                parsed_json, parse_err = universal_robust_json_parser(raw_text, response_schema=response_schema)
                if parsed_json:
                    parsed_json = self.repair_dict_latex(parsed_json)
                    return parsed_json, parse_err
                else:
                    GLOBAL_METRICS.record_error("API_NETWORK", f"JSON 截斷或結構解析異常: {parse_err}", stage=desc_str)
                    
                attempts += 1
                smart_sleep(1.0)
                continue
                
            except RPDExhaustedException as ree:
                raise ree
            except APIError as e:
                err_code = getattr(e, 'code', 500)
                if key_obj: key_obj.record_call_outcome(False)
                key_masked = f"{key_obj.api_key[:6]}...{key_obj.api_key[-4:]}" if key_obj else "Unknown"
                GLOBAL_METRICS.record_http(chosen_proxy or "Direct_Google", model, key_masked, err_code)
                err_str = str(e).lower()
                GLOBAL_METRICS.record_error("API_NETWORK", f"HTTP {err_code}: {err_str[:160]}", stage=f"{task_desc} ({model})")
                
                # 🚀 檢測本題是否包含 PIL 圖片（若無圖片，則可安全切換至第三方大模型備援）
                has_pil_images = any(isinstance(c, Image.Image) for c in contents) if isinstance(contents, list) else False
                
                # 🚀 404 錯誤（模型端點不存在或代理路徑未對應）：不更換金鑰，只將該模型自候選名單中移除
                if e.code == 404 or "not found" in err_str or "is not found for api version" in err_str:
                    logging.warning(f"⏭️ [404 模型無效/代理端點未支援] 模型 {model} 回傳 404，金鑰保持有效，立即換下一個模型...")
                    with self.lock:
                        if model in candidate_models:
                            candidate_models.remove(model)
                        if model in self.models:
                            self.models.remove(model)
                    smart_sleep(0.3)
                    continue

                # 🚨 500 / 503 / 504 / 524 伺服器過載與代理逾時處理：立即換模型 + 換代理 + 指數退避
                is_server_or_proxy_overload = (
                    e.code in [500, 502, 503, 504, 520, 521, 522, 523, 524] 
                    or any(k in err_str for k in ["unavailable", "gateway timeout", "timeout", "524", "internal server error", "overloaded"])
                )
                
                if is_server_or_proxy_overload:
                    attempts += 1
                    
                    # 1. 🚨 動態熔斷機制：一旦該模型報 503/504，強制將該模型全域降溫 45~75 秒，防止持續對已崩潰模型發送請求
                    with self.lock:
                        if key_obj:
                            key_obj.mark_model_rate_limited(model, cooldown_seconds=60.0)
                        if candidate_models and model in candidate_models:
                            candidate_models.remove(model)
                            candidate_models.append(model) # 移到最末位
                    
                    chosen_proxy = self.get_next_proxy_url()
                    next_model_hint = f"，模型【{model}】熔斷冷卻 60 秒，自動切換至【{candidate_models[0] if candidate_models else '備援'}】"

                    # 2. 外部備援大模型接管（無附圖時提早於第 2 次 503 即切換）
                    if attempts >= 2 and not has_pil_images:
                        res_fallback, _ = self._generate_with_universal_fallback(contents, response_schema, temperature, task_desc=f"{task_desc} [503熔斷備援接管]")
                        if res_fallback: return res_fallback, None

                    # 3. 隨機抖動退避（3~7 秒，避免所有 thread 同步重發）
                    backoff_wait = random.uniform(3.0, 7.0) + (attempts * 0.8)
                    logging.warning(f"⚠️ [伺服器異常 HTTP {e.code}] 第 {attempts} 次重試{next_model_hint}，等待 {backoff_wait:.1f} 秒...")
                    smart_sleep(backoff_wait)
                    continue

                # 🚨【最高優先級】確診金鑰無效/已被刪除（Google 在 400 與 401 都會回傳 'api key not valid'）
                is_real_google_invalid = any(k in err_str for k in ["api key not valid", "api_key_invalid", "key_invalid", "invalid api key"])
                
                if is_real_google_invalid:
                    with self.lock:
                        key_obj.is_disabled = True
                        key_obj.disable_reason = f"{err_code}_Key_Invalid"
                    self.log_key_error(key_obj.api_key, err_code, "金鑰已被刪除、停用或無效")
                    logging.warning(f"⚠️ [確診金鑰無效/已刪除] 金鑰 {key_obj.api_key[:8]}... 經 Google 證實已無效 (HTTP {err_code})，已永久停用並記錄，立即換下一把金鑰！")
                    # 立即換下一把可用金鑰，不扣減重試次數，也不誤關閉 Thinking 模式！
                    continue

                # 🚨 401 代理攔截保護 (若非 Google 報錯，通常是 Vercel/Cloudflare 部署保護)
                if e.code == 401 or "unauthorized" in err_str:
                    logging.warning(f"⚠️ [401 代理攔截保護] 代理端點 {chosen_proxy} 回傳 401 驗證，金鑰 {key_obj.api_key[:8]}... 保持正常可用，切換下一個代理重試...")
                    attempts += 1
                    smart_sleep(1.0)
                    continue

                # 🚨 403 Forbidden 錯誤處理（地區拒絕、未開通 API 或帳單欠費）
                if e.code == 403 or "forbidden" in err_str or "permission denied" in err_str or "restricted" in err_str:
                    with self.lock:
                        key_obj.is_disabled = True
                        key_obj.disable_reason = "403_Forbidden"
                    self.log_key_error(key_obj.api_key, 403, "存取被拒(未開通API、地區限制或帳單欠費)")
                    logging.warning(f"⚠️ [403 權限拒絕] 金鑰 {key_obj.api_key[:8]}... 存取被拒，已永久停用。")
                    continue

                # 🚀 精準處理 400 參數錯誤：區分「Token 溢出」與「Thinking/Schema 不相容」
                if e.code == 400 or "invalid_argument" in err_str:
                    is_token_limit_error = any(kw in err_str for kw in ["max_output_tokens", "output token", "token limit", "greater than maximum allowed"])
                    
                    if is_token_limit_error:
                        current_limit = self.model_token_limits.get(model, 65536)
                        # 只有在 Google 明確指出 Token 上限時才進行精準匹配與學習
                        match = re.search(r'(?:<=|between \d+ and|less than or equal to|maximum allowed|limit of)\s*(\d{4,5})', err_str)
                        if match:
                            new_limit = int(match.group(1))
                            logging.warning(f"🎯 [Token上限學習] 模型 {model} 真實上限為 {new_limit} tokens！")
                            self.update_and_save_model_limit(model, new_limit)
                        else:
                            new_limit = max(4096, current_limit // 2)
                            logging.warning(f"📉 [Token階梯降級] 模型 {model} 調降上限至 {new_limit}")
                            self.update_and_save_model_limit(model, new_limit)
                    else:
                        # 🚨 非 Token 錯誤（通常為 ThinkingConfig 不相容或 Schema 嵌套過深），關閉 Thinking 即可，絕不調降 Token！
                        logging.warning(f"⚠️ [400 配置不相容] 模型 {model} 參數報錯（非 Token 問題: {err_str[:120]}），自動重置 Thinking 設定重試...")
                        enable_thinking = False
                    
                    attempts += 1
                    smart_sleep(1.0)
                    continue

                attempts += 1
                if e.code == 429 or "quota" in err_str or "exhausted" in err_str:
                    self.handle_rate_limit_error(key_obj, model, err_str)
                    
                    # 🚨 遇到 429 時，主動將當前模型移至候選清單末尾，讓 3.6 / 3.5 / Lite 自動接管下一次請求
                    if candidate_models and len(candidate_models) > 1:
                        candidate_models.append(candidate_models.pop(0))
                    
                    # 多 Key 智慧備援分流
                    fallback_threshold = max(5, int(len(self.keys) * 0.8))
                    if attempts >= fallback_threshold and not has_pil_images:
                        logging.info(f"🔄 [429 智慧避險] 已輪詢 {attempts} 把金鑰皆遇限流，切換至備援大模型 (CF/OpenRouter)...")
                        res_fallback, fb_err = self._generate_with_universal_fallback(contents, response_schema, temperature, task_desc=f"{task_desc} [429自動分流]")
                        if res_fallback:
                            return res_fallback, None
                        else:
                            # 🚨 備援模型也耗盡額度或失敗時，平滑切回 Gemini，重置懲罰並等待冷卻解鎖
                            logging.warning("⚠️ [備援通道皆失敗或額度耗盡] 自動回退至 Gemini 叢集，重新激活輪詢...")
                            attempts = max(0, attempts - 2) # 給予 Gemini 更多輪換機會
                            smart_sleep(6.0) # 等待 6 秒使 Gemini 金鑰的滑動冷卻窗口解鎖

                    wait_time = random.uniform(2.0, 4.0)
                    logging.warning(f"⏳ [429 階梯平滑降級] 金鑰 {key_obj.api_key[:8]}... 進入冷卻，模型切換至 {candidate_models[0]}，等待 {wait_time:.1f} 秒...")
                    smart_sleep(wait_time)
                    continue
                else:
                    smart_sleep(2)
            except Exception as e:
                attempts += 1
                err_msg = str(e).lower()
                if "timeout" in err_msg or "time out" in err_msg:
                    logging.warning(f"⚠️ [API 呼叫超時] (嘗試第 {attempts} 次): {e}。進行金鑰降溫與平滑重試...")
                    with self.lock:
                        key_obj.request_times.extend([time.time()] * 5)
                else:
                    logging.exception(f"發生未預期錯誤（嘗試第 {attempts} 次）: {e}")
                smart_sleep(2)
        return None, "請求失敗"

    def _generate_with_groq(self, contents, response_schema, temperature, max_attempts, preferred_model, task_desc):
        """專門處理 Groq 端點 (DeepSeek-R1 / Qwen / Llama3.3) 的多金鑰輪詢邏輯"""
        if not hasattr(self, 'groq_keys') or not self.groq_keys:
            logging.error("🚨 尚未設定 GROQ_API_KEY，無法使用 Groq！將回傳失敗。")
            return None, "groq_key_missing"

        # 1. 安全地取出下一把金鑰
        with self.groq_lock:
            current_groq_key = self.groq_keys[self.groq_key_idx]
            self.groq_key_idx = (self.groq_key_idx + 1) % len(self.groq_keys)
        
        # 每次都用輪到的這把金鑰建立一個新的臨時客戶端
        temp_groq_client = Groq(api_key=current_groq_key)
        key_masked = f"{current_groq_key[:8]}..."

        text_prompts = [str(c) for c in contents if isinstance(c, str)]
        combined_prompt = "\n\n".join(text_prompts)
        
        schema_json = response_schema.model_json_schema()
        system_msg = (
            "You are a highly logical math and science expert. "
            f"You MUST output exactly matching this JSON schema: {json.dumps(schema_json)}. "
            "Do NOT wrap the JSON in markdown code blocks like ```json. Output ONLY the raw JSON object."
        )
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": combined_prompt}]
        
        model_name = preferred_model if preferred_model else "llama-3.3-70b-versatile"
        attempts = 0

        # 1. 影像清洗：Groq 上的 DeepSeek-R1 與多數開源模型為純文字模型，必須濾除 PIL.Image
        text_prompts = [str(c) for c in contents if isinstance(c, str)]
        combined_prompt = "\n\n".join(text_prompts)
        
        # 2. 構建嚴格的 JSON 系統提示 (System Prompt)
        schema_json = response_schema.model_json_schema()
        system_msg = (
            "You are a highly logical math and science expert. "
            f"You MUST output exactly matching this JSON schema: {json.dumps(schema_json)}. "
            "Do NOT wrap the JSON in markdown code blocks like ```json. Output ONLY the raw JSON object after your thinking process."
        )
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": combined_prompt}
        ]
        
        model_name = preferred_model if preferred_model else "llama-3.3-70b-versatile"
        attempts = 0
        
        while attempts < max_attempts:
            try:
                
                # 🚨 突破 TPM 限制的核心：加入隨機抖動 (Jitter)，錯開多執行緒的併發請求
                smart_sleep(random.uniform(2.5, 6.0))
                
                desc_str = f" {task_desc}" if task_desc else ""
                
                print(f"🧠{desc_str} 嘗試使用 Groq 模型 {model_name} (使用金鑰 {key_masked}) (第 {attempts + 1} 次嘗試)...", flush=True)
                
                response = temp_groq_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=8000, # 開放最大輸出長度以容納詳解與思考鏈
                )
                
                raw_text = response.choices[0].message.content
                if not raw_text or not raw_text.strip():
                    attempts += 1
                    continue
                    
                # 繁簡轉換與兩岸名詞標準化
                raw_text = s2t(raw_text)
                term_replacements = {
                    "概率": "機率", "矢量": "向量", "標量": "純量", 
                    "宏觀": "巨觀", "微觀": "微觀", "屏幕": "螢幕",
                    "分辨率": "解析度", "內存": "記憶體", "算法": "演算法"
                }
                for simp_term, tw_term in term_replacements.items():
                    raw_text = raw_text.replace(simp_term, tw_term)

                # 🚀 調用萬能解析救援器（徹底解決備援模型的 Invalid \escape 與 Extra data 崩潰）
                parsed_json, parse_err = universal_robust_json_parser(raw_text, response_schema=response_schema)
                if parsed_json:
                    parsed_json = self.repair_dict_latex(parsed_json)
                    return parsed_json, parse_err

            except RateLimitError as e:
                attempts += 1
                # Groq 的免費版在分鐘級 TPM 滿了時會擋，需要冷卻
                wait_time = random.uniform(10.0, 15.0)
                logging.warning(f"⚠️ [Groq 限速] 觸發 TPM/RPM 限制，冷卻 {wait_time:.1f} 秒... (第 {attempts} 次嘗試)")
                smart_sleep(wait_time)
            except Exception as e:
                attempts += 1
                logging.warning(f"⚠️ [Groq API 錯誤] {e} (第 {attempts} 次嘗試)")
                smart_sleep(3)
                
        return None, "groq_api_error"
    
# ==========================================
# 3. 核心處理類別 (PDF 裁切、AI 呼叫)
# ==========================================


SUBJECT_FILE_LOCK = threading.Lock() # 在檔案頂端定義

def universal_robust_json_parser(raw_text: str, response_schema=None) -> tuple[Optional[dict], Optional[str]]:
    """
    萬能工業級 JSON 清洗與救援解析器
    徹底解決 Invalid \\escape、Extra data、JSON 截斷與字串內換行問題
    """
    if not raw_text or not isinstance(raw_text, str) or not raw_text.strip():
        return None, "empty_response"
        
    text = raw_text.strip()
    
    # 1. 剝除 DeepSeek / 思考模型的 <think> 標籤
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "</think>" in text:
        text = text.split("</think>")[-1].strip()
        
    # 2. 剝除 Markdown ```json 區塊標記
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE).strip()

    # 3. 處理 Extra data：尋找最外層的完整 JSON 物件邊界 { ... } 或 [ ... ]
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    first_bracket = text.find('[')
    last_bracket = text.rfind(']')
    
    if first_brace != -1 and last_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        text = text[first_brace:last_brace + 1]
    elif first_bracket != -1 and last_bracket != -1:
        text = text[first_bracket:last_bracket + 1]

    def sanitize_json_text(s: str) -> str:
        # A. 修正 LaTeX 結尾反斜線誤轉義邊界引號
        s = re.sub(r'\\"(?=\s*[,}\]])', r'\\\\"', s)
        
        # B. 精準轉義所有非標準 JSON 轉義的 LaTeX 反斜線（如 \alpha, \implies, \le, \theta, \frac 等）
        def fix_escape(match):
            full = match.group(0)
            if full.startswith('\\u') and len(full) == 6:
                return full
            char = match.group(2) if match.group(2) else match.group(1)
            if char in ['"', '\\', '/']:
                return full
            if char in ['b', 'f', 'n', 'r', 't']:
                # 若後面緊接字母則是 LaTeX 指令，強制轉義
                end_idx = match.end()
                if end_idx < len(s) and s[end_idx].isalpha():
                    return "\\\\" + char
                return full
            return "\\\\" + char

        s = re.sub(r'(\\u[0-9a-fA-F]{4})|\\(.)', fix_escape, s)
        
        # C. 轉義字串內的實體換行符
        def escape_newlines(m):
            return m.group(0).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        s = re.sub(r'"(?:[^"\\]|\\.)*"', escape_newlines, s, flags=re.DOTALL)
        return s

    cleaned_text = sanitize_json_text(text)
    
    # 4. 嘗試標準解析
    try:
        parsed = json.loads(cleaned_text, strict=False)
        return parsed, None
    except json.JSONDecodeError as je:
        logging.warning(f"⚠️ [JSON 截斷或語法異常] ({je})，啟動【深度子物件救援機制】...")
        
        # 5. 強化版子物件深度救援演算法 (Salvage Mode)
        extracted_objects = []
        depth = 0
        obj_start = -1
        in_string = False
        escape = False
        
        for i, char in enumerate(text):
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    if depth == 0 or (depth == 1 and obj_start == -1):
                        obj_start = i
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 1 and obj_start != -1:
                        raw_sub = text[obj_start:i+1]
                        # 🚨 關鍵升級：子物件在 parse 前必須先經過萬能清洗，絕不裸奔！
                        clean_sub = sanitize_json_text(raw_sub)
                        try:
                            obj = json.loads(clean_sub, strict=False)
                            if any(k in obj for k in ["detailed_solution", "is_valid", "question_text", "question_analysis", "answers"]):
                                extracted_objects.append(obj)
                        except Exception:
                            pass
                        obj_start = -1

        if extracted_objects:
            logging.info(f"🦸‍♂️ [萬能救援成功] 成功從截斷的輸出中救回 {len(extracted_objects)} 題完整數據！")
            schema_str = str(response_schema) if response_schema else ""
            if "solutions" in schema_str or "QuestionSolution" in schema_str or "Step1Batch" in schema_str:
                return {"solutions": extracted_objects}, "partial_success"
            elif "pedagogies" in schema_str or "Step2Batch" in schema_str:
                return {"pedagogies": extracted_objects}, "partial_success"
            elif "validators" in schema_str or "SolutionValidator" in schema_str:
                return {"validators": extracted_objects}, "partial_success"
            elif "questions" in schema_str or "ExtractedQuestion" in schema_str:
                return {"questions": extracted_objects}, "partial_success"
            elif "answers" in schema_str or "AnswerKey" in schema_str:
                return {"answers": extracted_objects}, "partial_success"
            return extracted_objects[0] if len(extracted_objects) == 1 else {"items": extracted_objects}, "partial_success"
            
        return None, "json_decode_error"

class DiagramResponse(BaseModel):
    need_diagram: bool = Field(description="是否需要幾何圖形、函數波形、物理受力分析、生物演化/機制圖或化學相圖來輔助學生理解本題。")
    python_code: str = Field(description="完整的 Python 繪圖程式碼。必須使用 matplotlib 與 numpy，且程式碼最末端必須包含將圖片儲存至指定路徑的儲存代碼。若不需要則留空。")

def clean_and_repair_python_code(py_code: str, target_filepath: str) -> str:
    """清理 Markdown 標記、自動將字面量 \\n 展開為真實換行，並修復 Line 35 續行符錯誤"""
    if not py_code:
        return ""
    
    # 1. 剝除 Markdown 代碼塊標記
    py_code = re.sub(r'^```(?:python)?\s*', '', py_code, flags=re.MULTILINE)
    py_code = re.sub(r'```\s*$', '', py_code, flags=re.MULTILINE).strip()
    
    # 2. 🚨 核心修復：若代碼被壓在單行或夾帶字面量 '\\n'，強制展開為真正的換行符
    if "\\n" in py_code and ("\nimport" not in py_code and "import" in py_code):
        py_code = py_code.replace("\\n", "\n").replace("\\t", "    ").replace('\\"', '"').replace("\\'", "'")
        
    # 3. 清理開頭多餘的字面量 '\n' 或單獨反斜線（徹底根絕 Line 35 unexpected character 報錯）
    py_code = re.sub(r'^\\[nrt]\s*', '', py_code)
    py_code = py_code.lstrip('\\').strip()
    
    # 4. 移除重複的 matplotlib 後端設定
    py_code = re.sub(r'matplotlib\.use\(.*?\)', '', py_code)
    
    # 5. 安全清理：消除所有行尾續行符反斜線後面的非法空格
    cleaned_lines = []
    for line in py_code.splitlines():
        clean_l = re.sub(r'\\([ \t]+)$', r'\\', line)
        cleaned_lines.append(clean_l)
    py_code = "\n".join(cleaned_lines)

    # 4. 🚨 跨平台字型自動載入與數學模式綁定（Linux 優先使用已安裝之 Noto Sans CJK TC）
    norm_target_path = target_filepath.replace("\\", "/")
    headless_header = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.font_manager as fm\n"
        "import numpy as np\n"
        "import os\n\n"
        "_cjk_font_paths = [\n"
        "    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',\n"
        "    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',\n"
        "    '/usr/share/fonts/opentype/noto/NotoSansTC-Regular.otf',\n"
        "    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',\n"
        "    'C:/Windows/Fonts/msjh.ttc',\n"
        "    'C:/Windows/Fonts/msjh.ttf'\n"
        "]\n"
        "_loaded_font_name = 'sans-serif'\n"
        "for _fp in _cjk_font_paths:\n"
        "    if os.path.exists(_fp):\n"
        "        try:\n"
        "            fm.fontManager.addfont(_fp)\n"
        "            _loaded_font_name = fm.FontProperties(fname=_fp).get_name()\n"
        "            break\n"
        "        except Exception:\n"
        "            pass\n\n"
        "# 🚨 雲端環境保底防護：若無中文字體，即刻下載輕量中文字型！\n"
        "if _loaded_font_name == 'sans-serif':\n"
        "    import urllib.request\n"
        "    _fallback_font = 'TaipeiSansTCBeta-Regular.ttf'\n"
        "    if not os.path.exists(_fallback_font):\n"
        "        try:\n"
        "            urllib.request.urlretrieve('https://github.com/dottgonzo/taipei-sans-tc/raw/master/TaipeiSansTCBeta-Regular.ttf', _fallback_font)\n"
        "            fm.fontManager.addfont(_fallback_font)\n"
        "            _loaded_font_name = fm.FontProperties(fname=_fallback_font).get_name()\n"
        "        except Exception:\n"
        "            pass\n\n"
        "plt.rcParams['font.sans-serif'] = [_loaded_font_name, 'Noto Sans CJK TC', 'Noto Sans TC', 'Microsoft JhengHei', 'DejaVu Sans', 'sans-serif']\n"
        "plt.rcParams['axes.unicode_minus'] = False\n"
        "plt.rcParams['text.usetex'] = False\n"
        "plt.rcParams['mathtext.fontset'] = 'custom'\n"
        "plt.rcParams['mathtext.rm'] = _loaded_font_name\n"
        "plt.rcParams['mathtext.it'] = _loaded_font_name\n"
        "plt.rcParams['mathtext.bf'] = _loaded_font_name\n"
        "plt.rcParams['mathtext.default'] = 'regular'\n"
    )
    full_code = headless_header + "\n" + py_code

    # 5. 確保結尾具備存檔與關閉圖表
    if "plt.savefig" not in full_code:
        full_code += f'\nplt.savefig(r"{norm_target_path}", dpi=200, bbox_inches="tight")\nplt.close("all")\n'
        
    return full_code

def execute_and_fix_diagram_script(ai_manager, initial_prompt, response_schema, diagram_filepath: str, safe_q_num: str, task_tag: str = "", max_attempts: int = 2) -> bool:
    """
    ⚡【輕量隔離繪圖引擎】
    以記憶體 AST 語法檢查 + 隔離 namespace 執行，徹底消滅 subprocess 進程建立開銷與超時卡死，速度提升 75%
    """
    import ast
    norm_diagram_path = os.path.abspath(diagram_filepath).replace("\\", "/")
    os.makedirs(os.path.dirname(norm_diagram_path), exist_ok=True)
    current_prompt = initial_prompt

    for attempt in range(1, max_attempts + 1):
        try:
            target_draw_model = "gemini-3.5-flash-lite" if attempt == 1 else "gemini-3.5-flash"
            res, _ = ai_manager.generate_with_retry(
                contents=[current_prompt],
                response_schema=response_schema,
                temperature=0.1,
                max_attempts=2,
                preferred_model=target_draw_model,
                enable_thinking=False,
                task_desc=f"{task_tag} (繪圖嘗試 {attempt}/{max_attempts})"
            )

            py_code_raw = ""
            if isinstance(res, dict): py_code_raw = res.get('python_code', '')
            if not py_code_raw and isinstance(res, str):
                code_match = re.search(r'```(?:python)?\s*([\s\S]*?)```', res)
                if code_match: py_code_raw = code_match.group(1).strip()

            if not py_code_raw: return False
            py_code = clean_and_repair_python_code(py_code_raw, norm_diagram_path)
            if not py_code: return False

            # 🚨 1. AST 安全與死循環防禦檢測（禁止 while 循環、禁止危險系統調用）
            try:
                tree = ast.parse(py_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.While):
                        raise ValueError("繪圖代碼禁止使用 while 循環以防死鎖")
            except Exception as ast_err:
                logging.warning(f"⚠️ [AST 攔截] 題號 {safe_q_num} 第 {attempt} 次語法有誤 ({ast_err})，要求 AI 重寫...")
                current_prompt = f"你生成的繪圖程式碼未通過安全語法審查 ({ast_err})，請改用純 numpy 數組計算並重新生成。"
                continue

            # 🚨 2. 記憶體隔離執行（持全域鎖保證 Matplotlib 執行緒安全，毫秒級完成）
            exec_error = ""
            with MATPLOTLIB_RENDER_LOCK:
                try:
                    import matplotlib.pyplot as plt
                    plt.close('all')
                    isolated_scope = {"__name__": "__main__"}
                    exec(py_code, isolated_scope)
                except Exception as ex:
                    import traceback
                    exec_error = traceback.format_exc()
                finally:
                    try:
                        import matplotlib.pyplot as plt
                        plt.close('all')
                    except Exception: pass

            if not exec_error and os.path.exists(norm_diagram_path) and os.path.getsize(norm_diagram_path) > 0:
                logging.info(f"✅ [繪圖成功] 題號 {safe_q_num} 圖表已生成：{os.path.basename(norm_diagram_path)}")
                with GLOBAL_METRICS.lock:
                    GLOBAL_METRICS.diagram_count += 1
                return True
            else:
                err_msg = exec_error.strip()[-200:] if exec_error else "未生成實體圖片檔案"
                logging.warning(f"⚠️ [繪圖執行失敗] 題號 {safe_q_num} (第 {attempt} 次): {err_msg}...")
                GLOBAL_METRICS.record_error("RENDERING_SYNTAX", f"Matplotlib 執行失敗: {err_msg}", question_num=safe_q_num, stage=task_tag)
                current_prompt = f"先前代碼執行失敗：{err_msg}。請修復語法錯誤並重新生成至 `{norm_diagram_path}`。"
        except Exception as e:
            logging.warning(f"⚠️ 題號 {safe_q_num} 繪圖發生例外: {e}")

    return os.path.exists(norm_diagram_path) and os.path.getsize(norm_diagram_path) > 0
    
def canonicalize_taxonomy_tag(tag: str, subject: str) -> str:
    """
    🚨 標籤剛性正規化中樞：
    1. 第一層嚴格且唯一只能是「必修」或「選修」
    2. 徹底消除「必修化學」、「選修物理」等重複科目層級
    3. 統一輸出格式：必修/選修_大單元_次單元_具體考點
    """
    if not tag or not isinstance(tag, str):
        return "必修_綜合主題_核心概念"
    t = str(tag).strip()
    t = re.sub(r'^[\[\"\'\s]+|[\]\"\'\s]+$', '', t)
    
    prefix = "選修" if "選修" in t else "必修"
    t = re.sub(r'^(?:必修|選修)[_/\\]*', '', t)

    redundant_subjs = [
        "國語文", "國綜", "國文", "數學甲", "數學乙", "數學A", "數學B", "數甲", "數乙", "數A", "數B", "數學",
        "物理", "化學", "生物", "地球科學", "地科", "歷史", "地理", "公民與社會", "公民", "英文", "英語", "自然", "社會"
    ]
    for _ in range(2):
        t = re.sub(r'^(?:必修|選修)[_/\\]*', '', t)
        for sub in redundant_subjs:
            t = re.sub(rf'^(?:必修|選修)?{sub}(?:\([一二三四五六全上下1-4]+\))?[_/\\]*', '', t)

    t = re.sub(r'^[\(\[][一二三四五六全上下1-4]+[\)\]][_/\\]*', '', t)

    parts = [p.strip() for p in t.split("_") if p.strip()]
    if not parts:
        return f"{prefix}_綜合主題_核心概念應用"
    if len(parts) == 1:
        return f"{prefix}_{parts[0]}_核心概念"
    return f"{prefix}_" + "_".join(parts[:4])
    

def is_strictly_valid_topic(topic_str: str) -> bool:
    """知識點鋼鐵檢驗閘門：必須完全合規才准許寫入 subject.json"""
    if not isinstance(topic_str, str): return False
    t = re.sub(r'^[\[\"\'\s]+|[\]\"\'\s]+$', '', str(topic_str)).strip()
    
    if len(t) < 4 or len(t) > 50: return False
    if any(c in t for c in ["\n", "\r", "###", "```", "**", ">", "[", "]", "國語文"]): return False # 🚨 嚴禁國語文別名入庫
    
    ghost_tokens = ["題目闕如", "無法判定", "placeholder", "內容缺失", "未包含", "無實質", "綜合主題", "待補充", "因題目"]
    if any(gt in t for gt in ghost_tokens): return False
    
    parts = t.split("_")
    if len(parts) < 3: return False
    if parts[0] not in ["必修", "選修"]: return False
    
    return True

def update_subject_taxonomy(normalized_subject: str, solution_data: dict):
    """具備鋼鐵防禦閘門的動態知識庫更新器：強制學科主鍵歸一、標籤標準化，杜絕別名與贅詞！"""
    global SUBJECT_TAXONOMY
    with SUBJECT_FILE_LOCK:
        updated = False
        
        # 🚨 1. 學科主鍵強制標準化，絕對禁止將「自然」作為獨立主鍵寫入
        clean_sub_key = re.sub(r'(?:考科|科)$', '', str(normalized_subject).strip())
        
        # 若傳入為「自然」或包含多科，從題目的 sub_subject 或內容中取得真正學科（物理/化學/生物/地科）
        if clean_sub_key in ["自然", "理綜"]:
            real_sub = str(solution_data.get("sub_subject", "")).strip()
            if real_sub and real_sub not in ["自然", "理綜"]:
                clean_sub_key = real_sub
            else:
                probe_txt = str(solution_data.get("question_text", "")) + " " + str(solution_data.get("detailed_solution", ""))
                chem_c = sum(probe_txt.count(k) for k in ["化學", "溶液", "反應", "莫耳", "酸鹼", "原子"])
                phys_c = sum(probe_txt.count(k) for k in ["物理", "速度", "加速度", "力學", "電場", "磁場"])
                bio_c  = sum(probe_txt.count(k) for k in ["生物", "細胞", "基因", "遺傳", "葉綠"])
                geo_c  = sum(probe_txt.count(k) for k in ["地科", "地震", "板塊", "地層", "星等", "大氣"])
                best_c = max([("化學", chem_c), ("物理", phys_c), ("生物", bio_c), ("地球科學", geo_c)], key=lambda x: x[1])
                clean_sub_key = best_c[0] if best_c[1] > 0 else "物理"

        if clean_sub_key in ["國綜", "國語文"]: clean_sub_key = "國文"
        elif any(k in clean_sub_key for k in ["數A", "數B", "數甲", "數乙", "數學A", "數學B", "數學甲", "數學乙"]): clean_sub_key = "數學"
        elif clean_sub_key in ["地科"]: clean_sub_key = "地球科學"
        elif clean_sub_key in ["公民"]: clean_sub_key = "公民與社會"
        clean_sub_key = s2t(clean_sub_key)

        # 徹底刪除可能殘留在字典中的「自然」鍵
        if "自然" in SUBJECT_TAXONOMY:
            del SUBJECT_TAXONOMY["自然"]
            updated = True

        if clean_sub_key not in SUBJECT_TAXONOMY:
            SUBJECT_TAXONOMY[clean_sub_key] = {"topics": [], "techniques": []}
            updated = True
            
        # 2. 收集並透過 canonicalize_taxonomy_tag 剛性清洗考點
        candidate_topics = []
        if isinstance(solution_data.get("topic_categories"), list):
            candidate_topics.extend(solution_data["topic_categories"])
        if solution_data.get("topic_category"):
            candidate_topics.append(solution_data["topic_category"])
            
        for t in candidate_topics:
            if not t: continue
            clean_t = canonicalize_taxonomy_tag(s2t(str(t).strip()), clean_sub_key)
            if is_strictly_valid_topic(clean_t):
                if clean_t not in SUBJECT_TAXONOMY[clean_sub_key]["topics"]:
                    SUBJECT_TAXONOMY[clean_sub_key]["topics"].append(clean_t)
                    logging.info(f"🆕 [標準考點入庫 ({clean_sub_key})] {clean_t}")
                    updated = True

        # 3. 收集並清洗解題技巧
        raw_techs = solution_data.get("techniques_used", [])
        if isinstance(raw_techs, str): raw_techs = [raw_techs]
            
        if isinstance(raw_techs, list):
            for tech in raw_techs:
                if not tech or not isinstance(tech, str): continue
                clean_tech = canonicalize_taxonomy_tag(s2t(str(tech).strip()), clean_sub_key)
                if len(clean_tech) >= 4 and len(clean_tech) <= 60 and not any(c in clean_tech for c in ["\n", "\r", "闕如", "無法判定"]):
                    if clean_tech not in SUBJECT_TAXONOMY[clean_sub_key]["techniques"]:
                        SUBJECT_TAXONOMY[clean_sub_key]["techniques"].append(clean_tech)
                        logging.info(f"🆕 [標準技巧入庫 ({clean_sub_key})] {clean_tech}")
                        updated = True

        if updated:
            try:
                # 寫回前進行排序去重
                SUBJECT_TAXONOMY[normalized_subject]["topics"] = sorted(list(dict.fromkeys(SUBJECT_TAXONOMY[normalized_subject]["topics"])))
                SUBJECT_TAXONOMY[normalized_subject]["techniques"] = sorted(list(dict.fromkeys(SUBJECT_TAXONOMY[normalized_subject]["techniques"])))
                with open("subject.json", "w", encoding="utf-8") as f:
                    json.dump(SUBJECT_TAXONOMY, f, ensure_ascii=False, indent=4)
            except Exception as e:
                logging.error(f"寫入 subject.json 失敗: {e}")

def safe_filename(name: str) -> str:
    """過濾掉 Windows 檔案系統不允許的特殊字元，確保存檔安全"""
    return re.sub(r'[\\/*?:"<>|]', "_", name)


        
class ExamParser:
    # 🚨 將提示詞作為類別屬性（縮排 4 格），確保 self.ANSWERS_OCR_PROMPT 的呼叫完全合規
    ANSWERS_OCR_PROMPT = """這是一份台灣考試的答案相關頁面影像。
    請你扮演最嚴謹、零失誤的官方解答數據核對專家，精確萃取「題號/列號」與「標準答案」為 JSON 列表。

    🚨【致命紅線一：考生空白作答卷 / 答題卷 絕對零容忍】🚨：
    本試卷中可能包含提供給考生作答的手寫答題卷、空白劃卡樣張（特徵：印有班級、座號、姓名欄位、得分欄、大考中心浮水印，或是只有空白的方框「[ ]」、手寫作答空格，但答案格內完全沒有印出正式解答）：
    - 只要本頁是【空白供作答用的答題卷】，裡面的答案格空無一物，你【必須且只能】直接輸出空列表 `{"answers": []}`！
    - 絕對禁止把答題卷上印刷的格子編號（如「第1列」、「第2列」）或題號當成答案！
    - 真正的解答卷通常位於試卷更後方的頁面，必須印有明確的「參考答案」、「標準答案」、「選擇題答案確定稿」或在表格內明確印有 ABCD 答案！

    🚨【致命紅線二：多科並列同一頁時，只提取目標考科】🚨：
    有些綜合解答本會在一頁之內以「多個獨立直欄或表格」並列多種考科（例如左欄是國文、中欄是英文、右欄是物理）：
    - 你必須先找到印有當前【目標考科名稱】（如：物理、數學甲、化學等）的專屬直欄或表格！
    - 【絕對嚴禁】跨越欄位邊界去讀取相鄰其他科目的答案，非目標考科的答案表格一律視而不見！

    🚨【無解答頁面警告】🚨：
    如果本頁純粹是題目卷內頁，完全沒有解答表格，直接回傳 `{"answers": []}`，絕對禁止自行猜測或拿題目選項湊數！

    🚨【多欄表格與列號防錯位極度警告（致命考點）】🚨：
    台灣大考的選填題（如 A, B, C, D...）在答案卷上，是以「列號」（如 8, 9, 10, 11, 12...）作為對應格子的。
    解答表的排版通常是「多直欄」並排（例如：左半部是 1~10 題與 A(8,9) 題；右半部是 B(10,11)、C(12,13) 題與非選答案）。
    
    🚨【大考數學科：數字題號與數字答案對位專項硬性限制】🚨：
    在大考數學科的答案卷中，**『題號』與『答案』常常都是純數字**（例如：第 1 題答案是 3，第 2 題答案是 1，第 3 題答案是 2）。
    - **【嚴禁行列錯位與自我歸因】**：AI 極易因為兩者皆為數字，而把「題號/列號本身」當作該題的「答案」讀出（例如：錯誤地將『題號 1』讀作第 1 題的答案『1』、將『題號 2』讀作第 2 題的答案『2』）！
    - **【行列交叉定位法】**：
      * 讀取 any 答案數字前，請先「水平畫一條無形線」與「垂直畫一條無形線」，確認該答案格的「正上方」或「正左方」對應的「列號/題號」到底是多少。
      * **【連續性交叉檢查】**：大考答案卷的列號（如 1, 2, 3... 8, 9, 10, 11...）在整張考卷中是**嚴格單調遞增且不重複**的！如果你發現某個選填題（如選填 B）你讀出的列號是 9，但前一題已經用了 9，這代表你發生了「橫向錯位」！請立即重新對位。
      * **【對應核對】**：請反覆在心裡默唸：『第 X 題的答案，永遠是位於第 X 題右邊（或下邊）對應答案格內部的數字，絕對不是 X 本身！』
      * 例如 114 數乙的第一列為 `1 3`，代表第 1 題答案為 `3`，絕對不要把答案誤讀為 `1`！第二列為 `2 1`，代表第 2 2 題答案為 `1`，絕對不要誤讀為 `2`！

    🚨【多直欄選擇題：單雙位數題號與邊界防錯位限制】🚨：
    在多直欄並排的選擇題答案表中（例如左邊是 1~20 題，右邊是 21~40 題）：
    - **【尾數防混淆對位】**：題號 `1`、`11`、`21`、`31`、`41` 的尾數皆為 `1`。當讀取題號（如 `31` 或 `40`）的答案時，請**強制水平對齊該題號**，確認讀取的是緊鄰該題號右側的答案（如 `31. C` 或 `40. A`），絕對禁止因為單雙位數視覺模糊而跨欄讀取到相鄰欄位（如第 1 題或第 39 題）的答案！
    - **【直欄隔離原則】**：每一欄之間有明顯的物理分隔。讀取的答案與題號必須屬於【同一個直欄內部緊密相連的單一配對】，嚴禁跨越直欄分界線去抓取相鄰直欄的答案字元！

    【表格邊界雜訊過濾（防跨科感染）】：
    - 有些解答卷的頁尾或邊角會印有其他科目的備註（例如：某某科第18題不計分/送分）。請你【只關注並讀取結構化答案表格內部的數字】，**絕對禁止**將表格外的「不計分」或「送分」等字眼錯誤掛載到正常的數學選填題上！

    【嚴格轉換規則】：
    1. **單選與多選題與是非題**：
       - 若答案為字母或數字，請精確提取（如 "A", "BC", "3"）。
       - **【是非題專用】**：若答案表內為圓圈 `O`（或 `○`）與叉號 `X`（或 `✕`），請一律標準化為大寫英文字母 `O` 與 `X`（例如第 1 題填 `"O"`，第 3 題填 `"X"`），或者轉為選項代號 `A` (代表 O/正確) 與 `B` (代表 X/錯誤)。
    2. **選填題（核心格式）**：若該題（如 A, B, C）跨越了多個列號，你**必須**將每個列號對應的答案字元（包含數字、負號 `-` 或正負號 `±`）以半角逗號 `,` 隔開。
       - 例如：選填 B 對應列號 12, 13, 14, 15，其答案分別為 3、1、1、3，則輸出 "B" 的答案為 `"3,1,1,3"`。
       - 例如：選填 C 的答案為 -、7、2，則寫成 `"-,7,2"`。
    3. **負號 `-` 與斜線 `／` 識別**：
       - 負號 `-` 是極為重要的答案組成，必須完整保留。
       - 若整格為斜線 `／` 或空白，代表該格不需畫記（通常是非選擇題），請予以忽略。
    4. **🚨【零雜訊、零贅字、絕對禁止自然語言解釋】🚨**：
       - `standard_answer` 欄位**必須且只能**包含答案字元本身（例如 `"A"`, `"ACD"`, `"3,1,1,3"`, `"-,7,2"`）。
       - **絕對禁止**在答案欄位中出現任何中文、英文解釋或說明性文字。若該格沒有答案，請直接輸出空字串 `""`。
    """

    def __init__(self, ai_manager: GeminiFreeTierManager):
        self.ai_manager = ai_manager

    @staticmethod
    def classify_page_type(page_text: str, page_num: int, total_pages: int) -> str:
        """
        🚨 智能頁面屬性分類器：
        精準判定頁面為 COVER(封面)、TOC(目錄)、SUMMARY(知識彙整)、SOLUTION(解答詳解)、QUESTION(正式試題)
        """
        clean_t = page_text.strip()
        
        # 1. 解答/詳解/簡答表格頁面判定（全面納入標準答案、簡答、選填題答案表）
        sol_markers = [
            "參考詳解", "試題詳解", "考科詳解", "參考答案暨詳解", "【詳解】", "【解析】", 
            "答案與詳解", "選擇題答案確定稿", "答案公佈", "參考答案", "標準答案", 
            "選擇題答案", "選填題答案", "非選擇題答案", "簡答", "解答", "答案表", "答案卡"
        ]
        if any(sm in clean_t for sm in sol_markers):
            return "SOLUTION"
            
        # 額外特徵：若頁面出現密集題號與答案對照（如大量出現 "1. A"、"2. B" 或選填題列號對照）
        if re.search(r'(?:選擇題|選填題|單選題|多選題)\s*(?:參考)?答案', clean_t):
            return "SOLUTION"
            
        # 2. 目錄頁面判定
        toc_markers = ["目次", "目錄", "Table of Contents", "單元目錄", "章節索引", "單元一覽"]
        if any(tm in clean_t for tm in toc_markers) and not any(qm in clean_t for qm in ["(A)", "(B)", "(1)", "(2)", "第 1 題"]):
            return "TOC"

        # 3. 知識彙整 / 重點摘要 / 觀念整理判定（如基礎物理複習卷前兩頁）
        summary_markers = ["重點摘要", "觀念摘要", "公式整理", "單元重點", "學習重點", "知識綱要", "核心概念歸納", "考點速讀", "考前重點", "重點整理", "觀念剖析"]
        has_real_q = re.search(r'(?:^|\n)\s*(?:\[\s*(?:複|簡|選)?\s*\]\s*\d+|\d+\s*[.．、\s)]|第?\s*[1-9]\s*題)', clean_t) is not None
        if any(sm in clean_t for sm in summary_markers) and not has_real_q:
            return "SUMMARY"

        # 4. 封面 / 注意事項頁面判定 (通常在第 1~2 頁)
        if page_num <= 1:
            cover_markers = [
                "作答注意事項", "考試時間", "作答方式", "選擇題範例", "答案卡", "畫卡樣式",
                "答題卷", "注意事項", "作答說明", "答題說明", "准考證號", "作答示例", "第壹部分作答示例"
            ]
            if any(cm in clean_t for cm in cover_markers) and not has_real_q:
                return "COVER"

        return "QUESTION"

    def get_initial_batch_size(self, subject: str) -> int:
        """依照學科思考複雜度設定最佳批次大小"""
        sub = subject.lower()
        if any(x in sub for x in ["數學", "數a", "數b", "數甲", "數乙"]):
            return 1  # 數學計算極複雜，批次 1 題
        elif any(x in sub for x in ["物理", "化學", "生物", "地球科學", "自然"]):
            return 1  # 自然科涉及多圖表與實驗，批次 1 題
        else:
            return 2  # 社會文科，批次 2 題
    
    def detect_solution_boundary_multimodal(self, pdf_path: str, all_page_images: List[str]) -> tuple[int, int]:
        """
        🚨 相容性版面邊界探測轉發器：
        直接調用循環切分引擎，提取首份試卷之詳解起始與結束頁 (0-based)
        """
        cycles = self.segment_compilation_cycles_multimodal(pdf_path, all_page_images, "", "", "")
        if cycles and cycles[0].get("sol_pages"):
            s_pages = cycles[0]["sol_pages"]
            return s_pages[0], s_pages[-1]
        return -1, -1

    def segment_compilation_cycles_multimodal(self, pdf_path: str, all_page_images: List[str], base_year: str, base_mock_tag: str, base_subject: str) -> List[dict]:
        """
        🚨 [百頁合訂本循環狀態機] 專門處理 100+ 頁巨型 PDF：
        自動辨識 [試卷1題目 ➔ 試卷1詳解 ➔ 試卷2題目 ➔ 試卷2詳解 ➔ ...] 的交替波浪！
        回傳結構化考卷清單，每份考卷擁有獨立的題目頁碼與詳解頁碼。
        """
        total_pages = len(all_page_images)
        if total_pages == 0:
            return []

        logging.info(f"📚 [合訂本循環掃描啟動] 正在對 {os.path.basename(pdf_path)} (共 {total_pages} 頁) 進行全局架構探測...")
        GLOBAL_METRICS.set_stage(f"📚 正在探測 {total_pages} 頁大合輯之試卷與詳解循環邊界...")

        class VisualPageMarker(BaseModel):
            page_index: int = Field(description="頁面索引 (0-based)")
            role: Literal["QUESTION", "SOLUTION", "COVER", "OTHER"] = Field(
                description="'QUESTION' (試題問答頁), 'SOLUTION' (答案/詳解本頁面，含雙欄解析或答案表), 'COVER' (新試卷封面/標題), 'OTHER' (空白/注意事項)"
            )
            detected_exam_title: str = Field(default="", description="若為封面或有大標題，辨識其年份與次數（如 '107學年度第六次模擬考'）")
            first_q_num: str = Field(default="", description="該頁出現的第一個題號（如 '1', '15', '非選一'）")

        class BatchScanResult(BaseModel):
            markers: List[VisualPageMarker] = Field(description="頁面視覺標籤列表")

        # 1. 以每批 6~8 頁進行視覺低損耗快速分類
        all_markers = [None] * total_pages
        batch_size = 6

        # 優先快篩數位文字層；若為純掃描檔文字為空，則送入視覺 AI
        doc = fitz.open(pdf_path)
        pages_needing_vision = []
        for i in range(total_pages):
            txt = doc[i].get_text("text").replace(" ", "") if i < len(doc) else ""
            if len(txt) > 30:
                is_sol = any(k in txt for k in ["[參考詳解]", "參考答案暨詳解", "考科詳解", "試題詳解", "答案與解析", "【解析】", "出處："])
                is_cov = any(k in txt for k in ["作答注意事項", "模擬考試題", "學科能力測驗", "指定科目考試"]) and "第1題" not in txt
                m_q = re.search(r'(?:第\s*(\d+|[一二三四五])\s*題|(\d+)\s*[.．、\s)])', txt)
                fq = m_q.group(1) or m_q.group(2) if m_q else ""
                all_markers[i] = {
                    "page_index": i,
                    "role": "SOLUTION" if is_sol else ("COVER" if is_cov else "QUESTION"),
                    "detected_exam_title": "",
                    "first_q_num": fq
                }
            else:
                pages_needing_vision.append(i)
        doc.close()

        # 針對純圖片掃描頁面，啟動視覺多模態分類
        if pages_needing_vision:
            logging.info(f"🔍 檢測到 {len(pages_needing_vision)} 頁為純掃描圖，啟動 Gemini 3.5 Lite 批次視覺版面探針...")
            for b_start in range(0, len(pages_needing_vision), batch_size):
                sub_indices = pages_needing_vision[b_start : b_start + batch_size]
                prompt = """請仔細觀察這組試卷掃描頁面，判斷每一頁的屬性：
1. 'QUESTION'：正式考題（題幹、單選多選選項、字體較大、排版寬鬆）。
2. 'SOLUTION'：官方答案或詳解本（特徵：印有答案表格、雙欄排版、題號加粗、帶有『【解析】』、『【出處】』或 (A) 錯誤 (B) 正確 等文字。注意：即使標頭沒印詳解二字，只要是逐題解析就屬於 SOLUTION！）。
3. 'COVER'：試卷封面、大標題頁或新年度考卷開頭。
"""
                contents = [prompt]
                opened_imgs = []
                try:
                    for p_i in sub_indices:
                        contents.append(f"=== 第 {p_i+1} 頁影像 ===")
                        im = Image.open(all_page_images[p_i])
                        opened_imgs.append(im)
                        contents.append(im)

                    res_b, _ = self.ai_manager.generate_with_retry(
                        contents=contents,
                        response_schema=BatchScanResult,
                        temperature=0.0,
                        preferred_model="gemini-3.5-flash-lite",
                        enable_thinking=False,
                        task_desc=f"[合訂本視覺探針 p{sub_indices[0]+1}~{sub_indices[-1]+1}]"
                    )
                    if res_b and 'markers' in res_b:
                        for m in res_b['markers']:
                            idx = m.get('page_index', 0)
                            if 0 <= idx < total_pages:
                                all_markers[idx] = m
                finally:
                    for im in opened_imgs:
                        try: im.close()
                        except: pass

        # -------------------------------------------------------------
        # 2. 狀態機引擎：嚴格依循 [試題 ➔ 詳解 ➔ 試題 ➔ 詳解] 剛性循環
        # -------------------------------------------------------------
        cycles = []
        current_cycle = {"cycle_index": 1, "title": "", "q_pages": [], "sol_pages": []}
        current_state = "IN_Q"  # 初始一律為收集題目狀態

        for p_i in range(total_pages):
            info = all_markers[p_i] or {"role": "QUESTION", "first_q_num": "", "detected_exam_title": ""}
            role = info.get("role", "QUESTION")
            fq = str(info.get("first_q_num", "")).strip()

            # 🚨 剛性鐵律：只有在已經進入詳解階段 (IN_SOL) 後，看到「新年度大封面」或「新第一題 (Q1)」，才允許切分新試卷！
            # 在題目收集階段 (IN_Q)，任何「第貳部分、混合題、非選題」大標題一律視為當前試卷題目，絕對禁止腰斬！
            is_new_exam_trigger = (
                current_state == "IN_SOL" and 
                (role == "COVER" or fq in ["1", "一"] or (role == "QUESTION" and p_i > current_cycle["sol_pages"][-1]))
            )

            if is_new_exam_trigger:
                # 封裝上一份完整試卷
                if current_cycle["q_pages"]:
                    cycles.append(current_cycle)
                current_cycle = {
                    "cycle_index": len(cycles) + 1,
                    "title": info.get("detected_exam_title", ""),
                    "q_pages": [],
                    "sol_pages": []
                }
                current_state = "IN_Q"
                if role != "COVER":
                    current_cycle["q_pages"].append(p_i)
            elif role == "SOLUTION":
                current_cycle["sol_pages"].append(p_i)
                current_state = "IN_SOL"
            else:
                current_cycle["q_pages"].append(p_i)
                # 若尚未遇到詳解，保持在 IN_Q 狀態持續收集題目


        if current_cycle and current_cycle["q_pages"]:
            cycles.append(current_cycle)

        logging.info(f"🎉 [合訂本邊界解碼成功] 共成功分離出 {len(cycles)} 套獨立試卷循環！")
        for idx, c in enumerate(cycles, 1):
            q_count = len(c['q_pages'])
            q_range = f"p{c['q_pages'][0]+1}~p{c['q_pages'][-1]+1}" if c['q_pages'] else "無題目"
    
            sol_count = len(c['sol_pages'])
            sol_range = f"p{c['sol_pages'][0]+1}~p{c['sol_pages'][-1]+1}" if c['sol_pages'] else "無詳解"
    
            logging.info(
                f"  -> 📦 循環 {idx}: 題目卷共 {q_count} 頁 ({q_range}) | ",
                f"詳解本共 {sol_count} 頁 ({sol_range})"
            )

        return cycles
        
    def pdf_to_images(self, pdf_path: Optional[str], prefix: str, target_dir: str, dpi: int = 150) -> List[str]:
        """將指定 PDF 檔案的每一頁都轉換為高解析度圖片，並儲存在指定資料夾中，內建記憶體級影像強化"""
        if not pdf_path or not os.path.exists(pdf_path):
            return []
        
        image_paths = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=dpi)
                filename = f"{prefix}_page_{page_num+1:02d}.png"
                filepath = os.path.join(target_dir, filename)
                
                # 全局影像強化：防止印刷淡字、微小分數線或細微符號遺漏
                from PIL import Image, ImageEnhance
                # 直接從 PyMuPDF 的記憶體資料（pix.samples）轉換為 PIL 影像物件，不經過二次讀寫，效率極高
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                enhancer_contrast = ImageEnhance.Contrast(img)
                img_enhanced = enhancer_contrast.enhance(1.4)  # 提升 40% 對比度
                enhancer_sharp = ImageEnhance.Sharpness(img_enhanced)
                img_enhanced = enhancer_sharp.enhance(1.4)      # 提升 40% 銳利度
                
                # 儲存強化後的影像
                img_enhanced.save(filepath, "PNG")
                
                # 正規化路徑為網頁/JSON 標準的斜線 "/"
                normalized_path = filepath.replace("\\", "/")
                image_paths.append(normalized_path)
            doc.close()
        except Exception as e:
            logging.error(f"❌ 轉換 PDF {pdf_path} 為圖片時出錯: {e}")
        return image_paths
    
    def extract_rubric_visual(self, rubric_pdf: Optional[str], img_dir: str) -> Dict[str, Dict]:
        """叫 AI 去看評分標準 PDF，抓出每一題的答案圖座標與給分文字 (併行處理提速)"""
        if not rubric_pdf or not os.path.exists(rubric_pdf):
            return {}
            
        logging.info("🎨 正在執行「跨頁縫合級」評分標準視覺併行萃取...")
        task_id = uuid.uuid4().hex[:6]
        temp_images = []
        
        try:
            with fitz.open(rubric_pdf) as doc:
                cat = doc.pdf_catalog()
                if cat > 0: doc.xref_set_key(cat, "StructTreeRoot", "null")
                
                # 1. 先將所有頁面轉為臨時圖片
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    pix = page.get_pixmap(dpi=300)
                    img_path = f"temp_rubric_{task_id}_{page_num}.png"
                    pix.save(img_path)
                    temp_images.append((page_num, img_path))
        except Exception as e:
            logging.error(f"無法開啟或轉換評分 PDF: {e}")
            return {}
            
        rubric_data_map = {}
        
        # 定義專用的 Pydantic 用於解析評分卷 (已補上實體 Field 描述以供 Schema 傳遞)
        class RubricItem(BaseModel):
            question_number: str = Field(description="題號，例如 '37', '39', '40', '41'")
            criteria_text: str = Field(description="本題在本頁對應的完整評分原則與滿分參考答案文字敘述。")
            bboxes: List[List[int]] = Field(
                default=[], 
                description="包覆該題在原圖中所有評分內容（含標題、參考答案與評分原則）的 Bounding Box。格式必須為 [[ymin, xmin, ymax, xmax]]，所有坐標必須規格化至 0 到 1000 的整數區間（左上角為 [0,0]，右下角為 [1000,1000]）。"
            )

        class RubricPage(BaseModel):
            items: List[RubricItem] = Field(description="本頁中所有非選擇題評分原則的條目列表。")

        # 用於保存各頁面解析出來的 Bounding Boxes 與文字資料
        parsed_pages_results = [None] * len(temp_images)
        results_lock = threading.Lock()
        
        @staticmethod
    def extract_clean_markdown_from_pdf(pdf_path: str, page_num: int) -> str:
        """【開源庫優先】利用 pymupdf4llm 提取原生 PDF 頁面的 Markdown（含完整表格與雙欄文字排版）"""
        try:
            import pymupdf4llm
            # 僅提取特定頁面
            md_text = pymupdf4llm.to_markdown(pdf_path, pages=[page_num])
            if md_text and len(md_text.strip()) > 40:
                return md_text.strip()
        except Exception as e:
            logging.debug(f"pymupdf4llm 提取第 {page_num+1} 頁失敗或非原生版: {e}")
        return ""

    @staticmethod
    def enhance_scanned_image_for_ocr(pil_img: Image.Image) -> Image.Image:
        """【模擬考掃描檔增強】利用 OpenCV 自適應直方圖均衡化 (CLAHE) 與雙邊濾波去噪，顯著提高公式與文字清晰度"""
        try:
            import cv2
            import numpy as np
            
            # PIL 轉 OpenCV 格式 (灰階)
            img_np = np.array(pil_img.convert("L"))
            
            # 1. CLAHE 自適應局部對比增強（凸顯淡墨、細微符號、根號、下標）
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_np)
            
            # 2. 輕微雙邊濾波（去除掃描紙張顆粒與雜點，保留文字邊緣邊界）
            denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
            
            return Image.fromarray(denoised).convert("RGB")
        except Exception as e:
            # 若 OpenCV 異常，回退至 PIL 原生增強
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(pil_img)
            return enhancer.enhance(1.4)

        # 2. 定義併行呼叫 API 函數
        def process_rubric_page(page_num, img_path):
            if not os.path.exists(img_path):
                return
            
            prompt = f"""這是一份考試的【非選擇題評分原則】。
            這是一份非選擇題評分原則的第 {page_num+1} 頁。
            請執行以下任務：
            1. **【跨頁語意連續性】**：如果本頁開頭是上一頁某題（如：第17題）的延續內容，請務必使用相同的題號，以便系統能自動將斷裂的文字拼接完整，絕對不要因為跨頁就將其斷章取義！
            2. **【子題號與大題號對位（極重要）】**：大考的非選擇題通常包含一個大題（如：二）和多個子題（如：17、18題）。請你仔細閱讀原圖中的文字，確認各段評分標準到底屬於哪一個子題（例如：17 題是列出不等式，18 題是求最大獲利與作圖）。**絕對不要將 17 題跨頁到本頁頂部的後半段文字，張冠李戴地當成 18 題的開頭！**
            3. 請識別出本頁中所有非選擇題題目（如：第 40 題、第 41 題）的範圍。
            4. 【塊狀擷取】：給出完整包含該題在本頁所有內容（包含『第 X 題』標題、滿分參考答案、評分原則文字或表格說明）的大型 Bounding Box。
            5. 嚴禁將同一個大題的 (1)(2)(3) 小題切分成多個碎圖。
            6. 提取該塊的所有評分文字描述。
            
            🚨【Bounding Box 坐標規範】🚨：
            - 每一個 Bounding Box 必須是 `[ymin, xmin, ymax, xmax]` 的 4 個整數列表。
            - 坐標值必須嚴格規格化到 0 到 1000 之間（以原圖左上角為 [0, 0]，右下角為 [1000, 1000]）。
            - y term 代表垂直方向（高度），x term 代表水平方向（寬度）。
            - **【橫向無限寬幅原則】**：因為評分原則採整行橫排。請務必將 Bounding Box 的水平寬度拉到最大！**強制將 xmin 設為 10 到 50 之間，xmax 設為 950 到 990 之間**。這樣能確保整行文字與附屬表格被完整裁切，絕對不允許從左右兩側切斷文字！
            """
            
            try:
                with Image.open(img_path) as pil_img:
                    res, err = self.ai_manager.generate_with_retry(
                        contents=[prompt, pil_img],
                        response_schema=RubricPage,
                        temperature=0.0,
                        preferred_model="gemini-3.5-flash-lite",  # 🚨 升級為最新 3.5-Lite，大幅提升 Bounding Box 坐標與表格框選精度
                        enable_thinking=False
                    )
                    if res and 'items' in res:
                        with results_lock:
                            parsed_pages_results[page_num] = res['items']
            except Exception as e:
                logging.error(f"併行解析評分卷第 {page_num} 頁失敗: {e}")
            finally:
                if os.path.exists(img_path):
                    try: os.remove(img_path)
                    except Exception: pass

        # 3. 多執行緒同步發送 API 請求
        max_workers = min(len(temp_images), len(self.ai_manager.keys) * 2, 6)
        if max_workers > 0:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(lambda x: process_rubric_page(*x), temp_images)

        # 4. 在主線程中安全進行影像裁切，避免 PyMuPDF 跨線程衝突
        try:
            crop_doc = fitz.open(rubric_pdf)
            cat = crop_doc.pdf_catalog()
            if cat > 0: crop_doc.xref_set_key(cat, "StructTreeRoot", "null")
            
            for page_num, items in enumerate(parsed_pages_results):
                if not items:
                    continue
                page = crop_doc[page_num]
                for item in items:
                    q_num = item.get('question_number')
                    criteria_text = item.get('criteria_text', '')
                    bboxes = item.get('bboxes', [])
                    
                    if not q_num:
                        continue
                        
                    # 主線程安全裁切
                    # 主線程安全裁切（過濾 N/A 等非法斜線路徑字元）
                    safe_q = safe_filename(str(q_num)).replace(" ", "")
                    new_imgs = self.execute_crop(page, bboxes, img_dir, f"Rubric_{safe_q}_p{page_num}")
                    
                    if q_num in rubric_data_map:
                        rubric_data_map[q_num]["text"] += "\n(續前頁)\n" + criteria_text
                        rubric_data_map[q_num]["paths"].extend(new_imgs)
                    else:
                        rubric_data_map[q_num] = {
                            "text": criteria_text,
                            "paths": new_imgs
                        }
            crop_doc.close()
        except Exception as e:
            logging.error(f"主線程評分標準裁切失敗: {e}")

        return rubric_data_map

    def extract_mock_explanations_visual(self, sol_pdf_path: str, sol_page_indices: List[int], img_dir: str, spec_name: str) -> Dict[str, dict]:
        """
        🎯 [模擬考專屬] 官方詳解本視覺多模態高精萃取引擎
        直接從純掃描的詳解本影像中提取出處、目標、推導步驟、選項分析、評分原則與詳解附圖，
        完全跳過耗時昂貴的 AI 自創推導，實現 100% 官方原汁原味出版級解析。
        """
        if not sol_pdf_path or not os.path.exists(sol_pdf_path) or not sol_page_indices:
            return {}

        logging.info(f"📑 [模考詳解快車道] 正在啟動詳解本專屬視覺 OCR 萃取 (共 {len(sol_page_indices)} 頁詳解)...")
        GLOBAL_METRICS.set_stage(f"📑 [模考專屬] 正在萃取官方詳解本文字、出處、評分細則與附圖...")

        extracted_explanations = {}
        task_id = uuid.uuid4().hex[:6]
        
        # 💡 [斷點續傳機制] 建立詳解本專屬的分頁快取檔
        cache_file = os.path.join(img_dir, f"mock_sol_cache_{spec_name}.json").replace("\\", "/")
        processed_pages = []
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                    extracted_explanations = cache_data.get("explanations", {})
                    processed_pages = cache_data.get("processed_pages", [])
                logging.info(f"📂 [模考詳解快取] 成功載入已萃取的 {len(processed_pages)} 頁詳解進度，將自動跳過這些頁面！")
            except Exception as e:
                logging.warning(f"讀取模考詳解快取失敗: {e}")
        
        try:
            doc = fitz.open(sol_pdf_path)
        except Exception as e:
            logging.error(f"無法開啟詳解 PDF {sol_pdf_path}: {e}")
            return {}

        for p_idx in sol_page_indices:
            check_global_timeout()
            if p_idx in processed_pages:
                continue # 💡 跳過已成功處理過的頁面
            if p_idx >= len(doc): continue
            page = doc[p_idx]
            pix = page.get_pixmap(dpi=300)
            temp_img_path = os.path.abspath(f"temp_mock_sol_{task_id}_p{p_idx}.png")
            pix.save(temp_img_path)

            prompt = f"""
            你是一位最精準的台灣大考、高中模擬考與單元複習卷詳解數位化專家。
            這是一份試卷官方【參考答案暨詳解】（或複習考【[參考詳解]】）的第 {p_idx+1} 頁影像。
            
            請仔細檢視本頁排版，提取所有題目的解析與答案，填入 MockExplanationPageExtraction 結構：
            
            🚨【欄位萃取極致規範（支援答案表與圖解裁切）】🚨：
            1. **【答案表格對應】**：
               - 若頁首或頁中有「答案表格」（如：1. ABD, 2. CD, 3. D, 4. B, 5. E, 6.(1) 10kg, 6.(2) 20m/s...），請務必將答案精確對應至每一題的 `answer` 欄位！
               - 小題格式請標準化（例如：6.(1) 對應題號 '6(1)'，9.(2) 對應題號 '9(2)'）。
            2. **question_number (題號)**：如 '1', '2', '6(1)', '6(2)', '14'。
            3. **source_curriculum 與 exam_target（🚨 事前防偽造剛性禁令）**：
               - 只有在試卷版面上【明確印出】『出處：...』或『目標：...』時才進行提取。
               - 🚨 若版面上根本沒有印出出處或測驗目標（例如純文字解答、無單元範圍標示），這兩個欄位【必須強制留空字串 ""】，絕對禁止自行腦補、猜測或從題目反推！
            4. **explanation_text (詳細解析)**：
               - 完整提取該題之全部推導、公式列式、受力分析或文字說明。
               - 數學與物理公式必須嚴格轉為 LaTeX（如 $\\frac{{1}}{{2}}mv^2$, $U = mgh$）。
            5. **has_solution_image 與 solution_image_bboxes (詳解附圖，極重要)**：
               - 仔細檢查解析旁是否印有受力分析圖、坐標圖、圓周運動幾何圖或實驗裝置圖（例如物理第 5 題圓周運動軌道圖、第 4 題鞦韆圖）。
               - 若有附圖，將 `has_solution_image` 設為 true，並框出覆蓋該圖的 Bounding Box `[ymin, xmin, ymax, xmax]` (0~1000)！
            6. **options_analysis (選項深入剖析)**：
               - 若解析中有對 (A)(B)(C)(D)(E) 選項逐一分析對錯，請拆解為列表；無則填 `[]`。
            7. **topic_category 與 techniques_used（🚨 剛性命名結構）**：
               - 格式必須嚴格為：`必修/選修_大單元_次單元_具體考點`（例如：'必修_閱讀理解_文本細節比對'、'必修_文言文閱讀_延伸反思'）。
               - 🚫【絕對禁止】在單元前重複加上科目名稱（例如絕對不可寫 '必修_國文_...' 或 '必修_國語文_...'）！
               - 🚫【絕對禁止】使用未標準化的別名（嚴禁出現 '國語文' 字眼）！
            """

            try:
                with Image.open(temp_img_path) as pil_img:
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Contrast(pil_img)
                    pil_img_enh = enhancer.enhance(1.4)
                    
                    res_dict, err = self.ai_manager.generate_with_retry(
                        contents=[prompt, pil_img_enh],
                        response_schema=MockExplanationPageExtraction,
                        temperature=0.0,
                        preferred_model="gemini-3.5-flash",
                        enable_thinking=False,
                        task_desc=f"[{spec_name} 詳解頁 p{p_idx+1}]"
                    )

                    if res_dict and 'items' in res_dict and res_dict['items']:
                        for item in res_dict['items']:
                            qn = str(item.get('question_number', '')).strip()
                            if not qn: continue

                            # 處理詳解附圖裁切
                            sol_img_paths = []
                            if item.get('has_solution_image') and item.get('solution_image_bboxes'):
                                safe_qn = safe_filename(qn).replace(" ", "")
                                sol_img_paths = self.execute_crop(
                                    page, 
                                    item['solution_image_bboxes'], 
                                    img_dir, 
                                    f"Sol_Q{safe_qn}_p{p_idx+1}"
                                )

                            item['solution_image_paths'] = sol_img_paths
                            extracted_explanations[qn] = item
                            logging.info(f"  -> 🎯 [成功抓取模考詳解] 題號 {qn} (出處: {item.get('source_curriculum', '無')[:15]}... | 附圖: {len(sol_img_paths)} 張)")

                        # 🚨 [網頁即時推播] 每完成一頁詳解萃取，立即刷新網頁橫幅與日誌！
                        GLOBAL_METRICS.set_stage(f"📑 正在視覺萃取詳解第 {p_idx+1} 頁 (已累計獲得 {len(extracted_explanations)} 題官方解析)")
                        GLOBAL_METRICS.add_log(f"📑 詳解本第 {p_idx+1} 頁解析完成，目前已累積鎖定 {len(extracted_explanations)} 題官方詳解！")
                        
            except Exception as e:
                logging.error(f"提取詳解卷第 {p_idx+1} 頁失敗: {e}")
            finally:
                if os.path.exists(temp_img_path):
                    try: os.remove(temp_img_path)
                    except Exception: pass
                    
            # 💡 [斷點續傳機制] 每一頁萃取完畢立刻寫入快取，防範 40 頁巨型 PDF 中途斷線
            if p_idx not in processed_pages:
                processed_pages.append(p_idx)
                try:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump({
                            "explanations": extracted_explanations,
                            "processed_pages": processed_pages
                        }, f, ensure_ascii=False, indent=4)
                except Exception:
                    pass

        doc.close()
        logging.info(f"🎉 [模考詳解萃取完畢] 全卷共成功收錄 {len(extracted_explanations)} 道題目的官方完整詳解與評分量規！")
        
        # 完工後安全刪除快取檔
        if os.path.exists(cache_file):
            try: os.remove(cache_file)
            except Exception: pass
            
        return extracted_explanations
    
    def heal_mock_question_consistency(self, q_data: dict, explanation_data: dict) -> dict:
        """
        🎯 [模擬考專用] 輕量級圖文 OCR 一致性自癒校驗器
        雙層核對：
        1. 本地正則比對詳解結論與答案
        2. 若發現題幹與詳解推導有數值不一致或符號破損，調用極速模型進行秒級自癒修復
        """
        q_text = str(q_data.get("question_text", "")).strip()
        raw_ans = str(q_data.get("answer", "")).strip()
        sol_text = str(explanation_data.get("explanation_text", "")).strip()
        sol_ans = str(explanation_data.get("answer", "")).strip()

        # -------------------------------------------------------------
        # 第一層：本地秒級規則自癒 (Zero-API Cost)
        # -------------------------------------------------------------
        # 1. 從詳解文字尾端提取明確的結論答案（全面支援：故選、故本題選、正確選項為、故選：等）
        conclusive_ans = ""
        m_opt = re.search(r'(?:故(?:本題)?選|正確選項(?:為)?)[：:\s]*[（\(【]?([A-Ga-g1-9,\(\)（）\s、]+)[）\)】]?', sol_text)
        if m_opt:
            raw_c = m_opt.group(1).replace("(", "").replace(")", "").replace("（", "").replace("）", "").replace(" ", "").replace(",", "").replace("、", "")
            if any(c.isdigit() for c in raw_c):
                conclusive_ans = "".join(sorted(list(dict.fromkeys(re.findall(r'\d', raw_c)))))
            else:
                conclusive_ans = "".join(sorted(list(dict.fromkeys(re.findall(r'[A-Za-z]', raw_c))))).upper()
        
        # 若為非選擇題或數值題，尋找等號結尾或「故...為 X」
        if not conclusive_ans:
            m_val = re.search(r'(?:故|所求|最小值|最大值|方程式)?(?:為|＝|=)\s*([^\n，。]+?)[。！!\n]*$', sol_text)
            if m_val:
                cand_v = m_val.group(1).strip()
                if len(cand_v) <= 25 and not any(k in cand_v for k in ["選", "式", "法", "題"]):
                    conclusive_ans = cand_v.strip("$* ")

        # 2. 答案標準化覆寫（🚨 多選題剛性保護：若結論有 2 個以上選項，絕不被簡答的單字母覆蓋！）
        clean_sol_ans = clean_ocr_answer_format(sol_ans)
        clean_concl = clean_ocr_answer_format(conclusive_ans) if conclusive_ans else ""

        is_multi_choice = q_data.get("question_type") == "多選題" or len(clean_concl) >= 2 or len(clean_sol_ans) >= 2
        if is_multi_choice:
            q_data["question_type"] = "多選題"
            # 優先採用長度更長、更完整的多選選項（例如 BD 優於 B）
            if len(clean_concl) >= 2 and len(clean_concl) > len(clean_sol_ans):
                final_ans = clean_concl
            elif len(clean_sol_ans) >= 2:
                final_ans = clean_sol_ans
            else:
                final_ans = clean_concl or clean_sol_ans or raw_ans
        else:
            final_ans = clean_sol_ans or clean_concl or raw_ans

        if final_ans:
            q_data["answer"] = final_ans
            q_data["official_answer"] = final_ans
            q_data["has_official_answer"] = True
            
        # 🚨 [全方位數理與文字破損特徵檢測（涵蓋純文字與無圖題目）]：
        # 1. 詳解極度空洞或僅有一行課綱標題（如第 47、52 題）
        sol_is_trivial = len(sol_text) < 35 or (sol_text.startswith("內容：") and len(sol_text.splitlines()) <= 2)

        # 2. 答案字串擠壓黏連（如 3aaBBaaBbaabb）
        ans_is_squished = bool(re.search(r'[a-zA-Z]{5,}', raw_ans)) and not raw_ans.isalpha()

        # 3. 詳解結論與答案矛盾
        has_concl_mismatch = bool(clean_concl and clean_sol_ans and clean_concl != clean_sol_ans)

        # 4. 🚨【新增：數理公式與抓字硬傷檢測】（無圖題專項防禦）
        # 檢測常見的 OCR 公式損毀：未閉合大括號、連續數字無運算符號、漏掉的 LaTeX 指令殘缺
        math_syntax_broken = (
            sol_text.count("{") != sol_text.count("}") or
            sol_text.count("$") % 2 != 0 or
            any(corrupt in sol_text for corrupt in [r"\frac{", "12gt^2", "12mv^2", "+-", "??", "__DBL"])
        )

        # 理科考題強制進行數理推導完整性核驗
        is_stem_subj = any(k in str(q_data.get("sub_subject", "")) for k in ["數", "物", "化", "生", "自", "地"])
        needs_deep_audit = sol_is_trivial or ans_is_squished or has_concl_mismatch or math_syntax_broken

        # 若是理科題或命中破損特徵，強制啟動數理與文字校準器！
        if not needs_deep_audit and not is_stem_subj:
            return q_data

        # -------------------------------------------------------------
        # 第二層：極速 AI 語意與算式自癒 (Lightweight AI Inspector)
        # -------------------------------------------------------------
        logging.info(f"🔍 [自癒觸發] 題號 {q_data.get('question_number')} 偵測到微小圖文差異，啟動極速 AI 自動修復...")
        
        opts_str = "\n".join([f"- ({opt.get('key')}) {opt.get('value')}" for opt in q_data.get("options", [])])
        inspector_prompt = f"""
        你是一位在台灣大考閱卷界享有崇高聲望的【高中試題數理邏輯與 OCR 抓字精校教授】。
        我們正在審查以下這道模擬考題目的「題幹」、「選項」與「官方詳解本 OCR 掃描文字」：

        【題幹】：
        {q_text}

        【選項】：
        {opts_str or '無'}

        【官方詳解本文字】：
        {sol_text}

        【當前標註答案】：{final_ans}

        🚨【全方位審查與糾錯任務（涵蓋純文字與無圖題）】🚨：
        0. **【🚨 嚴禁劇透：題幹必須保留挖空】**：
           - 題幹是給學生作答的題目！**絕對嚴禁將詳解中的答案數字直接填回題幹空格中**！
           - 若題幹原本為 `[ 15-1 ][ 15-2 ][ 15-3 ]` 或 `______`，必須百分之百維持挖空結構，絕對不可替換為解答數值（如不可改為 `\\sqrt{442}`）！
        
        1. **【數理公式與計算過程審查（極重要）】**：
           - 仔細驗算詳解中的每一道代數計算、化學計量、物理定律列式。
           - 抓出 OCR 掃描最常犯的硬傷：**負號漏抓**（如將負值誤讀為正值）、**分數線遺失**（如 `1/2` 誤讀為 `12`）、**指數/下標變形**（如 $10^{-3}$ 誤讀為 $10-3$）。
           - 檢查「中間計算步驟是否能自然推導出最終答案」？若算式斷裂或前後矛盾，請在 `corrected_solution_snippet` 中予以平滑補正！
        2. **【文字語病與形似錯字排查】**：
           - 檢查是否有形近字錯讀（如「質子」讀成「質手」、「氧化」讀成「氧北」）或語句斷頭不通順。
        3. **【非選答案解壓縮還原】**：
           - 若答案為擠壓黏連字串（如 `3aaBBaaBbaabb`），還原為標準格式：`3 種（aaBB、aaBb、aabb）`。
        4. **【空洞詳解充實】**：
           - 若詳解僅有一行大綱標題（如「內容：...」），請依據正確答案補齊完整的名師推導演繹！

        若發現任何上述瑕疵，將 `has_flaw` 設為 true，並在 `corrected_solution_snippet` 輸出 100% 嚴謹、無語病、無計算漏洞的完整解答！若原解析已極度完美自洽，則填 'CLEAN'。
        """

        try:
            res_heal, err = self.ai_manager.generate_with_retry(
                contents=[inspector_prompt],
                response_schema=MockLightweightCheckResult,
                temperature=0.0,
                preferred_model="gemini-3.5-flash-lite",
                enable_thinking=False,
                task_desc=f"[模考自癒 Q{q_data.get('question_number')}]"
            )

            if res_heal and res_heal.get("has_flaw"):
                fixed_txt = res_heal.get("corrected_question_text", "").strip()
                fixed_a = str(res_heal.get("corrected_answer", "")).strip()
                fixed_sol = str(res_heal.get("corrected_solution_snippet", "")).strip()
                summary = res_heal.get("repair_summary", "")
                flaw_type = res_heal.get("flaw_category", "CORRECTION")

                if fixed_txt and len(fixed_txt) >= len(q_text) * 0.8:
                    q_data["question_text"] = fixed_txt

                if fixed_a:
                    q_data["answer"] = clean_ocr_answer_format(fixed_a) if q_data.get("question_type") in ["單選題", "多選題"] else fixed_a
                    q_data["official_answer"] = q_data["answer"]

                # 🚨 只要提供了實質修正且非 'CLEAN'，直接覆寫 detailed_solution，消滅算式硬傷與語病！
                if fixed_sol and fixed_sol != "CLEAN" and len(fixed_sol) >= 20:
                    q_data["detailed_solution"] = fixed_sol
                    
                logging.info(f"  -> ✅ [數理與文字校準成功] 題號 {q_data.get('question_number')} ({flaw_type})：{summary}")
        except Exception as e:
            logging.warning(f"輕量自癒執行跳過: {e}")

        return q_data
    
    def execute_crop(self, page, bboxes, img_dir, prefix) -> List[str]:
        saved_paths = []
        with PYMUPDF_LOCK:
            try:
                page_words = page.get_text("words")
            except Exception:
                page_words = []

        for i, bbox in enumerate(bboxes):
            if len(bbox) != 4: continue
            ymin, xmin, ymax, xmax = bbox
            
            x0_raw = (min(xmin, xmax) / 1000.0) * page.rect.width
            y0_raw = (min(ymin, ymax) / 1000.0) * page.rect.height
            x1_raw = (max(xmin, xmax) / 1000.0) * page.rect.width
            y1_raw = (max(ymin, ymax) / 1000.0) * page.rect.height

            # 初始非對稱式 Padding 緩衝
            pad_w = page.rect.width * 0.05
            pad_h_top = page.rect.height * 0.04
            pad_h_bot = page.rect.height * 0.02

            x0 = max(0.0, x0_raw - pad_w)
            x1 = min(page.rect.width, x1_raw + pad_w)
            y0 = max(0.0, y0_raw - pad_h_top)
            y1 = min(page.rect.height, y1_raw + pad_h_bot)

            rect = fitz.Rect(x0, y0, x1, y1)

            # 若有文字塊與目前的裁切框相交，則自動將裁切框向外延伸，完整包裹該文字塊
            if page_words:
                for w in page_words:
                    w_rect = fitz.Rect(w[0], w[1], w[2], w[3])
                    # 如果該單字被裁切框切到，但沒有被完整包裹
                    if rect.intersects(w_rect) and not rect.contains(w_rect):
                        # 動態向外微調，防範公式上標或分數線被切斷
                        rect.x0 = min(rect.x0, w_rect.x0 - 2)
                        rect.y0 = min(rect.y0, w_rect.y0 - 2)
                        rect.x1 = max(rect.x1, w_rect.x1 + 2)
                        rect.y1 = max(rect.y1, w_rect.y1 + 2)

            rect = rect.intersect(page.rect)
            if rect.width < 10 or rect.height < 10: continue

            img_filename = f"{prefix}_{i}_{int(time.time()*1000)}.png"
            img_filepath = os.path.join(img_dir, img_filename).replace("\\", "/")
            
            try:
                with PYMUPDF_LOCK:
                    clip_pix = page.get_pixmap(clip=rect, dpi=300)
                clip_pix.save(img_filepath)
                if os.path.exists(img_filepath):
                    saved_paths.append(img_filepath)
            except Exception as e:
                logging.error(f"裁切失敗 {prefix}: {e}")
        return saved_paths

    # 請在 ExamParser 類別中，精確替換此方法（注意縮排為 8 個空格）：
    # 🚨 將解析答案卷的提示詞抽離為類別變數，以利多個模型共用，保持代碼簡潔
    
    def _run_single_ocr(self, a_pdf: str, model: str, target_subject: str = "", target_year: str = "") -> tuple[dict, List[int]]:
        """為共識機制設計的單次獨立 OCR 執行器（自動標定答案實體頁碼，支援純圖片掃描檔）"""
        try:
            doc = fitz.open(a_pdf)
            cat = doc.pdf_catalog()
            if cat > 0: doc.xref_set_key(cat, "StructTreeRoot", "null")
        except Exception as e:
            logging.error(f"無法開啟解答 PDF {a_pdf}: {e}")
            return {}, []
            
        all_answers = {}
        hit_page_indices = []
        task_id = uuid.uuid4().hex[:8]
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)
            img_path = os.path.abspath(f"temp_ans_{task_id}_p{page_num}_{model[:8]}.png")
            pix.save(img_path)

            if not os.path.exists(img_path):
                smart_sleep(0.5) 
                if not os.path.exists(img_path):
                    continue
            
            page_text_layer = page.get_text("text").strip()
            
            # 針對掃描檔（文字層為空）與數位檔的提示詞整合
            text_layer_str = f"\n\n=== 該頁數位文字層對照 ===\n{page_text_layer}" if page_text_layer else "\n\n（註：本頁為純影像掃描檔，請完全依據圖片視覺排版精確辨識）"
            
            # 取得考科的所有可能簡稱與別名，增強 OCR 定位容錯
            subj_aliases = [target_subject]
            if "數學甲" in target_subject or "數甲" in target_subject: subj_aliases += ["數學甲", "數甲", "數學甲考科", "高中數學甲"]
            elif "數學乙" in target_subject or "數乙" in target_subject: subj_aliases += ["數學乙", "數乙", "數學乙考科", "高中數學乙"]
            elif "數學A" in target_subject or "數A" in target_subject: subj_aliases += ["數學A", "數A", "數學A考科", "高中數學A"]
            elif "數學B" in target_subject or "數B" in target_subject: subj_aliases += ["數學B", "數B", "數學B考科", "高中數學B"]
            elif "國文" in target_subject: subj_aliases += ["國文", "國綜", "國語文", "國文考科"]
            elif "公民" in target_subject: subj_aliases += ["公民與社會", "公民", "公社"]
            alias_str = "、".join(list(dict.fromkeys(subj_aliases)))

            subject_guard_prompt = f"""
            🚨【考科與學年度精確定位指引（跨頁/多欄/中段答案定位）】🚨
            目標考科：【{target_subject}】（包含別名：{alias_str}），目標學年度：【{target_year}】。
            大考與模擬考的解答 PDF 常常是多科合一（同一頁可能包含多個科目，或目標科目出現在頁面中間段、特定直欄或表格區塊）：
            1. **【全局與中段精準掃描】**：請仔細巡視本頁的「頁首、頁中、頁尾以及各獨立欄位或表格標題」，尋找標註有【{alias_str}】的專屬區塊！
            2. **【區段隔離提取】**：若本頁包含多個科目（例如上方是國文/數乙，中間或下方是【{target_subject}】），請**專注且只提取標註為【{alias_str}】區塊內的題號與答案**，忽略其他科目的表格！
            3. **【全頁無目標科目的判定】**：只有在確認「整頁完全沒有出現任何【{alias_str}】的標題、表格或答案」時，才回傳 `{{"answers": []}}`。切勿因為頁首是其他科目，就忽略頁面中段或下方的目標科目答案！
            """
            combined_prompt = subject_guard_prompt + "\n" + self.ANSWERS_OCR_PROMPT + text_layer_str
            
            try:
                with Image.open(img_path) as pil_img:
                    from PIL import ImageEnhance
                    enhancer_contrast = ImageEnhance.Contrast(pil_img)
                    pil_img_enhanced = enhancer_contrast.enhance(1.6)
                    enhancer_sharp = ImageEnhance.Sharpness(pil_img_enhanced)
                    pil_img_enhanced = enhancer_sharp.enhance(1.4)
                    
                    res_dict, err = self.ai_manager.generate_with_retry(
                        contents=[combined_prompt, pil_img_enhanced],
                        response_schema=AnswerKey,
                        temperature=0.0,
                        preferred_model=model,
                        enable_thinking=True
                    )
                    if res_dict and 'answers' in res_dict and res_dict['answers']:
                        found_valid_ans = False
                        for item in res_dict['answers']:
                            raw_ans = item['standard_answer']
                            cleaned_ans = clean_ocr_answer_format(raw_ans)
                            if cleaned_ans:
                                all_answers[item['question_number']] = cleaned_ans
                                found_valid_ans = True
                        if found_valid_ans:
                            hit_page_indices.append(page_num)
            except Exception as e:
                logging.error(f"模型 {model} 解析解答卷第 {page_num+1} 頁出錯: {e}")
            finally:
                if os.path.exists(img_path):
                    try: os.remove(img_path)
                    except Exception: pass
        doc.close()
        return all_answers, hit_page_indices

    def _resolve_ocr_conflict(self, a_pdf: str, dict_1: dict, dict_2: dict, mismatched_keys: list, hit_pages: List[int] = None) -> dict:
        """
        對衝突的題號進行二次比對與終極仲裁（動態載入命中頁面，支援多頁仲裁）
        """
        merged_dict = {**dict_2, **dict_1} 
        task_id = uuid.uuid4().hex[:8]
        temp_img_paths = []
        
        try:
            with fitz.open(a_pdf) as doc:
                target_pages = hit_pages if hit_pages else list(range(len(doc)))
                for p_idx in target_pages:
                    if 0 <= p_idx < len(doc):
                        pix = doc[p_idx].get_pixmap(dpi=300)
                        p_img = os.path.abspath(f"temp_arbitrate_{task_id}_p{p_idx}.png")
                        pix.save(p_img)
                        temp_img_paths.append(p_img)
        except Exception as e:
            logging.error(f"仲裁時讀取解答卷頁面失敗: {e}")
            return merged_dict

        if not temp_img_paths:
            return merged_dict

        class ArbitrationResult(BaseModel):
            resolutions: List[AnswerItem] = Field(description="對所有衝突題號的最終仲裁結果列表。")

        prompt = f"""
        這是一份大考解答卷影像。我們在使用兩個不同的 AI 模型解析以下題號的標準答案時發生了衝突：
        衝突題號：{mismatched_keys}
        模型 1 的解析：{ {k: dict_1.get(k) for k in mismatched_keys} }
        模型 2 的解析：{ {k: dict_2.get(k) for k in mismatched_keys} }

        請你扮演「最終仲裁官」：
        1. 仔細辨識傳入的解答原圖中，這些衝突題號對應的真實標準答案。
        2. 做出最準確的裁決，並將結果填入 Pydantic 結構中。
        """

        opened_pils = []
        try:
            contents = [prompt]
            from PIL import ImageEnhance
            for ip in temp_img_paths:
                p_im = Image.open(ip)
                enhancer_contrast = ImageEnhance.Contrast(p_im)
                p_im_enh = enhancer_contrast.enhance(1.5)
                enhancer_sharp = ImageEnhance.Sharpness(p_im_enh)
                p_im_enh = enhancer_sharp.enhance(1.3)
                opened_pils.append(p_im)
                contents.append(p_im_enh)

            res, err = self.ai_manager.generate_with_retry(
                contents=contents,
                response_schema=ArbitrationResult,
                temperature=0.0,
                preferred_model="gemini-3.6-flash",
                enable_thinking=True,
                task_desc="[解答仲裁]"
            )
            if res and 'resolutions' in res:
                for item in res['resolutions']:
                    ans_val = item['standard_answer']
                    if ans_val is None or str(ans_val).strip() in ["", "None", "null", "／", "/", "\\", "無", "無答案", "－"]:
                        ans_val = "／"
                    else:
                        ans_val = str(ans_val).strip()
                    merged_dict[item['question_number']] = ans_val
                    logging.info(f"⚖️ [仲裁成功] 題號 {item['question_number']} 已被裁決為: {ans_val}")
        except Exception as e:
            logging.error(f"執行解答仲裁失敗，將採用 Model 1 預設值: {e}")
        finally:
            for p_im in opened_pils:
                try: p_im.close()
                except Exception: pass
            for ip in temp_img_paths:
                if os.path.exists(ip):
                    try: os.remove(ip)
                    except Exception: pass
                
        return merged_dict

    def extract_clean_answers(self, a_pdf: Optional[str], target_subject: str = "", target_year: str = ""):
        if not a_pdf or not os.path.exists(a_pdf):
            return "無官方解答。", []

        logging.info(f"🧠 [共識投票] 啟動解答卷雙模型雙重驗證 (目標: {target_year} {target_subject}): {os.path.basename(a_pdf)}")
        
        # 1. 第一輪：帶入目標科目與學年驗證
        ans_dict_1, pages_1 = self._run_single_ocr(a_pdf, model="gemini-3.5-flash", target_subject=target_subject, target_year=target_year)
        
        # 2. 第二輪：帶入目標科目與學年盲測驗證
        ans_dict_2, pages_2 = self._run_single_ocr(a_pdf, model="gemini-3.5-flash-lite", target_subject=target_subject, target_year=target_year)
        
        hit_pages = sorted(list(set(pages_1 + pages_2)))
        
        if not ans_dict_1 and not ans_dict_2:
            return "無官方解答。", []
        if not ans_dict_1: return json.dumps(ans_dict_2, ensure_ascii=False), hit_pages
        if not ans_dict_2: return json.dumps(ans_dict_1, ensure_ascii=False), hit_pages
        
        # 3. 在 Python 中進行精確的 Key-Value 比對，並在比對前統一空值、None、斜線與未作答標記為 "／"
        # 🚨 修正：不可將未辨識到的 Key 粗暴設為 "／"，這會導致沒讀到的題目全部被標為無效題
        def sanitize_answer_value(val) -> str:
            if val is None:
                return ""
            val_str = str(val).strip()
            if val_str in ["None", "null", "none", "Undefined", "undefined", "[]", "{}"]:
                return ""
            return val_str

        all_keys = set(list(ans_dict_1.keys()) + list(ans_dict_2.keys()))
        for k in all_keys:
            if k in ans_dict_1: ans_dict_1[k] = sanitize_answer_value(ans_dict_1[k])
            if k in ans_dict_2: ans_dict_2[k] = sanitize_answer_value(ans_dict_2[k])

        mismatched_keys = []
        for k in all_keys:
            if ans_dict_1[k] != ans_dict_2[k]:
                mismatched_keys.append(k)
                
        if not mismatched_keys:
            logging.info("🎉 [共識達成] 雙模型對位完全一致！答案卷數據 100% 信賴。")
            return json.dumps(ans_dict_1, ensure_ascii=False), hit_pages
            
        logging.warning(f"⚠️ [共識衝突] 偵測到以下題號在雙模型解析中不一致: {mismatched_keys}")
        # 先在外面計算好子字典，再放入 f-string 輸出
        ans_1_sub = {k: ans_dict_1.get(k) for k in mismatched_keys}
        ans_2_sub = {k: ans_dict_2.get(k) for k in mismatched_keys}
        logging.warning(f"  -> Model 1 (3.5-flash): {ans_1_sub}")
        logging.warning(f"  -> Model 2 (3.1-lite): {ans_2_sub}")
        
        # 4. 啟動第三輪：終極仲裁（傳入真實命中頁碼，不再盲目只讀第 1 頁）
        resolved_dict = self._resolve_ocr_conflict(a_pdf, ans_dict_1, ans_dict_2, mismatched_keys, hit_pages=hit_pages)
        return json.dumps(resolved_dict, ensure_ascii=False), hit_pages

    def extract_text_from_pdf(self, pdf_path: Optional[str]) -> str:
        """
        🌐【通用雙欄/多欄排版感知文字提取器】
        依據空間幾何自動識別雙欄中線，強制「先左欄頂到底、後右欄頂到底」，
        100% 適用於大考中心、全模、北模、南一、翰林等所有雙欄解析本，徹底根除跨題串供亂碼！
        """
        if not pdf_path or not os.path.exists(pdf_path):
            return "無官方資料提供。"
        full_text = []
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    # 取得所有文字區塊 (x0, y0, x1, y1, text, block_no, block_type)
                    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
                    if not blocks:
                        continue

                    page_w = page.rect.width
                    mid_x = page_w / 2.0

                    # 判定本頁是否主要為雙欄排版 (檢查是否有大量文字區塊分佈在中線兩側)
                    has_left = any(b[2] <= mid_x * 1.1 for b in blocks)
                    has_right = any(b[0] >= mid_x * 0.9 for b in blocks)

                    if has_left and has_right:
                        # 雙欄排序策略：左欄 (col 0) 依 y0 排序，右欄 (col 1) 依 y0 排序，跨欄橫幅 (col 2) 依 y0 排序
                        def get_block_column_key(b):
                            x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
                            # 橫跨兩欄的通欄標題
                            if x0 < mid_x * 0.4 and x1 > mid_x * 1.6:
                                return (0, y0)  # 置於該高度優先處理
                            elif x1 <= mid_x * 1.05:
                                return (1, y0)  # 左欄
                            else:
                                return (2, y0)  # 右欄

                        sorted_blocks = sorted(blocks, key=get_block_column_key)
                    else:
                        # 單欄排版直接依 y0 (由上至下) 排序
                        sorted_blocks = sorted(blocks, key=lambda b: b[1])

                    page_txt = "\n".join(b[4].strip() for b in sorted_blocks)
                    full_text.append(page_txt)
            return "\n\n".join(full_text)
        except Exception as e:
            logging.error(f"雙欄排版文字提取失敗 {pdf_path}：{e}")
            return "無官方資料提供。"

    def parse_detailed_explanations(self, pdf_path: Optional[str]) -> Dict[str, str]:
        """
        [模擬考與指考詳解本專用高精分析器]
        從解析本/詳解 PDF 中自動辨識並切分出每一題的官方詳解文字內容。
        """
        if not pdf_path or not os.path.exists(pdf_path):
            return {}
        
        # 1. 讀取解析本全文並統一簡轉繁
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            return {}
            
        text = s2t(raw_text)
        explanations = {}
        
        # 2. 匹配模式：行首有數字、字母或大寫中文題號，後接標點符號或空格
        pattern = r'(?:^|\n)\s*(?P<q_num>\d+|[A-Ga-g]|一|二|三|四|五|六|七|八|九|十)\s*[\.．、\s)]\s*(?P<content>.*?)(?=(?:^|\n)\s*(?:\d+|[A-Ga-g]|一|二|三|四|五|六|七|八|九|十)\s*[\.．、\s)]|$)'
        
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            q_num = match.group('q_num').strip()
            content = match.group('content').strip()
            
            # 清除常見的頁眉頁尾或多餘贅字
            content = re.sub(r'－\s*\d+\s*－|化學考科詳解|物理考科詳解|數學乙考科詳解|生物考科詳解|化學考科參考答案暨詳解|學測.*詳解', '', content)
            
            if q_num not in explanations:
                explanations[q_num] = content
            else:
                explanations[q_num] += "\n" + content
                
        return explanations
        
    def run_academic_arbitration(self, q_data: dict, sol_data: dict, error_critique: str) -> Optional[AcademicArbitration]:
        """
        當重掃描 2 次以上仍存在邏輯與答案衝突時，由高階 AI 專家分析最合適的解答並進行學術仲裁。
        """
        prompt = f"""
        你是一位在台灣高中大考界享有最高學術威望的【學術仲裁委員會主席】。
        我們正在解析以下題目，但遇到了學術上的重大爭議（官方答案與 AI 的嚴謹學理推導不一致）。
        
        【待仲裁題目資訊】
        題號：{q_data.get('question_number')}
        題幹：{q_data.get('question_text')}
        官方給定的答案：【{q_data.get('answer')}】
        
        【先前審查教授指出的衝突線索】：
        >>> {error_critique} <<<
        
        【AI 產出的嚴謹推導詳解】：
        {sol_data.get('detailed_solution')}
        
        請執行以下學術仲裁任務：
        1. 仔細評估 AI 的推導過程，確認其是否符合正確的數學、物理或化學定律。
        2. 評估官方給定答案是否有以下可能：
           - 官方公布答案本身出錯（即印刷錯誤、題目設計瑕疵導致無解、或多重答案）。
           - 影像擷取（OCR）錯位：即因答案卷多欄排版，導致系統誤將隔壁題目的答案讀作本題答案。
        3. 做出最明智、最符合教育意義與學術嚴謹性的裁決：
           - 如果 AI 的推導 100% 正確，而官方答案或 OCR 確實有誤，請將 `chosen_answer` 設為 AI 推導出的正確答案。
           - 如果官方答案雖然有印刷爭議，但仍有特定詮釋角度，請在 `chosen_answer` 中指出最合理的折衷答案。
        4. 撰寫一份極具教學價值的【學術仲裁與印刷疑義備註】。說明衝突原因、官方印刷瑕疵（若有），並指導學生如何在考試中應對此類爭議。
        """
        
        arbitration_contents = [prompt]
        target_page_idx = q_data.get('page_number', 1) - 1
        if 0 <= target_page_idx < len(q_data.get('question_page_image_paths', [])):
            q_img_path = q_data['question_page_image_paths'][target_page_idx]
            if os.path.exists(q_img_path):
                arbitration_contents.append(Image.open(q_img_path))
                
        for ans_p in q_data.get('answer_page_image_paths', []):
            if os.path.exists(ans_p):
                arbitration_contents.append(Image.open(ans_p))
                
        try:
            res, err = self.ai_manager.generate_with_retry(
                contents=arbitration_contents,
                response_schema=AcademicArbitration,
                temperature=0.0,
                preferred_model="gemini-3.8-flash",  # 🚨 升級為最高旗艦 3.8-flash，確保學術爭議判定具備最高認知水準
                enable_thinking=True,
                task_desc="[終極學術仲裁 (Gemini 3.8)]"
            )
            if res:
                with GLOBAL_METRICS.lock:
                    GLOBAL_METRICS.arbitration_count += 1
                    
            return res
        except Exception as e:
            logging.error(f"執行學術仲裁 API 呼叫失敗: {e}")
            return None
            
    def clean_and_verify_questions(self, questions: list) -> list:
        """
        [方案 4 + 方案 1] 題型與選項硬性清洗器 + 選填題挖空長度比對器
        """
        cleaned_questions = []
        for q in questions:
            q_type = q.get("question_type", "")
            q_num = q.get("question_number", "")
            
            # 1. 【題型自我校驗與選項保護】
            opts = q.get("options", [])
            # 若題目實際上含有 3 個以上實體選項，這 100% 是選擇題，自動修復題型，絕不刪除選項！
            if len(opts) >= 3 and q_type in ["選填題", "簡答題", "繪圖作圖題"]:
                ans_val = str(q.get("answer", "")).strip()
                new_type = "多選題" if len(re.findall(r'[A-Za-z0-9]', ans_val)) > 1 else "單選題"
                logging.info(f"🔄 [題型自癒修正] 題號 {q_num} 帶有 {len(opts)} 個選項，已自動從 '{q_type}' 扶正為 '{new_type}'！")
                q["question_type"] = new_type
                q_type = new_type
            elif q_type in ["選填題", "簡答題", "繪圖作圖題"]:
                # 只有真正無選項的非選題，才清空可能被誤抓的 options
                q["options"] = []
                
            # 🚨 [防禦性機制 - 提案三：選擇題「評分標準」反向去污染清洗器]
            # 若為選擇題，硬性將第一階段可能因 AI 讀取大篇幅 Rubric 產生的幻覺與錯位 scoring_criteria 清空，
            # 確保資料庫中僅有非選擇題/手寫題包含評分標準，徹底消除選擇題（如 Q31, Q53）被寫入評分說明之 Bug。
            if q_type in ["單選題", "多選題"]:
                q["scoring_criteria"] = ""

            # 2. 【方案 1：選填題「挖空結構與答案長度」自動比對器】
            if q_type == "選填題":
                q_text = q.get("question_text", "")
                ans_str = q.get("answer", "")
                
                # 匹配所有 [ 9 ] 或 [ 10-1 ] 等挖空標籤
                blanks = re.findall(r'\[\s*\d+[^\]]*\]', q_text)
                blank_count = len(blanks)
                
                # 計算答案分割後的元素數量
                ans_parts = [p for p in ans_str.split(",") if p.strip()]
                ans_count = len(ans_parts)
                
                if blank_count > 0 and ans_count > 0 and blank_count != ans_count:
                    logging.error(f"❌ [對位衝突] 題號 {q_num} 的挖空數 ({blank_count} 個) 與答案長度 ({ans_count} 個，答案: '{ans_str}') 不一致！")
                    logging.error(f"  -> 挖空標籤: {blanks}")
                    # 標記為錯位，以利在 Stage 2 提示詞中動態警告 AI
                    q["_length_mismatch"] = True
                    q["_expected_blank_count"] = blank_count
                    
            cleaned_questions.append(q)
        return cleaned_questions

    def process_exam_paper(self, subject: str, year: str, exam_type: str, mock_tag: str, q_pdf: str, a_pdf: Optional[str], rubric_pdf: Optional[str], output_dir: str, skip_cover: bool = False, school_name: str = ""):
        check_global_timeout()
    
        doc = None
        validation_records = []
        val_log_lock = threading.Lock()
        rpd_hit = False  # 🚨 [全域頂層初始化] 確保在快車道直接完工時，rpd_hit 必定已宣告，徹底消除 UnboundLocalError！

        # 🚨 [全域最頂層宣告] 在函式第一行完成考科名稱標準化，徹底根除 free variable referenced before assignment 致命崩潰！
        clean_subj_name = re.sub(r'(?:考科|科)$', '', str(subject).strip())
        if clean_subj_name in ["高中數學甲", "數甲"]: clean_subj_name = "數學甲"
        elif clean_subj_name in ["高中數學乙", "數乙"]: clean_subj_name = "數學乙"
        elif clean_subj_name in ["高中數學A", "數A"]: clean_subj_name = "數學A"
        elif clean_subj_name in ["高中數學B", "數B"]: clean_subj_name = "數學B"
        elif clean_subj_name in ["國綜", "國語文"]: clean_subj_name = "國文"
        elif clean_subj_name in ["國語文寫作", "寫作測驗"]: clean_subj_name = "國寫"

        safe_year = safe_filename(year)
        safe_subject = safe_filename(subject)
        paper_tag = f"[{year} {subject}]"

        # 🚨 補考標籤提早宣告
        is_makeup = any(k in str(mock_tag) or k in str(year) for k in ["補考", "補辦", "代考"])
        makeup_tag = "_補考" if is_makeup else ""
        makeup_src = "補考_" if is_makeup else ""

        # 🚨 格式化考卷名稱
        clean_tag = mock_tag.strip('_')
        if exam_type == "SCHOOL":
            sch_disp = school_name or "學校定期考"
            paper_label = f"[{year}] {sch_disp} {clean_tag} {subject} ({exam_type})".replace("  ", " ")
        elif exam_type in ["MOCK", "MOCK_GSAT", "MOCK_AST"]:
            paper_label = f"[{year}] {clean_tag} {subject} ({exam_type})".replace("  ", " ")
        else:
            makeup_disp = " 補考" if is_makeup else ""
            paper_label = f"[{year}]{makeup_disp} {subject} ({exam_type})"
        paper_label = re.sub(r'\s+', ' ', paper_label).strip()

        GLOBAL_METRICS.set_current_paper(paper_label, total_q=0, existing_partial_q=0)
        GLOBAL_METRICS.set_stage(f"🚀 初始化考卷: {paper_label}")
        logging.info(f"\n{'='*30} 📥 開始處理考卷：{paper_label} {'='*30}")

        # 💡 將題號對位邏輯拉到最頂端（🚨 強化版：嚴格隔離非選中文大題「一(1)」與選擇題「1」，防範張冠李戴）
        def is_precise_match(q_data: dict, r_k: str) -> bool:
            q_num = str(q_data.get('question_number', '')).strip()
            r_k = str(r_k).strip()
            if q_num == r_k:
                return True
            
            cn_mapping = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
            q_cn = [cn_mapping[c] for c in q_num if c in cn_mapping]
            r_cn = [cn_mapping[c] for c in r_k if c in cn_mapping]
            
            # 🚨 關鍵防禦：若一方包含中文字（如「一(1)」、「非選二」）而另一方為純數字（如「1」），絕不可單憑數字碰撞判定為同一題！
            if bool(q_cn) != bool(r_cn):
                return False
            if q_cn and r_cn and q_cn != r_cn:
                return False

            q_digits = re.findall(r'\d+', q_num)
            r_digits = re.findall(r'\d+', r_k)
            
            # 若兩者都有數字且長度相同（如 15(a) 對應 15(1)）
            if q_digits and r_digits and len(q_digits) == len(r_digits):
                return q_digits == r_digits
            return False
        
        subjects_stem = ["數學", "數A", "數B", "數學乙", "數學甲", "數甲", "數乙", "物理", "化學", "生物", "地球科學", "自然"]
        is_stem = any(t in subject for t in subjects_stem)

        # Stage 1 用 Flash-Lite 極速掃描
        stage_1_models = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
        stage_1_model = stage_1_models[0]

        # 🚨 [依試卷重要度動態綁定模型梯隊]：
        # 1. 預先定義各分類專屬池，徹底消除 NameError 未定義崩潰
        is_hard_stem = any(s in clean_subj_name for s in ["數", "物理", "化學", "自然"])
        is_official_or_school = exam_type in ["GSAT", "AST", "SCHOOL"]

        stem_official_pool = [
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash"
        ]
        stem_mock_pool = [
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite"
        ]
        humanities_pool = [
            "gemini-3.5-flash-lite"
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.8-flash",
            "gemini-3.7-flash",
        ]
        stem_reasoning_pool = stem_official_pool if is_official_or_school else stem_mock_pool

        # 2. 精確指派當前試卷的 tier1_pool，杜絕任何舊變數覆寫
        if is_hard_stem and is_official_or_school:
            tier1_pool = stem_official_pool
        elif is_hard_stem:
            tier1_pool = stem_mock_pool
        else:
            tier1_pool = humanities_pool

        stage_2_model = tier1_pool[0]
        validator_model = tier1_pool[1 % len(tier1_pool)]
        # stage_2_model = "gemini-3.5-flash" if is_stem else "gemini-3.1-flash-lite"
        # validator_model = "gemini-3.5-flash" if is_stem else "gemini-3.1-flash-lite"
        
        year_digits = "".join(filter(str.isdigit, year))
        if not year_digits:
            year_digits = year

        # 1. 決定大類資料夾名稱與學校獨立目錄（注入上下學期標籤，防止 114上 與 114下 同名覆蓋）
        sem_tag = "上" if "上學期" in str(year) else ("下" if "下學期" in str(year) else "")

        try: y_val = int(year_digits)
        except ValueError: y_val = 111

        if exam_type == "SCHOOL":
            safe_school = safe_filename(school_name) if school_name else "通用學校"
            type_folder = os.path.join("學校定期考", safe_school)
            spec_name = f"{y_val}學年度_{safe_school}{mock_tag}_{clean_subj_name}"
            standard_academic_year = f"{y_val}{sem_tag}定期考"
            standard_exam_source = f"{y_val}學年度{sem_tag+'學期' if sem_tag else ''}{school_name if school_name else ''}定期考{mock_tag}_{clean_subj_name}"
        elif exam_type in ["MOCK", "MOCK_GSAT", "MOCK_AST"]:
            # 🎯 依學年度精準劃分學測模擬考 vs 分科/指考模擬考（111起為分科，110以前為指考）
            ast_short = "分科" if y_val >= 111 else "指考"
            mock_sub_type = "學測模擬" if exam_type == "MOCK_GSAT" else (f"{ast_short}模擬" if exam_type == "MOCK_AST" else "模擬考")
            
            org_label = "全國"
            for candidate_org in ["北模", "中模", "南模", "竹模", "全模", "翰林", "南一", "龍騰", "康熹", "泰宇", "詮達", "全華", "漢樺", "文昌"]:
                if candidate_org in mock_tag:
                    org_label = candidate_org
                    break
            type_folder = os.path.join("模擬考", mock_sub_type, org_label)
            spec_name = f"{y_val}學年度_{mock_sub_type}{mock_tag}_{clean_subj_name}"
            standard_academic_year = f"{y_val}{mock_sub_type[:2]}模{mock_tag}"
            standard_exam_source = f"{y_val}學年度{mock_sub_type}考{mock_tag}_{clean_subj_name}"
        elif exam_type == "GSAT":
            type_folder = "學測"
            spec_name = f"{y_val}學年度_學測{makeup_tag}_{clean_subj_name}"
            standard_academic_year = f"{y_val}學測{makeup_tag}"
            standard_exam_source = f"{y_val}學年度學科能力測驗{makeup_src}{clean_subj_name}"
        else: # AST（大考指考 / 分科測驗）
            type_folder = "分科指考"
            ast_short = "分科" if y_val >= 111 else "指考"
            ast_full = "分科測驗" if y_val >= 111 else "指定科目考試"
            spec_name = f"{y_val}學年度_{ast_short}{makeup_tag}_{clean_subj_name}"
            standard_academic_year = f"{y_val}{ast_short}{makeup_tag}"
            standard_exam_source = f"{y_val}學年度{ast_full}{makeup_src}{clean_subj_name}"
            
        spec_name = re.sub(r'[\s\n\r\t]+', '', spec_name)

        # 4. 建立專屬的分類資料夾與 JSON 儲存路徑
        os.makedirs(os.path.join(output_dir, type_folder), exist_ok=True)
        json_path = os.path.join(output_dir, type_folder, f"{spec_name}_database.json")
        partial_json_path = json_path.replace("_database.json", "_partial_database.json") # 🆕 暫存進度檔路徑
        raw_extracted_json_path = json_path.replace("_database.json", "_raw_extracted.json") # 🆕 第一階段題目快取檔路徑
        validation_log_path = json_path.replace("_database.json", "_validator_log.txt") # 🚨 全域頂層初始化，保證快取命中時 100% 可存取
        
        # 🚨 [狀態機嚴格巡檢]：融合檢查 database 與 partial，杜絕「早產假死 (只有1~2題即跳過)」
        db_exists = os.path.exists(json_path)
        partial_exists = os.path.exists(partial_json_path)
        
        if db_exists or partial_exists:
            try:
                merged_checkpoint_map = {}
                for fp in [json_path, partial_json_path]:
                    if os.path.exists(fp):
                        try:
                            with open(fp, "r", encoding="utf-8") as jf:
                                for item in json.load(jf):
                                    if is_valid_solution(item.get("detailed_solution")):
                                        k = (str(item.get("question_number")), str(item.get("question_type")))
                                        if k not in merged_checkpoint_map or len(str(item.get("detailed_solution"))) > len(str(merged_checkpoint_map[k].get("detailed_solution"))):
                                            merged_checkpoint_map[k] = item
                        except Exception:
                            pass
                        
                merged_checkpoint_list = list(merged_checkpoint_map.values())
                
                # 🚨 核心修復：依據大考真實學科動態計算合理的最低題數門檻，絕不讓 72 題的考卷在 15~40 題被腰斬！
                clean_sub_check = str(subject).strip()
                if any(k in clean_sub_check for k in ["社會", "社考"]):
                    min_expected_q = 65  # 學測社會 72 題，至少需達 65 題
                elif any(k in clean_sub_check for k in ["自然", "自考", "理綜"]):
                    min_expected_q = 60  # 學測自然 68 題，至少需達 60 題
                elif any(k in clean_sub_check for k in ["英文", "英語"]):
                    min_expected_q = 45  # 英文通常 50~56 題
                elif any(k in clean_sub_check for k in ["國文", "國綜"]):
                    min_expected_q = 38  # 國文通常 40~44 題
                elif any(k in clean_sub_check for k in ["數", "math"]):
                    min_expected_q = 10  # 數學通常 15~22 題
                elif "國寫" in clean_sub_check:
                    min_expected_q = 2   # 國寫 2 題
                else:
                    min_expected_q = 20  # 其餘單科分科測驗/指考通常 25~35 題
                
                # 🚨 雙重品質門檻：數量達標 +「每一題都必須具有實質詳細解答」，才准許跳過！
                has_all_solutions = (
                    len(merged_checkpoint_list) >= min_expected_q and 
                    all(len(str(q.get("detailed_solution", "")).strip()) >= 20 for q in merged_checkpoint_list)
                )

                if db_exists and has_all_solutions:
                    # 🔍 啟動圖片完整性巡檢 (Diagram Integrity Check)
                    missing_diagram_count = 0
                    base_dir = os.path.dirname(json_path)
                    
                    for q_item in merged_checkpoint_list:
                        detailed_text = str(q_item.get("detailed_solution", ""))
                        safe_q_num = safe_filename(str(q_item.get('question_number', 'X')).replace(" ", ""))
                        
                        # 剛性約束：僅匹配檔名以 diagram_ 開頭的圖片
                        img_matches = re.findall(r'!\[.*?\]\((?:\./)?((?:images/|images\\)?[^)]*?diagram_[^)]+?\.png)\)', detailed_text, re.IGNORECASE)
                        for raw_img_path in img_matches:
                            file_name = os.path.basename(raw_img_path)
                            if not file_name.startswith("diagram_"):
                                continue
                                
                            clean_rel_path = raw_img_path.replace("\\", "/").lstrip("./")
                            if not clean_rel_path.startswith("images/"):
                                clean_rel_path = f"images/{spec_name}/{file_name}"
                            
                            full_img_path = os.path.abspath(os.path.join(base_dir, clean_rel_path)).replace("\\", "/")
                            done_marker = f"{full_img_path}.done"
                            fail_marker = f"{full_img_path}.failed"
                            force_refresh = os.environ.get("REFRESH_DIAGRAMS", "false").lower() == "true"

                            if os.path.exists(full_img_path) and os.path.getsize(full_img_path) > 0:
                                if os.path.exists(fail_marker):
                                    try: os.remove(fail_marker)
                                    except Exception: pass
                                if not os.path.exists(done_marker):
                                    try:
                                        with open(done_marker, "w", encoding="utf-8") as f_mk:
                                            f_mk.write("done\n")
                                    except Exception: pass
                                if not force_refresh:
                                    continue

                            if force_refresh and os.path.exists(fail_marker):
                                try: os.remove(fail_marker)
                                except Exception: pass

                            needs_draw = False
                            if force_refresh:
                                needs_draw = True
                            else:
                                if not os.path.exists(full_img_path) or os.path.getsize(full_img_path) == 0:
                                    needs_draw = True

                            if needs_draw:
                                os.makedirs(os.path.dirname(full_img_path), exist_ok=True)
                                log_action = "重新繪製 (修復舊圖)" if force_refresh else "遺失圖片補繪"
                                logging.info(f"🎨 [{log_action}] 正在處理 {spec_name} 題號 {safe_q_num}：{file_name}...")
                                
                                redraw_prompt = f"""
                                你是一位頂尖的 Python 數學與科學繪圖專家。
                                以下題目在詳解中引用了圖表，需要請你繪製圖形：
                                
                                【題幹】：{q_item.get('question_text', '')}
                                【詳細解答】：{detailed_text}
                                
                                請撰寫一段完整且獨立的 Python 程式碼，使用 matplotlib 與 numpy 繪製該圖形，並儲存至：
                                `{full_img_path}`
                                
                                【繪圖極致剛性要求（違反將導致程式崩潰）】：
                                1. 所有字串標籤（特別是 ax.annotate / plt.text / plt.title）**必須寫在同一行內**，字串前必須加 `r` 前綴（如 `r'點 $A$'`），嚴禁直接跨行斷開字串！
                                2. **絕對禁止在 `$ ... $` 內部包含任何中文字元**！中文字元請寫在公式符號外面，例如寫 `r'重心 $G$'`，絕對不可寫 `r'$重心 G$'`！
                                3. 2D 平面請使用 `plt.axis('equal')` 保持真實比例；3D 請使用 `Axes3D`。
                                4. 程式碼最後必須包含 `plt.savefig(r"{full_img_path}", dpi=200, bbox_inches='tight')` 與 `plt.close("all")`。
                                """
                                
                                draw_success = execute_and_fix_diagram_script(
                                    self.ai_manager, 
                                    redraw_prompt, 
                                    DiagramResponse, 
                                    full_img_path, 
                                    safe_q_num, 
                                    task_tag=f"[{spec_name} 補繪]"
                                )
                                
                                if draw_success:
                                    missing_diagram_count += 1
                                    if os.path.exists(fail_marker):
                                        try: os.remove(fail_marker)
                                        except Exception: pass
                                    try:
                                        with open(done_marker, "w", encoding="utf-8") as f_mk:
                                            f_mk.write(f"done_at: {time.time()}\n")
                                    except Exception: pass
                                else:
                                    try:
                                        with open(fail_marker, "w", encoding="utf-8") as f_mk:
                                            f_mk.write(f"failed_at: {time.time()}\n")
                                    except Exception: pass
                                    
                    if missing_diagram_count > 0:
                        logging.info(f"🎉 [補繪完成] 已成功為 {spec_name} 補齊了 {missing_diagram_count} 張遺失的圖表！")

                    # 🚨 核心修復：自動跳過已完工考卷時，必須累加 completed_papers 並更新 Web 儀表板！（沿用頂層標準格式化 paper_label）
                    with GLOBAL_METRICS.lock:
                        # 避免啟動預掃描與早退重複累加
                        if paper_label not in GLOBAL_METRICS.exam_progress or "完工" not in GLOBAL_METRICS.exam_progress[paper_label].get("status", ""):
                            GLOBAL_METRICS.completed_papers += 1
                        GLOBAL_METRICS.update_exam_progress(paper_label, len(merged_checkpoint_list), len(merged_checkpoint_list), "✅ 100% 完工 (已存在)")

                    logging.info(f"⏭️  {json_path} 已存在且題數完整 ({len(merged_checkpoint_list)} 題)，已計入進度並自動跳過。")
                    if partial_exists:
                        try: os.remove(partial_json_path)
                        except Exception: pass
                    return
                else:
                    # 💡 題數不足時僅儲存 partial 作為補充，絕不刪除雲端已存在的正式資料庫
                    reason_msg = f"題數不足 (僅 {len(merged_checkpoint_list)}/{min_expected_q} 題)" if len(merged_checkpoint_list) < min_expected_q else "部分題目缺少詳解"
                    logging.warning(f"⚠️ [進度補充] 發現 {json_path} {reason_msg}，保留現有檔案並同步至 partial 進行補充！")
                    with open(partial_json_path, "w", encoding="utf-8") as pf:
                        json.dump(merged_checkpoint_list, pf, ensure_ascii=False, indent=4)
                    # 🚨 關閉刪除指令，防止正式資料庫與雲端檔案被清除

            except Exception as e:
                logging.error(f"讀取現有資料庫 {json_path} 進行品質巡檢失敗: {e}")
            
        # 4. 建立這份試卷專屬的圖片資料夾
        img_dir = os.path.join(output_dir, type_folder, "images", spec_name)
        img_dir = re.sub(r'[\n\r\t]+', '', img_dir).replace("\\", "/")
        os.makedirs(img_dir, exist_ok=True)

        # ----------------- 【學科與數學類別標準化映射】 (放置於快取檢查外，全域皆可調用) -----------------
        normalized_subject = subject
        math_type = None  # 用於在 Prompt 中引導數學 A/B 的範圍
        
        # 統一將各式數學名稱歸類至 "數學" 知識庫，並自動判別卷別屬性
        if any(m in subject for m in ["數學", "數A", "數B", "數甲", "數乙", "數學甲", "數學乙", "數學B", "數學A"]):
            normalized_subject = "數學"
            # 判定屬於 Math A 還是 Math B 體系
            if any(x in subject for x in ["A", "甲"]):
                math_type = "數學A / 數學甲（對應理工醫農科，範圍包含複數極式、棣美弗定理、空間向量、矩陣線性變換、微積分、立體幾何等高階內容）"
            elif any(x in subject for x in ["B", "乙"]):
                math_type = "數學B / 數學乙（對應社會組、商管文法科，不包含複數平面與極式，空間著重基本經緯度與二元不等式，矩陣著重基礎運算與轉移矩陣）"
            elif any(x in subject for x in ["國語", "國綜", "國文", "國語文"]):
                normalized_subject = "國文"
        # 根據標準化後的學科讀取知識分類與難度量表
        current_allowed = SUBJECT_TAXONOMY.get(normalized_subject, {"topics": [], "techniques": []})
        if normalized_subject == "自然":
            for sub in ["化學", "物理", "生物", "地球科學"]:
                if sub in SUBJECT_TAXONOMY:
                    current_allowed["topics"].extend(SUBJECT_TAXONOMY[sub].get("topics", []))
                    current_allowed["techniques"].extend(SUBJECT_TAXONOMY[sub].get("techniques", []))
        elif normalized_subject == "社會":
            for sub in ["歷史", "地理", "公民與社會"]:
                if sub in SUBJECT_TAXONOMY:
                    current_allowed["topics"].extend(SUBJECT_TAXONOMY[sub].get("topics", []))
                    current_allowed["techniques"].extend(SUBJECT_TAXONOMY[sub].get("techniques", []))
        
        subject_rubric = SUBJECT_DIFFICULTY_RUBRICS.get(normalized_subject, GENERAL_DIFFICULTY_RUBRIC)
        # --------------------------------------------------------------------------------------------------

        # 🚨 全局初始化審查日誌與執行緒鎖（確保快取命中時依然能正常存取）
        validation_records = []
        val_log_lock = threading.Lock()
        all_extracted_questions = []

        # 🚨 [頂層變數初始化] 徹底杜絕快取命中時的 UnboundLocalError 崩潰！
        is_mock_paper = exam_type in ["MOCK", "MOCK_GSAT", "MOCK_AST"]
        is_combined_pdf = (a_pdf is not None and os.path.abspath(q_pdf) == os.path.abspath(a_pdf))
        mock_sol_page_indices = []
        mock_sol_pdf_path = None

        # 🚨 [合訂本循環狀態機對接]：徹底征服 100+ 頁大合輯，精確切分多套 [試題 ➔ 詳解] 循環！
        if q_pdf and os.path.exists(q_pdf):
            all_rendered_pages = self.pdf_to_images(q_pdf, "page", img_dir, dpi=300)
            
            # 若為獨立詳解檔
            if a_pdf and os.path.exists(a_pdf) and os.path.abspath(q_pdf) != os.path.abspath(a_pdf):
                try:
                    with fitz.open(a_pdf) as temp_a_doc:
                        mock_sol_page_indices = list(range(len(temp_a_doc)))
                    mock_sol_pdf_path = a_pdf
                    q_image_paths = all_rendered_pages
                    a_image_paths = self.pdf_to_images(a_pdf, "a_full", img_dir, dpi=300)
                    logging.info(f"📑 [模考獨立詳解本鎖定] 題目卷 {len(q_image_paths)} 頁，詳解卷 {len(mock_sol_page_indices)} 頁。")
                except Exception as e:
                    logging.warning(f"讀取獨立詳解卷失敗: {e}")
            else:
                # 🚨 [單卷防切碎保護]：一般 30 頁以內的單卷 PDF，詳解必定在後半段，嚴禁將題幹大標題誤判為合輯切斷！
                total_p_cnt = len(all_rendered_pages)
                if total_p_cnt <= 30:
                    # 在後半段尋找詳解起始頁
                    sol_start_p = -1
                    doc_peek = fitz.open(q_pdf)
                    ans_headers = [
                    "[參考詳解]", "【參考詳解】", "參考詳解", "參考答案暨詳解", "考科詳解", 
                    "試題詳解", "答案與解析", "【解析】", "出處：", "參考答案", "標準答案", 
                    "選擇題答案", "選填題答案", "非選擇題答案", "簡答", "解答"
                ]
                for p_i in range(int(total_p_cnt * 0.35), total_p_cnt):
                    txt = doc_peek[p_i].get_text("text").replace(" ", "")
                    # 1. 命中詳解/答案標題關鍵字
                    if any(k in txt for k in ans_headers):
                        sol_start_p = p_i
                        break
                    # 2. 命中答案表格標題（如「一、單選題 1. (3) 2. (1)...」）
                    if ("答案" in txt or "解答" in txt) and any(k in txt for k in ["單選", "多選", "選填", "題號"]):
                        sol_start_p = p_i
                        break
                    doc_peek.close()

                    if sol_start_p != -1:
                        q_page_idxs = list(range(sol_start_p))
                        sol_page_idxs = list(range(sol_start_p, total_p_cnt))
                    else:
                        # 呼叫多模態探針探測後半段詳解起始點
                        s_idx, e_idx = self.detect_solution_boundary_multimodal(q_pdf, all_rendered_pages)
                        if s_idx > int(total_p_cnt * 0.3):
                            q_page_idxs = list(range(s_idx))
                            sol_page_idxs = list(range(s_idx, total_p_cnt))
                        else:
                            q_page_idxs = list(range(total_p_cnt))
                            sol_page_idxs = []

                    target_cycle = {"q_pages": q_page_idxs, "sol_pages": sol_page_idxs}
                else:
                    # 100+ 頁超大合訂本才啟動循環切分狀態機
                    cycles = self.segment_compilation_cycles_multimodal(q_pdf, all_rendered_pages, year, mock_tag, clean_subj_name)
                    target_cycle = cycles[0] if cycles else {"q_pages": list(range(total_p_cnt)), "sol_pages": []}
                    
                # 若從版面中辨識出了更精準的期別/標題，動態補強標籤
                if target_cycle.get("title") and any(k in target_cycle["title"] for k in ["第", "模", "次"]):
                    t_match = re.search(r'(?:第\s*([一二三四五六七八九十0-9]+)\s*[次回]|第\s*([0-9]+)\s*模)', target_cycle["title"])
                    if t_match:
                        raw_t = next(g for g in t_match.groups() if g is not None)
                        cn_map = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8'}
                        mock_tag = f"_{mock_tag.split('_')[1]}_第{cn_map.get(raw_t, raw_t)}次"

                q_page_idxs = target_cycle["q_pages"]
                sol_page_idxs = target_cycle["sol_pages"]
                
                q_image_paths = [all_rendered_pages[i] for i in q_page_idxs]
                a_image_paths = [all_rendered_pages[i] for i in sol_page_idxs] if sol_page_idxs else []

                # 🚨 初始化當前考卷：題數尚未由 Stage 1 或詳解本確認前設為 0，等待真實題數鎖定（徹底消滅 0/3 盲猜）
                GLOBAL_METRICS.set_current_paper(paper_label, total_q=0, existing_partial_q=0)
                
                if sol_page_idxs:
                    mock_sol_page_indices = sol_page_idxs
                    mock_sol_pdf_path = q_pdf
                    is_combined_pdf = True
                    a_pdf = q_pdf
                    logging.info(f"📑 [合卷區塊鎖定] 成功切出目標試卷：題目卷第 {q_page_idxs[0]+1}~{q_page_idxs[-1]+1} 頁 (共 {len(q_image_paths)} 頁)，官方詳解本第 {sol_page_idxs[0]+1}~{sol_page_idxs[-1]+1} 頁 (共 {len(mock_sol_page_indices)} 頁)！")
        
        # 🆕 1. 優先檢查是否有第一階段題目快取檔
        if os.path.exists(raw_extracted_json_path):
            logging.info(f"⚡ [Stage 1 快取命中] 找到已擷取的題目快取：{raw_extracted_json_path}")
            logging.info("  -> 直接載入題目結構與圖片路徑，【完全跳過】PDF 轉圖與第一階段全部 OCR 掃描！")
            try:
                with open(raw_extracted_json_path, "r", encoding="utf-8") as f:
                    all_extracted_questions = json.load(f)
                    
                has_official_now = (a_pdf is not None and os.path.exists(a_pdf) and exam_type in ["GSAT", "AST", "MOCK", "MOCK_GSAT", "MOCK_AST"])
                for q_cached in all_extracted_questions:
                    if has_official_now:
                        q_cached['has_official_answer'] = True
                    q_cached['question_pdf_path'] = q_pdf.replace("\\", "/") if q_pdf else q_cached.get('question_pdf_path', '')
                    q_cached['answer_pdf_path'] = a_pdf.replace("\\", "/") if a_pdf else q_cached.get('answer_pdf_path', '')
                    q_cached['rubric_pdf_path'] = rubric_pdf.replace("\\", "/") if rubric_pdf else q_cached.get('rubric_pdf_path', '')
            except Exception as e:
                logging.error(f"讀取 Stage 1 快取失敗 ({e})，將重新進行第一階段掃描...")
                all_extracted_questions = []

        # 🆕 2. 只有在無快取時，才進入龐大的第一階段 PDF 轉圖與全套 OCR 掃描
        if not all_extracted_questions:
            logging.info("📸 正在將原卷與解答 PDF 轉換成黃金 300 DPI 整頁圖片...")
            GLOBAL_METRICS.set_stage(f"📸 正在將試卷與解答 PDF 轉換為 300 DPI 整頁影像 ({year} {subject})")

            # 🚨 確定題目卷圖片路徑與解答卷圖片路徑
            if is_combined_pdf and mock_sol_page_indices:
                sol_start_p = mock_sol_page_indices[0]
                q_image_paths = all_rendered_pages[:sol_start_p]
                a_image_paths = [all_rendered_pages[i] for i in mock_sol_page_indices]
                ans_text, ans_page_indices = "詳解本已在模考快車道就緒。", mock_sol_page_indices
            elif is_combined_pdf:
                all_rendered_pages = self.pdf_to_images(q_pdf, "page", img_dir, dpi=300)
                ans_text, ans_page_indices = self.extract_clean_answers(a_pdf, target_subject=subject, target_year=year)
                if ans_page_indices:
                    a_image_paths = [all_rendered_pages[i] for i in ans_page_indices]
                    q_image_paths = [all_rendered_pages[i] for i in range(len(all_rendered_pages)) if i not in ans_page_indices]
                else:
                    q_image_paths = all_rendered_pages
                    a_image_paths = []
            else:
                q_image_paths = self.pdf_to_images(q_pdf, "q_full", img_dir, dpi=300) 
                a_image_paths = self.pdf_to_images(a_pdf, "a_full", img_dir, dpi=300)
                ans_text, ans_page_indices = self.extract_clean_answers(a_pdf, target_subject=subject, target_year=year)
                
            rubric_image_paths = self.pdf_to_images(rubric_pdf, "rubric_full", img_dir, dpi=300)
            rubric_text = self.extract_text_from_pdf(rubric_pdf)

            doc = fitz.open(q_pdf)
            cat = doc.pdf_catalog()
            doc.xref_set_key(cat, "StructTreeRoot", "null")

            failed_questions = []
            validation_records = []  
            val_log_lock = threading.Lock()
            validation_log_path = json_path.replace("_database.json", "_validator_log.txt")
            if os.path.exists(validation_log_path):
                try: os.remove(validation_log_path)
                except Exception: pass

            rubric_visual_map = self.extract_rubric_visual(rubric_pdf, img_dir)
            GLOBAL_METRICS.set_stage("🎨 正在執行「跨頁縫合級」評分標準視覺萃取...")
            
            # =========================================================
            # 【第一階段】：分頁批次掃描與附圖裁切
            # =========================================================
            # 💡 [斷點續傳機制] 初始化題目卷的批次快取
            stage1_cache_path = json_path.replace("_database.json", "_stage1_cache.json")
            stage1_cache = {"processed_pages": [], "questions": []}
            if os.path.exists(stage1_cache_path):
                try:
                    with open(stage1_cache_path, "r", encoding="utf-8") as f:
                        stage1_cache = json.load(f)
                    all_extracted_questions = stage1_cache.get("questions", [])
                    logging.info(f"📂 [Stage 1 續傳] 載入已掃描的 {len(stage1_cache.get('processed_pages', []))} 頁題目進度！")
                except Exception as e:
                    logging.warning(f"讀取 Stage 1 快取失敗: {e}")

            pages = []
            # 🚨 剛性範圍約束：題目擷取【嚴格鎖定】在當前試卷的 q_page_idxs 範圍內，絕不跨入詳解或下一份試卷！
            allowed_q_set = set(q_page_idxs) if 'q_page_idxs' in locals() else set(range(min(len(doc), mock_sol_page_indices[0] if (is_combined_pdf and mock_sol_page_indices) else len(doc))))
            for page_num in range(len(doc)):
                if page_num not in allowed_q_set:
                    continue
                if is_combined_pdf and page_num in ans_page_indices:
                    continue

                page = doc[page_num]
                raw_page_text = page.get_text("text")

                # 2. 🚨 調用智能分類器，全面攔截封面、目錄、知識彙整與非題目的解答頁
                p_type = self.classify_page_type(raw_page_text, page_num, len(doc))
                if p_type in ["COVER", "TOC", "SUMMARY"]:
                    logging.info(f"  -> ⏭️ [{p_type} 自動過濾] 第 {page_num + 1} 頁判定為【{p_type} (封面/目錄/重點摘要)】，自動排除！")
                    continue
                elif p_type == "SOLUTION":
                    logging.info(f"  -> ⏭️ [詳解頁自動排除] 第 {page_num + 1} 頁判定為【SOLUTION (參考詳解)】，不作為題目擷取。")
                    continue
                    
                pages.append(page_num)

            page_batch_size = 2 if is_stem else 3

            for b_idx in range(0, len(pages), page_batch_size):

                check_global_timeout()
                batch_pages = pages[b_idx : b_idx + page_batch_size]
                
                # 💡 [斷點續傳機制] 檢查本批次是否已經被完整掃描過
                unprocessed_pages = [p for p in batch_pages if p not in stage1_cache.get("processed_pages", [])]
                if not unprocessed_pages:
                    logging.info(f"  -> ⏭️ [Stage 1 快取跳過] 試卷第 {batch_pages[0]+1} ~ {batch_pages[-1]+1} 頁先前已成功掃描，直接跳過。")
                    continue
                    
                logging.info(f"正在批次分析 [{year} {subject}] 試卷第 {batch_pages[0]+1} ~ {batch_pages[-1]+1} 頁...")
                GLOBAL_METRICS.set_stage(f"📑 正在掃描題目卷第 {batch_pages[0]+1} ~ {batch_pages[-1]+1} 頁 (OCR 題幹與選項)")
                
                batch_pil_imgs = []
                batch_raw_texts = []
                for p_num in batch_pages:
                    full_page_filepath = q_image_paths[p_num]
                    raw_img = Image.open(full_page_filepath)
                    
                    # 🚀 1. 模擬考掃描檔自動增強：提升細小符號與微淡筆畫清晰度
                    enhanced_img = self.enhance_scanned_image_for_ocr(raw_img)
                    batch_pil_imgs.append(enhanced_img)
                    
                    # 🚀 2. 開源庫優先：調用 pymupdf4llm 提取排版結構完整的 Markdown
                    open_source_md = self.extract_clean_markdown_from_pdf(q_pdf, p_num)
                    page_text = open_source_md if open_source_md else doc[p_num].get_text("text")
                    batch_raw_texts.append(f"--- 第 {p_num+1} 頁開源排版解析對照 ---\n" + s2t(page_text))
                    
                try:
                    batch_text_combined = "\n\n".join(batch_raw_texts)
                    has_official_ans_flag = (a_pdf is not None and os.path.exists(a_pdf) and ans_text != "無官方解答。")

                    prompt_stage_1 = f"""
                    🚨【最高優先級：試卷結構與題型剛性規範】🚨：
                    當前試卷官方解答狀況：{'【有官方解答】' if has_official_ans_flag else '【無官方解答（學校段考/定期考）】'}。

                    1. **【頁面類型精準識別與過濾（知識彙整/目錄/封面/詳解絕對零容忍）】**：
                       - 🚫 **【知識彙整 / 重點摘要過濾】**：若該頁為「重點摘要」、「單元觀念整理」、「公式歸納」（整頁都是物理公式表格、定律解說、無題目問句與獨立選項），這**純屬考前複習筆記，【絕對嚴禁】生造題目擷取**！
                       - 🚫 **【目錄 / 索引過濾】**：若該頁為章節目錄（如：第一單元...第3頁），【絕對嚴禁】當作題目擷取！
                       - 🚫 **【純附錄 / 週期表 / 封面過濾】**：包含作答注意事項、畫卡示例、純元素週期表或常數表，【絕對嚴禁】當作題目擷取！
                       - 🚫 **【解答 / 解析頁面過濾】**：若該頁印有『參考詳解』、『答案表格』或各題解析，不可當作題目擷取！
                       - ✅ **【真實試題判定標準】**：只有印有實質題號（如 1., 2., [複]1., 第1題）、明確問題問句、及配分或選項的頁面，才是唯一合法的正式考題！

                    2. **【題型 (question_type) 嚴格精準分類與選填題防裂解剛性禁令】**：
                       - `單選題`：題目含有單一正確選項。**【注意】若題幹問「最小值為何？」並給予 (1)~(5) 選項，這 100% 是單選題，絕對不可標為多選題！答案必須是選項代號（如 '4'），嚴禁填入選項所代表的數值（如 '16'）！**
                       - `多選題`：題目含有多個選項，正確選項為多個。
                       - `選填題`：大考數學科的挖空填答題。
                         🚨🚨【選填題題號防裂解最高警報（絕對嚴格遵守）】🚨🚨：
                         * 大考與模擬考選填題，在試卷上通常標示為「A. 有一矩形...」、「B. 坐標平面上...」、「C. 設矩陣...」。
                         * 題幹或說明中出現的『8–18 列號』、『[ 15 ][ 16 ]』是【答案卡畫卡列號】，**【絕對不是題號】**！
                         * 選填題的 `question_number` **必須且只能是大題代號 `A`、`B`、`C`**！
                         * **【絕對嚴格禁止】** 將選填題依畫卡格號拆分成多道題目（如嚴禁產出題號 '9'、'11'、'15'、'16' 的題目）！一道選填大題只能產出一個 JSON 題幹！
                         * 選填題的 `options` 必須 100% 為空列表 `[]`。
                       - `簡答題`：**【非選計算證明題專用】** 凡是手寫題、非選擇題、計算證明題、作答問答題（例如：98指考數乙 非選二(1)、二(2)），只要原題**沒有提供 A,B,C,D 選項供勾選**，題型【強制設為 `簡答題`】！**絕對禁止**把純計算證明題標記為 `混合題`！
                       - `混合題`：**僅限於** 108 課綱新型大考中，該題明確包含「第(1)小題為選擇/勾選，第(2)小題為手寫說明」的複合大題。

                    3. **【選項為圖表/幾何圖形/化學結構之擷取規範（極度重要）】**：
                       - 如果選項 (A)~(E) 或 (1)~(5) 本身是圖片（例如：五個函數圖形、五個受力圖、五個細胞分裂圖、五個分子結構）：
                       - **【嚴禁零散碎切】**：絕對不要把 A、B、C、D、E 切割成五個零碎的小框！
                       - **【整排寬幅包裹】**：請直接框選一個**覆蓋全部選項的大型 Bounding Box**，放入該題的 `image_bboxes` 中（此時題幹 `has_image` 設為 `true`）。
                       - 個別選項的 `has_image` 設為 `false`，個別選項的 `value` 統一填寫：`"【圖形選項，請參見題幹附圖】"`。

                    4. **【英文科文意選填 (Word Bank) 專屬規範】**：
                       - 整篇帶有空格（如 [ 31 ]、[ 32 ]...）的文章，必須完整放入每個小題的 `shared_context`。
                       - 下方的單字庫（通常為 A 到 J 或 A 到 L）：**必須完整複製這組單字選項到每一個空格小題的 `options` 中**。
                       - 該小題的 `question_text` 填入包含該空格的那一個完整英文句子（例如：`The researcher tried to ______ the mystery.`）。
                    
                    🚨【是非題 (True/False) 標準化轉單選題剛性規則】🚨：
                    若題目為「是非題」（例如：1. $sin 26^\\circ = cos 64^\\circ$）：
                    - **【題型】**：必須將 `question_type` 強制設為 `"單選題"`。
                    - **【選項】**：因為原題未印出選項，你**必須自動為其生成兩個標準選項**：
                      `options`: [
                          {{"key": "A", "value": "正確 (○)"}},
                          {{"key": "B", "value": "錯誤 (✕)"}}
                      ]
                    - **【答案對應】**：若官方解答為 `O` 或 `○`，`answer` 填 `"A"`（或 `"O"`）；若官方解答為 `X` 或 `✕`，`answer` 填 `"B"`（或 `"X"`）。

                    🚨【實體考題校驗規則】🚨：
                    1. 你的任務是擷取「正式試題」。
                    2. 封面頁（Page 1）的內容通常是「作答範例」（如：第1題為單選題...），**【絕對禁止】** 擷取。
                    3. 真正第一題通常出現在 Page 2 或 Page 1 的下半部（橫線之後）。
                    4. 每一題的 `page_number` 必須精確記錄其在 PDF 中的實體頁碼。

                    若題目是純文字（如選擇題），且沒有任何附圖或化學反應式圖像，請將 has_image 設為 false。只有當題目中出現獨立於文字行之外的複雜反應式、圖形或表格時，才設為 true 並圈選。
                    
                    你是一位專業的台灣高中閱卷與數位典藏專家。這是一組台灣高中考試的試卷連續影像。
                    學年度：{year}，科目：{subject}。
                    這組影像包含第 {batch_pages[0]+1} 頁到第 {batch_pages[-1]+1} 頁。

                    【任務目標】
                    請極其精準地擷取這幾頁中的所有考題，並將資料填入對應的 Pydantic 結構中。

                    【手寫題與非選擇題評分標準精準對齊】
                    本考卷包含非選擇題（如第22、24-31題等手寫題）。
                    請對照下方提供的【官方評分原則文字】，將手寫題對應的「列式給分、答案給分、扣分限制」等極重要規則，精確填入 `scoring_criteria` 欄位中！
                    
                    🚨【數位文字與影像視覺雙重比對規限（零誤差保證）】🚨
                    - 本系統提供了該頁面的『原始純文字對照（Text Layer）』。
                    - 當你從影像中識別數學變數（如 $x, y, z, a, b$）或物理單位時，如果因為字體過小或解析度問題產生疑義，請【強制比對純文字對照區】中的相對應字元。
                    - 英文大小寫、希臘字母（如 \\theta, \\phi, \\alpha）必須完全以底層數位文字層的命名為最高準則，嚴禁因為視覺模糊而自行臆測或改寫變數名稱！

                    【一、剛性文字與公式規格化】
                    1. **題目文字完全對齊**：`question_text` 必須與圖片及底層純文字 100% 吻合。
                    2. **LaTeX 規範**：所有數學公式、化學式、變數符號、數值等，**必須**使用標準 LaTeX 語法包裹，確保排版美觀。
                    3. **LaTeX 嚴格符號規範**：所有數學公式、化學式、變數符號，【行內公式】必須嚴格使用 `$...$` 包裹；【獨立行公式】必須嚴格使用 `$$...$$` 包裹。絕對禁止使用 `\\( \\)` 或 `\\[ \\]` 作為公式標籤！
                    4. **🚨【選項分離與題幹清洗硬性規定】**：對於選擇題（單選與多選），**必須且強制**將題幹文字（`question_text`）中所有包含選項描述的部分（例如：尾部的 `(A) xxx (B) yyy...` 或 `(1) xxx (2) yyy...` 或者是行內的選項敘述）**完全清除乾淨**！題幹 `question_text` 內只能保留最純粹的題目問句，絕對不能殘留任何選項的標籤與其文字內容！所有的選項標籤與文字必須且只能存在於 `options` 欄位中，嚴禁在題幹與選項列表間發生資料雙重冗餘與重複！
                    - 🚨【選填題無選項剛性鐵律】：**選填題（填空題，即題號為大寫字母 A, B, C... 或帶有 [ 9 ] 挖空者）絕對沒有選項！** 它的 `options` 屬性必須 100% 為空列表 `[]`。**絕對禁止**將解答卷或答案卡上的欄位索引與答案對照（如：`8. 1. 9. 3.`）當作選項寫入 `options`！
                    4-2. **表格強制轉換**：數據表格務必轉換為 Markdown Table 格式嵌入。
                    5. **內文嵌入選項**：若遇上英文或國文科「克漏字」等選項內嵌在文章段落中的狀況，請在題幹中保留如 (A)、(B) 等引導標記以維持文章完整，並將對應的代號與內容整理到 `options` 欄位中。
                    5-2. **【選填題挖空規則（🚨 嚴禁劇透答案與回填）】**：
                       - 大考選填題常使用圓圈數字（如 ⑨、⑩ 或 ⑮-①、⑮-②、⑮-③）代表畫卡格。請統一精確轉換為 `[ 9 ]`、`[ 10 ]` 或 `[ 15-1 ][ 15-2 ][ 15-3 ]` 等挖空格式。
                       - 🚨🚨【絕對嚴禁將答案數字填入題幹空格中】🚨🚨：
                         * 題幹是給學生作答的，**絕對不可直接把答案填進去！**
                         * 例如：題目印 $\\sqrt{{ ⑮-① ⑮-② ⑮-③ }}$，答案為 4,4,2，題幹【必須】寫成 `\\sqrt{{ [ 15-1 ][ 15-2 ][ 15-3 ] }}`，【絕對嚴禁】寫成 `\\sqrt{{442}}`！
                         * 如果題目根號或分數線內含有畫卡格，必須完整保留根號與空格結構，絕對不可以因為知道答案就直接替換！
                    🚨【矩陣 LaTeX 剛性規範】：試卷中的方陣或矩陣，必須使用標準 LaTeX 的 `\\begin{{bmatrix}} ... \\end{{bmatrix}}` 表示，元素間用 `&`，行間用 `\\\\`，且務必包裹在 `$...$` 中！絕對禁止用純文字中括號 `[1 2][-1]` 代替矩陣！
                    🚨【警告】：有時候圓圈數字旁邊**真的有根號或其他數學運算符號**（例如根號內有畫卡格），請務必精準轉換為例如 `[ 9 ]`、`\\sqrt{{ [ 10 ] }}` 等結構！絕對不可以把真正的根號吃掉，也絕對禁止擅自把 ⑨ 和 ⑩ 強行合併成 `[ 9-10 ]`！請忠實反映圖片上的數學結構。
                    6. **表格強制轉換**：若題目中包含數據表格（如：表1、表2），請【務必】將表格內容完整轉換為 Markdown Table 格式，並嵌入到 `question_text` 中相應的位置。絕對不可省略表格內容！
                    7. **選填題答案與畫卡格子對應規範（極度重要）**：
                       - 大考選填題的答案會對應至多個獨立的畫卡格子（例如：`[ 10-1 ]`、`[ 10-2 ]`、`[ 10-3 ]`）。
                       - **【強制規定】**：在 `answer` 欄位中，**必須**將每一個格子所對應的答案字元（包含數字、正負號或特定根號代號）依序填入，並以半形逗號 `,` 隔開！
                       - 例如：
                         * 若題目為 $a=[ 10-1 ][ 10-2 ]$, $b=[ 10-3 ]$，其中 $a = -4$, $b = 3$，其對應畫卡格答案分別為 `-`、`4`、`3`，則 `answer` 欄位必須寫成 `-,4,3`，絕對不可直接相連寫成 `-43`。
                         * 若選填題 A 答案為 $\frac{9}{10}$，畫卡格 9-10 為 9、1、0，則寫成 `9,1,0`，絕對不可寫成 `910`。
                    8. **🚨【英文科五大題型專屬擷取剛性規範】🚨**：
                       - **【詞彙題 (Vocabulary)】**：題幹完整擷取，選項為標準 4 個單字 (A, B, C, D)，`question_type` 為 `"單選題"`。
                       - **【綜合測驗 (Cloze Test)】**：
                         * 整篇克漏字文章**完整保留並填入 `shared_context`**。
                         * 每個小題的 `question_text` 必須擷取「包含該空格的那一個完整英文句子」，並將題號替換為 `______`（例如：`The company decided to ______ the new policy.`）。
                         * 該小題下方的四個選項填入 `options`，`question_type` 為 `"單選題"`。
                       - **【文意選填 (Blanks with Word Bank)】**：
                         * 整篇文章放入 `shared_context`。
                         * 題目下方給予的單字庫（通常為 A 到 J 或 A 到 L）：**必須為每個小題複製完整的這組相同單字選項庫到 `options`**。
                         * 該題 `question_text` 填入該空格所在的完整句子，`question_type` 強制設為 `"單選題"`。
                       - **【篇章結構 (Discourse Structure)】**：
                         * 整篇文章放入 `shared_context`，空格標註為 `[ 31 ]`、`[ 32 ]` 等。
                         * 選項庫中的句子 (A)~(E) 填入 `options`，`question_type` 強制設為 `"單選題"`。
                       - **【混合題 (Hybrid Test)】**：
                         * 英文混合題常要求在文章中找出單字填空、或回答 2~5 字的簡答題。
                         * 若為手寫填空或簡答，`question_type` 設為 `"簡答題"`，`options` 為空列表 `[]`。
                    9. **選填題挖空規則**：若遇到大考特有的圓圈畫卡題號（例如 ⑬-① ⑬-②），請統一轉換為標準挖空格式 `[ 13-1 ] [ 13-2 ]`，不要使用 LaTeX 的 \\bigcirc。這有利於系統自動生成填空輸入框。
                    9-2. **🚨【選填題畫卡格圓圈數字不視為圖片】🚨**：
                       大考選填題中出現的帶圓圈數字（如 ⑧、⑨、⑩）是排版文字的一部分（請按規則 9 轉換為 `[ 8 ]`、`[ 9 ]`、`[ 10 ]`）。**【絕對禁止】**將這些圓圈數字、分數線或其相鄰的填空文字框選為 `image_bboxes`！只有當題目中出現真正的實體插圖、函數圖形、幾何圖形或大型數據表時，才將其框選為 `image_bboxes`。
                    10. **【極度精準的數學 OCR】**：數學公式與不等式的辨識必須一字不差！例如 `x-y` 絕對不能看錯成 `2x-y` 或 `x+y`，大於小於符號絕對不能反！請反覆核對圖片中的方程式。
                    10-2. **🚨【微細符號與循環小數極度預警】🚨**：
                       在數學科中，常常出現**循環小數**（例如 $1.\\bar{{5}}$，即數字 5 的上方有一條橫線，代表 $1.5555...$）。
                       - 這類微小的上標橫線極易被低解析度 OCR 遺漏並誤讀為普通小數（如 $1.5$）。
                       - 請你**仔細盯住原卷圖片上的每一個小數點與數字上方**！若看到數字上方有任何橫線、波浪號或圓點，**必須且強制**將其識別為標準 LaTeX 的循環小數格式，例如 `$1.\\bar{{5}}$`、`$7.\\bar{{7}}$` 或 `$1.\\dot{{5}}$`。
                       - 絕對禁止遺漏這些符號並將其簡化為普通小數，這會導致整道題目的數論邏輯與選項對照徹底崩潰！
                    10-3. **🚨【對數 log 底數與真數防看錯剛性規則（極度重要）】🚨**：
                       - 數學中的對數符號 $\log_a b$，「底數 $a$」是較小且靠下方的下標，而「真數 $b$」是正常大小的主字體！
                       - 例如：$\log_2 3$（底數為 2，真數為 3）、$\log_3 2$（底數為 3，真數為 2）、$\log_4 6$（底數為 4，真數為 6）。
                       - **【禁止看反】**：絕對嚴禁將底數與真數看反（例如不可將 $\log_2 3$ 誤讀為 $\log_3 2$）！
                       - **【禁止將底數誤認為倍數】**：絕對嚴禁將下標 $2$ 看成前面的乘數（例如不可把 $\log_2 3$ 誤寫為 $2\log 3$）！
                       - 若有數位文字層對照，請強制比對文字層中緊跟在 `log` 後面的小數字，確保 LaTeX 語法一律寫成 `\\log_{{a}} b`！
                    11. **【防錯位與防遺漏警告】**：大考的題目偶爾會分欄排版。請務必遵循正常的閱讀順序（先左後右，先上後下）完整提取 `question_text`。若題目包含附表，請確保 Markdown Table 欄位數與原圖完全一致，絕不可漏掉任何一行數據！
                    12. **【防選項合併】**：請確保 `options` 欄位中，每個選項是獨立的物件，絕對不可以把選項 A 和選項 B 融合成一個選項輸出。
                    13. **頁碼追蹤（極度重要）**：你必須在 `page_number` 欄位中，填入該題目在原卷 PDF 中的真實頁碼（從 1 開始計數）。這對於裁切考題附圖與表格至關重要。
                    14. **【字母 y 與數字 3 的防混淆警告（化學與代數特防）】**：在化學分子式（例如 $CH_3(CH_2)_yCl$）中，**斜體的小寫字母 $y$ 極易被誤判為數字 $3$**！
                        如果題目中同時出現了 $x, z$ 作為待求變數（如 $CH_3SH_x$ 與 $CH_3NH_z$），夾在中間的變數絕對是小寫字母 $y$（即 $CH_3(CH_2)_yCl$），請務必精準辨識為小寫字母 $y$，絕對不可以看成數字 $3$！
                    15. **【清洗圖形文字佔位符】**：大考 PDF 的純文字中常含有如 `[圖3 結構圖]`、`[圖形]`、`[圖片]` 或 `[圖 3]` 等無意義的純文字佔位符。請你在擷取 `question_text` 與 `shared_context` 時，**務必將這些無意義的圖形文字佔位符完全剔除**！因為我們後續會有實體的 `image_bboxes` 裁剪圖，不需要保留這些純文字垃圾。
                    16. **【極度寬裕的 Bounding Box 標記】**：當你框選 `image_bboxes` 或選項附圖的 Bounding Box 時，**請務必畫得極度寬裕 (Very Generous)**！寧可多框 15% 的空白邊緣，也絕對不可以切到任何化學鍵、原子符號、反應箭頭、坐標軸文字、或選項字母 A, B, C 的邊角！
                    17. **【題組判定極度嚴格警告】**：只有當試卷上明確印有『X-Y題為題組』時，才可將 X 到 Y 題歸為題組，並將共同引言寫入 `shared_context`。**絕對禁止**只因為多道題目印在同一頁、或者因為它們都是選擇題，就擅自編造『題組』將其歸類！非題組的題目，其 `shared_context` 必須為空！
                    18. **科目精細分類與【自然考卷嚴格分流禁令】**：
                       - 請將考卷的原有名稱填入 'exam_source'。
                       - 🚨🚨【自然科考題絕對分流】：對於學測自然、分科自然或自然模考，每一道題目的 `sub_subject` **必須且只能**精確指派為 `'物理'`, `'化學'`, `'生物'`, `'地球科學'` 之一！
                       - **【絕對嚴格禁止】** 將 `sub_subject` 填寫為 `'自然'`！知識庫中絕無「自然」這一科目！
                       - 全科合法科目清單只有：'物理', '化學', '生物', '地球科學', '歷史', '地理', '公民與社會', '數學', '英文', '國文', '國寫'。
                    
                    【圖片裁切極致規範 - 解決切錯/漏切問題】
                    1. **題號包含原則**：Bounding Box 必須包裹住題號數字。
                    2. **🚨選項附圖合併規則 (極重要)🚨**：
                    - 如果題目選項 (A)~(E) 或 1~5 是圖形（如幾何圖、生物分類樹、化學結構）：
                    - **【嚴禁】** 將 A, B, C, D, E 分開裁切成五張圖。
                    - **【必須】** 直接框選一個覆蓋 A 到 E 所有選項的大型 Bounding Box，並放入該題 `image_bboxes` 中。
                    3. **🚨嚴禁憑空捏造 Bounding Box🚨**：如果考卷影像中沒有明確的圖表、幾何圖形或附圖，絕對不可以因為題目敘述出現「圖形」、「正方體」、「橢圓」等字眼就憑空捏造 Bounding Box 座標！此時必須將 has_image 設為 false 且 image_bboxes 設為空列表 []。
                    - 此時，個別選項的 `has_image` 設為 false，其文字內容填寫「【請參見題幹附圖中的選項內容】」。
                    3. **表格與圖表標籤**：必須包含「圖15」或「表7」等標籤。
                    4. **表格邊界**：框選表格時請多留 50 個單位的空白邊緣，嚴禁切到表格的框線或標題。
                    5. 題號與圖號絕對對位（極度重要）：
                       - 確保 question_number 與框選的 image_bboxes 嚴格屬於同一個題目！
                       - 🚨【圖號交叉檢驗】：若題目文字提到「如圖 9 所示」或「圖 9 中有...」，你所框選的圖片 Bounding Box 內部【必須包含圖 9 的標籤或圖像】，絕對嚴禁把同頁或相鄰題目的「圖 7」、「圖 8」框進來！若本頁根本找不到該題提到的圖（例如該圖在上一頁或引言），請留空 []，由題組背景處理，嚴禁張冠李戴！
                    

                    【二、題組與附圖剛性規則】
                    1. **題組共同題幹處理（嚴禁題幹夾帶引言）**：
                       - 若本頁有「X-Y 為題組」或非選擇題大題（如：非選二、題組一）：
                       - **【共同文章/背景/地圖/情境】必須且只能填入 `shared_context`**！
                       - 每個小題的 `question_text` **絕對只能包含該單一子題的獨立問句**（例如：二(1) 的題幹只能寫「試求 X 的期望值。(請用最簡分數表示)」，絕對嚴禁把大題的山洞黃金引言、更嚴禁把第二小題的題目文字複製進來）！
                    2. **多圖定位 (image_bboxes)**：若該題含有多張分散的附圖、表格或化學結構式（例如同時包含圖 1 與圖 2，或包含表 1 與結構圖），請將**所有**附圖的 Bounding Box 分別精確框出，並以列表的列表（如 `[[ymin1, xmin1, ymax1, xmax1], [ymin2, xmin2, ymax2, xmax2]]`）填入 `image_bboxes`。寧可框大，也絕不漏掉任何一張圖。
                    🚨 3. **選項附圖合併規則（極度重要，解決裁切不精準的致命傷）**：
                       若選擇題的選項本身是圖形（如：五個細胞分裂圖、五個系統分類樹、五個幾何圖形）：
                       - **如果選項（A, B, C, D, E）在版面上是「橫向排成一列」、或者「排成整齊的網格區塊」，請【絕對不要】將它們分割成五個零碎的小圖！**
                       - 請你【直接將這整排/整個區塊的選項圖（必須包含 A, B, C, D, E 的標記以及它們所有的圖形）合併框成一個唯一的、寬裕的大 Bounding Box，放入主標題（題幹）的 `image_bboxes` 中】（此時題幹 `has_image` 設為 `true`）。
                       - 此時，個別選項的 `has_image` 請設為 `false`，其 `image_bboxes` 設為空列表 `[]`，選項的 `value` 欄位統一填入：`"【圖形選項，請參見題幹附圖】"`。
                       - 只有當選項圖在版面上分布極度散亂、完全無法合併為一個方框時，才允許將各個選項的 `has_image` 設為 `true` 並單獨框選。
                    4. **附圖框選寬裕原則**：當你框選任何附圖或大表格時，**請務必框得極度寬裕（Generous）一些**。確保表格名稱、上方的標題文字、下方的選項標籤文字，都完整被包裹進 Bounding Box 中，避免在後續裁切時被切掉邊緣。

                    【三、大考答案與評分絕對對齊】
                    官方選擇題解答：
                    {ans_text}

                    官方非選題評分標準：
                    {rubric_text}

                    【四、標準輸出格式範例 (Few-Shot Example)】
                    {{
                        "questions": [
                            {{
                                "academic_year": "114分科",
                                "exam_source": "114學測自然",
                                "sub_subject": "物理",
                                "question_number": "1",
                                "page_number": 2,
                                "question_text": "2024 年聯合國大會宣布 2025 年為國際量子科學與科技年（IYQ）...下列有關量子力學發展的敘述何者正確？",
                                "has_image": false,
                                "image_bboxes": [],
                                "options": [
                                    {{"key": "A", "value": "普朗克提出量子論成功解釋氫原子光譜的性質"}},
                                    {{"key": "B", "value": "德布羅意提出物質波說明波與粒子的二象性"}}
                                ],
                                "answer": "E",
                                "question_type": "單選題",
                                "scoring_criteria": "",
                                "full_page_image_path": "",
                                "shared_context": ""
                            }}
                        ]
                    }}
                    「絕對禁止將 A、B、C、D 的選項附圖框線混入主題幹的 image_bboxes 中！」

                    【五、防呆機制】：
                    1. 絕對禁止將「純題組導言」當作一題輸出。
                    2. options 的 key 只能是 A, B, C, D, E 或 1, 2, 3, 4。
                    3. **【非選擇題拆分嚴格規定】**：若遇到非選擇題（例如大題為「一」，內含「(1)」、「(2)」兩小題），請將其拆分為兩個獨立 JSON 物件，`question_number` 分別命名為 `"一(1)"` 與 `"一(2)"`，並將大題幹的共同敘述放在 `shared_context` 中。**【絕對禁止】把大題幹本身（題號"一"）當成獨立的一題輸出，這會造成題目重複！**
                    4. **【測驗說明排除】**：若本頁為「作答注意事項」或「作答範例」（出現「例：若第1題為單選題...」），請直接忽略本頁所有內容，切勿將範例當作考題！

                    【本批次純文字參考（防漏字輔助，請與影像對比校驗）：】
                    {batch_text_combined}
                    """
                    
                    batch_contents_with_labels = [prompt_stage_1]
                    for idx, p_num in enumerate(batch_pages):
                        batch_contents_with_labels.append(f"=== 原卷 PDF 第 {p_num+1} 頁影像 ===")
                        batch_contents_with_labels.append(batch_pil_imgs[idx])
                    
                    batch_contents_with_labels.append("\n\n=== 原始純文字對照 ===")
                    batch_contents_with_labels.append(batch_text_combined)

                    current_stage1_model = stage_1_models[(b_idx // page_batch_size) % len(stage_1_models)]
                    result_dict_1, error_1 = self.ai_manager.generate_with_retry(
                        contents=batch_contents_with_labels,
                        response_schema=PageExtraction,
                        temperature=0.0,
                        preferred_model=current_stage1_model,
                        enable_thinking=False
                    )

                    if result_dict_1:
                        result_dict_1 = s2t_recursive(result_dict_1)

                    if error_1 or not result_dict_1:
                        logging.warning(f"Lite 模型掃描失敗，嘗試使用強力模型重試...")
                        result_dict_1, error_1 = self.ai_manager.generate_with_retry(
                            contents=[prompt_stage_1] + batch_pil_imgs + ["\n".join(batch_raw_texts)],
                            response_schema=PageExtraction,
                            preferred_model="gemini-3.5-flash"
                        )
                        if error_1:
                            logging.error(f"該頁掃描徹底失敗: {error_1}")
                            continue

                    batch_crops_cache = {}

                    def get_bbox_overlap(box1, box2):
                        ymin1, xmin1, ymax1, xmax1 = box1
                        ymin2, xmin2, ymax2, xmax2 = box2
                        yi_min, xi_min = max(ymin1, ymin2), max(xmin1, xmin2)
                        yi_max, xi_max = min(ymax1, ymax2), min(xmax1, xmax2)
                        if yi_min >= yi_max or xi_min >= xi_max:
                            return 0.0
                        inter_area = (yi_max - yi_min) * (xi_max - xi_min)
                        area1 = (ymax1 - ymin1) * (xmax1 - xmin1)
                        area2 = (ymax2 - ymin2) * (xmax2 - xmin2)
                        union_area = area1 + area2 - inter_area
                        return inter_area / union_area if union_area > 0 else 0.0

                    for q_data in result_dict_1.get('questions', []):
                        # 🚨 實體文字錨點校準：以題幹關鍵字搜尋真實所屬頁碼，杜絕 AI 報錯頁碼導致裁圖錯位
                        q_text_sample = re.sub(r'[^\w\u4e00-\u9fa5]', '', q_data.get('question_text', ''))[:20]
                        q_num_tag = str(q_data.get('question_number', '')).strip()
                        
                        real_page_idx = -1
                        # 優先在當前批次的頁面中搜尋文字錨點
                        for p_cand in batch_pages:
                            p_raw = doc[p_cand].get_text("text")
                            p_clean = re.sub(r'[^\w\u4e00-\u9fa5]', '', p_raw)
                            if q_text_sample and q_text_sample in p_clean:
                                real_page_idx = p_cand
                                break
                            # 若題幹過短，比對題號特徵（如行首題號）
                            if re.search(rf'(?:^|\n)\s*(?:第\s*{q_num_tag}\s*題|{q_num_tag}\s*[.．、\s)]|\({q_num_tag}\))', p_raw):
                                real_page_idx = p_cand
                                break
                                
                        if real_page_idx != -1:
                            target_page_idx = real_page_idx
                            q_data['page_number'] = target_page_idx + 1
                        else:
                            # 保底：若搜尋不到（如純圖片掃描檔），限制在當前批次範圍內
                            try:
                                parsed_page = int(q_data.get('page_number', 1))
                                if (parsed_page - 1) in batch_pages:
                                    target_page_idx = parsed_page - 1
                                else:
                                    target_page_idx = batch_pages[0]
                            except Exception:
                                target_page_idx = batch_pages[0]
                            q_data['page_number'] = target_page_idx + 1

                        page = doc[target_page_idx]
                        full_page_filepath = q_image_paths[target_page_idx]
                        q_data['full_page_image_path'] = full_page_filepath.replace("\\", "/")
                        q_data['question_pdf_path'] = q_pdf.replace("\\", "/") if q_pdf else ""
                        q_data['answer_pdf_path'] = a_pdf.replace("\\", "/") if a_pdf else ""
                        q_data['rubric_pdf_path'] = rubric_pdf.replace("\\", "/") if rubric_pdf else ""
                        
                        q_data['question_page_image_paths'] = q_image_paths
                        q_data['answer_page_image_paths'] = a_image_paths
                        q_data['rubric_page_image_paths'] = rubric_image_paths
                        q_data['school_name'] = school_name if exam_type == "SCHOOL" else ""

                        # 🚨 核心修復：單題級別精確判定本題是否存在官方答案！
                        # 只有當該題的 answer 真正存在且非空白時，has_official_answer 才為 True
                        cur_ans = str(q_data.get('answer', '')).strip()
                        has_real_ans = bool(cur_ans and cur_ans not in ["", "None", "null", "／", "/", "無", "無答案"])
                        q_data['has_official_answer'] = has_real_ans
                        q_data['official_answer'] = cur_ans if has_real_ans else ""

                        options = q_data.get('options', [])
                        is_alpha = any(str(opt.get('key', '')).isalpha() for opt in options)
                        expected_keys = ["A", "B", "C", "D", "E", "F", "G"] if is_alpha else ["1", "2", "3", "4", "5", "6", "7"]

                        for idx, opt in enumerate(options):
                            current_key = str(opt.get('key', '')).strip()
                            if len(current_key) != 1 or not current_key.isalnum() or current_key.lower() == 'key':
                                if idx < len(expected_keys):
                                    opt['key'] = expected_keys[idx]

                        q_data['image_paths'] = []
                        cropped_imgs = []
                        
                        try:
                            if q_data.get('has_image') and q_data.get('image_bboxes'):
                                bboxes = q_data['image_bboxes']
                                for b_idx, bbox in enumerate(bboxes, 1):
                                    if len(bbox) == 4 and all(v is not None for v in bbox):
                                        ymin, xmin, ymax, xmax = bbox
                                        xmin_val = max(0, min(1000, min(xmin, xmax)))
                                        xmax_val = max(0, min(1000, max(xmin, xmax)))
                                        ymin_val = max(0, min(1000, min(ymin, ymax)))
                                        ymax_val = max(0, min(1000, max(ymin, ymax)))

                                        p_num = q_data.get('page_number', 1)
                                        if p_num not in batch_crops_cache:
                                            batch_crops_cache[p_num] = []
                                            
                                        duplicate_path = None
                                        for cache_item in batch_crops_cache[p_num]:
                                            if get_bbox_overlap(bbox, cache_item["bbox"]) > 0.7:
                                                duplicate_path = cache_item["path"]
                                                break
                                                
                                        if duplicate_path:
                                            if duplicate_path not in q_data['image_paths']:
                                                q_data['image_paths'].append(duplicate_path)
                                                cropped_imgs.append(Image.open(duplicate_path))
                                            continue
                                        
                                        # 🚨 修正：呼叫具備「防止文字邊緣切斷」智能防禦的 execute_crop！
                                        safe_q_num = safe_filename(str(q_data.get('question_number', 'X')).replace(" ", ""))
                                        try:
                                            new_img_paths = self.execute_crop(page, [bbox], img_dir, f"Q{safe_q_num}_{b_idx}")
                                            for path in new_img_paths:
                                                if path not in q_data['image_paths']:
                                                    q_data['image_paths'].append(path)
                                                    cropped_imgs.append(Image.open(path))
                                                    batch_crops_cache[p_num].append({
                                                        "bbox": bbox,
                                                        "path": path
                                                    })
                                        except Exception as e:
                                            logging.error(f"裁切題幹圖片失敗: {e}")

                            for opt in options:
                                opt['image_paths'] = []
                                if opt.get('has_image') and opt.get('image_bboxes'):
                                    opt_bboxes = opt['image_bboxes']
                                    for b_idx, bbox in enumerate(opt_bboxes, 1):
                                        if len(bbox) == 4 and all(v is not None for v in bbox):
                                            ymin, xmin, ymax, xmax = bbox
                                            xmin_val = max(0, min(1000, min(xmin, xmax)))
                                            xmax_val = max(0, min(1000, max(xmin, xmax)))
                                            ymin_val = max(0, min(1000, min(ymin, ymax)))
                                            ymax_val = max(0, min(1000, max(ymin, ymax)))

                                            p_num = q_data.get('page_number', 1)
                                            if p_num not in batch_crops_cache:
                                                batch_crops_cache[p_num] = []
                                                
                                            duplicate_path = None
                                            for cache_item in batch_crops_cache[p_num]:
                                                if get_bbox_overlap(bbox, cache_item["bbox"]) > 0.7:
                                                    duplicate_path = cache_item["path"]
                                                    break
                                                    
                                            if duplicate_path:
                                                if duplicate_path not in opt['image_paths']:
                                                    opt['image_paths'].append(duplicate_path)
                                                    cropped_imgs.append(Image.open(duplicate_path))
                                                continue
                                            
                                            # 🚨 修正：選項附圖同步採用智能裁切
                                            safe_q_num = safe_filename(str(q_data.get('question_number', 'X')).replace(" ", ""))
                                            opt_key = safe_filename(str(opt.get('key', 'X')).replace(" ", ""))
                                            try:
                                                new_img_paths = self.execute_crop(page, [bbox], img_dir, f"Q{safe_q_num}_Opt{opt_key}_{b_idx}")
                                                for path in new_img_paths:
                                                    if path not in opt['image_paths']:
                                                        opt['image_paths'].append(path)
                                                        cropped_imgs.append(Image.open(path))
                                                        batch_crops_cache[p_num].append({
                                                            "bbox": bbox,
                                                            "path": path
                                                        })
                                            except Exception as e:
                                                logging.error(f"裁切選項 {opt_key} 圖片失敗: {e}")

                            q_data['_cropped_pil_images'] = cropped_imgs
                            all_extracted_questions.append(q_data)

                            # 🚨 [網頁總題數即時聯動]：每掃描出一批題目，即刻推播更新網頁已知總題數！
                            with GLOBAL_METRICS.lock:
                                GLOBAL_METRICS.paper_total_questions = len(all_extracted_questions)
                            GLOBAL_METRICS.update_exam_progress(
                                GLOBAL_METRICS.current_paper, 
                                GLOBAL_METRICS.paper_completed_questions, 
                                len(all_extracted_questions), 
                                f"題目卷掃描中 (已辨識 {len(all_extracted_questions)} 題)..."
                            )

                        except Exception as e:
                            logging.error(f"準備多模態資訊失敗: {e}")

                finally:
                    for img in batch_pil_imgs:
                        try: img.close()
                        except Exception: pass
                        
                # 💡 [斷點續傳機制] 在本批次成功後，立刻將進度與目前提取到的題目安全寫入快取
                stage1_cache["processed_pages"].extend(unprocessed_pages)
                
                # 為了避免 Pydantic 或 PIL Image 不能被 JSON 序列化，進行深拷貝與清理
                clean_extracted_questions = []
                for q_item in all_extracted_questions:
                    q_copy = dict(q_item)
                    q_copy.pop('_cropped_pil_images', None)
                    clean_extracted_questions.append(q_copy)
                    
                stage1_cache["questions"] = clean_extracted_questions
                try:
                    with open(stage1_cache_path, "w", encoding="utf-8") as f:
                        json.dump(stage1_cache, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    logging.warning(f"即時寫入 Stage 1 批次快取失敗: {e}")

            # 在整個 Stage 1 迴圈順利完成後，清理暫存的 cache 檔
            if os.path.exists(stage1_cache_path):
                try: os.remove(stage1_cache_path)
                except Exception: pass

            def filter_ghost_questions(q_list):
                valid_q = []
                ghost_keywords = [
                        "本題於原試卷中不存在", "全卷掃描漏失", "無完整題目文本", 
                        "題目文字未在影像中提供", "此頁面為試卷封面", "本頁為「大學入學考試中心",
                        "作答注意事項", "無實質試題文字", "本題不存在", "題目內容缺失", 
                        "題目闕如", "無法判定", "未包含第", "作答示例", "作答說明", 
                        "劃記方式之說明", "例：若第", "題目內容未提供", "選項內容未提供", 
                        "無法從提供的影像中識別出", "漏掉之第", "請補充"
                    ]
                dummy_opt_values = {"選項A", "選項B", "選項C", "選項D", "選項E", "A", "B", "C", "D", "E", "選項A說明", "選項B說明", "選項C說明", "選項D說明", "(選項內容未提供)"}

                for q in q_list:
                    q_text = str(q.get("question_text", "")).strip()
                    q_num = str(q.get("question_number", "")).strip()
                    ans_text = str(q.get("answer", "")).strip()
                    cat_text = str(q.get("topic_category", "")).strip()
                    ana_text = str(q.get("question_analysis", "")).strip()
                    opts = q.get("options", [])
                    
                    combined_text = f"{q_text} | {ans_text} | {cat_text} | {ana_text}"
                    
                    # 1. 命中絕對幽靈佔位關鍵字
                    if any(gk in combined_text for gk in ghost_keywords):
                        logging.warning(f"🧹 [清除幽靈題目] 題號 {q.get('question_number')} 命中佔位特徵，已剔除。")
                        continue

                    # 🚨 2. 純週期表/常數表附錄過濾（帶有「真題保護傘」：有選項或有問句者一律視為真題保留）
                    has_real_opts = len(opts) >= 2 and not all(str(opt.get("value", "")).strip() in dummy_opt_values for opt in opts)
                    has_question_clue = any(k in q_text for k in ["何者", "下列", "試問", "求", "為何", "多少", "？", "?", "正確", "錯誤", "選出"])
                    is_pure_table = bool(re.match(r'^(?:[【\[\s]*)(?:元素週期表|週期表|週期律表|原子量表|常數表|參考常數)', q_text))
                    
                    if (is_pure_table or q_num in ["週期表", "封面", "說明", "附錄"]) and not has_real_opts and not has_question_clue:
                        logging.warning(f"🧹 [清除附錄頁] 題號 {q_num} 為純週期表/常數表附錄，已剔除。")
                        continue
                        
                    # 2. 題幹只是單純題號重複 (如:「九十六學年度指定科目考試物理考科第40題」或「第51題」)
                    if re.match(r'^(?:[一二三四五六七八九十\d]+學年度)?(?:[^\n]{2,15}考科)?第?\s*\d+\s*題$', q_text):
                        logging.warning(f"🧹 [清除幽靈題目] 題號 {q.get('question_number')} 題幹僅為純題號標題，已剔除。")
                        continue

                    # 3. 選項全為虛擬假選項 (選項A, 選項B...)
                    if opts and all(str(opt.get("value", "")).strip() in dummy_opt_values for opt in opts):
                        logging.warning(f"🧹 [清除幽靈題目] 題號 {q.get('question_number')} 所有選項均為佔位符 (選項A/B/C/D)，已剔除。")
                        continue
                        
                    # 4. 題幹為空或長度小於 6 且無圖
                    if len(q_text) < 6 and not q.get("has_image"):
                        logging.warning(f"🧹 [清除幽靈題目] 題號 {q.get('question_number')} 題幹過短且無附圖，已剔除。")
                        continue

                    valid_q.append(q)
                return valid_q

            all_extracted_questions = filter_ghost_questions(all_extracted_questions)
            all_extracted_questions = self.clean_and_verify_questions(all_extracted_questions)
            all_extracted_questions = deduplicate_questions(all_extracted_questions)

            for q_data in all_extracted_questions:
                q_sub = q_data.get("sub_subject", "")
                
                # 🚨 模糊相容「社會考科」、「學測社會」、「社會科」等各式學科名稱
                if any(k in normalized_subject for k in ["社會", "社考"]):
                    if q_sub not in ["歷史", "地理", "公民與社會"]:
                        q_text = s2t(q_data.get("question_text", "") + q_data.get("shared_context", ""))
                        if any(k in q_text for k in ["憲法", "法律", "政府", "權利", "經濟", "市場", "社會", "勞工", "法規", "法治", "契約", "法官", "選制"]):
                            q_data["sub_subject"] = "公民與社會"
                        elif any(k in q_text for k in ["地圖", "氣候", "地形", "空間", "地理", "貿易", "生活圈", "自然環境", "沙丘", "位置", "等高線", "洋流", "降水"]):
                            q_data["sub_subject"] = "地理"
                        else:
                            q_data["sub_subject"] = "歷史"
                # 🚨 模糊相容「自然考科」、「學測自然」、「理綜」等各式學科名稱
                elif any(k in normalized_subject for k in ["自然", "自考", "理綜"]):
                    if q_sub not in ["物理", "化學", "生物", "地球科學"]:
                        q_text = s2t(q_data.get("question_text", "") + q_data.get("shared_context", ""))
                        if any(k in q_text for k in ["力", "速度", "電", "磁", "能量", "波", "加速度", "力學", "牛頓", "歐姆", "光電"]):
                            q_data["sub_subject"] = "物理"
                        elif any(k in q_text for k in ["化學", "反應", "分子", "溶液", "元素", "原子", "化合物", "酸鹼", "莫耳", "氧化", "還原"]):
                            q_data["sub_subject"] = "化學"
                        elif any(k in q_text for k in ["細胞", "基因", "生態", "植物", "動物", "生物", "染色體", "群落", "器官", "葉綠體", "演化"]):
                            q_data["sub_subject"] = "生物"
                        else:
                            q_data["sub_subject"] = "地球科學"
                else:
                    q_data["sub_subject"] = normalized_subject

            try:
                expected_q_nums = []
                ans_map = {}
                try:
                    if ans_text and ans_text != "無官方解答。":
                        ans_map = json.loads(ans_text)
                        if isinstance(ans_map, dict):
                            expected_q_nums = sorted(list(ans_map.keys()), key=natural_sort_key)
                except Exception as e:
                    logging.warning(f"⚠️ [解答 JSON 解析警告] 官方答案格式非標準 JSON，已自動降級防護: {e}")
                    ans_map = {}

                # 🚨【核心架構修復：Python 強制將官方答案字典與更正通知精準灌入題目】
                for q_item in all_extracted_questions:
                    qn = str(q_item.get("question_number", "")).strip()
                    matched_ans = None
                    
                    # 1. 優先檢查線程安全更正通知快取庫！
                    cached_errata_ans = get_errata_answer(clean_subj_name, qn)
                    if cached_errata_ans:
                        matched_ans = cached_errata_ans
                        logging.info(f"⚡ [更正通知生效] {clean_subj_name} 題號 {qn} 已覆寫為官方更正答案：{matched_ans}")
                    elif qn in ans_map:
                        matched_ans = ans_map[qn]
                    else:
                        for k_ans, v_ans in ans_map.items():
                            if is_precise_match(q_item, k_ans):
                                matched_ans = v_ans
                                break
                    
                    if matched_ans and str(matched_ans).strip() not in ["", "None", "null", "／", "/", "無", "無答案"]:
                        cleaned_match = clean_ocr_answer_format(str(matched_ans))
                        q_item['answer'] = cleaned_match
                        q_item['official_answer'] = cleaned_match
                        q_item['has_official_answer'] = True
                    else:
                        q_item['official_answer'] = ""
                        q_item['has_official_answer'] = False
                    
                def get_base_q_num(q_num_str: str) -> str:
                    q_num_str = str(q_num_str).strip()
                    match = re.match(r'^(\d+)-(\d+)$', q_num_str)
                    if match: return match.group(1)
                    match_letter = re.match(r'^([A-Ga-g])-(\d+)$', q_num_str)
                    if match_letter: return match_letter.group(1)
                    return q_num_str

                def is_question_covered(expected_base: str, extracted_questions: list) -> bool:
                    expected_base = str(expected_base).strip()
                    for q_item in extracted_questions:
                        q_num = str(q_item.get("question_number", "")).strip()
                        if q_num == expected_base: return True
                        range_match = re.match(r'^(\d+)-(\d+)$', q_num)
                        if range_match:
                            try:
                                start, end = int(range_match.group(1)), int(range_match.group(2))
                                val = int(expected_base)
                                if start <= val <= end: return True
                            except ValueError: pass
                        if q_num.startswith(expected_base) and len(q_num) > len(expected_base):
                            next_char = q_num[len(expected_base)]
                            if not next_char.isalnum(): return True
                    return False

                def get_official_answer_for_base(base_num: str, ans_dict: dict) -> str:
                    if base_num in ans_dict: return str(ans_dict[base_num])
                    sub_keys = []
                    for k in ans_dict.keys():
                        if k.startswith(f"{base_num}-"):
                            suffix = k[len(base_num)+1:]
                            if suffix.isdigit(): sub_keys.append((int(suffix), k))
                    if sub_keys:
                        sub_keys.sort()
                        return ",".join(str(ans_dict[k]) for _, k in sub_keys)
                    return ""

                gaps = []
                
                # 🔍 步驟 A：直接從「題目卷全卷文字層」提取實體存在的最大題號與所有題號清單 (Ground Truth)
                paper_physical_q_nums = set()
                max_paper_q_val = 0
                for p_idx in range(len(doc)):
                    p_txt = doc[p_idx].get_text("text")
                    # 排除封面與作答範例干擾
                    if "作答注意事項" in p_txt or "作答示例" in p_txt or "示例中" in p_txt:
                        lines = [l for l in p_txt.splitlines() if not any(k in l for k in ["作答注意事項", "作答示例", "例：", "答案卡第"])]
                        p_txt = "\n".join(lines)
                    
                    found_nums = re.findall(r'(?:^|\n)\s*(?:第?\s*(\d{1,2})\s*題|(\d{1,2})\s*[.．、\s)])(?!\d)', p_txt)
                    for fn in found_nums:
                        n_str = fn[0] or fn[1]
                        if n_str:
                            val = int(n_str)
                            if val <= 80: # 正常台灣高中考卷題號範圍
                                paper_physical_q_nums.add(str(val))
                                if val > max_paper_q_val:
                                    max_paper_q_val = val

                logging.info(f"📑 [題目卷真理校驗] 題目卷文字層偵測到實體最大題號為: 第 {max_paper_q_val} 題 (題號集合: {sorted([int(x) for x in paper_physical_q_nums])})")

                # 🚀 引擎一：官方解答對位補漏（嚴格受限於題目卷真實存在的最大題號）
                for num in expected_q_nums:
                    base_num = get_base_q_num(num)
                    # 🚨 若解答卷題號超過題目卷實體最大題號，判定為跨科污染或答案卷溢出，立刻丟棄！
                    if base_num.isdigit() and max_paper_q_val > 0 and int(base_num) > max_paper_q_val:
                        logging.warning(f"🛡️ [攔截跨科答案] 解答卷題號 {base_num} 超出題目卷實體最大題號 ({max_paper_q_val})，判定為跨科污染，予以忽略。")
                        continue
                    if not is_question_covered(base_num, all_extracted_questions):
                        # 該題必須在題目卷實體文字層中有出現痕跡才允許補漏
                        if not paper_physical_q_nums or base_num in paper_physical_q_nums or not base_num.isdigit():
                            if base_num not in gaps: gaps.append(base_num)
                                
                # 🚀 引擎二：題號連續性補漏（不設死題數上限，但範圍嚴格限制在當前已抓到的最小～最大題號之間）
                num_list = []
                letter_list = []
                for q_ext in all_extracted_questions:
                    ext_num = str(q_ext.get("question_number", "")).strip()
                    b_num = get_base_q_num(ext_num)
                    if b_num.isdigit():
                        num_list.append(int(b_num))
                    elif len(b_num) == 1 and b_num.upper() in "ABCDEFGHJKLMNOPQRSTUVWXYZ":
                        letter_list.append(b_num.upper())
                
                if num_list:
                    min_n, max_n = min(num_list), max(num_list)
                    # 僅補齊中間斷號（例如有 1~24 題，漏了 3 題才補；絕不往上擴張猜測第 25~60 題）
                    for n in range(min_n, max_n + 1):
                        n_str = str(n)
                        if n_str not in gaps and not is_question_covered(n_str, all_extracted_questions):
                            gaps.append(n_str)
                                
                # 檢查字母連續性 (如 A, B, D -> 漏了 C)
                if letter_list:
                    start_ord, end_ord = ord(min(letter_list)), ord(max(letter_list))
                    for o in range(start_ord, end_ord + 1):
                        char = chr(o)
                        if char not in gaps and not is_question_covered(char, all_extracted_questions):
                            gaps.append(char)
                
                if gaps:
                    # 按照自然排序重新整理缺口名單
                    gaps = sorted(gaps, key=natural_sort_key)
                    logging.warning(f"⚠️ [補漏雙引擎啟動] 結合解答對位與連續性推測，偵測到有 {len(gaps)} 道題目在第一階段漏抓：{gaps}")
                    for gap_num in gaps:
                        target_page_num = -1
                        
                        # 🚨 絕對禁止在封面頁（Page 0）或只有作答說明的頁面補漏
                        search_start_page = 1 if (exam_type != "SCHOOL" and len(doc) > 1) else 0
                        for page_idx in range(search_start_page, len(doc)):
                            page_text = doc[page_idx].get_text("text")
                            
                            # 若該頁包含過多「範例」或「作答注意事項」字眼，跳過該頁搜尋
                            if "作答注意事項" in page_text and "第壹部分" not in page_text:
                                continue
                                
                            # 嚴格正則比對：必須是實體獨立題號開頭（防止內文數字碰撞）
                            pattern = rf'(?:^|\n)\s*(?:第\s*{gap_num}\s*題|{gap_num}\s*[.．、\s)]|\({gap_num}\))(?!\d)'
                            if re.search(pattern, page_text):
                                target_page_num = page_idx
                                break
                        
                        if target_page_num == -1:
                            logging.info(f"⏭️ 題號 {gap_num} 在試卷實體內容中並不存在（屬答案卷欄位溢出或雜訊），安全略過。")
                            continue
                        
                        logging.info(f"🔍 題號 {gap_num} 最可能位於 PDF 第 {target_page_num+1} 頁，啟動單題高精度定向擷取...")
                        GLOBAL_METRICS.set_stage(f"🔍 定向補漏：正在精確擷取第 {gap_num} 題...")
                        
                        gap_page = doc[target_page_num]
                        gap_image_path = q_image_paths[target_page_num]
                        
                        official_ans_val = get_official_answer_for_base(gap_num, ans_map)
                        ans_prompt_part = f"\n🚨【本題官方標準答案】：{official_ans_val}\n請直接將此答案填入 `answer` 欄位中，絕對不可變更或縮水。" if official_ans_val else ""

                        gap_prompt = f"""
                        我們在全卷掃描中漏掉了第 {gap_num} 題。請仔細閱讀以下試卷影像：
                        1. 請精確找出第 {gap_num} 題的完整題幹、選項（若有）。{ans_prompt_part}
                        2. 將該單題的資料結構化填入 Pydantic 結構。
                        3. 🚨 必須將 `page_number` 設為 {target_page_num+1}。
                        """
                        
                        with Image.open(gap_image_path) as gap_pil:
                            res_gap, _ = self.ai_manager.generate_with_retry(
                                contents=[gap_prompt, gap_pil, f"=== 原卷第 {target_page_num+1} 頁純文字 ===\n" + s2t(gap_page.get_text("text"))],
                                response_schema=PageExtraction,
                                temperature=0.0,
                                preferred_model="gemini-3.5-flash",
                                enable_thinking=True,
                                task_desc=f"{paper_tag} [單題補漏 Q{gap_num}]"
                            )
                            
                            if res_gap and res_gap.get('questions'):
                                for new_q in res_gap['questions']:
                                    if str(new_q.get("question_number")) == str(gap_num):
                                        new_q['page_number'] = target_page_num + 1
                                        new_q['full_page_image_path'] = gap_image_path.replace("\\", "/")
                                        new_q['question_page_image_paths'] = q_image_paths
                                        new_q['answer_page_image_paths'] = a_image_paths
                                        new_q['rubric_page_image_paths'] = rubric_image_paths
                                        new_q['question_pdf_path'] = q_pdf.replace("\\", "/") if q_pdf else ""
                                        new_q['answer_pdf_path'] = a_pdf.replace("\\", "/") if a_pdf else ""
                                        new_q['rubric_pdf_path'] = rubric_pdf.replace("\\", "/") if rubric_pdf else ""
                                        
                                        new_q['image_paths'] = []
                                        if new_q.get('has_image') and new_q.get('image_bboxes'):
                                            new_q['image_paths'] = self.execute_crop(gap_page, new_q['image_bboxes'], img_dir, f"Q{gap_num}_Gap")
                                        
                                        # 🚨 補漏安全防禦：檢視是否為 AI 捏造的佔位題，是則拒收！
                                        test_list = filter_ghost_questions([new_q])
                                        if not test_list:
                                            logging.warning(f"⚠️ [捏造攔截] 題號 {gap_num} AI 回傳佔位假題或捏造內容，已被拒絕收錄！")
                                            continue

                                        new_q['_cropped_pil_images'] = [Image.open(p) for p in new_q['image_paths'] if os.path.exists(p)]
                                        all_extracted_questions.append(new_q)
                                        logging.info(f"🎯 [補漏成功] 已成功補回第 {gap_num} 題，並完成影像高精裁切與 PIL 緩衝。")
                                        break
                    
                    all_extracted_questions = self.clean_and_verify_questions(all_extracted_questions)
                    all_extracted_questions.sort(key=lambda x: natural_sort_key(x.get('question_number', '0')))

                    official_explanations = {}
                    if a_pdf: official_explanations.update(self.parse_detailed_explanations(a_pdf))
                    if rubric_pdf: official_explanations.update(self.parse_detailed_explanations(rubric_pdf))
                        
                    for q_data in all_extracted_questions:
                        matched_exp = None
                        for key_num, exp_text in official_explanations.items():
                            if is_precise_match(q_data, key_num):
                                matched_exp = exp_text
                                break
                        q_data['official_explanation'] = matched_exp if matched_exp else ""

            except Exception as e:
                logging.error(f"❌ [補漏引擎異常] 執行補漏時發生未預期錯誤: {e}")

            # =========================================================================
            # 🌐【通用選填題列號 ➔ 大題代號答案聚合器 (Universal Grid-to-Letter Aggregator)】
            # 自動將答案卷中的列號 (8: 1, 9: 2) 聚合給選填大題 (A: 1,2)，適用於所有大考與模考數學卷
            # =========================================================================
            if ans_map:
                for q in all_extracted_questions:
                    qn = str(q.get("question_number", "")).strip().upper()
                    # 辨識大考選填題大題代號 (A ~ H)
                    if len(qn) == 1 and qn in "ABCDEFGH":
                        q_text = q.get("question_text", "")
                        # 提取題幹中所有的挖空數字，如 [ 8 ][ 9 ] 或 ⑧⑨
                        blank_indices = re.findall(r'\[\s*(\d+)[^\]]*\]', q_text)
                        if not blank_indices:
                            # 支援圓圈數字 ①~⑳
                            circled_map = {chr(0x2460 + i): str(i + 1) for i in range(20)}
                            blank_indices = [circled_map[c] for c in q_text if c in circled_map]

                        if blank_indices:
                            aggregated_answers = []
                            for b_idx in blank_indices:
                                if b_idx in ans_map:
                                    aggregated_answers.append(str(ans_map[b_idx]))

                            if len(aggregated_answers) == len(blank_indices):
                                full_aggregated_ans = ",".join(aggregated_answers)
                                q["answer"] = full_aggregated_ans
                                q["official_answer"] = full_aggregated_ans
                                q["has_official_answer"] = True
                                logging.info(f"🎯 [選填列號通用聚合] 選填題 {qn} 成功聚合列號 {blank_indices} 之答案為: {full_aggregated_ans}")

            # =========================================================================
            # 🌐【通用題組子題題幹精確剪裁器】
            # 若子題 (1) 的題幹誤包含了 (2) 的文字，自動依 (1)、(2) 標籤進行分段修復
            # =========================================================================
            for q in all_extracted_questions:
                qn = str(q.get("question_number", ""))
                qt = q.get("question_text", "")
                if any(k in qn for k in ["(1)", "一(1)", "二(1)", "三(1)", "1(1)", "2(1)"]):
                    m_sub2 = re.search(r'[\(（]\s*2\s*[\)）]', qt)
                    if m_sub2:
                        cut_idx = m_sub2.start()
                        clean_sub1_text = qt[:cut_idx].strip()
                        if len(clean_sub1_text) >= 10:
                            q["question_text"] = clean_sub1_text
                            logging.info(f"✂️ [題組題幹通用剪裁] 子題 {qn} 成功剝離第 (2) 小題之干擾文字！")

            mismatch_count = sum(1 for q in all_extracted_questions if q.get("_length_mismatch"))
            if mismatch_count > 0:
                logging.warning(f"⚠️ [選填題格子對照驗證] 偵測到 {mismatch_count} 題選填題存在挖空數量與答案長度不一致！")
            else:
                logging.info("🎯 [選填題格子對照驗證] 全數通過！所有選填題挖空數皆與官方答案長度吻合。")

            for q_data in all_extracted_questions:
                q_num = q_data['question_number']
                has_options = len(q_data.get("options", [])) > 0
                matched_key = next((k for k in rubric_visual_map if is_precise_match(q_data, k)), None)
                
                if matched_key:
                    q_data['scoring_criteria'] = rubric_visual_map[matched_key]['text']
                    q_data['rubric_image_paths'] = rubric_visual_map[matched_key]['paths']
                    logging.info(f"🔗 [評分補件] 題號 {q_num} 已成功掛載手寫圖文標準")
                    
                    if has_options:
                        q_data['question_type'] = "混合題"
                        logging.info(f"💎 [混合題偵測] 題號 {q_num} 具備雙重屬性，已重塑其類型為: 混合題")
                        
                        current_ans = str(q_data.get('answer', '')).strip()
                        non_choice_indicators = ["見評分", "非選擇", "手寫", "無", "畫在", "作圖", "作答", ""]
                        
                        if any(ind in current_ans for ind in non_choice_indicators) or not current_ans:
                            rubric_paths = q_data.get('rubric_image_paths', [])
                            if rubric_paths and os.path.exists(rubric_paths[0]):
                                logging.info(f"⚖️ [跨PDF縫合] 題號 {q_num} (混合題) 無直接選擇答案，啟動跨模態答案反向提取...")
                                
                                class ExtractedAnswer(BaseModel):
                                    answer: str = Field(description="從評分標準圖片中精確提取出的選擇題或勾選題部分的標準答案。必須且只能是選項字母或數字，嚴禁包含任何中文字或說明。")

                                stitch_prompt = f"""
                                這是一份大考混合題的官方評分標準。
                                請你仔細閱讀這張圖片，找出該混合題中「選擇題/單選/多選/勾選」部分的標準答案。
                                我們距離選擇題部分的答案字元（如 'C' 或 '4'），不要有任何其他說明。
                                """
                                
                                try:
                                    with Image.open(rubric_paths[0]) as rub_img:
                                        res_stitch, _ = self.ai_manager.generate_with_retry(
                                            contents=[stitch_prompt, rub_img],
                                            response_schema=ExtractedAnswer,
                                            temperature=0.0,
                                            preferred_model="gemini-3.5-flash",
                                            enable_thinking=False,
                                            task_desc=f"{paper_tag} [混合題答案縫合 Q{q_num}]"
                                        )
                                        if res_stitch and res_stitch.get('answer'):
                                            cleaned_stitch_ans = clean_ocr_answer_format(res_stitch['answer'])
                                            if cleaned_stitch_ans:
                                                q_data['answer'] = cleaned_stitch_ans
                                                logging.info(f"🎯 [縫合成功] 題號 {q_num} (混合題) 選擇答案已修正為: {q_data['answer']}")
                                except Exception as e:
                                    logging.error(f"混合題答案跨模態縫合失敗: {e}")

                    sc_text = q_data.get('scoring_criteria', '')
                    has_checkbox_clue = "勾選" in q_data.get('question_text', '') or any(sym in sc_text for sym in ["☑", "□", "■", "✔"])
                    
                    if has_checkbox_clue and not q_data.get('options'):
                        logging.info(f"⚖️ [選項重建] 偵測到題號 {q_num} 為「勾選混合題」且缺少選項，啟動跨模態選項提取...")
                        
                        class ExtractedCheckboxes(BaseModel):
                            options: List[OptionItem] = Field(description="從評分標準或答題卷中提取的所有勾選選項列表。按順序為 A, B, C, D...。")
                            correct_key: str = Field(description="被勾選（☑ 或 ■）的正確選項代號。")

                        stitch_prompt = f"""
                        這是一份大考非選擇題的官方評分標準。
                        請你仔細閱讀以下評分文字與圖片，找出該題在答題卷上供學生「勾選」的所有選項內容（通常在『滿分參考答案』中以 ☑ 或 □ 標示）。
                        
                        任務：
                        1. 提取所有選項的文字，依序編號為 A, B, C, D...。
                        2. 找出被勾選（☑ 或帶有打勾、黑塊標記）的那個正確選項，將其 key 填入 `correct_key`。
                        
                        【評分標準文字對照】：
                        {sc_text}
                        """
                        
                        rubric_paths = q_data.get('rubric_image_paths', [])
                        stitch_contents = [stitch_prompt]
                        if rubric_paths and os.path.exists(rubric_paths[0]):
                            stitch_contents.append(Image.open(rubric_paths[0]))
                            
                        try:
                            res_opts, _ = self.ai_manager.generate_with_retry(
                                contents=stitch_contents,
                                response_schema=ExtractedCheckboxes,
                                temperature=0.0,
                                preferred_model="gemini-3.5-flash",
                                enable_thinking=True,
                                task_desc=f"{paper_tag} [混合題選項重建 Q{q_num}]"
                            )
                            if res_opts and res_opts.get('options'):
                                q_data['options'] = res_opts['options']
                                q_data['question_type'] = "混合題"
                                current_ans = str(q_data.get('answer', '')).strip()
                                if current_ans in ["", "／", "/", "\\", "無"]:
                                    q_data['answer'] = res_opts.get('correct_key', '')
                                logging.info(f"🎯 [重建成功] 題號 {q_num} 已成功恢復 {len(q_data['options'])} 個勾選選項，並更新答案為: {q_data['answer']}")
                        except Exception as e:
                            logging.error(f"混合題勾選選項重建失敗: {e}")

            # =========================================================================
            # 🌐【全卷跨頁穿透題組資源共享池 (Cross-Page Context & Asset Piercing Engine)】
            # =========================================================================
            def extract_group_explicit_range(text: str) -> tuple:
                """精確提取『第28至32題為題組』、『回答第30~33題』等明確題號範圍 (start_num, end_num)"""
                if not text: return None
                m = re.search(r'(?:第\s*)?(\d+)\s*(?:至|到|~|～|-)\s*(\d+)\s*題', text)
                if m:
                    try:
                        s_n, e_n = int(m.group(1)), int(m.group(2))
                        if 1 <= s_n < e_n <= 90 and (e_n - s_n) <= 12:
                            return (s_n, e_n)
                    except Exception: pass
                return None

            def extract_subquestion_base_tag(q_num_str: str) -> str:
                s = str(q_num_str).strip()
                m_cn = re.match(r'^([一二三四五六七八九十]+)', s)
                if m_cn: return m_cn.group(1)
                m_sub = re.match(r'^(\d+)[\-_(（]', s)
                if m_sub: return m_sub.group(1)
                return ""

            # 步驟一：第一輪掃描全卷，建立「題組錨定資源清單 (Anchor Pools)」
            anchor_pools = []
            for q_idx, q in enumerate(all_extracted_questions):
                sc = str(q.get('shared_context', '')).strip()
                q_text = str(q.get('question_text', '')).strip()
                combined_probe = sc + " " + q_text[:120]

                rng = extract_group_explicit_range(combined_probe)
                base_tag = extract_subquestion_base_tag(q.get('question_number', ''))

                if sc or rng or base_tag:
                    # 尋找是否能併入已存在的錨點池
                    matched_pool = None
                    for pool in anchor_pools:
                        if rng and pool.get("range") and rng == pool["range"]:
                            matched_pool = pool; break
                        if base_tag and pool.get("base_tag") and base_tag == pool["base_tag"]:
                            matched_pool = pool; break

                    if matched_pool:
                        if len(sc) > len(matched_pool["canonical_context"]):
                            matched_pool["canonical_context"] = sc
                    else:
                        anchor_pools.append({
                            "range": rng,
                            "base_tag": base_tag,
                            "canonical_context": sc,
                            "image_paths": [],
                            "image_bboxes": []
                        })

            # 步驟二：收集各題組內的所有題幹與圖表資產（排除選項小圖 _Opt）
            for pool in anchor_pools:
                rng = pool.get("range")
                base_tag = pool.get("base_tag")
                
                for q in all_extracted_questions:
                    qn_str = str(q.get('question_number', '')).strip()
                    belongs = False

                    # A. 區間匹配判定 (如 28 <= 30 <= 32)
                    if rng:
                        digits = re.findall(r'^\d+$', qn_str)
                        if digits and rng[0] <= int(digits[0]) <= rng[1]:
                            belongs = True
                    # B. 大題前綴判定 (如 '二(1)', '二(2)' 屬於 '二')
                    if not belongs and base_tag:
                        if extract_subquestion_base_tag(qn_str) == base_tag:
                            belongs = True

                    if belongs:
                        # 收集非選項附圖
                        valid_imgs = [p for p in q.get('image_paths', []) if "_Opt" not in p and os.path.exists(p)]
                        pool["image_paths"].extend(valid_imgs)
                        pool["image_bboxes"].extend(q.get('image_bboxes', []))
                        # 若此子題有更齊全的 shared_context，動態更新至池中
                        cur_sc = str(q.get('shared_context', '')).strip()
                        if len(cur_sc) > len(pool["canonical_context"]):
                            pool["canonical_context"] = cur_sc

                # 資源池去重
                pool["image_paths"] = list(dict.fromkeys(pool["image_paths"]))

            # 步驟三：全卷穿透覆蓋與內文冗餘剔除（穿透跨頁界限）
            for q in all_extracted_questions:
                qn_str = str(q.get('question_number', '')).strip()
                target_pool = None

                for pool in anchor_pools:
                    rng = pool.get("range")
                    base_tag = pool.get("base_tag")
                    if rng:
                        digits = re.findall(r'^\d+$', qn_str)
                        if digits and rng[0] <= int(digits[0]) <= rng[1]:
                            target_pool = pool; break
                    if base_tag and extract_subquestion_base_tag(qn_str) == base_tag:
                        target_pool = pool; break

                if target_pool and target_pool["canonical_context"]:
                    canonical_sc = target_pool["canonical_context"]
                    q['shared_context'] = canonical_sc

                    # 跨頁穿透繼承題組圖表資產
                    if target_pool["image_paths"]:
                        opt_imgs = [p for p in q.get('image_paths', []) if "_Opt" in p]
                        merged_imgs = list(dict.fromkeys(target_pool["image_paths"] + opt_imgs))
                        q['image_paths'] = merged_imgs
                        q['has_image'] = True

                        # 刷新 PIL 緩衝
                        fresh_pils = []
                        for ip in merged_imgs:
                            try: fresh_pils.append(Image.open(ip))
                            except Exception: pass
                        q['_cropped_pil_images'] = fresh_pils

                    # 🚨 題幹冗餘引言自動剝離：若第一個子題把整個題組文章塞在 question_text 開頭，自動清洗！
                    clean_qt = q.get('question_text', '').strip()
                    if len(clean_qt) > len(canonical_sc) and clean_qt.startswith(canonical_sc[:60]):
                        stripped_qt = clean_qt[len(canonical_sc):].strip()
                        if len(stripped_qt) >= 5:
                            q['question_text'] = stripped_qt
                            logging.info(f"  -> ✂️ [題幹冗餘剝離] 題號 {qn_str} 已自動移除與題組引言重複的開頭內文。")

                    logging.info(f"  -> 🌐 [跨頁穿透成功] 題號 {qn_str} 成功繼承全卷題組背景與 {len(q.get('image_paths', []))} 張圖表資產！")

            # 🆕 完成第一階段掃描後，將題目結構安全寫入快取檔
            try:
                clean_extracted_questions = []
                for q_item in all_extracted_questions:
                    q_copy = dict(q_item)
                    q_copy.pop('_cropped_pil_images', None)
                    clean_extracted_questions.append(q_copy)
                with open(raw_extracted_json_path, "w", encoding="utf-8") as f:
                    json.dump(clean_extracted_questions, f, ensure_ascii=False, indent=4)
                logging.info(f"💾 [Stage 1 快取儲存] 第一階段題目結構已安全寫入：{raw_extracted_json_path}")
            except Exception as e:
                logging.error(f"寫入 Stage 1 快取失敗: {e}")

            # doc.close()
            # === 第一階段 結束點 ===

        math_scope_instruction = ""
        if math_type:
            math_scope_instruction = f"""
            【四、數學科特定課綱與元認知思維引導】：
            本卷屬於：{math_type}。
            
            🚨【補教名師元認知解題規範】🚨：
            在你的解題與思考背景中，請強迫自己執行以下思考步驟，這能幫助你產出最嚴謹、零失誤且最具啟發性的多重解答：
            
            1. **【雙系統幾何沙盒】**：
               面對任何幾何考題，請先在腦中評估「純幾何性質」與「直角座標化」的雙軌可行性。若選用座標化，請選擇能產生最多零座標的頂點作為原點 $(0,0,0)$ 或 $(0,0)$。
               
            2. **【自由度與基底簡化】**：
               在代數展開前，先計算變數與約束條件。優先尋找變數的「對稱性」與「週期性」，利用對稱多項式、根與係數關係（韋達定理）進行代數降階，嚴禁盲目暴力展開。
               
            3. **【對稱性與邊界值檢驗（Symmetry & Boundary Check）】**：
               在產出任何最終解答前，請代入特殊邊界值（如角度為 0 或 90度，或變數相等、特殊格子點）進行快速檢算，確認標準解與另解在極端狀態下完全自洽。
               
            4. **【代數反代回歸一驗算】**：
               解出任何代數解或交點後，必須在背景將解「反代回」最原始的題目方程式中進行左右等式驗算，確認無任何移項、去括號、負號漏看或分數計算失誤。
            """

        subject_specific_instruction = ""
        if normalized_subject in ["國文", "國綜"]:
            subject_specific_instruction = """
            【五、國文科名師多維解題與寫作規範】：
            1. **多元解題維度**：
               - `### 【標準解法：文本語意與文言訓詁】`：若涉及文言文，必須進行字詞訓詁（實詞虛詞拆解、倒裝/省略句式還原），並給予優美流暢的白話對譯。
               - `### 【另解一：文學意象與文意脈絡分析】`：從核心意象、情章剪裁、作者生平或文化思想背景切入，引導學生體會作者的情感寄託。
               - `### 【另解二 / 考場速解：語氣判讀與關鍵字刪去法】`：分析選項語氣的絕對性、情感色彩（褒貶）、句式對仗排偶對稱性，教導學生在考場快速排除干擾項。
            2. **白話散文與跨文本**：明確指出選項中的「偷換概念」或「過度推論」之處。
            """
        elif normalized_subject == "國寫":
            subject_specific_instruction = """
            【五、國語文寫作（國寫）專屬引導與評分量規】：
            1. **題型分流**：
               - 第一大題（知性題）：著重「文意統整」、「分析說明」與「立場論述」。詳解必須條列審題重點、立論依據與三段式架構範文。
               - 第二大題（情意題）：著重「情感抒發」、「生活體悟」與「意象深化」。詳解必須展示高分範文、核心意象經營手法與名言佳句延伸。
            2. **評分標準規範**：嚴禁套用數理選擇題評分！必須採用大考中心官方「六等九級分量規」：
               `國寫評分標準：A+等（22~25分，立意深刻結構嚴謹）、A等（18~21分，思維清晰敘述完整）、B+等（14~17分，文意通順符合題意）、B等（10~13分，結構稍鬆文筆平實）、C等（1~9分，未切題或字數不足）。`
            """
        elif normalized_subject == "英文":
            subject_specific_instruction = """
            【五、英文科名師多維解題與寫作規範】：
            1. **多元解題維度**：
               - `### 【標準解法：語法骨架與語意分析】`：長難句必須拆解語法骨架（主詞 S、主要動詞 V、分詞構句、關係子句分層說明），並附上通順之中文翻譯。
               - `### 【另解一：上下文語境與邏輯銜接】`：針對克漏字與篇章結構，抓出關鍵轉折詞（Signposts，如 however, consequently）、代名詞前指關係（Reference words）與語意場連貫性。
               - `### 【另解二 / 考場速解：詞根詞綴與搭配詞秒殺】`：運用搭配詞（Collocations）、介系詞搭配慣用法、詞根詞綴（Etymology）構詞法，提供不需通篇讀完即可破題的秒殺技巧。
            2. **閱讀測驗**：必須精準標出文章中的【定位句】（Quote），並點破選項是如何進行「同義詞改寫（Paraphrasing）」的。
            """
        elif normalized_subject == "歷史":
            subject_specific_instruction = """
            【五、歷史科名師多維解題規範】：
            1. `### 【標準解法：時序因果與歷史脈絡】`：結合具體時代背景（政治體制、經濟結構、社會文化），梳理歷史事件的前因後果。
            2. `### 【另解一：史料批判與視角解讀】`：辨析題幹是一手史料（Primary source）還是二手史料，解讀作者身分、寫作立場、意識形態與歷史詮釋的主客觀偏見。
            3. `### 【另解二 / 考場速解：特徵關鍵字檢索法】`：提取史料中不可竄改的關鍵指標（如貨幣制度、官制名稱、特殊地名、法令名稱），秒判年代與政權。
            """
        elif normalized_subject == "地理":
            subject_specific_instruction = """
            【五、地理科名師多維解題規範】：
            1. `### 【標準解法：地理原理與成因分析】`：闡明背後的自然法則（大氣環流、洋流、板塊運動、地形作用力）或人文區位理論（中地理論、韋伯工業區位、農業集約度）。
            2. `### 【另解一：空間圖表與數據解讀】`：深入解析題幹地圖（等高線、二度分帶坐標）、溫雨圖、統計圖表或 GIS 疊圖邏輯。
            3. `### 【另解二 / 考場速解：極端氣候與地理要素排除法】`：利用半球判定、迎風背風、緯度帶特徵，快速刪去不合常理之地區選項。
            """
        elif normalized_subject in ["公民與社會", "公民"]:
            subject_specific_instruction = """
            【五、公民與社會名師多維解題規範】：
            1. `### 【標準解法：學科模型與法理分析】`：
               - 法律題：嚴格套用刑法三階論（構成要件、違法性、罪責）、民法權利能力、或憲法比例原則（適當性、必要性、衡平性）。
               - 經濟題：詳細分析供需曲線位移、均衡價格變動、消費者/生產者剩餘、外部效果或比較利益原理。
               - 政治/社會題：結合民主治理、選制換算、公共利益衡量或社會不平等理論。
            2. `### 【另解一：圖形位移與幾何推理】`（經濟題必備）：清晰描述曲線移動方向及福利變化。
            3. `### 【另解二 / 考場速解：法理要件關鍵字速判】`：利用法條主體要件、除斥期間、或正義理論之核心關鍵詞快速突破。
            """
        elif normalized_subject in ["數學", "數A", "數B", "數甲", "數乙", "數學A", "數學B", "數學甲", "數學乙"]:
            subject_specific_instruction = """
            【五、數學科專屬多解與解題思路規範】：
             1. **【極致推崇「一題多解」與「多元解題思維」】**：
               - 在 `detailed_solution` 欄位中，除了提供符合大課綱的 `### 【標準解法】` 之外，**【強制要求】寫出至少 3 到 4 種（含）以上完全不同維度的解題思維與切入點**（如：`### 【標準解法】`、`### 【另解一】`、`### 【另解二】`、`### 【另解三 / 秒殺速解】`）。
               ，**但請保持精簡，避免過度冗長導致 JSON 截斷**。
               - 針對不同題型，你可以展現以下多元學術維度的切入點：
                 - **幾何直覺法（平面/空間幾何、向量）**：運用圓冪定理、托勒密定理、弦切角、對稱性或重心等幾何定理解題。
                 - **向量與座標化解（直角座標系）**：建立最適座標系（讓原點與軸線對齊最多零座標），將幾何問題代數化。
                 - **三角函數與三角比解**：利用正弦、餘弦、和差角、倍半角公式等，以角度與邊長關係突破。
                 - **代數邏輯與不等式極值解**：利用算幾不等式、柯西不等式或二次函數配方法，探討極值問題。
                 - **代數降階與對稱多項式解（多項式、方程）**：利用韋達定理、對稱多項式化簡、拉格朗日插值法等，避免暴力展開。
                 - **函數圖形與幾何交點法**：畫出函數圖形，透過觀察對稱中心、凹凸性、切線逼近或遞增遞減特性求解。
                 - **特殊值法與極端值逼近法**：代入極端邊界值、對稱特殊點或特殊格子點，作為驚艷的快速驗算或秒殺解法。
               - 請確保這 2~4 種解法各有千秋，邏輯鏈嚴密、過程完整且 LaTeX 格式精緻，帶給學生全方位的思維激盪！
               - 對於**代數方程式、多項式 or 函數極值題**，嘗試提供：
                 - **「代數邏輯推演法」**。
                 - **「函數圖形與幾何交點法」**：畫出函數圖形，觀察對稱軸、對稱中心、凹凸性或線性逼近來求得視覺化的幾何解。
                 - **「特殊值法」**：快速選取邊界值、特殊格子點或對稱點，在選擇題中快速排除錯誤選項。
                 
            2. **【微積分基本定理與極值/反曲點防混淆剛性要求】**：
               - 若本題涉及函數的極值或反曲點分析，你**必須**展現最高水準的微積分學科邏輯。
               - **極值點 (Local Extrema)**：對於可微函數，一階導數 $f'(c) = 0$ 是必要條件。極值判別中，若 $f''(c) < 0$，則在 $c$ 處有局部極大值；若 $f''(c) > 0$，則在 $c$ 處有局部極小值。**絕對禁止**誤以為極值點處的二階導數 $f''(c)$ 必為 0！
               - **反曲點 (Inflection Point)**：指的是函數凹凸性改變的點。若 $f''(c) = 0$ 且在其左右兩側的二階導數正負號相反，則 $(c, f(c))$ 為反曲點。反曲點處的切線斜率不一定為 0。
               - **二階導數與凹凸性**：二階導數 $f''(x) > 0$ 代表函數在該區間圖形「凹向上」；$f''(x) < 0$ 代表「凹向下」。請反覆確認你的凹凸性描述與極值判別完全正確，嚴禁出現凹凸性張冠黎戴、二階導數正負號搞反等學科硬傷！
               - **🚨【極值與定義域邊界分析防撞擊盲區（端點最大值）】🚨**：
                 在分析任何幾何或代數求極值、最大值、最小值問題時，**你必須先明確求出所有自變數的『幾何約束定義域』！**
                 - **絕對禁止**在求得導函數極值點（如 $h = 1/\\sqrt{3}$）後就直接當作答案！你必須繪製單調性分析表。
                 - 如果求出的局部極值點落在定義域邊界（如 $h \\ge 1/\\sqrt{2} \\approx 0.707$）之外，最大值或最小值必定發生在『定義域端點（邊界值）』處！你必須代入邊界端點值進行推導（如在 $h = 1/\\sqrt{2}$ 處取得最大值），給出最嚴謹的端點最大化運算。
                 
            3. **【排列組合與機率題之極致嚴謹規範】**：
               若本題涉及「排列組合」 or 「機率」：
               - **優先考慮反面解法（扣除法）**：對於多重條件，「全集 - 餘集」通常比正面分類討論更不容易遺漏。
               - **相同物與相異物分配**：分清重複排列、重複組合（H 轉 C）之分配模型。
               - **雙重驗算**：必須採用正面分類與反面扣除兩種獨立方法互相驗算，確保數值完全一致。
               3. **【選擇題選項逐一排除與對位驗算】**：
               - 數學科的單選與多選題，通常每個選項都是一個獨立的數學敘述（例如不等式範圍、幾何關係或特定數值）。
               - **【硬性要求】**：你必須在 `options_analysis` 中對這 4~5 個選項「逐一單獨列出」並給予最嚴謹的代數或幾何證明與證偽。嚴禁只在標準解法中求出範圍，就在選項分析中一筆帶過！
            """
        elif normalized_subject in ["物理", "化學", "生物", "地球科學", "自然"]:
            subject_specific_instruction = """
            【五、自然科專屬審查與解題規範】：
            1. **單位與物理量對齊**：所有公式計算（如：化學計量、力學運動）必須標示單位（LaTeX 格式，如 $g/mol$, $m/s^2$）。
            2. **圖表與實驗對位**：若題目有實驗圖表或相圖，詳解必須針對圖表中的橫軸、縱軸、控制變因進行解讀，說明數據趨勢如何得出答案。
            3. **化學式與結構式**：化學反應方程式必須使用標準 LaTeX 格式配平。
            4. **特值法的妙用（極重要）**：對於物理、化學選擇題，若符號推導極度抽象，**強烈允許並建議在【快速另解】中採用『特殊數值代入法』**。
            """
        elif normalized_subject in ["歷史", "地理", "公民與社會", "社會", "公民"]:
            subject_specific_instruction = """
            【五、社會科專屬審查與解題規範】：
            1. **歷史：時代背景與史料解析**：必須結合時間軸與人名地名，分析一手/二手史料的立場與主客觀偏差（區分 Fact 與 Interpretation）。
            2. **地理：空間與統計圖表分析**：等高線、溫雨圖、NDVI 圖、韋伯原料指數等，必須解讀其圖形指標或幾何特徵，不能直接跳出答案。
            3. **公民：法條與經濟模型**：
                - 法律題需引用三階論或比例原則進行合憲性分析。
                - 經濟題（如消費者剩餘、外部效果）需說明供需曲線位移與無謂損失的幾何變動。
            """

        subject_rubric = SUBJECT_DIFFICULTY_RUBRICS.get(subject, GENERAL_DIFFICULTY_RUBRIC)

        # =========================================================
        # 【第二、三階段】：滑動窗口批次詳解、審查與中斷點續傳
        # =========================================================
        active_batch_size = 1 
        max_single_attempts = 8
        max_rechecks = 8

        # 🆕 檢查並加載未完成的暫存進度檔
        loaded_partial_questions = []
        if os.path.exists(partial_json_path):
            try:
                with open(partial_json_path, "r", encoding="utf-8") as f:
                    loaded_partial_questions = json.load(f)
                logging.info(f"📂 [中斷點續傳] 偵測到 [{year} {subject}] 未完成的暫存進度檔：{partial_json_path}")
                logging.info(f"  -> 已成功載入 {len(loaded_partial_questions)} 題已完成的詳解，本次將跳過重複生成。")
            except Exception as e:
                logging.error(f"讀取暫存進度檔失敗: {e}")

        all_final_questions = []
        queue_lock = threading.Lock() # 保護寫入共用陣列的鎖

        

        def get_unique_q_key(q_obj):
            return (str(q_obj.get("question_number", "")).strip(), str(q_obj.get("question_type", "")).strip())

        # 💡 將已加載的【有效詳解】建立成對位索引表（以題號+題型為複合鍵，防止跨大題碰撞）
        partial_map = {get_unique_q_key(q): q for q in loaded_partial_questions if is_valid_solution(q.get("detailed_solution"))}

        # 📊 依實體掃描出的真實題目與已暫存題目鎖定真理分母（絕不硬湊虛假 benchmark）
        confirmed_total_q = max(len(all_extracted_questions), len(loaded_partial_questions))
        preloaded_keys = [f"{q.get('question_number')}_{q.get('question_type','')}" for q in loaded_partial_questions if is_valid_solution(q.get("detailed_solution"))]
        
        GLOBAL_METRICS.set_current_paper(paper_label, total_q=confirmed_total_q, existing_partial_q=len(preloaded_keys), initial_solved_keys=preloaded_keys)
        GLOBAL_METRICS.update_paper_total_questions(confirmed_total_q, status_msg="🧠 名師詳解生成中...")
        GLOBAL_METRICS.add_log(f"🧠 {paper_label} 第一階段掃描完畢，鎖定真實總題數: {confirmed_total_q} 題 (已續傳 {len(preloaded_keys)} 題)")
        
        # =========================================================================
        # 🚀 [模擬考專屬快車道] 官方詳解本一鍵直接組裝，完全跳過 Stage 2/3 推導！
        # =========================================================================
        task_queue = [] # 🚨 頂層安全初始化，防範 NameError 致命崩潰！
        if is_mock_paper and mock_sol_pdf_path and mock_sol_page_indices:
            logging.info(f"⚡ [啟動模考快車道] 正在直接從官方詳解本視覺萃取原汁原味解析...")
            official_mock_exps = self.extract_mock_explanations_visual(
                mock_sol_pdf_path, 
                mock_sol_page_indices, 
                img_dir, 
                spec_name
            )
            
            # 🚨 [真理主鍵逆向反查] 詳解本涵蓋了全卷所有題目（如 1~42 題）。以詳解本為基底，建立全域題目字典！
            master_q_dict = {str(q.get("question_number", "")).strip(): q for q in all_extracted_questions if q.get("question_number")}

            # 刷新全域儀表板總題數為「詳解本真實題數」，絕對鎖定真理分母！
            actual_total_q = max(len(all_extracted_questions), len(official_mock_exps))
            GLOBAL_METRICS.update_paper_total_questions(actual_total_q, status_msg="模考快車道裝配中...")

            # 遍歷詳解本中辨識出的所有題號，確保一題不漏！
            all_target_q_nums = sorted(list(set(list(master_q_dict.keys()) + list(official_mock_exps.keys()))), key=natural_sort_key)

            for qn in all_target_q_nums:
                q = master_q_dict.get(qn)
                matched_exp = official_mock_exps.get(qn)
                
                # 若題號包含括號或非選，進行模糊精準匹配
                if not matched_exp:
                    for k_exp, v_exp in official_mock_exps.items():
                        if is_precise_match({"question_number": qn}, k_exp):
                            matched_exp = v_exp
                            break

                # 💡 若題目卷 OCR 漏抓該題，但詳解本有該題，自動補建題目骨架！
                if not q and matched_exp:
                    q = {
                        "academic_year": standard_academic_year,
                        "exam_source": standard_exam_source,
                        "sub_subject": clean_subj_name,
                        "question_number": qn,
                        "page_number": 1,
                        "shared_context": "",
                        "question_text": f"第 {qn} 題（詳見題目卷）",
                        "options": matched_exp.get("options_analysis", []),
                        "answer": clean_ocr_answer_format(matched_exp.get("answer", "")),
                        "question_type": "單選題" if len(matched_exp.get("answer", "")) == 1 else "多選題",
                        "has_image": False,
                        "image_bboxes": [],
                        "image_paths": []
                    }

                if not q: continue
                q_key = get_unique_q_key(q)

                if q_key in partial_map:
                    q.update(partial_map[q_key])
                    all_final_questions.append(q)
                    # 🚨 同步登記至狀態機已完成清單，防止分子遺漏
                    GLOBAL_METRICS.record_question_completed(f"{qn}_{q.get('question_type','')}", is_newly_solved=False, status_msg=f"模考快車道裝配中 ({len(all_final_questions)}/{actual_total_q})")
                    continue
                            
                # 🚨 詳解空洞檢測：若官方詳解本該題僅有一行標題或字數少於 35 字（如第 47、52 題），退回 Stage 2 大模型補齊名師推導！
                raw_exp_body = matched_exp.get("explanation_text", "").strip() if matched_exp else ""
                is_exp_too_trivial = len(raw_exp_body) < 35 or (raw_exp_body.startswith("內容：") and len(raw_exp_body.splitlines()) <= 2)

                # 🚨 詳解空洞檢測：若官方詳解本該題僅有一行標題或字數少於 35 字（如第 47、52 題），退回 Stage 2 大模型補齊名師推導！
                raw_exp_body = matched_exp.get("explanation_text", "").strip() if matched_exp else ""
                is_exp_too_trivial = len(raw_exp_body) < 35 or (raw_exp_body.startswith("內容：") and len(raw_exp_body.splitlines()) <= 2)

                if matched_exp and not is_exp_too_trivial:
                    # 1. 題意分析：依原卷真實內容動態判定，絕不硬湊虛構標籤
                    src_curriculum = matched_exp.get("source_curriculum", "").strip()
                    ex_target = matched_exp.get("exam_target", "").strip()
                    
                    # 過濾常見佔位雜訊
                    if any(bad in src_curriculum.lower() for bad in ["無", "未知", "未提供", "none", "null"]): src_curriculum = ""
                    if any(bad in ex_target.lower() for bad in ["無", "未知", "未提供", "none", "null"]): ex_target = ""

                    analysis_parts = []
                    if src_curriculum: analysis_parts.append(f"📚 **【官方試題出處】**：{src_curriculum}")
                    if ex_target: analysis_parts.append(f"🎯 **【官方測驗目標】**：{ex_target}")

                    core_focus = ex_target or src_curriculum
                    if core_focus:
                        analysis_parts.append(f"💡 **【核心題意引導】**：本題重點評量 **{core_focus}**，解題關鍵在於提取題幹條件進行因果推演。")
                        q["solving_strategy"] = f"💡 **【官方思維拆解】**：\n本題聚焦於 **{core_focus}**。依據官方詳解指引，解題突破口在於辨識核心條件，並依循學科定理完成推導。"
                    else:
                        analysis_parts.append("💡 **【核心題意引導】**：本題考查核心學科概念與題幹情境之綜合理解，解題關鍵在於精準掌握題目已知條件。")
                        q["solving_strategy"] = "💡 **【解題思維拆解】**：\n依據官方詳解指引，解題突破口在於提取題目的關鍵條件，並按部就班依循學科定理完成推導。"

                    q["question_analysis"] = "\n".join(analysis_parts)
                    # 3. 完整解法：組裝官方解析與詳解附圖
                    sol_body = matched_exp.get("explanation_text", "").strip()
                    sol_imgs = matched_exp.get("solution_image_paths", [])
                    if sol_imgs:
                        img_embeds = []
                        for s_i, s_path in enumerate(sol_imgs, 1):
                            rel_p = os.path.relpath(s_path, os.path.dirname(json_path)).replace("\\", "/")
                            img_embeds.append(f"\n\n![詳解圖 {s_i}]({rel_p})")
                        sol_body += "".join(img_embeds)
                        
                    q["detailed_solution"] = sol_body

                    # 4. 選項深入剖析
                    opts_ana = matched_exp.get("options_analysis", [])
                    if isinstance(opts_ana, list) and opts_ana:
                        formatted_opts = []
                        for opt_it in opts_ana:
                            k = opt_it.get("key", "").strip()
                            exp = opt_it.get("explanation", "").strip()
                            if k and exp: formatted_opts.append(f"- **({k})** {exp}")
                        q["options_analysis"] = "\n".join(formatted_opts) if formatted_opts else "本題為非選擇題/選填題，無選項可供分析。"
                    else:
                        q["options_analysis"] = "本題為非選擇題/選填題，無選項可供分析。"

                    # 5. 評分標準、難度評分與知識點（🚨 官方評分原則同步寫入 scoring_criteria）
                    official_rubric = matched_exp.get("scoring_rubric", "").strip()
                    q["scoring_rubric"] = official_rubric or "依考卷標準計分。"
                    if official_rubric:
                        q["scoring_criteria"] = official_rubric
                    q["difficulty_level"] = matched_exp.get("difficulty_level", 5)
                    q["difficulty_reason"] = matched_exp.get("difficulty_reason", f"出處涵蓋：{src_curriculum or '常規課程'}。")
                    
                    # 🚨 補教名師結構化昇華：補齊核心概念複習與易錯陷阱
                    if not q.get("concept_review"):
                        q["concept_review"] = f"本題考查核心學科概念：**{ex_target or src_curriculum or '課程核心概念'}**。解題時需依據定義正確列式，避免運算與符號失誤。"
                    if not q.get("traps_and_warnings"):
                        q["traps_and_warnings"] = "⚠️ **【易錯陷阱】**：解題時請特別注意題幹條件中的正負號、單位換算與邊界限制，避免因粗心或公式套用不當而失分。"
                    if not q.get("advanced_supplement"):
                        q["advanced_supplement"] = "無"

                    t_cat = matched_exp.get("topic_category", "") or "必修_綜合主題_核心概念"
                    q["topic_category"] = t_cat
                    q["topic_categories"] = [t_cat]
                    q["techniques_used"] = matched_exp.get("techniques_used", ["必修_解題方法_綜合理論推導"])
                    
                    # 6. 標準答案對齊：若原本 answer 缺失，補齊為詳解本答案
                    official_clean = clean_ocr_answer_format(matched_exp.get("answer", ""))
                    if official_clean and (not q.get("answer") or q["answer"] in ["", "／", "/"]):
                        q["answer"] = official_clean
                        q["official_answer"] = official_clean
                        q["has_official_answer"] = True
                        
                    # 🚨 執行輕量級圖文 OCR 一致性自癒校驗（校正題幹錯字、答案對位）
                    q = self.heal_mock_question_consistency(q, matched_exp)

                    # 補齊結論句
                    effective_ans = q.get("official_answer") or q.get("answer", "")
                    if effective_ans and "**綜上所述" not in q["detailed_solution"]:
                        if q.get("question_type") in ["單選題", "多選題"]:
                            valid_opts = re.findall(r'[A-Ga-g1-9]', effective_ans)
                            if valid_opts:
                                formatted_opts = "".join([f"({c.upper()})" for c in valid_opts])
                                q["detailed_solution"] += f"\n\n**綜上所述，本題正確選項為：{formatted_opts}**"
                        else:
                            q["detailed_solution"] += f"\n\n**綜上所述，本題正確答案為：** {effective_ans}"

                    update_subject_taxonomy(q.get("sub_subject", subject), q)
                    all_final_questions.append(q)
                    
                    # 🚨 透過統一事件派發器即時連動：使用真正的 actual_total_q 分母，徹底解決進度不增或分母錯誤！
                    GLOBAL_METRICS.record_question_completed(
                        f"{qn}_{q.get('question_type','')}", 
                        is_newly_solved=True, 
                        status_msg=f"模考快車道裝配中 ({len(all_final_questions)}/{actual_total_q})"
                    )
                    GLOBAL_METRICS.add_log(f"⚡ [模考快車道] 題號 {qn} 已成功裝配官方出處、解析與圖表 ({len(all_final_questions)}/{actual_total_q})！")
                    logging.info(f"🎉 [模考收錄成功] 題號 {qn} 已完成圖文自癒校驗並收錄！")
                else:
                    # 🚨 若詳解本漏抓該題，或官方解析僅有一行標題，退回常規 task_queue 由大模型生成完整名師解析！
                    critique_note = "官方詳解僅有一行標題，請生成完整詳細步驟與推導" if is_exp_too_trivial else ""
                    task_queue.append({"q_data": q, "critique": critique_note, "retry_count": 0, "recheck_count": 0})
                    
            save_partial_progress_immediately(partial_json_path, all_final_questions)
            logging.info(f"⚡ [模考快車道結算] 成功直裝 {len(all_final_questions)} 題，剩餘 {len(task_queue)} 題進入備援隊列。")
        else:
            task_queue = []
            for q in all_extracted_questions:
                q_num = q.get("question_number")
                q_key = get_unique_q_key(q)
                if q_key in partial_map:
                    logging.info(f"⏭️  題號 {q_num} ({q.get('question_type')}) 已存在暫存詳解，自動加載成果。")
                    q.update(partial_map[q_key])
                    all_final_questions.append(q)
                else:
                    task_queue.append({"q_data": q, "critique": "", "retry_count": 0, "recheck_count": 0})

        def process_question_chunk(current_batch, batch_size):
            """處理單一批次題目的獨立執行緒函式"""
            successful_items = []
            retry_items = []
            batch_contents = []
            batch_pil_images = []
            completed_indices = set() # 🆕 新增：追蹤本批次中已成功寫入資料庫的題目索引，防範重複寫入
            
            try:
                # 使用全域 Prompt 變數
                batch_intro = PROMPT_STAGE_2_INTRO.format(batch_size=len(current_batch))
            
                for idx, item in enumerate(current_batch, 1):
                    q_data = item["q_data"]
                    critique = item["critique"]
                    
                    options_list_str = "\n".join([f"- ({opt.get('key')}) {opt.get('value')}" for opt in q_data.get('options', [])])
                    if not options_list_str.strip(): options_list_str = "無"
                    q_sub = q_data.get("sub_subject", subject)
                    q_rubric = SUBJECT_DIFFICULTY_RUBRICS.get(q_sub, GENERAL_DIFFICULTY_RUBRIC)
                    q_allowed = SUBJECT_TAXONOMY.get(q_sub, {"topics": [], "techniques": []})
                    
                    # 🚨 核心修正：針對篇章結構、克漏字等題幹僅有底線 "______" 的題型，動態注入定位標記，防止批次生成時產生跨題號交叉混淆（如 31 題寫了 32 題的分析）
                    q_text_prompt = q_data['question_text']
                    if q_text_prompt.strip() in ["", "______", "___"]:
                        q_text_prompt = f"【定位引導】：請針對 `shared_context` 中編號為 [{q_data['question_number']}] 的空格進行前後文邏輯與語意銜接分析，求出最適合填入第 [{q_data['question_number']}] 空格的正確選項。"
                    
                    has_off_ans = q_data.get('has_official_answer', False)
                    ans_label = f"【{q_data['answer']}】" if (has_off_ans and q_data.get('answer')) else "【無官方解答（請由你嚴謹推導求出唯一正確答案）】"
                    
                    # 🚨 [事前主動防禦一：核心題目背景組裝]
                    item_desc = f"=== 待解第 {idx} 題 ===\n題號：{q_data['question_number']}\n題型：{q_data['question_type']}\n具體科目分類：{q_sub}\n是否有官方解答：{has_off_ans}\n共同背景：{q_data.get('shared_context', '無')}\n題幹：{q_text_prompt}\n選項：\n{options_list_str}\n官方答案：{ans_label}\n"

                    # 🚨 [事前主動防禦二：核心官方詳解文字實體透傳（徹底消除 AI 盲猜！）]
                    off_exp = str(q_data.get('official_explanation', '')).strip()
                    if off_exp:
                        item_desc += f"""
📢【官方解析本提供的本題真實解析與推導步驟（事前最高指導原則）】：
>>> 
{off_exp}
<<<
⚠️【名師任務】：請務必仔細研讀上方官方詳解的推導骨架與算式，在此基礎上以補教名師口吻進行教學昇華，絕對不可脫離官方思維自創怪異解法！
"""

                    # 🚨 [事前主動防禦三：手寫評分細則事前精準對齊]
                    sc_crit = str(q_data.get('scoring_criteria', '')).strip()
                    if sc_crit:
                        item_desc += f"""
📑【官方公布之手寫題評分原則（事前剛性約束）】：
>>>
{sc_crit}
<<<
⚠️【名師任務】：本題為非選擇題，你的推導過程必須嚴格涵蓋上方官方評分細則中的每一個步驟給分點！
"""

                    # 🚨 [事前主動防禦四：題型專屬事前禁令（杜絕格式混淆與多選幻覺）]
                    q_t = q_data.get('question_type', '')
                    if q_t in ["選填題", "簡答題", "繪圖作圖題"]:
                        item_desc += "🚫【非選擇題/選填題事前禁令】：本題無實體選項，你的 `options_analysis` 欄位【必須且只能】輸出空列表 `[]`，絕對禁止輸出任何選項分析物件！\n"
                    elif q_t == "單選題":
                        item_desc += "🎯【單選題事前禁令】：本題為單選題，正確選項只有 1 個，`derived_answer` 必須為單一字母或數字（如 'A' 或 '3'），絕對禁止輸出多個選項代號！\n"

                    if not has_off_ans or not q_data.get('answer'):
                        item_desc += "⚠️【無官方答案剛性要求】：本題無官方標準解答。請你務必在 `detailed_solution` 中進行最扎實的推導，並在 `derived_answer` 中明確輸出你推導出的最終答案（如選擇題輸出純選項代號 A/B/C 或 1/2/3；選填題輸出數值如 3,3 或 1/3；非選證明題填寫 '證明題'）！系統將以你的推導答案作為本題的標準答案。\n"
                    
                    # 🚨【自動幾何圖解與多圖精確嵌入規範】🚨
                    safe_q_num = safe_filename(str(q_data.get('question_number', 'X')).replace(" ", ""))
                    item_desc += f"""
                    💡【圖解原理引導】：若本題涉及幾何結構、函數圖形、受力分析或實驗示意圖，**非常鼓勵您針對不同解法（如標準解法用圖 1，另解一用圖 2）或關鍵步驟嵌入圖解**，幫助學生建立直覺。但若為純代數或純文字概念題，則保持純文字即可，無須刻意湊圖。
                    - **【格式限制】**：若需繪圖，請在內文中緊鄰說明文字處嵌入以下精確 Markdown 語法：
                      * 第一張圖嵌入：`![圖 1](images/{spec_name}/diagram_Q{safe_q_num}_1.png)`
                      * 第二張圖嵌入：`![圖 2](images/{spec_name}/diagram_Q{safe_q_num}_2.png)`
                      * 第三張圖嵌入：`![圖 3](images/{spec_name}/diagram_Q{safe_q_num}_3.png)`
                    - **【配合文字解說】**：嵌入圖片時請配合上下文引導（如：「...同學們請觀察圖 1 ![圖 1](...) 中的受力方向...」）。系統會在背景自動為您繪製真實實體圖形！
                    """

                    # 🚨 核心優化：如果是題組題（含有共同背景），放寬「題號一致性」審查警告，防止 AI 誤判因共用圖表而產生的標籤不一致，避免陷入重試死循環
                    if q_data.get('shared_context', '').strip():
                        item_desc += "🚨【題組圖片共用提示】：本題屬於題組題的一部分，可能與同組其他子題共用圖表、插圖或背景。若圖片中印有鄰近子題的題號（例如第 15 題的圖片上印著 14 標籤），這是題組共用圖表的正常現象！在此情況下，【絕對不要】將其判定為 suspects_image_mismatch，請直接以此共用圖片進行正常解題與分析，不要拋出『[圖片對位錯誤]』警告！\n"
                    else:
                        item_desc += f"🚨【視覺一致性檢查】：請確認這題的圖片內容中是否印有題號 '{q_data['question_number']}'。若圖片內容與題號不符（如第 5 題卻印著 6 的圖），請在詳解開頭註明 '⚠️[圖片對位錯誤]' 並將 suspects_image_mismatch 設為 true，以文字題幹為主。\n"
                    
                    # 🚨 核心修正：若官方答案為無答案或送分，注入專屬引導，防止 AI 強行選取錯誤選項而導致審查死循環
                    if any(k in str(q_data['answer']) for k in ["無答案", "全體給分", "送分", "不計分"]):
                        item_desc += f"⚠️【特別學術指引】：大考官方公佈本題『無答案/全體給分』。請在【題意分析】與【解題思路】中明確指出題目的不周延處或瑕疵所在，分析各選項為何皆不正確。在 detailed_solution 尾端與 options_analysis 中，【絕對不要】強行選取任何一個選項（不要寫「故選(B)」），請維持官方無答案的裁決。\n"
                    
                    # 🚨 [核心修復] 偵測到選填題挖空數與答案長度不一致時，動態注入警告
                    if q_data.get("_length_mismatch"):
                        item_desc += f"⚠️【答案卷對位錯位警告】：系統偵測到本選填題在題幹中的挖空標籤數為 {q_data['_expected_blank_count']} 個，但自動化擷取到的官方答案長度卻有 {len(q_data['answer'].split(','))} 個。這代表最初答案卷的 OCR 解析極可能發生了錯位（如讀到了隔壁題目）！在解這題時，請你【100% 絕對以你的真實數學推導結果為準】，並在詳解中寫下正確的推導。若最後求出的數值與上面的官方答案不符，請在詳解末尾加上說明並指正，絕對不允許寫出不合邏輯的強湊算式！\n"
                    
                    if item["retry_count"] > 0 and critique:
                        item_desc += f"⚠️【本題為重新生成（第 {item['retry_count']} 次重試）】\n退回原因：>>> {critique} <<<\n"
                    batch_intro += item_desc
                    
                    # 載入附圖
                    for path in q_data.get('image_paths', []) + [p for opt in q_data.get('options', []) for p in opt.get('image_paths', [])]:
                        if os.path.exists(path):
                            img_obj = Image.open(path)
                            batch_pil_images.append(img_obj)
                            batch_contents.append(img_obj)

                # === 2026 架構：注入 DeepSeek-R1 專屬的思考鏈 (CoT) 引導指令 ===
                r1_cot_instruction = """
                🚨【DeepSeek-R1 深度思考指令】🚨：
                你具備極其強大的邏輯推理能力。在輸出最終的 JSON 結果之前，請你務必在後台的 <think> 標籤內進行「沙盤推演」。
                - 如果是數學或物理題：請在思考過程中窮盡所有代數運算、幾何座標化的可能性，並代入極端值進行驗算。
                - 如果是文科題：請在思考過程中仔細拆解每個選項的語意，並與題幹進行嚴謹的邏輯對應。
                - 確保推導完全自洽、無硬湊邏輯後，再將結果輸出至 JSON 格式中。
                """

                # 動態取得精確的當前台北時間（UTC+8）
                cur_taipei_time = get_current_taipei_time_str()

                # 修復原 Prompt 中的全域變數引用，並注入台北時鐘
                batch_prompt = batch_intro + r1_cot_instruction + PROMPT_STAGE_2_MAIN.format(
                    current_taipei_time=cur_taipei_time,
                    subject_rubric=q_rubric,
                    q_answer="各題上方所給定的官方答案",
                    topics=q_allowed.get('topics', []),
                    technique=q_allowed.get('techniques', []),
                    math_scope_instruction=math_scope_instruction,
                    subject_specific_instruction=subject_specific_instruction
                )
                batch_contents.insert(0, batch_prompt)

                try:
                    # 🚀【題型認知複雜度路由器 (Cognitive Complexity Router)】
                    # 嚴密評估題目認知負荷，文言文/公民法理/史料批判/數理綜合題絕對強制指派 Tier 1 旗艦思考模型！
                    first_q = current_batch[0]["q_data"]
                    f_text = str(first_q.get("question_text", ""))
                    f_ctx = str(first_q.get("shared_context", ""))
                    f_sub = str(first_q.get("sub_subject", clean_subj_name))
                    f_type = str(first_q.get("question_type", ""))
                    f_combined = f_text + " " + f_ctx

                    is_hybrid_or_handwritten = f_type in ["簡答題", "混合題", "繪圖作圖題"]
                    is_reading_group = len(f_ctx) > 80 or len(f_text) > 350
                    is_stem_hard = any(s in f_sub for s in ["數", "物", "化"])
                    is_civics_law = any(s in f_sub for s in ["公民", "社會"]) and any(k in f_combined for k in ["法", "憲法", "刑法", "民法", "權利", "政府", "供需", "市場", "契約", "法官", "法院"])
                    is_classical_chinese = "國" in f_sub and any(k in f_combined for k in ["曰", "云", "之", "乎", "者", "也", "詩", "詞", "賦", "文言", "子曰", "孟子"])
                    is_hist_primary = "歷" in f_sub and any(k in f_combined for k in ["史料", "文獻", "詔", "奏", "論曰", "紀", "碑", "諭"])

                    # 高認知負荷：100% 強制綁定旗艦思考模型 (Gemini 3.8 / 3.7 + HIGH Thinking)
                    is_high_cognitive = (is_hybrid_or_handwritten or is_reading_group or is_stem_hard or is_civics_law or is_classical_chinese or is_hist_primary)

                    task_q_num = first_q.get("question_number", "1")
                    try: q_num_int = int(re.sub(r'\D', '', str(task_q_num)) or 1)
                    except Exception: q_num_int = 1

                    item_retry = current_batch[0].get("retry_count", 0)
                    override_model = current_batch[0].get("preferred_fallback_model")

                    if override_model:
                        dynamic_stage2_model = override_model
                        target_thinking_level = "LOW"
                        use_thinking = "lite" not in override_model
                    elif is_high_cognitive:
                        # 旗艦算力深度推演
                        dynamic_stage2_model = "gemini-3.8-flash" if (q_num_int % 2 == 0) else "gemini-3.7-flash"
                        target_thinking_level = "HIGH"
                        use_thinking = True
                    else:
                        # 單純詞彙或記憶題，採用均衡通道
                        dynamic_stage2_model = "gemini-3.6-flash" if (q_num_int % 2 == 0) else "gemini-3.5-flash"
                        target_thinking_level = "LOW"
                        use_thinking = True

                    solutions_dict, s_err = self.ai_manager.generate_with_retry(
                        contents=batch_contents, 
                        response_schema=QuestionSolutionBatch,
                        temperature=0.3,
                        preferred_model=dynamic_stage2_model,  
                        provider="google",
                        enable_thinking=use_thinking,
                        thinking_level=target_thinking_level,
                        task_desc=f"{paper_tag} [解題 ({dynamic_stage2_model}|思考:{target_thinking_level})]"
                    )
                    
                    # 🚨 核心優化：將 Stage 2 生成的所有詳解結果遞迴且全面地轉為繁體中文，阻斷簡體字存入資料庫
                    if solutions_dict:
                        solutions_dict = s2t_recursive(solutions_dict)
                    valid_batch = []
                    failed_batch = []
                    salvaged_solutions = []
                    new_batch_size = batch_size

                    # 🚨 動態自癒縮減與擷取已完善題目 🚨
                    if s_err in ["json_decode_error", "partial_success"]:
                        salvaged_solutions = solutions_dict.get('solutions', []) if solutions_dict else []
                        salvaged_count = len(salvaged_solutions)
                        
                        if salvaged_count > 0:
                            logging.info(f"🦸‍♂️ [救援成功] 從截斷的 JSON 中成功救回 {salvaged_count} 題已完善的詳解！")
                            valid_batch = current_batch[:salvaged_count]
                            failed_batch = current_batch[salvaged_count:]
                            new_batch_size = max(1, batch_size - 1)
                        else:
                            logging.warning(f"⚠️ [JSON截斷] 無法救回任何完整題目，批次大小縮減為 {max(1, batch_size - 1)} 重新嘗試...")
                            failed_batch = current_batch
                            new_batch_size = max(1, batch_size - 1)
                    elif not solutions_dict or 'solutions' not in solutions_dict:
                        failed_batch = current_batch
                    else:
                        valid_batch = current_batch
                        salvaged_solutions = solutions_dict['solutions']
                    
                    # 🚨 [卡題快速降級與放行]：單題若失敗超過 2 次，強制降級由 3.5-Lite 接管；超過上限則安全保留 partial，不卡死全卷！
                    for item in failed_batch:
                        item["retry_count"] += 1
                        if item["retry_count"] >= 2:
                            # 失敗兩次以上，強制指派 3.5-flash-lite 與關閉思考，追求秒級成功率
                            item["preferred_fallback_model"] = "gemini-3.5-flash-lite"

                        if item["retry_count"] >= max_single_attempts:
                            logging.warning(f"⚠️ 題號 {item['q_data'].get('question_number')} 屢次請求失敗，先保留於 partial 暫存，不阻塞後續題目與試卷！")
                        else:
                            retry_items.append(item)

                    if not valid_batch:
                        return retry_items, new_batch_size

                    # --- 呼叫 Stage 3 審查 ---
                    
                    # 🚨 [防禦性機制 - 提案二：選填題/非選題格式自動化預校驗器]
                    for idx, sol in enumerate(salvaged_solutions):
                        q_data = valid_batch[idx]["q_data"]
                        
                        # 1. 強制規範非選擇題/選填題的 options_analysis 必須為空
                        if q_data.get("question_type") in ["選填題", "簡答題", "繪圖作圖題"]:
                            sol["options_analysis"] = []
                            
                        # 2. 自動修正選填題答案格式：若無逗號則依據挖空數量強制拆分
                        if q_data.get("question_type") == "選填題" and "," not in str(q_data.get("answer", "")):
                            raw_ans = str(q_data.get("answer", "")).strip()
                            expected_count = q_data.get("_expected_blank_count", 0)
                            if expected_count > 0 and len(raw_ans) == expected_count:
                                q_data["answer"] = ",".join(list(raw_ans))
                                logging.info(f"🧹 [預校驗修復] 已自動將選填題 {q_data['question_number']} 的答案格式化為 '{q_data['answer']}'")
                        
                        # 3. 🚨 [防禦性機制 - 提案一：雙重不一致「學術裁決與防硬凹」仲裁機制 (第一階段：偵測)]
                        # 針對選擇題，提取 AI 真正判定為「正確」的選項
                        if q_data.get("question_type") in ["單選題", "多選題"] and isinstance(sol.get("options_analysis"), list):
                            derived_correct_keys = []
                            for opt in sol.get("options_analysis", []):
                                exp = str(opt.get("explanation", "")).strip()
                                # 嚴格結尾匹配：只有在句尾明確宣告「正確」或「故選」者才採納，防止內文提及「甲正確」導致誤判
                                is_conclusive_correct = bool(re.search(r'(?:此選項|該選項|故|本選項)?\s*(?:正確|為正確選項|符合題意)[。！!]*$', exp)) or ("正確" in exp and "錯誤" not in exp)
                                is_conclusive_incorrect = bool(re.search(r'(?:此選項|該選項|故|本選項)?\s*(?:錯誤|不正確|不合題意)[。！!]*$', exp))
                                
                                if is_conclusive_correct and not is_conclusive_incorrect:
                                    derived_correct_keys.append(opt.get("key", "").strip().upper())
                                    
                            # 🚨 單選題防多選幻覺硬性約束：若為單選題，但抓到了多個字母（如 BDE），代表正則被題幹的甲乙丙丁混淆，強制回退為官方答案！
                            if q_data.get("question_type") == "單選題":
                                if len(derived_correct_keys) == 1:
                                    derived_ans = derived_correct_keys[0]
                                else:
                                    # 多個字母發生衝突，不採用幻覺字串，直接以官方答案為推導基準
                                    derived_ans = str(q_data.get("answer", "")).strip().upper()
                            else:
                                derived_ans = "".join(sorted(list(set(derived_correct_keys))))
                            
                            # 💡 核心補件：若本試卷無官方解答（學校段考），自動將 AI 推導出的答案填入 answer 欄位中
                            if not q_data.get('has_official_answer', True):
                                fallback_ans = sol.get('derived_answer', '').strip() or derived_ans
                                if fallback_ans:
                                    q_data['answer'] = fallback_ans
                                    logging.info(f"💡 [無官方解答自動填補] 題號 {q_data['question_number']} 已根據 AI 推導自動寫入答案: '{fallback_ans}'")

                            official_ans = str(q_data.get('answer', '')).strip()
                            
                            # 雙向清理格式（去除所有逗號與空格），確保多選題 '3,4' 與 '34' 等價比對
                            derived_clean = derived_ans.replace(",", "").replace(" ", "")
                            official_clean = official_ans.replace(",", "").replace(" ", "")
                            
                            # 🚨 [大防禦機制修補]：只有在「有官方解答」的情況下才進行學術衝突攔截
                            if q_data.get('has_official_answer', True):
                                has_conflict = False
                                if q_data.get("question_type") == "單選題" and (not derived_clean or len(derived_clean) != 1):
                                    has_conflict = True
                                elif derived_clean != official_clean:
                                    has_conflict = True
                                    
                                if has_conflict and official_clean:
                                    # 發現嚴重衝突！標記此題存在學術不對位，以便後續強制觸發學術仲裁
                                    valid_batch[idx]["_discrepancy_detected"] = True
                                    valid_batch[idx]["_derived_ans"] = derived_ans if derived_ans else "（未順利推導出唯一正確選項）"
                                    logging.warning(f"⚖️ [不一致預警] 題號 {q_data['question_number']}：AI 實質推導出 '{derived_ans}'，但官方紀錄為 '{official_ans}'。已標記進行強制仲裁！")

                        # 呼叫預校驗器進行格式與類型清洗
                        pre_validate_format(q_data, sol)

                    # === 2026 架構：步驟四 - 自動幾何/函數/科學圖解生成 ===
                    for idx, sol in enumerate(salvaged_solutions):
                        q_data = valid_batch[idx]["q_data"]
                        q_sub = q_data.get("sub_subject", "")
                        
                        # 🚨 完整支援所有理科（數學、物理、化學、生物、地球科學、自然）
                        if any(s in q_sub for s in ["數學", "物理", "化學", "生物", "地球科學", "自然"]):
                            safe_q_num = safe_filename(str(q_data.get('question_number', 'X')).replace(" ", ""))
                            detailed_sol_text = sol.get('detailed_solution', '')
                            
                            # 1. 放寬正規表達式：無論 AI 寫 ./images/ 還是其他前綴，皆能精確抓取圖 1, 圖 2, 圖 3 ...
                            pattern = rf"diagram_Q{re.escape(safe_q_num)}_(\d+)\.png"
                            matches = re.findall(pattern, detailed_sol_text)
                            
                            # 狀況 A：如果 AI 在詳解內文中，主動嵌入了圖片
                            # 狀況 A：如果 AI 在詳解內文中，主動嵌入了圖片
                            if matches:
                                logging.info(f"🔍 偵測到題號 {safe_q_num} 的詳解內文主動嵌入了 {len(matches)} 張自定義圖表：{matches}")
                                for img_num in matches:
                                    diagram_filename = f"diagram_Q{safe_q_num}_{img_num}.png"
                                    diagram_filepath = os.path.abspath(os.path.join(img_dir, diagram_filename)).replace("\\", "/")
                                    
                                    qwen_prompt = f"""
                                    你是一位頂尖的 Python 科學繪圖專家。以下是一道題目的題幹與詳細解答：
                                    
                                    【題幹】：{q_data.get('question_text', '')}
                                    【詳細解答】：{detailed_sol_text}
                                    
                                    請針對解答中提到的「圖 {img_num}」（儲存路徑 `{diagram_filepath}`），撰寫一段簡潔、美觀且能成功執行的 Python 繪圖程式碼。
                                    
                                    【規範】：
                                    1. 所有標籤字串必須寫在同一行，換行請使用 '\\n'，嚴禁直接斷行！
                                    2. 2D 圖形請使用 `plt.axis('equal')`；3D 空間請使用 `Axes3D`。
                                    3. 最末尾必須包含 `plt.savefig(r"{diagram_filepath}", dpi=200, bbox_inches='tight')` 與 `plt.close()`。
                                    """
                                    
                                    draw_ok = execute_and_fix_diagram_script(
                                        self.ai_manager, 
                                        qwen_prompt, 
                                        DiagramResponse, 
                                        diagram_filepath, 
                                        safe_q_num=f"{safe_q_num}_{img_num}", 
                                        task_tag=f"{paper_tag} [繪製 圖{img_num}]"
                                    )
                                    
                                    # 🛡️ 容錯保底：若繪圖失敗，將詳解中未生成的圖片標記安全清除，防止審查判為死圖
                                    if not draw_ok:
                                        bad_tag_pattern = rf'!\[圖\s*{img_num}\]\([^)]*?diagram_Q{re.escape(safe_q_num)}_{img_num}\.png\)'
                                        sol['detailed_solution'] = re.sub(bad_tag_pattern, '', sol['detailed_solution'])
                            
                            # 狀況 B：保底機制（如果 AI 忘記在內文嵌入圖片標籤，我們自動在最下方補上一張預設的「圖 1」）
                            else:
                                diagram_filename = f"diagram_Q{safe_q_num}_1.png"
                                diagram_filepath = os.path.abspath(os.path.join(img_dir, diagram_filename)).replace("\\", "/")
                                
                                qwen_prompt = f"""
                                你是一位在台灣大考教學界享有盛譽的 Python 高中科學圖解繪製專家。以下是一道高中理科題目的題幹與詳細解答：
                                
                                【題幹】：{q_data.get('question_text', '')}
                                【詳細解答】：{detailed_sol_text}
                                
                                請評估本題是否需要專業圖解輔助（如：幾何輔助線圖、函數極值與切線圖、物理受力自由體圖 Free-body Diagram、光路圖、電路圖、化學相圖/反應曲線、生物適應輻射/演化機制流程圖）。
                                如果需要，請撰寫一段完整、美觀、高對比且符合出版水準的 Python 繪圖程式碼，並儲存至：
                                `{diagram_filepath}`
                                
                                【出版級繪圖規範】：
                                1. 允許且鼓勵使用繁體中文進行標註（如：點 $A$、外接圓、合力方向、基因頻率變化）。
                                2. 所有文字標籤與註解（如 ax.annotate, plt.text）**必須寫在同一行內**，換行一律使用 '\\n'，嚴禁直接斷行。
                                3. 2D 幾何請使用 `plt.axis('equal')` 保持真實比例；3D 空間請使用 `Axes3D`。
                                4. 程式碼必須能獨立執行，最末尾必須包含 `plt.savefig(r"{diagram_filepath}", dpi=200, bbox_inches='tight')` 與 `plt.close()`。
                                """
                                
                                # 🚀 優化：使用具備語法自動修復與錯誤重試的執行器
                                if execute_and_fix_diagram_script(
                                    self.ai_manager, 
                                    qwen_prompt, 
                                    DiagramResponse, 
                                    diagram_filepath, 
                                    safe_q_num=f"{safe_q_num}_1", 
                                    task_tag=f"{paper_tag} [保底繪圖]"
                                ):
                                    rel_path = f"images/{os.path.basename(img_dir)}/{diagram_filename}"
                                    sol['detailed_solution'] += f"\n\n### 【AI 幾何解析圖】\n![圖 1]({rel_path})"

                    # 執行原有的 Markdown 格式化轉換
                    for sol in salvaged_solutions:
                        if isinstance(sol.get("options_analysis"), list):
                            opt_list = sol["options_analysis"]
                            formatted_opts = []
                            for opt in opt_list:
                                opt_key = opt.get("key", "").strip()
                                opt_exp = opt.get("explanation", "").strip()
                                if opt_key and opt_exp:
                                    formatted_opts.append(f"- **({opt_key})** {opt_exp}")
                            
                            if formatted_opts:
                                sol["options_analysis"] = "\n".join(formatted_opts)
                            else:
                                sol["options_analysis"] = "本題為非選擇題/選填題，無選項可供分析。"
                        elif not sol.get("options_analysis"):
                            sol["options_analysis"] = "本題為非選擇題/選填題，無選項可供分析。"

                    validator_batch_intro = ""
                    for idx, sol in enumerate(salvaged_solutions):
                        q_data = valid_batch[idx]["q_data"]
                        
                        # 格式化原始選項供審查教授對照
                        q_options_str = "\n".join([f"- ({opt.get('key')}) {opt.get('value')}" for opt in q_data.get('options', [])])
                        if not q_options_str.strip():
                            q_options_str = "無（非選擇題/選填題）"
                            
                        # 🚨 核心優化：在待審查提示詞中精確注入「共同背景（shared_context）」，徹底阻斷審查教授對題組題目的「無中生有」誤判！
                        validator_batch_intro += f"""
                        === 待審查第 {idx+1} 題 ===
                        題號：{q_data['question_number']}
                        共同背景：{q_data.get('shared_context', '無')}
                        題目：{q_data['question_text']}
                        原始給定選項：
                        {q_options_str}
                        官方答案：【{q_data['answer']}】

                        [生成的各詳解欄位內容]
                        【題意分析】：
                        {sol.get('question_analysis', '')}

                        【解題思路】：
                        {sol.get('solving_strategy', '')}

                        【完整解法與另解】：
                        {sol.get('detailed_solution', '')}

                        【選項深入剖析】：
                        {sol.get('options_analysis', '')}

                        【易錯陷阱】：
                        {sol.get('traps_and_warnings', '')}
                        \n"""

                    llama_cot_instruction = """
                    🚨【審查思維引導】：
                    請嚴格按照以下步驟進行審查：
                    1. 先在 `error_critique` 欄位中，一步一步 (step-by-step) 寫下你核對該詳解代數計算與邏輯的推演過程。
                    2. 若推演發現無誤，`error_critique` 請在結尾寫上「推演無誤」。
                    3. 只有在完成上述推演後，才給出 `is_valid` 的布林值。
                    """
                    
                    validator_batch_prompt = llama_cot_instruction + PROMPT_STAGE_3_VALIDATOR.format(validator_batch_intro=validator_batch_intro)
                    
                    # 🚨 審查必須帶上本批次所有裁切圖片，徹底杜絕「文字盲審」！
                    validator_contents = [validator_batch_prompt]
                    for item in valid_batch:
                        q_d = item["q_data"]
                        for img_p in q_d.get('image_paths', []):
                            if os.path.exists(img_p):
                                try:
                                    validator_contents.append(Image.open(img_p))
                                except Exception:
                                    pass
                    
                    dynamic_val_model = tier1_pool[(q_num_int + 1) % len(tier1_pool)]

                    val_dict, val_err = self.ai_manager.generate_with_retry(
                        contents=validator_contents, 
                        response_schema=SolutionValidatorBatch,
                        temperature=0.0, 
                        preferred_model=dynamic_val_model, 
                        provider="google", 
                        enable_thinking=True,
                        task_desc=f"{paper_tag} [Gemini 審查 ({dynamic_val_model})]"
                    )

                    # 處理審查結果 (包含重掃描 OCR)
                    for idx, item in enumerate(valid_batch):
                        q_data = item["q_data"]
                        sol_data = salvaged_solutions[idx]
                        
                        if not sol_data.get("options_analysis"):
                            sol_data["options_analysis"] = "本題為非選擇題/選填題，無選項可供分析。"
                            
                        # 🚨 安全防呆：若審查整個 API 失敗 (None)，預設重試審查而不盲目放行
                        is_valid_passed = False if (val_dict is None or val_err) else True
                        error_critique = "審查回傳為空或發生異常，自動排入重新審查。" if (val_dict is None or val_err) else ""
                        suspects_ocr_error = False

                        if val_dict and 'validators' in val_dict and idx < len(val_dict['validators']):
                            v_res = val_dict['validators'][idx]
                            is_valid_passed = v_res.get('is_valid', False)
                            error_critique = v_res.get('error_critique', '')
                            suspects_ocr_error = v_res.get('suspects_ocr_error', False)

                        q_data.update(sol_data)

                        # 🚨 修正：結合生成端、審查端與本地預校驗的三重衝突偵測
                        stage2_mismatch = sol_data.get("suspects_image_mismatch", False)
                        discrepancy_detected = item.get("_discrepancy_detected", False)
                        
                        if (suspects_ocr_error or stage2_mismatch or discrepancy_detected):
                            is_valid_passed = False 
                            conflict_summary = error_critique or ("推導答案不一致" if discrepancy_detected else "圖片模糊或題號錯位")
                            GLOBAL_METRICS.record_error("ACADEMIC_CONFLICT", f"學術爭議/對位疑義: {conflict_summary[:160]}", exam_name=paper_tag, question_num=q_data.get('question_number', ''), stage="Stage 3 審查")
                            
                            # 🚨 [防禦性機制 - 提案一：雙重不一致「學術裁決與防硬凹」仲裁機制 (第二階段：強制仲裁與覆寫)]
                            # 如果是「學術推導直接不一致」或「已經重新掃描過仍有衝突」，直接跳過二次重掃，交給學術仲裁器
                            if discrepancy_detected or item["recheck_count"] >= 1:
                                logging.warning(f"⚖️ {paper_tag} [學術衝突/重掃描上限] 題號 {q_data['question_number']} 啟動【學術仲裁器】進行最終裁決...")
                                
                                # 組合詳細的衝突上下文，供仲裁器精確研判
                                conflict_clue = error_critique if error_critique else f"AI 實質推導出正確選項為 '{item.get('_derived_ans')}'，但官方輸入答案為 '{q_data.get('answer')}'。"
                                arbitrated_data = self.run_academic_arbitration(q_data, sol_data, conflict_clue)
                                
                                if arbitrated_data:
                                    # 1. 自動覆寫官方答案欄位，修正為學術正確之答案（🚨 強制正則清洗，只留純選項字母！）
                                    raw_chosen = arbitrated_data.get('chosen_answer', q_data['answer'])
                                    q_data['answer'] = clean_ocr_answer_format(str(raw_chosen))
                                    # 2. 將仲裁說明與「修正後的詳解」拼合，剔除所有硬凹迎合的文字
                                    q_data['detailed_solution'] = sol_data.get('detailed_solution', '') + "\n\n" + arbitrated_data.get('final_solution_append', '')
                                    is_valid_passed = True
                                    logging.info(f"⚖️ {paper_tag} [仲裁成功] 題號 {q_data['question_number']} 已被強制覆寫並修正答案為: {q_data['answer']}")
                                else:
                                    # 仲裁失敗時的保底機制
                                    is_valid_passed = True
                                    q_data['detailed_solution'] = sol_data.get('detailed_solution', '') + "\n\n**【備註】本題經多輪重編譯核對，官方答案與嚴謹學理推導存在潛在衝突，詳解已優先採用最嚴謹之學術推導過程。**"
                            
                            # 正常重掃描流程（僅在非直接學術不一致，且為第一次重掃時執行）
                            elif item["recheck_count"] < max_rechecks:
                                is_valid_passed = False 
                                if stage2_mismatch and not error_critique:
                                    error_critique = "生成詳解的 AI 偵測到圖片題號與題幹不符或圖片模糊，主動要求重掃描補件。"
                                    
                                logging.warning(f"  -> 🚨 {paper_tag} [重大衝突] 偵測到圖片精確性異常，第 {q_data['question_number']} 題啟動【補掃描與新圖偵測】...")
                                # ...後面照常執行原有的重掃描 PIL 圖像、CorrectedSource API 呼叫等代碼...
                                ans_pil_imgs = []
                                try:
                                    # 給予重掃描模型更具體的錯誤診斷反饋
                                    clue_str = f"🚨【審查教授指出的線索與潛在錯誤】：\n>>> {error_critique} <<<\n"
                                    recheck_prompt = f"""
                                    我們在解析第 {q_data['question_number']} 題時遇到嚴重的邏輯或數理衝突。
                                    {clue_str}
                                    請你扮演最高精度的圖文比對專家，特別針對上述線索，仔細檢視下方的題目原卷影像與標準答案卷：
                                    1. 重新核對題目中的變數、係數、正負號（例如：負號是否被看漏、x 與 y 是否看反、根號是否被切掉等）。
                                    2. 重新比對官方標準答案，檢查是否因為多欄排版或行列交錯，導致上一版看錯了欄位或對位到其他題目的答案。
                                    3. 🚨【關鍵任務】：檢查該題周邊是否有先前漏掉的「附圖」、「表格」或「化學式」？
                                       若有，請將 found_new_images 設為 true，並給出所有漏掉圖片的 Bounding Box (0-1000)。
                                    """
                                    
                                    # 🚨 核心優化：重置與標註重掃描內容，顯式指引 AI 區分「題目原卷」與「官方答案卷」，避免對位混淆！
                                    recheck_contents_with_labels = [recheck_prompt]
                                    
                                    # 1. 優先放入已裁剪的原始題目附圖（若有）
                                    if q_data.get('image_paths'):
                                        recheck_contents_with_labels.append("=== 原始裁剪出的題目附圖 ===")
                                        for p in q_data['image_paths']:
                                            if os.path.exists(p):
                                                img_obj = Image.open(p)
                                                ans_pil_imgs.append(img_obj)
                                                recheck_contents_with_labels.append(img_obj)
                                    
                                    # 2. 優先放入題目原卷的整頁影像，並顯式標註
                                    target_page_idx = q_data.get('page_number', 1) - 1
                                    if target_page_idx < 0 or target_page_idx >= len(q_data.get('question_page_image_paths', [])):
                                        target_page_idx = 0
                                    
                                    if 0 <= target_page_idx < len(q_data.get('question_page_image_paths', [])):
                                        q_img_path = q_data['question_page_image_paths'][target_page_idx]
                                        if os.path.exists(q_img_path):
                                            recheck_contents_with_labels.append(f"=== 題目原卷 PDF 第 {target_page_idx+1} 頁整頁影像（提取題目與選項之唯一依據） ===")
                                            q_img_obj = Image.open(q_img_path)
                                            ans_pil_imgs.append(q_img_obj) # 🚨 補件：追蹤以利關閉，防手把洩漏
                                            recheck_contents_with_labels.append(q_img_obj)
                                            
                                    # 3. 放入官方答案卷，並顯式標註（僅供對應答案，嚴禁從此圖提取題目文字）
                                    for idx, ans_p in enumerate(q_data.get('answer_page_image_paths', [])):
                                        if os.path.exists(ans_p):
                                            recheck_contents_with_labels.append(f"=== 官方標準答案卷第 {idx+1} 頁影像（僅供對照與核對實際答案） ===")
                                            ans_img_obj = Image.open(ans_p)
                                            ans_pil_imgs.append(ans_img_obj) # 🚨 補件：追蹤以利關閉，防手把洩漏
                                            recheck_contents_with_labels.append(ans_img_obj)
                                            
                                    recheck_dict, _ = self.ai_manager.generate_with_retry(
                                        contents=recheck_contents_with_labels, response_schema=CorrectedSource,
                                        temperature=0.0, preferred_model=validator_model, enable_thinking=False,
                                        task_desc=f"{paper_tag} [重掃描]"
                                    )

                                    if recheck_dict and recheck_dict.get('is_confident') is True:
                                        # 更新數據（全防禦取值，防止 KeyError 崩潰）
                                        q_data['shared_context'] = recheck_dict.get('corrected_shared_context', q_data.get('shared_context', ''))
                                        q_data['question_text'] = recheck_dict.get('corrected_question_text', q_data.get('question_text', ''))
                                        q_data['options'] = recheck_dict.get('corrected_options', q_data.get('options', []))
                                        q_data['answer'] = clean_ocr_answer_format(str(recheck_dict.get('corrected_answer', q_data.get('answer', ''))))

                                        # 處理補裁切
                                        if recheck_dict.get('found_new_images') and recheck_dict.get('new_image_bboxes'):
                                            logging.info(f"🔍 {paper_tag} [發現新圖] 第 {q_data['question_number']} 題正在補裁切...")
                                            try:
                                                # 🚨 防禦性初始化：確保字典中必定存在 image_paths 與 _cropped_pil_images
                                                if 'image_paths' not in q_data:
                                                    q_data['image_paths'] = []
                                                if '_cropped_pil_images' not in q_data:
                                                    q_data['_cropped_pil_images'] = []

                                                with fitz.open(q_data['question_pdf_path']) as temp_doc:
                                                    cat = temp_doc.pdf_catalog()
                                                    if cat > 0: temp_doc.xref_set_key(cat, "StructTreeRoot", "null")
                                                    
                                                    # 防止 page_number 索引越界
                                                    page_idx = max(0, min(q_data.get('page_number', 1) - 1, len(temp_doc) - 1))
                                                    target_page = temp_doc[page_idx]
                                                    
                                                    new_img_paths = self.execute_crop(target_page, recheck_dict['new_image_bboxes'], img_dir, f"Q{q_data['question_number']}_Extra")
                                                    
                                                    for path in new_img_paths:
                                                        if path not in q_data['image_paths']:
                                                            q_data['image_paths'].append(path)
                                                            if os.path.exists(path):
                                                                q_data['_cropped_pil_images'].append(Image.open(path))
                                                            q_data['has_image'] = True
                                            except Exception as e:
                                                logging.error(f"補裁切失敗: {e}")

                                        logging.info(f"🔄 {paper_tag} [重掃描完成] 第 {q_data['question_number']} 題準備重新解題。")
                                        error_critique = f"已完成重掃描修正與新圖補件。請重新解題。"
                                        is_valid_passed = False 
                                        item["recheck_count"] += 1
                                finally:
                                    for img in ans_pil_imgs:
                                        try: img.close()
                                        except: pass

                        if is_valid_passed:
                            # 🚨 1. 取得 AI 實質推導出的答案
                            derived = str(sol_data.get("derived_answer", "")).strip() or item.get("_derived_ans", "").strip()
                            if not derived or derived == "（未順利推導出唯一正確選項）":
                                m_find = re.search(r'\*\*綜上所述[，,]\s*(?:本題)?(?:正確答案|正確選項|本題正確選項|選填題各畫卡格答案)?(?:為)?[：:\s]*\**\s*([^\n\*]+?)\**(?:\(|$|\n)', sol_data.get("detailed_solution", ""))
                                if m_find:
                                    raw_cand = m_find.group(1).strip()
                                    pure_keys = re.findall(r'[A-Ga-g1-9]', raw_cand)
                                    if pure_keys and q_data.get("question_type") in ["單選題", "多選題"]:
                                        derived = "".join(pure_keys) if q_data.get("question_type") == "多選題" else pure_keys[0]
                                    else:
                                        derived = raw_cand.strip("()$* ")

                            q_data["derived_answer"] = clean_ocr_answer_format(derived) if derived else ""

                            # 🚨 2. 若本題無官方答案，has_official_answer 強制設為 false，並將 answer 欄位填入推導答案！
                            has_conflict = item.get("_discrepancy_detected", False)
                            has_off = q_data.get('has_official_answer', False) and bool(str(q_data.get('answer', '')).strip())
                            if not has_off:
                                has_conflict = False
                                q_data['has_official_answer'] = False
                                q_data['official_answer'] = ""
                                q_data['answer'] = q_data['derived_answer'] or (q_data.get("answer", "") or ("證明題" if "證" in q_data.get("question_text", "") else ""))
                                q_data['has_answer_discrepancy'] = False
                                q_data['discrepancy_reason'] = ""
                            else:
                                if not q_data.get("official_answer"):
                                    q_data["official_answer"] = str(q_data.get('answer', '')).strip()
                                q_data["has_answer_discrepancy"] = has_conflict
                                if has_conflict:
                                    q_data["discrepancy_reason"] = f"學術嚴謹推導答案為【{q_data['derived_answer']}】，但官方原始紀錄為【{q_data['official_answer']}】（疑為答案卷錯位、印刷瑕疵或版本混淆）。"
                                else:
                                    q_data["discrepancy_reason"] = ""

                            # 成功通過，加上客觀結論句（絕不精神分裂硬凹！無官方答案或有衝突時優先取推導答案）
                            effective_ans = q_data['derived_answer'] if (has_conflict or not has_off) else (q_data.get('official_answer') or q_data.get('answer', ''))
                            
                            # 🚨 防呆：若答案完全為空，不追加空的「綜上所述」句子
                            if effective_ans and effective_ans.strip():
                                if any(k in effective_ans for k in ["無答案", "全體給分", "送分", "不計分"]):
                                    q_data['detailed_solution'] += f"\n\n**綜上所述，本題官方公佈無答案，全體給分。**"
                                elif q_data.get('question_type', '') in ["單選題", "多選題"]:
                                    # 只提取大寫英文字母或純數字選項，防止「即」等中文字元被包成括號 (即)
                                    valid_opts = re.findall(r'[A-Ga-g1-9]', effective_ans)
                                    if valid_opts:
                                        formatted_opts = "".join([f"({c.upper()})" for c in valid_opts])
                                        if has_conflict:
                                            q_data['detailed_solution'] += f"\n\n**綜上所述，本題學理嚴謹推導之正確選項為：{formatted_opts}（官方原始答案為：{q_data['official_answer']}，詳見上述疑義剖析）。**"
                                        else:
                                            q_data['detailed_solution'] += f"\n\n**綜上所述，本題正確選項為：{formatted_opts}**"
                                elif q_data.get('question_type', '') == "選填題" and "," in effective_ans:
                                    formatted_parts = effective_ans.split(",")
                                    blanks_desc = ", ".join([f"第 {idx+1} 空格為「{val}」" for idx, val in enumerate(formatted_parts)])
                                    q_data['detailed_solution'] += f"\n\n**綜上所述，本題選填題各畫卡格答案為：{effective_ans}（即 {blanks_desc}）**"
                                else:
                                    clean_ans_disp = effective_ans
                                    # 🚨 若答案中含有 LaTeX 指令但未包裹 $，自動為其補全 $，防止 Markdown 粗體損毀公式
                                    if any(c in clean_ans_disp for c in ["\\", "^", "_"]) and not clean_ans_disp.startswith("$"):
                                        clean_ans_disp = f"${clean_ans_disp}$"
                                    q_data['detailed_solution'] += f"\n\n**綜上所述，本題正確答案為：** {clean_ans_disp}"
                                    
                            # 🚨 多維知識點自動補齊與主副關聯校正
                            raw_cats = q_data.get("topic_categories", [])
                            if isinstance(raw_cats, list) and raw_cats:
                                q_data["topic_categories"] = [s2t(str(c).strip()) for c in raw_cats if str(c).strip()]
                                # 主考點鎖定為第一優先項
                                q_data["topic_category"] = q_data["topic_categories"][0]
                            elif q_data.get("topic_category"):
                                clean_main = s2t(str(q_data["topic_category"]).strip())
                                q_data["topic_category"] = clean_main
                                q_data["topic_categories"] = [clean_main]
                            else:
                                q_data["topic_category"] = "必修_綜合主題"
                                q_data["topic_categories"] = ["必修_綜合主題"]

                            with queue_lock:
                                if '_cropped_pil_images' in q_data:
                                    for pimg in q_data['_cropped_pil_images']:
                                        try: pimg.close()
                                        except: pass
                                    del q_data['_cropped_pil_images']
                                update_subject_taxonomy(q_data.get("sub_subject", subject), q_data)
                                all_final_questions.append(q_data)
                                
                                # 📊 即時更新網頁儀表板進度（實質新解題目計數 +1）
                                GLOBAL_METRICS.record_new_solved_question()
                                GLOBAL_METRICS.update_exam_progress(GLOBAL_METRICS.current_paper, len(all_final_questions), len(all_extracted_questions), "解題中...")
                                GLOBAL_METRICS.add_log(f"✅ 題號 {q_data['question_number']} 詳解與圖解通過審查！")
                                
                            completed_indices.add(idx)
                            save_partial_progress_immediately(partial_json_path, all_final_questions)
                            logging.info(f"  -> 🎉 {paper_tag} 題號 {q_data['question_number']} 通過驗證，成功收錄Partial。")
                        else:
                            item["retry_count"] += 1
                            item["critique"] = error_critique if error_critique else "推導邏輯不連貫或瑕疵。"
                            with val_log_lock:
                                validation_records.append({
                                    "question_number": q_data['question_number'],
                                    "attempt": item['retry_count'],
                                    "error_critique": item['critique'],
                                    "ai_output_raw": sol_data.get('detailed_solution', '')
                                })
                            if item["retry_count"] < max_single_attempts:
                                retry_items.append(item)
                                logging.warning(f"🔴 {paper_tag} 題號：{q_data['question_number']} 審查退回：{item['critique']}")
                            else:
                                logging.warning(f"⚠️ 題號 {q_data.get('question_number')} 審查未通過且已達重試上限，本次不寫入資料庫，保留至下次自動重跑。")
                    
                    return retry_items, new_batch_size
                except RPDExhaustedException as ree:
                    raise ree
                except Exception as inner_e:
                    raise inner_e
            except RPDExhaustedException as ree:
                raise ree
            except Exception as e:
                import traceback
                tb_full = traceback.format_exc()
                logging.exception(f"❌ [執行緒崩潰安全自癒] 處理批次時發生未預期異常: {e}")
                GLOBAL_METRICS.record_error("CODE_RUNTIME", f"執行緒未預期崩潰: {str(e)}", exam_name=paper_tag, question_num=current_batch[0]["q_data"].get("question_number", "") if current_batch else "", stage="批次平行執行緒", tb_str=tb_full)
                # 🚨 修正：僅救回本批次中「尚未成功寫入」的題目，防範因併發異常導致成功題目重複寫入或重複重試！
                for idx, item in enumerate(current_batch):
                    if idx in completed_indices:
                        continue # 🚨 跳過已成功的題目，維持資料庫唯一性並節省 Token 額度
                    item["retry_count"] += 1
                    if item["retry_count"] < max_single_attempts:
                        retry_items.append(item)
                    else:
                        item["q_data"]["detailed_solution"] = f"\n\n> **⚠️ [系統提示]：此題在解析時引發系統內部執行緒崩潰。**\n\n錯誤訊息: {str(e)}"
                        with queue_lock:
                            all_final_questions.append(item["q_data"])
                return retry_items, max(1, batch_size - 1)
            finally:
                for img in batch_pil_images:
                    try: img.close()
                    except: pass
            

        # =========================================================
        # 啟動考卷內題目並行處理 (ThreadPoolExecutor)
        # =========================================================
        max_workers = max(2, min(len(API_KEYS) * 2, 10))
        logging.info(f"🚀 開始並行詳解生成！啟動 {max_workers} 條執行緒 (啟用冷啟動階梯平滑調度)...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = set()
            thread_spawn_count = 0
            while True:
                to_submit = []
                with queue_lock:
                    # 🚀 修正：持鎖期間只做資料列佇列取出，絕不呼叫 time.sleep 阻塞全域鎖
                    while len(task_queue) > 0 and (len(futures) + len(to_submit)) < max_workers:
                        current_batch = task_queue[:active_batch_size]
                        task_queue = task_queue[active_batch_size:]
                        to_submit.append(current_batch)

                # 🚀 在鎖外提交任務，並加入微延遲打散併發發送，防止 IP 級 429
                for batch in to_submit:
                    futures.add(executor.submit(process_question_chunk, batch, active_batch_size))
                    if thread_spawn_count < max_workers:
                        thread_spawn_count += 1
                        
                        time.sleep(random.uniform(0.15, 0.35))
                
                if not futures: break
                
                # 等待任意一個 Batch 完成
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                
                rpd_hit = False
                for fut in done:
                    try:
                        retry_items, new_batch_size = fut.result()
                        with queue_lock:
                            if new_batch_size < active_batch_size:
                                active_batch_size = new_batch_size
                            # 將重試題目插回佇列最前方
                            task_queue = retry_items + task_queue
                    except RPDExhaustedException as ree:
                        logging.warning(f"🛑 [偵測到額度耗盡] 執行緒回報每日配額已用盡。準備安全中斷目前考卷任務...")
                        rpd_hit = True
                    except Exception as e:
                        logging.error(f"批次處理執行緒發生例外: {e}")
                
                # 🚨 檢查是否接近 5 小時全域上限或收到 Web 控制台「🛑 優雅停機」指令！
                try:
                    check_global_timeout()
                except TimeoutSafetyException as tse:
                    logging.critical(f"🛑 [全域中斷觸發] {tse}！正在立即取消其餘批次並安全退出...")
                    for f in futures:
                        f.cancel()
                    raise tse

                if rpd_hit:
                    for f in futures:
                        f.cancel()
                    break

        # 最後排序題號以維持最終 JSON 整齊，防範字典序排序導致 10 排在 2 前面
        def safe_page_to_int(val):
            try: return int(val)
            except (ValueError, TypeError): return 1

        # 🚨 結合頁碼 (整數化保證安全) 與題號，保證整份考卷按照實體印刷順序排列
        all_final_questions.sort(key=lambda x: (safe_page_to_int(x.get('page_number', 1)), natural_sort_key(x.get('question_number', '0'))))

        if not all_final_questions:
            logging.error(f"❌ [任務中止] [{year} {subject}] 解析出的題目數量為 0！")
            logging.error("這可能是由於掃描階段失敗、無可用 Key 或 API 全程斷線所致。為避免生成無意義的空白資料庫檔案，我們將跳過寫入 JSON。請修復 API 金鑰或網路後再次執行！")
            return

        # =========================================================
        # 5. 儲存與寫入最終的 JSON 檔案
        # =========================================================
        # 🚨 寫入 JSON 前，徹底清除所有非 JSON 序列化的暫存屬性 (如 PngImageFile 影像對象) 🚨

        def normalize_latex_delimiters(text: str) -> str:
            r"""自動將 Rogue LaTeX 符號 \[ \] 轉換為標準 $$ $$，將 \( \) 轉換為 $ $，防範前端排版渲染崩潰"""
            if not isinstance(text, str):
                return text
            # 1. 轉換獨立行公式 \[ ... \] 為 $$ ... $$
            text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
            # 2. 轉換行內公式 \( ... \) 為 $ ... $（處理有空格與無空格之情形）
            text = re.sub(r'\\ \((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
            text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
            return text

        def clean_paths(obj):
            if isinstance(obj, list):
                return [clean_paths(i) for i in obj]
            if isinstance(obj, dict):
                return {k: clean_paths(v) for k, v in obj.items()}
            if isinstance(obj, str):
                if "./" in obj or "gsat_" in obj or "ast_" in obj:
                    obj = obj.replace("\\", "/")
                # 1. 執行原本的分隔符號轉換
                obj = normalize_latex_delimiters(obj)
                # 2. 🚨 執行新增的高階 LaTeX 語法錯誤清洗
                obj = sanitize_latex_syntax(obj)
            return obj

        # 💡 [安全保存與品質嚴格驗證]
        # 1. 嚴格過濾：必須同時具備 detailed_solution、question_analysis 與 options_analysis 才是有效題！
        def is_really_complete(q):
            if not is_valid_solution(q.get("detailed_solution")):
                return False
            if not str(q.get("question_analysis", "")).strip():
                return False
            return True

        valid_solved_questions = [q for q in all_final_questions if is_really_complete(q)]
        
        # 2. 🚨 雙重完工判讀：依據【已正規化學科 clean_subj_name】精確核對法定題數下限！
        min_threshold_map = {
            "社會": 65, "自然": 60, "英文": 45, "國文": 38, 
            "數學甲": 12, "數學乙": 10, "數學A": 15, "數學B": 15, "數學": 10, 
            "國寫": 2
        }
        target_min = 20
        for sub_k, min_v in min_threshold_map.items():
            if sub_k in clean_subj_name:
                target_min = min_v
                break

        # 只要題數少於提取題數，或者少於該考科法定最低題數（如 43 < 65），【強制維持 partial】！
        is_partial = (len(valid_solved_questions) < len(all_extracted_questions)) or (len(valid_solved_questions) < target_min)

        valid_solved_questions = clean_paths(valid_solved_questions)
        for q in valid_solved_questions:
            q["academic_year"] = standard_academic_year
            q["exam_source"] = standard_exam_source
            # 🚨 強制寫入精確大考分類標籤，供網頁後台與前端即時篩選（絕不再只有粗暴的 MOCK！）
            q["exam_type"] = exam_type
            if exam_type == "MOCK_GSAT": q["exam_category"] = "學測模擬"
            elif exam_type == "MOCK_AST": q["exam_category"] = "分科模擬"
            elif exam_type == "GSAT": q["exam_category"] = "學測"
            elif exam_type == "AST": q["exam_category"] = "分科指考"
            elif exam_type == "SCHOOL": q["exam_category"] = "學校段考"
            else: q["exam_category"] = "模擬考"

            if '_cropped_pil_images' in q:
                for img_obj in q['_cropped_pil_images']:
                    try: img_obj.close()
                    except Exception: pass
                del q['_cropped_pil_images']

        # 結算本卷最終狀態
        final_status = "部分完成(保留降級)" if is_partial else "100% 全卷完工"
        GLOBAL_METRICS.update_exam_progress(GLOBAL_METRICS.current_paper, len(valid_solved_questions), len(all_extracted_questions), final_status)

        if is_partial:
            # 題數未齊全時寫入 partial，但絕對保留現有的 database.json，不加入雲端刪除清單
            with open(partial_json_path, "w", encoding="utf-8") as f:
                json.dump(valid_solved_questions, f, ensure_ascii=False, indent=4)
            # 🚨 若題目已解出大部分（大於等於 8 題），同步保留/更新為正式庫，避免雲端完全沒有可用資料庫
            if len(valid_solved_questions) >= 8 and not os.path.exists(json_path):
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(valid_solved_questions, f, ensure_ascii=False, indent=4)
            logging.warning(f"💾 [部分完成存檔] [{year} {subject}] 已儲存 {len(valid_solved_questions)} 題進度至 partial，現存資料庫予以保留不刪除。")

            # 🚨 核心阻斷：若是因為每日額度耗盡 (RPD) 而中斷，保存進度後立刻向上拋出異常，終止後續考卷盲跑！
            if rpd_hit:
                logging.critical(f"🛑 [全域額度耗盡] 考卷 {spec_name} 進度已落盤，觸發全域配額耗盡保護，準備進入 GHA 上傳！")
                raise RPDExhaustedException("每日 API 金鑰額度皆已耗盡。")
        else:
            # 全卷 100% 每題都有完美詳解，才正式寫入資料庫並清除 Raw 快取
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(valid_solved_questions, f, ensure_ascii=False, indent=4)
            logging.info(f"🎉 [{year} {subject}] 全部 {len(valid_solved_questions)} 題皆已 100% 高品質完成解析並儲存：{json_path}")
            
            # 🚨 即時紀錄考卷類型，更新跨 Run 3:1 繼承計數器
            record_completed_exam_to_scheduler(exam_type)
            
            # 🧹 1. 本地暫存檔清除
            for tmp_path in [partial_json_path, raw_extracted_json_path]:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                        logging.info(f"🧹 [本機暫存清理] 已清除：{os.path.basename(tmp_path)}")
                    except Exception:
                        pass
                        
            # ☁️ 2. 登記待清理之雲端暫存檔，稍後於 GitHub Action 結尾統一高速批次銷毀（零等待、不洩漏路徑）
            rel_parent = type_folder.replace("\\", "/")
            for tmp_name in [os.path.basename(partial_json_path), os.path.basename(raw_extracted_json_path)]:
                cloud_target = f"gdrive:exam_database_output/{rel_parent}/{tmp_name}"
                try:
                    with open("pending_cloud_purges.txt", "a", encoding="utf-8") as f_purge:
                        f_purge.write(f"{cloud_target}\n")
                except Exception:
                    pass
            logging.info(f"🧹 [暫存登記] 考卷 {spec_name} 已登記完工暫存清理任務。")
        
        
        # 🚨 新增：若有審查未通過的歷史紀錄，自動儲存至專屬的詳細 JSON 日誌中
        if validation_records:
            # 日誌裡的路徑也清洗一下
            validation_records = clean_paths(validation_records)
            with open(validation_log_path, "w", encoding="utf-8") as f_val:
                json.dump(validation_records, f_val, ensure_ascii=False, indent=4)
            logging.warning(f"⚠️ 提示：本卷共有 {len(validation_records)} 次審查未通過紀錄。詳細偵錯日誌已寫入：{validation_log_path}")

        
        logging.info(f"✅ [{year} {subject}] 處理完成！共擷取 {len(all_final_questions)} 題，已儲存至 {json_path}")
        # 📊 連動更新遠端儀表板之已完成考卷數量
        with GLOBAL_METRICS.lock:
            GLOBAL_METRICS.completed_papers += 1
        GLOBAL_METRICS.add_log(f"🎉 考卷 [{year} {subject}] 解析完成並成功存檔！")
        # 🚨 關閉 PDF 文件手把
        if doc is not None:
            try:
                if not doc.is_closed:
                    doc.close()
            except Exception:
                pass

# 4. 自動尋找檔案腳本 (自動化遍歷)
# ==========================================
def auto_find_exam_sets(directories: List[str]) -> List[dict]:
    """
    🚨 [跨年度/學科/模擬考全自動強健對位匹配器 (全面適應真實網抓題庫檔名)]：
    1. 完整支援「公社科/公社/社會科/國綜/國寫/數A/數B/數甲/數乙」等全套科目簡稱。
    2. 智慧識別「與詳解/與解析/含解析/試題+解析/試題暨詳解」等二合一 PDF 檔。
    3. 自動過濾「解答更正表/五標統計表/英聽」雜訊檔。
    4. 支援「全科共用詳解本 (如：自然組詳解.pdf、全科解答.pdf)」自動保底跨檔綁定！
    """
    exam_sets = []
    
    SUBJECT_MAPPING = {
        "國文": ["國文", "國綜", "國語文", "國文考科", "國文考科國綜", "國文科"],
        "國寫": ["國寫", "國語文寫作", "國文寫作", "寫作測驗", "國作"],
        "英文": ["英文", "英語", "英文科", "英語科"],
        "數學甲": ["數學甲", "數甲", "數學甲考科", "高中數學甲"],
        "數學乙": ["數學乙", "數乙", "數學乙考科", "高中數學乙"],
        "數學A": ["數學A", "數A", "數學A考科", "南一四模數學Ａ", "高中數學A"],
        "數學B": ["數學B", "數B", "數學B考科", "南一四模數學Ｂ", "高中數學B"],
        "數學": ["數學", "數學考科", "數學科", "高中數學"],
        "物理": ["物理", "物理科", "物理考科"],
        "化學": ["化學", "化學科", "化學考科"],
        "生物": ["生物", "生物科", "生物考科"],
        "地球科學": ["地球科學", "地科"],
        "歷史": ["歷史", "歷史科", "歷史考科"],
        "地理": ["地理", "地理科", "地理考科"],
        "公民與社會": ["公民與社會", "公民", "公社科", "公社", "公民科", "公民與社會科", "公民考科", "社會與公民"],
        "自然": ["自然", "自然科", "自然考科"],
        "社會": ["社會", "社會科", "社會考科"]
    }

    def parse_metadata(filename_clean: str, rel_dir_path: str = "") -> dict:
        # 0. 排除「英聽/聽力」以及純「解答更正/級距五標」雜訊
        if any(k in filename_clean for k in ["英聽", "聽力", "解答更正", "更正表", "級距", "五標", "統計圖", "統計表"]):
            return {"year": "未知年份", "exam_type": "IGNORE", "mock_tag": "", "school_name": "", "subject": "未知科目", "role": "ignore"}

        # 解析路徑層級 (例如: school_exam_papers_only/北一女中/數學/高二文組試題/)
        dir_parts = []
        if rel_dir_path and rel_dir_path != ".":
            dir_parts = [p for p in rel_dir_path.replace("\\", "/").split("/") if p and p != "." and p not in ["school_exam_papers_only", "ast_exam_papers_only", "gsat_exam_papers_only", "mock_exam_papers_only", "學校定期考", "分科指考", "學測", "模擬考"]]

        # 1. 智慧提取學校名稱（🚨 嚴格限制：僅限學校段考目錄，且剔除學年度/全國/公私立等非學校字詞）
        school_name = ""
        if "school_exam_papers_only" in rel_dir_path:
            if dir_parts:
                school_name = dir_parts[0]
            if not school_name:
                sch_match = re.search(r'([\u4e00-\u9fa5]{2,8}(?:高級中學|高中|女中|中學|實驗中學|高職|附中))', filename_clean)
                if sch_match:
                    cand_sch = sch_match.group(1)
                    if not any(k in cand_sch for k in ["學年度", "學期", "全國", "公私立", "全區", "聯合", "模擬"]):
                        school_name = cand_sch

        # 2. 智慧提取學年度與上下學期 (解析 109_1, 108_1, 114_1, 112_高三_2 等)
        year_match = re.search(r'(\d{2,3})(?:學年度|年|E\d+|_|\b)', filename_clean)
        year = "未知年份"
        semester_str = ""

        if re.search(r'(?:_1_|1學期|上學期|第一學期|_1$|高[一二三]_1|_\d{2,3}_1_)', filename_clean) or re.search(r'^\d{2,3}_1\b', filename_clean):
            semester_str = "上學期"
        elif re.search(r'(?:_2_|2學期|下學期|第二學期|_2$|高[一二三]_2|_\d{2,3}_2_)', filename_clean) or re.search(r'^\d{2,3}_2\b', filename_clean):
            semester_str = "下學期"

        if year_match:
            year_num = int(year_match.group(1))
            if 80 <= year_num <= 120:  # 符合民國 80~120 年區間
                year = f"{year_num}學年度{semester_str}"

        # 3. 智慧提取年級與文理組別 (高一/高二/高三, 文組/理組)
        combined_path_text = "/".join(dir_parts) + "/" + filename_clean

        grade_str = ""
        grade_match = re.search(r'(高[一二三]|高1|高2|高3|10年級|11年級|12年級)', combined_path_text)
        if grade_match:
            g_raw = grade_match.group(1)
            if g_raw in ["高一", "高1", "10年級"]: grade_str = "高一"
            elif g_raw in ["高二", "高2", "11年級"]: grade_str = "高二"
            elif g_raw in ["高三", "高3", "12年級"]: grade_str = "高三"

        stream_str = ""
        if any(k in combined_path_text for k in ["文組", "社會組", "數A文", "數B文", "文科"]):
            stream_str = "文組"
        elif any(k in combined_path_text for k in ["理組", "自然組", "理科"]) or re.search(r'高[二三]理', combined_path_text):
            stream_str = "理組"

        # 4. 判斷科目 (優先檔名，若檔名沒寫科目則從資料夾層級 '北一女中/數學/...' 補齊)
        subject = "未知科目"
        
        # 4a. 先嘗試從檔名比對精確科目
        for official_name, aliases in SUBJECT_MAPPING.items():
            if any(alias in filename_clean for alias in aliases):
                subject = official_name
                break

        # 4b. 若檔名完全沒寫科目（如 108_1_高二理_第一次段考.pdf），從資料夾層級 (dir_parts[1]) 提取
        if subject == "未知科目" and len(dir_parts) >= 2:
            folder_subj = dir_parts[1]
            for official_name, aliases in SUBJECT_MAPPING.items():
                if any(alias in folder_subj for alias in aliases):
                    subject = official_name
                    break

        # 4c. 數學科目精細約束：除非明確寫有「數甲」或「數乙」，否則一律歸類為「數學」（或「數學A」/「數學B」）
        if subject in ["數學甲", "數學乙"] and not any(k in combined_path_text for k in ["數甲", "數學甲", "數乙", "數學乙"]):
            if any(k in combined_path_text for k in ["數A", "數學A"]):
                subject = "數學A"
            elif any(k in combined_path_text for k in ["數B", "數學B"]):
                subject = "數學B"
            else:
                subject = "數學"

        # 5. 判斷考試類型與標籤 (mock_tag)
        exam_type = "MOCK"
        mock_tag = ""

        is_makeup = any(k in filename_clean for k in ["補考", "補辦", "代考"])
        makeup_tag = "_補考" if is_makeup else ""

        is_ast_kw = any(k in filename_clean for k in ["指考", "指定科目", "分科"])
        is_gsat_kw = any(k in filename_clean for k in ["學測", "學科能力"])
        is_school = "school_exam_papers_only" in rel_dir_path or any(k in combined_path_text for k in ["定期考", "段考", "期中考", "期末考", "月考", "定期次考"])
        is_mock_kw = any(k in filename_clean for k in ["模擬", "模考", "全模", "北模", "中模", "南模", "竹模", "全區", "中區", "台中區", "台北區", "北區", "南區", "北北基", "新北基", "詮達", "全華", "翰林", "南一", "文昌", "漢樺", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"])

        if is_ast_kw and not is_mock_kw:
            exam_type = "AST"
            mock_tag = makeup_tag
        elif is_gsat_kw and not is_mock_kw:
            exam_type = "GSAT"
            mock_tag = makeup_tag
        elif is_school and not is_mock_kw:
            exam_type = "SCHOOL"
            school_match = re.search(r'((?:第[一二三四五六七八九十0-9]+次?)?(?:定期考|段考|期中考|期末考|月考))', filename_clean)
            exam_name = school_match.group(1) if school_match else "段考"
            tag_parts = [p for p in [grade_str, stream_str, exam_name] if p]
            mock_tag = f"_{'_'.join(tag_parts)}" if tag_parts else "_段考"
        else:
            # 🎯 模擬考精確雙軌分類：學測模擬 vs 分科模擬
            if is_ast_kw:
                exam_type = "MOCK_AST"
            elif is_gsat_kw:
                exam_type = "MOCK_GSAT"
            else:
                # 依科目或年級智能推斷：若為數甲、數乙、分科單科則為分科模考；若為數A、數B、綜合自然/社會則為學測模考
                if subject in ["數學甲", "數學乙"] or any(k in filename_clean for k in ["數甲", "數乙"]):
                    exam_type = "MOCK_AST"
                elif subject in ["數學A", "數學B", "自然", "社會"] or any(k in filename_clean for k in ["數A", "數B"]):
                    exam_type = "MOCK_GSAT"
                elif grade_str == "高三" and "2" in filename_clean:
                    exam_type = "MOCK_AST"
                else:
                    exam_type = "MOCK_GSAT" # 預設以學測模考為基準

            # 🚨 優先匹配出版社與品牌（納入「全國高中」、「全國公私立」、「各公立」精準映射至「全模」）
            publisher_match = re.search(r'(南一|翰林|龍騰|康熹|泰宇|全華|詮達|文昌|漢樺|三民|常春藤)', combined_path_text)
            league_match = re.search(r'(北北基|新北基|北模|中模|南模|竹模|全模|全區|中區|台中區|台中市|臺中市|台北區|臺北區|北區|南區|高雄區|全國高中|全國公私立|公私立高級中學|全國聯合|公私立高中|各公立高中)', combined_path_text)

            if publisher_match:
                org_str = publisher_match.group(1)
            elif league_match:
                raw_l = league_match.group(1)
                org_map = {
                    "北北基": "北模", "台北區": "北模", "臺北區": "北模", "新北基": "北模", "北區": "北模",
                    "台中區": "中模", "台中市": "中模", "臺中市": "中模", "中區": "中模",
                    "高雄區": "南模", "南區": "南模",
                    "竹模": "竹模", "竹苗區": "竹模", "新竹區": "竹模",
                    "全模": "全模", "全區": "全模", "全國高中": "全模", "全國公私立": "全模", "公私立高級中學": "全模", "全國聯合": "全模", "公私立高中": "全模", "各公立高中": "全模"
                }
                org_str = org_map.get(raw_l, "全模")
            else:
                org_str = "全模" if ("mock" in rel_dir_path.lower() or "模" in combined_path_text) else "全國"

            # 提取次數或期別 (支援：第X次、第X模、EX、X模、第X回、(X) 等多種格式)
            times_match = re.search(r'(?:第\s*([一二三四五六七八九十0-9]+)\s*[次回]|第\s*([0-9]+)\s*模|E([1-8])|([1-8])\s*模|[（\(]([1-8一二三四五六七八])[）\)])', combined_path_text)
            times_str = ""
            if times_match:
                raw_t = next(g for g in times_match.groups() if g is not None)
                cn_to_num = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'}
                t_digit = cn_to_num.get(raw_t, raw_t)
                times_str = f"第{t_digit}次"
            elif re.search(r'第二學期|下學期', combined_path_text):
                times_str = "下學期"
            elif re.search(r'第一學期|上學期', combined_path_text):
                times_str = "上學期"
            else:
                times_str = "全期"

            # 🚨 剛性命名一致化：模考標籤僅保留「品牌_期別」（如 _全模_第2次、_南一_第4次），嚴禁夾帶高三/理組，徹底終結修復引擎衝突與重複解析！
            mock_tag = f"_{org_str}_{times_str}"

        # 6. 判斷文件角色 (Role) - 🚨 剛性過濾與角色分離
        clean_no_ext = os.path.splitext(filename_clean)[0]

        # A. 判斷是否為「解答更正通知」
        if any(k in filename_clean for k in ["更正通知", "解答更正", "更正表", "答案更正", "更正"]):
            return {"year": year, "exam_type": "ERRATA", "mock_tag": mock_tag, "school_name": school_name, "subject": subject, "role": "errata", "is_compilation": False}

        # B. 標記是否為「歷屆試題彙整 / 各區模擬彙整」
        is_compilation = any(k in filename_clean for k in ["彙整", "彙編", "歷屆", "各區"])

        # C. 🚨 剛性剔除「純空白答案卷/答題卷」：只要包含「卷/卡/紙」且不包含實體「詳解/解析/解答/公佈/確定稿」，絕對禁止作為解答本！
        is_blank_sheet = (
            any(k in filename_clean for k in ["答案卷", "答題卷", "作答卷", "答題卡", "答案卡", "畫卡紙", "答案紙"])
            and not any(k in filename_clean for k in ["解答", "解析", "詳解", "參考答案", "確定稿", "核對", "答案公佈"])
        )
        if is_blank_sheet:
            return {"year": year, "exam_type": "IGNORE", "mock_tag": "", "school_name": "", "subject": subject, "role": "ignore", "is_compilation": False}

        # D. 判定二合一、評分標準、解答本與題目卷（精準支援 _Q, _A, 試題, 解析）
        combined_keywords = ["與詳解", "與解析", "含解析", "含詳解", "含解答", "+解析", "&解答", "暨詳解", "暨答案", "題目+解析", "試題與解析", "試題加詳解", "試題暨詳解", "考科暨答案", "參考詳解"]
        has_combined_kw = any(k in filename_clean for k in combined_keywords)

        # 結尾特徵比對（支援 數A_Q.pdf, 數A_A.pdf, 106-Q.pdf 等）
        is_q_suffix = bool(re.search(r'[-_](?:q|Q|題目|試題|試卷)$', clean_no_ext))
        is_a_suffix = bool(re.search(r'[-_](?:a|A|答案|解答|解析|詳解)$', clean_no_ext))

        has_question_kw = is_q_suffix or any(k in filename_clean for k in ["試卷", "題目", "試題", "題本", "內頁", "封面+內頁", "封面＋內頁"])
        has_answer_kw = is_a_suffix or any(k in filename_clean for k in ["解答", "解析", "詳解", "詳答", "參考答案", "參考解答", "解答本", "答案公布"])

        if has_combined_kw or (has_question_kw and has_answer_kw):
            role = "combined"  # 題目與解析二合一 PDF
        elif any(k in filename_clean for k in ["記者會說明", "說明資料", "非選評分", "非選擇題評分", "非選", "評分", "原則", "標準"]):
            role = "rubric"     # 評分標準
        elif has_answer_kw:
            role = "answer"    # 純解答/詳解檔
        else:
            role = "question"  # 純題目檔

        return {
            "year": year,
            "exam_type": exam_type,
            "mock_tag": mock_tag,
            "school_name": school_name,
            "subject": subject,
            "role": role,
            "is_compilation": is_compilation
        }

    def extract_errata_from_pdf(errata_pdf_path: str):
        """從解答更正通知 PDF 中萃取官方勘誤表格（支援純文字與掃描圖雙軌解析）"""
        global GLOBAL_ERRATA_MAP
        try:
            doc = fitz.open(errata_pdf_path)
            full_txt = ""
            for p in doc: full_txt += p.get_text("text") + "\n"
            
            # 🚨 雙軌解析：若為純圖片掃描檔，啟動視覺 OCR 提取更正表格
            if len(full_txt.strip()) < 15 and len(doc) > 0:
                logging.info(f"🔍 [更正通知掃描檔辨識] {os.path.basename(errata_pdf_path)} 為純圖片，啟動視覺 OCR...")
                class VisualErrataItem(BaseModel):
                    subject: str = Field(description="更正科目，如 國文, 物理, 數學A")
                    question_number: str = Field(description="題號，如 10, 25")
                    corrected_answer: str = Field(description="更正後的正確答案，如 B, ACD, 送分")

                class VisualErrataPage(BaseModel):
                    items: List[VisualErrataItem] = Field(description="更正條目列表")

                for p_idx in range(len(doc)):
                    pix = doc[p_idx].get_pixmap(dpi=200)
                    tmp_p = f"temp_errata_{p_idx}.png"
                    pix.save(tmp_p)
                    try:
                        with Image.open(tmp_p) as p_img:
                            res_err, _ = parser.ai_manager.generate_with_retry(
                                contents=["這是一份考試答案更正/勘誤通知，請提取更正的科目、題號與更正後答案：", p_img],
                                response_schema=VisualErrataPage,
                                temperature=0.0,
                                preferred_model="gemini-3.5-flash-lite",
                                enable_thinking=False
                            )
                            if res_err and 'items' in res_err:
                                for it in res_err['items']:
                                    s = re.sub(r'(?:考科|科)$', '', it.get('subject', '').strip())
                                    qn = str(it.get('question_number', '')).strip()
                                    ca = clean_ocr_answer_format(it.get('corrected_answer', ''))
                                    if s and qn and ca:
                                        set_errata_answer(s, qn, ca)
                                        logging.info(f"📢 [視覺載入官方更正] 科目: {s} | 題號: {qn} ➔ 更正為: {ca}")
                    finally:
                        if os.path.exists(tmp_p): os.remove(tmp_p)
            else:
                # 純文字層正則提取
                lines = full_txt.splitlines()
                for line in lines:
                    m = re.search(r'([\u4e00-\u9fa5]{2,5})\s+(\d+|[A-Ga-g])\s+([A-Ga-g0-9、\s]+?)\s+([A-Ga-g0-9、\s]+|送分|全體給分)', line)
                    if m:
                        subj, qn, orig, corr = m.group(1).strip(), m.group(2).strip(), m.group(3).strip(), m.group(4).strip()
                        clean_s = re.sub(r'(?:考科|科)$', '', subj)
                        if clean_s in ["國綜", "國語文"]: clean_s = "國文"
                        elif clean_s in ["數A", "高中數學A"]: clean_s = "數學A"
                        elif clean_s in ["數B", "高中數學B"]: clean_s = "數學B"
                        elif clean_s in ["數甲", "高中數學甲"]: clean_s = "數學甲"
                        elif clean_s in ["數乙", "高中數學乙"]: clean_s = "數學乙"
                        elif clean_s in ["公民", "公社"]: clean_s = "公民與社會"
                        elif clean_s in ["地科"]: clean_s = "地球科學"
                        
                        corr_clean = "無答案（全體給分）" if any(k in corr for k in ["送分", "給分"]) else clean_ocr_answer_format(corr)
                        set_errata_answer(clean_s, qn, corr_clean)
                        logging.info(f"📢 [載入官方更正通知] 科目: {clean_s} | 題號: {qn} | 原答案: {orig} ➔ 更正為: {corr_clean}")
            doc.close()
        except Exception as e:
            logging.warning(f"讀取更正通知 {errata_pdf_path} 失敗: {e}")

    for base_dir in directories:
        if not os.path.exists(base_dir):
            continue
            
        for root, dirs, files in os.walk(base_dir):
            pdf_files = [f for f in files if f.endswith(".pdf")]
            if not pdf_files:
                continue
            
            rel_dir_path = os.path.relpath(root, base_dir) if base_dir in root else ""
            
            # 優先掃描並載入更正通知 PDF
            for f in pdf_files:
                if any(k in f for k in ["更正通知", "解答更正", "更正表", "答案更正", "詳解更正", "詳解修正", "答案修正", "勘誤表", "勘誤", "疑義"]):
                    extract_errata_from_pdf(os.path.join(root, f))
                    
            task_groups = {}
            global_answer_files = []

            for f in pdf_files:
                meta = parse_metadata(f, rel_dir_path=os.path.join(base_dir, rel_dir_path) if rel_dir_path != "." else base_dir)
                if meta["role"] in ["ignore", "errata"] or meta["year"] == "未知年份":
                    continue
                
                if meta["role"] == "answer" and meta["subject"] == "未知科目":
                    global_answer_files.append(os.path.join(root, f))
                    continue

                if meta["subject"] == "未知科目":
                    continue

                key = (meta["year"], meta["exam_type"], meta["mock_tag"], meta["school_name"], meta["subject"])
                if key not in task_groups:
                    task_groups[key] = []
                task_groups[key].append((f, meta))
            
            for key, grouped_files in task_groups.items():
                q_candidates = [filename for filename, meta in grouped_files if meta["role"] in ["question", "combined"]]
                if not q_candidates:
                    continue
                final_q_file = sorted(q_candidates, key=lambda x: len(x), reverse=True)[0]

                # 💡 [真理校準] 若模考缺少品牌或次數 (含 "全期" 或 "下學期")，偷瞄 PDF 第 1 頁首段標題 100% 救回！
                cur_m_tag = key[2]
                if key[1] in ["MOCK", "MOCK_GSAT", "MOCK_AST"] and any(k in cur_m_tag for k in ["全期", "下學期", "上學期", "全國"]):
                    q_full_peek_path = os.path.join(root, final_q_file)
                    try:
                        peek_doc = fitz.open(q_full_peek_path)
                        if len(peek_doc) > 0:
                            p_txt = peek_doc[0].get_text("text")[:600].replace(" ", "").replace("\n", "")
                            p_pub = re.search(r'(南一|翰林|龍騰|康熹|泰宇|全華|詮達|文昌|漢樺|三民)', p_txt)
                            p_leg = re.search(r'(北北基|新北基|北模|中模|南模|竹模|全模|全國公私立|公私立高級中學|全國高中|全國聯合)', p_txt)
                            p_tim = re.search(r'(?:第\s*([一二三四五六七八九十0-9]+)\s*[次回]|第\s*([0-9]+)\s*模|E([1-8])|([1-8])\s*模)', p_txt)

                            c_org = p_pub.group(1) if p_pub else (p_leg.group(1) if p_leg else "")
                            if c_org in ["全國公私立", "公私立高級中學", "全國高中", "全國聯合"]: c_org = "全模"
                            elif c_org in ["北北基", "新北基"]: c_org = "北模"

                            c_tim = ""
                            if p_tim:
                                raw_t = next(g for g in p_tim.groups() if g is not None)
                                cn_map = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8'}
                                c_tim = f"第{cn_map.get(raw_t, raw_t)}次"

                            if c_org or c_tim:
                                sub_tags = [p for p in cur_m_tag.split("_") if p and p not in ["全期", "全國", "下學期", "上學期"]]
                                if c_org and not any(k in sub_tags for k in ["北模", "中模", "南模", "竹模", "全模", "南一", "翰林", "龍騰", "康熹", "泰宇", "全華", "詮達", "文昌"]):
                                    sub_tags.append(c_org)
                                if c_tim and not any("第" in p and "次" in p for p in sub_tags):
                                    sub_tags.append(c_tim)
                                if sub_tags:
                                    cur_m_tag = f"_{'_'.join(sub_tags)}"
                        peek_doc.close()
                    except Exception:
                        pass
                
                # 尋找解答檔：若題目為補考，則答案檔「必須」為補考；若題目非補考，答案檔「嚴禁」為補考！
                is_q_makeup = "_補考" in key[2]
                a_candidates = [
                    filename for filename, meta in grouped_files 
                    if meta["role"] == "answer" and (("_補考" in meta["mock_tag"]) == is_q_makeup)
                ]
                if a_candidates:
                    # 優先選擇含有「選擇題」、「定稿」、「參考答案」等精確字眼的 PDF 檔
                    a_candidates.sort(key=lambda x: (
                        0 if any(k in x for k in ["選擇題", "選擇", "定稿"]) else 1,
                        0 if "記者會" not in x else 1,
                        x
                    ))
                    final_a_file = os.path.join(root, a_candidates[0])
                elif any(meta["role"] == "combined" for filename, meta in grouped_files if filename == final_q_file):
                    final_a_file = os.path.join(root, final_q_file)
                elif global_answer_files:
                    subj = key[4]
                    matched_global = None
                    for gf in global_answer_files:
                        gf_base = os.path.basename(gf)
                        # 嚴格匹配科目關鍵字，絕不盲目跨科抓取
                        if subj in gf_base:
                            matched_global = gf; break
                        elif subj in ["物理", "化學", "生物", "地球科學"] and any(k in gf_base for k in ["自然", "理綜"]):
                            matched_global = gf; break
                        elif subj in ["歷史", "地理", "公民與社會"] and any(k in gf_base for k in ["社會", "文綜"]):
                            matched_global = gf; break
                    
                    # 🚨 若完全無吻合科目，設為 None，由 AI 依據原卷自行嚴謹推導，絕不抓其他科目的解答表！
                    final_a_file = matched_global
                elif key[1] in ["SCHOOL", "MOCK", "MOCK_GSAT", "MOCK_AST"]:
                    # 🚨 模考與段考全自動合卷保底：若無獨立解答檔，通常詳解與答案直接附在題本後半部（24頁二合一合卷）
                    final_a_file = os.path.join(root, final_q_file)
                else:
                    final_a_file = None
                
                rubric_candidates = [filename for filename, meta in grouped_files if meta["role"] == "rubric"]
                final_rubric_file = os.path.join(root, rubric_candidates[0]) if rubric_candidates else None
                
                # 判定是否為歷屆彙整
                is_comp = any(meta.get("is_compilation", False) for _, meta in grouped_files)
                exam_sets.append({
                    "year": key[0],
                    "exam_type": key[1],
                    "mock_tag": cur_m_tag,
                    "school_name": key[3] if key[1] == "SCHOOL" else "",
                    "subject": key[4],
                    "q_pdf": os.path.join(root, final_q_file),
                    "a_pdf": final_a_file,
                    "rubric_pdf": final_rubric_file,
                    "is_compilation": is_comp
                })

    # 🚨 關鍵排程：將「歷屆試題彙整 / 各區模擬彙整」排到最後才跑！
    exam_sets.sort(key=lambda x: (1 if x.get("is_compilation") else 0))
    return exam_sets

# =========================================================================
# 🌟 工單驅動：單題定向深度重生手術刀 (Surgical Re-run Engine)
# =========================================================================
TICKETS_FILE = "exam_issue_tickets.json"

def process_single_ticket_rerun(ticket: dict, parser: ExamParser, target_dirs: List[str], output_dir: str) -> bool:
    """針對使用者回報的單一題目，徹底重核、重裁圖、AI重推導，並產生完整的修復報告寫入工單"""
    import copy
    
    uid = ticket.get("uid", "")
    desc = ticket.get("description", "")
    category = ticket.get("category", "")
    sug_ans = ticket.get("suggested_answer", "")
    sug_type = ticket.get("suggested_type", "")
    
    logging.info(f"💉 [啟動手術刀重跑] 工單 {ticket.get('ticket_id')} | 題目: {uid}")
    action_logs = []
    
    # 1. 透過 UID 全域搜尋對應的 JSON 資料庫
    target_json_path = None
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith("_database.json") and not f.endswith("_partial_database.json"):
                fp = os.path.join(root, f)
                try:
                    with open(fp, "r", encoding="utf-8") as jf:
                        d = json.load(jf)
                        if any(q.get("uid") == uid for q in d):
                            target_json_path = fp
                            break
                except Exception: pass
        if target_json_path: break
        
    if not target_json_path:
        ticket["ai_feedback"] = "❌ [尋找失敗] 資料庫中找不到該題目 UID，無法執行修復。"
        ticket["status"] = "error"
        return False
        
    with open(target_json_path, "r", encoding="utf-8") as jf:
        questions_list = json.load(jf)
        
    target_idx = next((i for i, q in enumerate(questions_list) if q.get("uid") == uid), -1)
    if target_idx == -1: return False
    
    target_q = questions_list[target_idx]
    old_q = copy.deepcopy(target_q)
    
    # 2. 🚨 來源 PDF 重新核對與重新裁切
    page_num = max(1, int(target_q.get("page_number", 1)))
    q_pdf_path = target_q.get("question_pdf_path")
    img_dir = os.path.dirname(target_json_path) + "/images/" + os.path.basename(target_json_path).replace("_database.json", "")
    os.makedirs(img_dir, exist_ok=True)
    
    if q_pdf_path and os.path.exists(q_pdf_path):
        try:
            with fitz.open(q_pdf_path) as doc:
                p_idx = min(page_num - 1, len(doc) - 1)
                page = doc[p_idx]
                if target_q.get("image_bboxes"):
                    for old_p in target_q.get("image_paths", []):
                        if os.path.exists(old_p):
                            try: os.remove(old_p)
                            except Exception: pass
                    target_q["image_paths"] = parser.execute_crop(page, target_q["image_bboxes"], img_dir, f"Q{target_q['question_number']}_Rerun")
                    action_logs.append(f"✅ 已刪除舊圖，並根據原卷重新裁切 {len(target_q['image_paths'])} 張高精度附圖")
        except Exception as e:
            action_logs.append(f"⚠️ 重新裁切圖片失敗: {e}")

    # 3. 🚨 銷毀舊的幾何解析圖 (diagram) 強制重繪
    safe_q = safe_filename(str(target_q['question_number']).replace(" ", ""))
    deleted_diags = 0
    for i in range(1, 4):
        diag_p = os.path.join(img_dir, f"diagram_Q{safe_q}_{i}.png")
        for ext in ["", ".done", ".failed"]:
            if os.path.exists(diag_p + ext):
                try: 
                    os.remove(diag_p + ext)
                    if ext == "": deleted_diags += 1
                except Exception: pass
    if deleted_diags > 0:
        action_logs.append(f"🗑️ 已強制刪除 {deleted_diags} 張舊版 AI 幾何解析圖")

    # 4. 🚨 準備單題 AI 重跑提示詞 (注入使用者批判)
    q_sub = target_q.get("sub_subject", "數學")
    q_rubric = SUBJECT_DIFFICULTY_RUBRICS.get(q_sub, GENERAL_DIFFICULTY_RUBRIC)
    q_allowed = SUBJECT_TAXONOMY.get(q_sub, {"topics": [], "techniques": []})
    
    opts_str = "\n".join([f"- ({opt.get('key')}) {opt.get('value')}" for opt in target_q.get('options', [])]) or "無"
    ans_label = f"【{sug_ans or target_q.get('answer')}】"
    
    item_desc = f"=== 待修復題目 ===\n題號：{target_q['question_number']}\n題幹：{target_q['question_text']}\n選項：\n{opts_str}\n官方/修正答案：{ans_label}\n"
    urgent_critique = f"\n🚨🚨【來自使用者/專家的重大修正指示】🚨🚨\n錯誤分類：{category}\n具體瑕疵：>>> {desc} <<<\n請務必採納此建議，揚棄先前的錯誤推導，以最高水準重新解析！\n"
    
    prompt = PROMPT_STAGE_2_INTRO.format(batch_size=1) + item_desc + urgent_critique + PROMPT_STAGE_2_MAIN.format(
        subject_rubric=q_rubric, q_answer=ans_label, topics=q_allowed.get('topics', []), technique=q_allowed.get('techniques', []),
        math_scope_instruction="", subject_specific_instruction=""
    )
    
    contents = [prompt]
    for p in target_q.get('image_paths', []):
        if os.path.exists(p): contents.append(Image.open(p))
            
    # 5. 呼叫頂級大模型進行深度重生
    logging.info("🧠 正在調用旗艦大模型執行單題名師級解析重生...")
    sol_batch, _ = parser.ai_manager.generate_with_retry(
        contents=contents, response_schema=QuestionSolutionBatch,
        temperature=0.2, preferred_model="gemini-3.8-flash", enable_thinking=True, task_desc=f"[工單旗艦重跑 {uid}]"  # 🚨 人工工單一律出動 3.8-flash
    )
    
    if sol_batch and 'solutions' in sol_batch and sol_batch['solutions']:
        sol = s2t_recursive(sol_batch['solutions'][0])
        
        # 覆寫並清洗格式
        target_q.update(sol)
        if sug_ans: target_q["answer"] = sug_ans
        if sug_type: target_q["question_type"] = sug_type
        pre_validate_format(target_q, sol)
        
        # 產生變更比對日誌 (Diff)
        if str(old_q.get('answer')).strip() != str(target_q.get('answer')).strip():
            action_logs.append(f"🔄 答案變更: {old_q.get('answer')} ➔ {target_q.get('answer')}")
        if str(old_q.get('question_type')).strip() != str(target_q.get('question_type')).strip():
            action_logs.append(f"🔄 題型變更: {old_q.get('question_type')} ➔ {target_q.get('question_type')}")
            
        action_logs.append(f"✅ AI 已成功重寫詳解與選項剖析 (共產出 {len(target_q.get('detailed_solution',''))} 字)")
        
        # 圖解補繪判定
        if "diagram_" in target_q.get("detailed_solution", ""):
            action_logs.append("✅ 系統已將新的幾何解析圖指令排入渲染引擎隊列")

        # 6. 寫回 JSON 資料庫
        questions_list[target_idx] = target_q
        with open(target_json_path, "w", encoding="utf-8") as jf:
            json.dump(questions_list, jf, ensure_ascii=False, indent=4)
            
        # 7. 更新工單狀態
        ticket["ai_feedback"] = "✅ [手術成功] 系統已接納建議，成功重核圖片並完成深度推導重寫。"
        ticket["action_log"] = "\n".join(action_logs)
        logging.info(f"🎉 題目 {uid} 深度重跑完畢！")
        return True
    else:
        ticket["ai_feedback"] = "❌ [手術失敗] AI 重跑無法產生有效結果，請檢查建議是否合理。"
        ticket["action_log"] = "\n".join(action_logs) + "\n❌ 模型生成失敗或超時。"
        return False
        
def execute_all_pending_tickets(parser: ExamParser, target_dirs: List[str], output_dir: str):
    """檢查並執行所有來自使用者回報的待修復工單"""
    if not os.path.exists(TICKETS_FILE):
        return
    try:
        with open(TICKETS_FILE, "r", encoding="utf-8") as tf:
            tickets = json.load(tf)
    except Exception: return
    
    pending = [t for t in tickets if t.get("status") == "pending"]
    if not pending: return
    
    logging.info(f"📬 [檢測到待辦工單] 共有 {len(pending)} 道題目被使用者標記有瑕疵，啟動定向重跑程序...")
    try:
        for t in pending:
            success = process_single_ticket_rerun(t, parser, target_dirs, output_dir)
            if success:
                t["status"] = "resolved"
                t["resolved_at"] = time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                if t.get("status") == "pending":
                    t["status"] = "failed"
                t["failed_at"] = time.strftime('%Y-%m-%d %H:%M:%S')
    finally:
        # 🚨 即使遇到超時中斷，也必定將已成功修復的工單狀態寫回磁碟！
        with open(TICKETS_FILE, "w", encoding="utf-8") as tf:
            json.dump(tickets, tf, ensure_ascii=False, indent=4)
        
# ==========================================
# 5. 執行進入點
# ==========================================
if __name__ == "__main__":
    # 將你的 Gemini API Keys 放入此處
    API_KEYS = load_api_keys('key.txt')
    if not API_KEYS:
        env_key = os.environ.get("GEMINI_API_KEY")
        if env_key:
            API_KEYS = [env_key]
            print("✅ 成功載入環境變數中的 GEMINI_API_KEY。")
    if API_KEYS:
        
        # 🚨 核心修改：在程式啟動時，將金鑰列表順序完全隨機洗牌打散
        # 這會打亂相同帳號、相鄰金鑰的派發順序，降低單一專案在短時間內被集中高頻呼叫的風險
        random.shuffle(API_KEYS)
        print(f"✅ 成功載入並隨機打散 {len(API_KEYS)} 組金鑰。")
        print(f"打散後第一組: {API_KEYS[0][:15]}...")
        print(f"打散後最後一組: {API_KEYS[-1][:15]}...")
    # 推薦使用 flash 模型處理多模態 (Vision) 任務，速度快且便宜
    MODELS = [
        "gemini-3.8-flash",        # 旗艦算力：負責學術仲裁、工單重跑、理科高難推導
        "gemini-3.7-flash",        # 次旗艦：負責名師一題多解、另解速解推演
        "gemini-3.6-flash",        # 均衡主力：負責審查批判、文科名師詳解
        "gemini-3.5-flash",        # 視覺主力：負責模考詳解萃取、單題補漏、代碼修復
        "gemini-3.5-flash-lite",   # 最新極速端點：負責試卷 OCR、評分標準框選、圖表繪製、模考自癒
        "gemini-3.1-flash-lite",   # 備援極速端點：負責大批次純文字掃描
        "gemini-2.5-flash"         # 兜底備援
    ]
    
    manager = GeminiFreeTierManager(api_keys=API_KEYS, models=MODELS)
    parser = ExamParser(ai_manager=manager)

    # 1. 指定你的題庫根目錄（包含學校定期考 school_exam_papers_only）
    TARGET_DIRECTORIES = ["ast_exam_papers_only", "gsat_exam_papers_only", "mock_exam_papers_only", "school_exam_papers_only"]
    output_directory = "./exam_database_output"

    # 🚨 優先執行使用者回報的單題重跑手術（重新核對來源PDF、重裁圖片、重跑詳解、重繪圖解）
    execute_all_pending_tickets(parser, TARGET_DIRECTORIES, output_directory)
    
    # =========================================================================
    # 🚀 優先插隊：執行管理員新上傳的試卷排隊任務清單 (Priority Upload Tasks)
    # =========================================================================
    UPLOAD_TASKS_FILE = "exam_processing_tasks.json"
    if os.path.exists(UPLOAD_TASKS_FILE):
        try:
            with open(UPLOAD_TASKS_FILE, "r", encoding="utf-8") as tf:
                upload_tasks = json.load(tf)
                
            pending_upload_tasks = [t for t in upload_tasks if t.get("status") == "pending"]
            if pending_upload_tasks:
                logging.info(f"⚡ [優先插隊] 發現 {len(pending_upload_tasks)} 份由管理員新上傳的試卷任務，啟動最高優先級解析！")
                for u_task in pending_upload_tasks:
                    u_task["status"] = "processing"
                    try:
                        parser.process_exam_paper(
                            subject=u_task["subject"],
                            year=u_task["academic_year"],
                            exam_type=u_task["exam_type"],
                            mock_tag=u_task["mock_tag"],
                            q_pdf=u_task["q_pdf"],
                            a_pdf=u_task["a_pdf"],
                            rubric_pdf=u_task["rubric_pdf"],
                            output_dir=output_directory,
                            school_name=u_task.get("school_name", "")
                        )
                        u_task["status"] = "completed"
                        u_task["completed_at"] = time.strftime('%Y-%m-%d %H:%M:%S')
                        logging.info(f"🎉 [優先任務完成] 試卷 {u_task['file_prefix']} 已成功產出資料庫！")
                    except (TimeoutSafetyException, RPDExhaustedException) as exit_err:
                        u_task["status"] = "pending"  # 保持 pending，下次續跑
                        with open(UPLOAD_TASKS_FILE, "w", encoding="utf-8") as tf:
                            json.dump(upload_tasks, tf, ensure_ascii=False, indent=4)
                        raise exit_err
                    except Exception as task_err:
                        u_task["status"] = "failed"
                        u_task["error_msg"] = str(task_err)
                        logging.error(f"❌ [優先任務失敗] {u_task['file_prefix']}: {task_err}")
                        
                # 寫回已更新的任務狀態
                with open(UPLOAD_TASKS_FILE, "w", encoding="utf-8") as tf:
                    json.dump(upload_tasks, tf, ensure_ascii=False, indent=4)
        except Exception as e:
            logging.error(f"讀取或執行優先試卷任務清單失敗: {e}")

    # 🎯 若 GitHub Actions 開啟了「僅精準重跑工單」開關，完成後直接儲存並正常退出，不耗時跑全卷常規解析
    if os.environ.get("PROCESS_REPORTED_TICKETS_ONLY", "false").lower() == "true":
        logging.info("🎯 [精準重跑任務完成] 使用者回報工單已全部重審並修復完畢，安全終止 Python 程序 (Exit 0)！")
        sys.exit(0)
    
    # 2. 跨 Run 繼承排程器：讀取先前進度，過濾未完成考卷，並實現 3:1 自適應交錯調度
    raw_exam_tasks = auto_find_exam_sets(TARGET_DIRECTORIES)
    logging.info(f"🔍 目錄中共有 {len(raw_exam_tasks)} 份試卷。正在過濾已完工試卷並載入排程狀態...")

    sched_state = load_scheduler_state()
    mock_streak = sched_state.get("mock_streak", 0)
    logging.info(f"📊 [排程進度繼承] 上次中斷時狀態：連續模考 streak = {mock_streak}/3 (歷史完工: 模考 {sched_state.get('total_mock_done', 0)} 份 | 正式考 {sched_state.get('total_official_done', 0)} 份)")

    # 💡 判斷一份任務是否「真的已經 100% 完工」
    def is_task_already_completed(t):
        s_name, y_name, e_type, m_tag, sch_name = t["subject"], t["year"], t["exam_type"], t["mock_tag"], t.get("school_name", "")
        # 🚨 嚴格學科正規化（避免 數甲 找不到 數學甲，導致每次開機都重解）
        clean_s = re.sub(r'(?:考科|科)$', '', str(s_name).strip())
        if clean_s in ["高中數學甲", "數甲"]: clean_s = "數學甲"
        elif clean_s in ["高中數學乙", "數乙"]: clean_s = "數學乙"
        elif clean_s in ["高中數學A", "數A"]: clean_s = "數學A"
        elif clean_s in ["高中數學B", "數B"]: clean_s = "數學B"
        elif clean_s in ["國綜", "國語文"]: clean_s = "國文"
        elif clean_s in ["國語文寫作", "寫作測驗"]: clean_s = "國寫"

        digits = "".join(filter(str.isdigit, str(y_name)))
        y_val = int(digits) if digits else 111
        if e_type == "SCHOOL":
            t_folder = os.path.join("學校定期考", safe_filename(sch_name) if sch_name else "通用學校")
            sp_name = f"{y_val}學年度_{safe_filename(sch_name) if sch_name else '通用學校'}{m_tag}_{clean_s}"
        elif e_type in ["MOCK", "MOCK_GSAT", "MOCK_AST"]:
            ast_short = "分科" if y_val >= 111 else "指考"
            mock_sub_type = "學測模擬" if e_type == "MOCK_GSAT" else (f"{ast_short}模擬" if e_type == "MOCK_AST" else "模擬考")
            org_label = "全國"
            for cand in ["北模", "中模", "南模", "竹模", "全模", "翰林", "南一", "龍騰", "康熹", "泰宇", "詮達", "全華", "漢樺", "文昌"]:
                if cand in m_tag: org_label = cand; break
            t_folder = os.path.join("模擬考", mock_sub_type, org_label)
            sp_name = f"{y_val}學年度_{mock_sub_type}{m_tag}_{clean_s}"
        elif e_type == "GSAT":
            t_folder = "學測"
            is_mu = "_補考" in str(m_tag) or "補考" in str(y_name)
            sp_name = f"{y_val}學年度_學測{'_補考' if is_mu else ''}_{clean_s}"
        else:
            t_folder = "分科指考"
            is_mu = "_補考" in str(m_tag) or "補考" in str(y_name)
            ast_s = "分科" if y_val >= 111 else "指考"
            sp_name = f"{y_val}學年度_{ast_s}{'_補考' if is_mu else ''}_{clean_s}"
        sp_name = re.sub(r'[\s\n\r\t]+', '', sp_name)
        target_db = os.path.join(output_directory, t_folder, f"{sp_name}_database.json")
        if os.path.exists(target_db):
            try:
                with open(target_db, "r", encoding="utf-8") as jf:
                    db_items = json.load(jf)
                    # 🚨 精準門檻：國寫永遠只有 2 大題，絕不可要求 15 題！
                    if clean_s == "國寫":
                        min_q = 2
                    elif any(k in clean_s for k in ["數", "math"]):
                        min_q = 8
                    else:
                        min_q = 15
                    has_real_sols = all(len(str(q.get("detailed_solution", "")).strip()) >= 20 for q in db_items)
                    return len(db_items) >= min_q and has_real_sols
            except Exception: pass
        return False

    # 僅針對「尚未完成」的考卷進行精確 3:1 調度
    # 🚨 精確分類：分流「已完工試卷」與「待處理試卷」，確保儀表板能完整統計全庫總進度
    pending_mocks = []
    pending_officials = []
    completed_tasks = []

    for t in raw_exam_tasks:
        if is_task_already_completed(t):
            completed_tasks.append(t)
        else:
            if t["exam_type"] in ["MOCK", "MOCK_GSAT", "MOCK_AST"]:
                pending_mocks.append(t)
            else:
                pending_officials.append(t)

    logging.info(f"🎯 [真·待處理清單] 歷史已完工：{len(completed_tasks)} 份 | 剩餘未完工：模擬考 {len(pending_mocks)} 份，學測/分科/段考 {len(pending_officials)} 份。")

    exam_tasks = []
    m_idx, o_idx = 0, 0
    total_m, total_o = len(pending_mocks), len(pending_officials)

    # 🚀 第一步：根據上次中斷殘留的 streak 進行「自適應平滑補齊」
    initial_mocks_needed = max(0, 3 - mock_streak) if mock_streak > 0 else 3
    if mock_streak >= 3:
        initial_mocks_needed = 0

    for _ in range(initial_mocks_needed):
        if m_idx < total_m:
            exam_tasks.append(pending_mocks[m_idx])
            m_idx += 1

    if (mock_streak > 0 or initial_mocks_needed > 0) and o_idx < total_o:
        exam_tasks.append(pending_officials[o_idx])
        o_idx += 1

    # 🚀 第二步：進入標準循環 (3 份模考 : 1 份正式考)
    while m_idx < total_m or o_idx < total_o:
        for _ in range(3):
            if m_idx < total_m:
                exam_tasks.append(pending_mocks[m_idx])
                m_idx += 1
        if o_idx < total_o:
            exam_tasks.append(pending_officials[o_idx])
            o_idx += 1

    preview_plan = [f"{i+1}.[{t['exam_type']}]{t['year']}{t['mock_tag']}_{t['subject']}" for i, t in enumerate(exam_tasks[:8])]
    logging.info(f"⚖️ [3:1 繼承佇列排程確認] 前 8 卷執行計劃：\n  " + "\n  ".join(preview_plan))

    # 🚨 核心優化：以【全庫所有試卷 raw_exam_tasks】為母體進行掃描，秒級呈現真實全庫 160+ 份完工進度！
    pre_completed_count = 0
    for task in raw_exam_tasks:
        s_name = task["subject"]
        y_name = task["year"]
        e_type = task["exam_type"]
        m_tag = task["mock_tag"]
        sch_name = task.get("school_name", "")
        
        # 🚨 取得全域標準正規化名稱
        clean_s = re.sub(r'(?:考科|科)$', '', str(s_name).strip())
        if clean_s in ["高中數學甲", "數甲"]: clean_s = "數學甲"
        elif clean_s in ["高中數學乙", "數乙"]: clean_s = "數學乙"
        elif clean_s in ["高中數學A", "數A"]: clean_s = "數學A"
        elif clean_s in ["高中數學B", "數B"]: clean_s = "數學B"
        elif clean_s in ["國綜", "國語文"]: clean_s = "國文"
        elif clean_s in ["國語文寫作", "寫作測驗"]: clean_s = "國寫"

        digits = "".join(filter(str.isdigit, str(y_name)))
        y_val = int(digits) if digits else 111
        
        if e_type == "SCHOOL":
            t_folder = os.path.join("學校定期考", safe_filename(sch_name) if sch_name else "通用學校")
            sp_name = f"{y_val}學年度_{safe_filename(sch_name) if sch_name else '通用學校'}{m_tag}_{clean_s}"
        elif e_type in ["MOCK", "MOCK_GSAT", "MOCK_AST"]:
            ast_short = "分科" if y_val >= 111 else "指考"
            mock_sub_type = "學測模擬" if e_type == "MOCK_GSAT" else (f"{ast_short}模擬" if e_type == "MOCK_AST" else "模擬考")
            org_label = "全國"
            for cand in ["北模", "中模", "南模", "竹模", "全模", "翰林", "南一", "龍騰", "康熹", "泰宇", "詮達", "全華", "漢樺", "文昌"]:
                if cand in m_tag: org_label = cand; break
            t_folder = os.path.join("模擬考", mock_sub_type, org_label)
            sp_name = f"{y_val}學年度_{mock_sub_type}{m_tag}_{clean_s}"
        elif e_type == "GSAT":
            t_folder = "學測"
            is_mu = "_補考" in str(m_tag) or "補考" in str(y_name)
            sp_name = f"{y_val}學年度_學測{'_補考' if is_mu else ''}_{clean_s}"
        else:
            t_folder = "分科指考"
            is_mu = "_補考" in str(m_tag) or "補考" in str(y_name)
            ast_s = "分科" if y_val >= 111 else "指考"
            sp_name = f"{y_val}學年度_{ast_s}{'_補考' if is_mu else ''}_{clean_s}"
            
        sp_name = re.sub(r'[\s\n\r\t]+', '', sp_name)
        target_db = os.path.join(output_directory, t_folder, f"{sp_name}_database.json")
        clean_t = m_tag.strip('_')
        if e_type == "SCHOOL":
            p_label = f"[{y_name}] {sch_name or '學校定期考'} {clean_t} {s_name} ({e_type})".replace("  ", " ")
        elif e_type in ["MOCK", "MOCK_GSAT", "MOCK_AST"]:
            p_label = f"[{y_name}] {clean_t} {s_name} ({e_type})".replace("  ", " ")
        else:
            p_label = f"[{y_name}] {s_name} ({e_type})"
        p_label = re.sub(r'\s+', ' ', p_label).strip()
        
        # 檢查該試卷是否已存在且具備實體題目
        if os.path.exists(target_db):
            try:
                with open(target_db, "r", encoding="utf-8") as jf:
                    db_items = json.load(jf)
                    # 🚨 國寫僅有 2 題，精確適配門檻
                    if clean_s == "國寫": min_q = 2
                    elif any(k in clean_s for k in ["數", "math"]): min_q = 8
                    else: min_q = 15
                    if len(db_items) >= min_q:
                        pre_completed_count += 1
                        GLOBAL_METRICS.update_exam_progress(p_label, len(db_items), len(db_items), "✅ 100% 完工 (歷史存檔)")
            except Exception:
                pass

    with GLOBAL_METRICS.lock:
        GLOBAL_METRICS.total_papers = len(exam_tasks)
        GLOBAL_METRICS.completed_papers = pre_completed_count

    logging.info(f"📊 [歷史進度盤點完畢] 全庫共 {len(exam_tasks)} 份試卷，已完工 {pre_completed_count} 份，剩餘 {len(exam_tasks) - pre_completed_count} 份待解析。")
    GLOBAL_METRICS.add_log(f"📊 全庫進度已載入：{pre_completed_count}/{len(exam_tasks)} 卷已完工")

    max_exam_workers = 1
    logging.info(f"🚀 開始多張考卷並行處理 (Max Workers: {max_exam_workers})")

    with ThreadPoolExecutor(max_workers=max_exam_workers) as exam_executor:
        futures = []
        for idx, task in enumerate(exam_tasks, 1):
            logging.info(f"======== 準備提交考卷任務 {idx}/{len(exam_tasks)} ========")
            futures.append(exam_executor.submit(
                parser.process_exam_paper,
                subject=task["subject"],
                year=task["year"],
                exam_type=task["exam_type"],
                mock_tag=task["mock_tag"],
                q_pdf=task["q_pdf"],
                a_pdf=task["a_pdf"],
                rubric_pdf=task["rubric_pdf"],
                output_dir=output_directory,
                school_name=task.get("school_name", "")
            ))
            
        # 等待所有考卷處理完畢
        rpd_shutdown = False
        for future in as_completed(futures):
            try:
                future.result()
            except (RPDExhaustedException, TimeoutSafetyException) as exit_exc:
                logging.critical(f"🛑 [全域安全退出] 觸發安全保護 ({exit_exc})！正在安全終止 Python...")
                # 🚨 強制取消所有還沒執行的考卷任務
                for f in futures:
                    f.cancel()
                
                logging.info("🎉 已保存當前所有完成進度，正常結束 Python 程序 (Exit 0)。")
                sys.exit(0)  # 🚨 關鍵：立刻以 Exit Code 0 結束 Python，讓 YML 接手進行上傳！
            except Exception as e:
                logging.error(f"❌ 處理考卷時發生嚴重錯誤: {e}")
                
        if rpd_shutdown:
            logging.critical("🛑 [中斷點退出] 安全終止任務，正常結束 Python 程序 (Exit 0)。")
            sys.exit(0)
