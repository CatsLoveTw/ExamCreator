import os
import re
import sys
import json
import time
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
import queue
import uuid
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        '简': '簡', '两': '兩', '个': '個', '实': '實', '根': '根', '筛': '篩', '单': '單', '西': '西',
        '峰': '峰', '一': '一', '维': '維', '据': '據', '析': '析', '相': '相', '次': '次', '累': '累',
        '加': '加', '判': '判', '读': '讀', '均': '均', '位': '位', '众': '眾', '散': '散', '布': '布',
        '强': '強', '弱': '弱', '联': '聯', '過': '過', '濾': '濾', '處': '處', '理': '理', '資': '資',
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
        '标': '純', '宏': '巨', '观': '觀'
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
    """用於題號自然排序（數值大小），防範字元排序 Bug（如 '10' 排在 '2' 前面）"""
    import re
    parts = re.split(r'(\d+)', str(s))
    return [int(text) if text.isdigit() else text.lower() for text in parts]

def clean_ocr_answer_format(ans_str: str) -> str:
    """清理多欄解答排版干擾下的格式污染與中文字洩漏，並排除手寫題的斜線與無效字元"""
    import re
    ans_str = str(ans_str).strip()
    
    # 🚨 核心修正：若包含大考官方的「無答案」、「送分」、「全體給分」、「不計分」等字樣，完整保留並標準化，避免被後續的正則剝除成無意義的殘留數字
    if any(k in ans_str for k in ["無答案", "全體給分", "送分", "不計分"]):
        return "無答案（全體給分）"
        
    # 🚨 修正：若答案卷上標示為斜線「／」、「/」、「\」或空白，代表該非選擇題無劃記答案，直接過濾為空字串，防止污染
    if ans_str in ["/", "／", "\\", "無", "無題目", ""] or re.match(r'^[\s/／\\–\-_]+$', ans_str):
        return "" 
    
    # 1. 偵測並處理中文說明的拼接格式，例如 "14題答案為2，15題答案為6" 或 "第14列為2，第15列為6"
    pattern = r'(?:第)?\s*(\d+)\s*(?:題|列)?\s*(?:答案)?\s*(?:為|是|:)?\s*([A-Ga-g0-9\-±])'
    matches = re.findall(pattern, ans_str)
    if matches:
        matches_sorted = sorted(matches, key=lambda x: int(x[0]))
        return ",".join(m[1] for m in matches_sorted)
        
    # 2. 如果非中文拼接格式，則移除所有中文字元與不合規字元
    cleaned = re.sub(r'[\u4e00-\u9fa5]+', '', ans_str)
    cleaned = re.sub(r'[^\w,\-±]', '', cleaned)
    return cleaned.strip(",")

def deduplicate_questions(questions: list) -> list:
    if not questions:
        return []
        
    def get_text_similarity(s1, s2):
        def clean_set(s):
            return set(re.sub(r'[^\w\u4e00-\u9fa5]', '', str(s)))
        set1 = clean_set(s1)
        set2 = clean_set(s2)  # 修正：移除原本的 set2 = clean_text = clean_set(s2) typo
        if not set1 or not set2:
            return 0.0
        return len(set1.intersection(set2)) / float(len(set1.union(set2)))

    deduped = []
    for q in questions:
        is_duplicate = False
        q_text = q.get("question_text", "")
        q_num = str(q.get("question_number", "")).strip()
        
        for existing in deduped:
            ext_text = existing.get("question_text", "")
            ext_num = str(existing.get("question_number", "")).strip()
            
            # 修正：只有當「題號完全相同」時，才允許進行文本相似度去重
            if q_num == ext_num:
                similarity = get_text_similarity(q_text, ext_text)
                if similarity > 0.85:
                    is_duplicate = True
                    logging.warning(f"🔄 [跨批次去重] 偵測到重複擷取！題號 {q_num} 與已存在題號相同且內容相似，自動執行去重合併。")
                    if len(q_text) > len(ext_text):
                        existing.update(q)
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
    [防禦性機制 - 提案二：選填題/非選題格式自動化預校驗器]
    """
    # 0. 自動就地修正 LaTeX 中 \\text{} 包裹數學公式等瑕疵，避免觸發 validator 的審查退回
    fixed_sol = recursive_fix_latex(sol_data)
    if isinstance(fixed_sol, dict):
        for k, v in fixed_sol.items():
            sol_data[k] = v

    # 1. 修正：當目前題目並非「選擇與非選並存」的混合題，且為非選擇題型時，清空 options_analysis 避免格式混淆
    is_hybrid = len(q_data.get("options", [])) > 0 and len(q_data.get("scoring_criteria", "")) > 0
    if q_data.get("question_type") in ["選填題", "簡答題", "繪圖作圖題"] and not is_hybrid:
        sol_data["options_analysis"] = []
        
    # 2. 自動修正選填題答案格式，若無半形逗號則依記挖空數量強制拆分
    if q_data.get("question_type") == "選填題" and "," not in str(q_data.get("answer", "")):
        raw_ans = str(q_data.get("answer", "")).strip()
        expected_count = q_data.get("_expected_blank_count", 0)
        if expected_count > 0 and len(raw_ans) == expected_count:
            q_data["answer"] = ",".join(list(raw_ans))

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

PROMPT_STAGE_2_MAIN = """
你是一位在台灣大考講義編撰領域享有崇高聲譽、解題思路極具啟發性的高中學科補教名師。請針對上述每一道題目，為我們撰寫極致詳細、邏輯縝密且充滿引導性的詳解。

🚨【語言字體與兩岸名詞絕對剛性規定（致命紅線）】🚨：
1. 全篇必須 100% 使用繁體中文（zh-TW）撰寫，絕對禁止夾雜任何簡體字！
2. 絕對禁止使用大陸學術名詞或用語。例如：必須寫「機率」而非「概率」、寫「向量」而非「矢量」、寫「純量」而非「標量」、寫「解析度」而非「分辨率」、寫「伏特計/安培計」而非「電壓表/電流表」。若違反將遭到系統嚴厲退件！

【一、詳解結構化與寫作規限（極度重要）】
1. **question_analysis (題意分析)**：
    - 精確剖析題目的核心考點與已知條件。
    - **必須**使用 Markdown 的 `**雙星號粗體**` 標示出題幹中最核心的關鍵概念與限制條件（如：**同溫同壓**、**飽和溶液**等）。
2. **solving_strategy (解題思路)**：
    - 詳細引導學生如何從題目已知條件聯想到解題突破點，建立嚴謹的因果推理鏈。
3. **detailed_solution (完整解法與另解)**：
    - 請使用 Markdown 標題區分多種解法，以激發學生的思維創造力，我們強烈要求「一題多解」：
            - `### 【標準解法】`：必須按部就班、寫出最正規且符合課綱的求解過程。涉及計算必須呈現完整的 LaTeX 公式推導，嚴禁直接跳出答案。**【極重要硬性規定】**：`### 【標準解法】` 必須是一個**完整的、能獨立推導或判斷出所有選項（如選項 1 至 5，或 A 至 E）對錯的完整解題路徑**。絕對不允許在標準解法中只判斷選項 1, 2，而把其餘選項的判斷拆分到其他另解中！
            - `### 【另解一：[命名你的思維，例如：座標轉換/幾何投影法]】`：提供第二套完整且嚴謹的推導思維。**如果某個另解因為方法特質限制，只能用來判斷部分選項，你必須在該另解的標題或開頭極其明確地註明：『本解法專用於快速判斷選項 X, Y』。**
            - `### 【另解二：[命名...]】`、`### 【另解三：[命名...]】`：依此類推。
            - `### 【另解 / 秒殺速解：[命名...]】`：提供極度直覺、極富創意或能在大考中快速破題的秒殺方法。
            - 我們期望你**竭盡所能提供 3 到 5 種完全不同的切入角度與解法**，以最豐富且深具啟發性的思維碰撞，展現補教名師的風範。
            - `### 【手寫題評分對照】`：若本題為非選擇題且含有「手寫評分標準」，你必須在標準解法中明確用粗體標示出各給分與扣分點（例如：`**【列式給分點（得1分）】**`、`**【計算與數值給分點（得1分）】**`）。
    - 🚨【選擇題選項對位硬性規定】：對於單選題、多選題、以及有提供選項的選填題，你必須在 `options_analysis` 列表中，為 `options` 中的**每一個**選項（如 A, B, C 或 1, 2, 3，必須與題目給定的選項 key 嚴格對齊）提供一個獨立的分析對象。
    - 每個對象包含 `key`（選項標籤）與 `explanation`（該選項的詳細分析、公式推導或對錯判斷，並在結尾明確指出該選項為『正確』或『錯誤』）。
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
  為了防止 JSON 傳輸與解析時 LaTeX 的反斜線 `\\` 被損毀、遺失或誤判為 JSON 控制字元（如 \\t, \\f, \\n），**你必須在輸出所有 LaTeX 公式時，將所有 LaTeX 中的反斜線 `\\` 替換為特殊的預留安全標記 `__LTXS__`**！
  - 例如：`\\frac{{1}}{{2}}` 必須寫成 `__LTXS__frac{{1}}{{2}}`。
  - `\\theta` 必須寫成 `__LTXS__theta`。
  - `\\rightarrow` 必須寫成 `__LTXS__rightarrow`。
  - `\\text{{...}}` 寫成 `__LTXS__text{{...}}`。
  - `\\begin{{matrix}}` 寫成 `__LTXS__begin{{matrix}}`。
  - 🚨【絕對禁止】在公式中輸出任何真正的單反斜線 `\\`，一律且強制使用 `__LTXS__`！
- 嚴格遵守標準 LaTeX 語法。
- 🚨【指數與冪次 LaTeX 規範】：當表示指數函數、高次冪或含有多個字元的上標（如 2 的 x+1 次方，或 e 的 -x 次方）時，**必須**將整個指數/上標部分用 LaTeX 的大括號 `{{}}` 完整包裹，例如寫成 `$2^{{x+1}}$`、`$e^{{-x}}$`，**絕對禁止**寫成 `$2^x+1$` 或 `$2^-x$`（這會被渲染/解讀為 $2^x + 1$ 或 $2^- \cdot x$，造成嚴重的學術邏輯與網頁渲染錯誤）。在非 LaTeX 純文字語境下，必須使用括號表示，如 `2^(x+1)`。
- 🚨【化學式專屬指令】：所有化學反應式必須使用 LaTeX 格式。
  - 務必使用下標語法，例如：$Mg_{{(s)}} + 2HCl_{{(aq)}} __LTXS__rightarrow MgCl_{{2(aq)}} + H_{{2(g)}}$。
  - 嚴禁使用 ->，必須使用 __LTXS__rightarrow 或 __LTXS__ce{{->}}。
  - 所有離子電荷必須使用上標，例如：$Ca^{{2+}}$。
- 絕對禁止使用 Tab 鍵縮排公式，防止 __LTXS__text 變形損毀。
- 所有變數與數值必須嚴格包裹在 $...$ 或 $$...$$ 之中。

【三、客觀難度評分量表與分析規範 (1-10分)】
為了確保所有科目的難度評分具有高度一致性與客觀性，請你**嚴格**按照以下當前學科的客觀級距給予 `difficulty_level`：
{subject_rubric}

撰寫 `difficulty_reason` 時，**必須**使用以下固定格式來證明你的評分是客觀的：
「本題評為 X 分。
1. 概念跨度：[說明涉及該學科多少個具體章節或理論]
2. 思考/運算負擔：[說明推導步驟多寡、圖表複雜度或計算量]
3. 陷阱與干擾：[說明干擾選項的強度或隱藏條件]」

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
           
【六、學科分類與解題技巧命名剛性規範（🚨最高優先級：違反將導致系統崩潰🚨）】

「🚨【視覺細節注意】：圖片已採用 300 DPI 高解析度裁切。請仔細辨識題目中的數學上標、下標、化學價電荷、以及圖表座標軸的小字，嚴禁看錯數值。」

🚨【知識庫最大化重用與防分叉剛性規定（非常重要）】：
1. 下方提供的【當前學科現有可用題型分類清單】和【當前學科現有可用解題技巧清單】是系統長期累積、沉澱的黃金標準知識庫。
2. **【嚴格禁止隨意發明新名稱】**：在你決定「自建考點」或「自建技巧」前，你必須在內部進行一次「語意映射檢索」。只要現有清單中，有任何一項在「學科觀念」或「解題方法」上與本題高度重合或相似，**你必須 100% 強制使用現有的名稱，絕對不能因為字眼稍微不同就另立新名（這會導致知識點分叉損毀）！**
3. **【合併優於拆分】**：即使現有清單的名稱比你學術上想的稍微寬泛（例如現有 `必修_排列組合_計數原理`，而你想寫 `必修_排列組合_計數原理_加法原理應用`），也請**優先選擇現有的寬泛名稱**，而不是隨意創建一項只多出兩三個字的新考點！
4. 只有在現有清單與本題考點**完全風馬牛不相及（例如：數學題考到化學，或者現有清單完全為空）**時，你才被允許「自建考點」，且自建考點必須嚴格遵循相同層級命名規範。

1. **題型分類 (topic_category) 命名規範**：
    - **首字強制約束**：此欄位輸出的字串**必須且只能**以 `必修_` 或 `選修_` 作為開頭（例如：`選修_化學平衡_平衡計算`、`必修_形音義_字形辨正_形近易混淆字辨析`）。
    - **階層化結構**：必須嚴格遵循 `必修/選修_大單元_次單元_具體觀念` 的底線（`_`）連結格式。
    - 🚨 **【底線 (_) 使用嚴格限制】**：底線 `_` **僅限於**作為「階層（大/次單元）」之間的**分隔符號**！若是同階層中並列的詞彙，或是完整的句子、片語，**絕對禁止**使用底線 `_` 分隔！
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

🚨【審查教授特別注意：容忍 OCR 解答錯位】🚨
大考中心官方答案有時會因雙欄排版或橫排導致本系統的 OCR 抓錯格子（例如將第 22 題看錯行）。
- 若你發現 AI 的學術推導**物理邏輯或代數推導完全正確**，但它最終算出來的答案卻與「官方答案」不符，請你務必力挺 AI，判定 `is_valid = true`！
- 同時，如果發現官方答案存在嚴重的 OCR 解析錯位（例如解答卷上該格分明是斜線或空白，卻被誤讀為數字），請將該題的 `suspects_ocr_error` 設為 true，啟動影像重新掃描機制。

【審查標準（若有以下任一項，is_valid 必為 false）】：
1. **無中生有**：AI 沒有去解題目的「共同資訊」或方程式，自己憑空捏造數字。
2. **邏輯斷層硬湊**：AI 算到一半算不出結果，突然神來一筆。
3. **知識性錯誤與計算失誤**：AI 的代數推導、英文文法、化學結構有明顯硬傷。
4. **放寬結論句限制**：若 AI 的『過程分析』已經 100% 正確，僅僅是最後沒有總結字眼，請判定為 is_valid = true。
🚨 5. **持續重掃描判定（連續報警）**：若你發現雖然上一輪重掃描更新了數據，但詳解中依然存在嚴重的化學式/數學結構矛盾（這代表上一次重掃描依然沒有看清楚、或者重掃描也看錯了），你【必須繼續將 suspects_ocr_error 設為 true】！引導系統進行更精確的二次或三次重掃描比對，絕對不要輕易放棄！
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


class PageExtraction(BaseModel):
    questions: List[ExtractedQuestion] = Field(description="這頁 PDF 中的所有題目列表。")

class OptionAnalysisItem(BaseModel):
    key: str = Field(description="選項標籤。必須且只能是一個大寫英文字母（如 'A', 'B', 'C', 'D', 'E'）或數字，必須與題目 options 列表中的 key 完全對齊。")
    explanation: str = Field(description="針對該選項的專屬分析與對錯判斷。必須包含公式推導、數值代入或文意對照，並在結尾明確指出該選項本身是『正確』或『錯誤』。")

# === 第二階段：大腦層 (Temperature = 0.7) ===
class QuestionSolution(BaseModel):
    suspects_image_mismatch: bool = Field(default=False, description="【圖片精確性確認】若你發現裁剪出來的圖片與本題題號不符（例如圖中印的題號不是本題題號），或者圖片模糊不清、公式嚴重錯位、表格被切掉時，請設為 true，系統將自動啟動重新掃描與補件機制。")
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
    topic_category: str = Field(description="題型分類。優先使用清單，無則自行發明。")
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
    
class QuestionSolutionBatch(BaseModel):
    solutions: List[QuestionSolution] = Field(description="批次產出的多題詳細解答列表，順序必須與輸入題目嚴格對齊。")

class SolutionValidatorBatch(BaseModel):
    validators: List[SolutionValidator] = Field(description="批次審查多題的驗證結果，順序必須與輸入解答嚴格對齊。")

# ==========================================
# 2. Gemini API 輪詢與排程管理器
# ==========================================
class FreeTierKey:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=900_000))
        self.rpd_exhausted = False
        self.request_times = []  # 記錄每次請求的 UNIX 時間戳

        # 本地限制追蹤 (此處預設為 Gemini 3.1-Flash-lite 等免費配額，可依需求自行微調)
        self.limit_rpm = 15
        self.limit_tpm = 250_000
        self.limit_rpd = 500
        
        self.is_disabled = False # 🚨 新增：是否被永久停用（如 401, 403 權限錯誤）
        self.disable_reason = "" # 🚨 新增：停用原因說明
        
        # 滑動視窗計數器
        self.request_times = []  # 記錄 RPM (60秒內的請求時間戳記)
        self.token_usage = []    # 記錄 TPM (60秒內的 token 使用，格式：(時間戳記, Token數))
        self.daily_request_count = 0

    def get_status(self) -> dict:
        """回傳當前 Key 的 RPD/RPM/TPM 統計狀態"""
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t <= 60.5]
        self.token_usage = [item for item in self.token_usage if current_time - item[0] <= 60.5]
        
        # 若已標記耗盡且時間已過重置點，則重新恢復
        if self.rpd_exhausted and current_time >= self.rpd_reset_time:
            self.rpd_exhausted = False
            self.daily_request_count = 0
            
        current_rpm = len(self.request_times)
        current_tpm = sum(item[1] for item in self.token_usage)
        
        return {
            "key": f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else self.api_key,
            "rpm_status": f"{current_rpm}/{self.limit_rpm}",
            "tpm_status": f"{current_tpm}/{self.limit_tpm}",
            "rpd_status": f"{self.daily_request_count}/{self.limit_rpd}" + (" (已耗盡，等待重置)" if self.rpd_exhausted else ""),
        }

    def can_request(self, current_time: float, estimated_tokens: int) -> bool:
        if self.rpd_exhausted:
            if current_time < self.rpd_reset_time:
                return False
            else:
                self.rpd_exhausted = False
                self.daily_request_count = 0
                
        # 進行時間滑動視窗清理
        self.request_times = [t for t in self.request_times if current_time - t <= 60.5]
        self.token_usage = [item for item in self.token_usage if current_time - item[0] <= 60.5]
        
        # 檢查 RPM 與 TPM 限制
        if len(self.request_times) >= self.limit_rpm:
            return False
        current_tpm = sum(item[1] for item in self.token_usage)
        if current_tpm + estimated_tokens > self.limit_tpm:
            return False
            
        return True

    def add_request(self, current_time: float, estimated_tokens: int):
        self.request_times.append(current_time)
        self.token_usage.append((current_time, estimated_tokens))
        self.daily_request_count += 1
        if self.daily_request_count >= self.limit_rpd:
            self.mark_rpd_exhausted()

    def mark_rpd_exhausted(self):
        self.rpd_exhausted = True
        self.rpd_reset_time = get_next_rpd_reset_timestamp()

class GeminiFreeTierManager:
    def __init__(self, api_keys: List[str], models: List[str]):
        self.keys = [FreeTierKey(k) for k in api_keys]
        self.models = models
        self.model_idx = 0
        self.lock = threading.Lock()
        self.last_model_used = models[0] if models else ""  # 🚨 新增：追蹤上一次呼叫成功的模型
        
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
        僅排除標準 JSON 允許的轉義字元（如 \\n, \\t, \\r, \\b, \\f, \\"）。
        這能徹底避免 \\le 變形損毀或 \\pm 被暴力轉義成 \\neq 的問題。
        """
        if not raw_text:
            return raw_text
            
        # 1. 暫時保護已有的雙反斜線
        raw_text = raw_text.replace("\\\\", "__DBL_SLASH__")
        
        # 2. 定義轉義邏輯
        def replace_backslash(match):
            full_match = match.group(0)
            # 如果是標準的 6 字元 Unicode 轉義字元，直接完整保留，不作轉義
            if full_match.startswith('\\u') and len(full_match) == 6:
                return full_match
            
            char = match.group(1) if match.group(1) else match.group(2)
            if char in ['b', 'f', 'n', 'r', 't']:
                return match.group(0)
            return "\\\\" + char

        # 優先匹配 \\uXXXX，其次匹配單個 \\(.)
        raw_text = re.sub(r'(\\u[0-9a-fA-F]{4})|\\(.)', replace_backslash, raw_text)
        
        # 3. 還原保護的雙反斜線
        raw_text = raw_text.replace("__DBL_SLASH__", "\\\\")
        return raw_text
    
    def repair_latex_control_chars(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        # 還原 LaTeX 安全預留字串
        text = text.replace("__LTXS__", "\\")
        # 🚨 核心修復：將 JSON 解析時被誤判為控制字元的 LaTeX 指令（如 \f, \r, \b, \t）還原為正常 LaTeX 反斜線
        replacements = {
            "\x0c": r"\f",  # 修正 \frac, \forall 被誤轉為 form feed (\x0c) 的問題
            "\x0d": r"\r",  # 修正 \rightarrow, \right 被誤轉為 carriage return (\x0d) 的問題
            "\x08": r"\b",  # 修正 \big, \beta 被誤轉為 backspace (\x08) 的問題
            "\x09": r"\t",  # 修正 \text, \theta 被誤轉為 tab (\x09) 的問題
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
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
                    tokens += 258  # 預留一張圖所需的 Token (Gemini 讀圖基本消秏)
        elif isinstance(contents, str):
            tokens += len(contents) // 2
        return max(100, tokens)

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

    def get_current_resource(self, preferred_model: Optional[str] = None, estimated_tokens: int = 1000):
        while True:
            sleep_time = 0
            with self.lock:
                current_time = time.time()
                chosen_model = self.models[self.model_idx]
                if preferred_model and preferred_model in self.models:
                    chosen_model = preferred_model
                    
                # 🚨 過濾出當前每日限額尚未用盡，且未被永久停用的有效金鑰
                active_keys = [k for k in self.keys if not k.rpd_exhausted and not k.is_disabled]
                
                # 🚨 狀況 0：主動防禦空金鑰檔案引發的非預期崩潰
                if not self.keys:
                    logging.critical("🚨 [系統致命錯誤] 找不到任何學術金鑰！請在 key.txt 中配置您的 Gemini API 金鑰後重新執行。")
                    os._exit(1)
                if not active_keys and all(k.is_disabled for k in self.keys):
                    logging.critical("🚨 [系統致命錯誤] 偵測到所有已載入的 API 金鑰皆已因 401/403 權限錯誤被永久停用！請檢查 key_errors_summary.txt。程式將中斷執行...")
                    os._exit(1) # 立即退出避免無限等待

                # 狀況 A：如果所有未停用的 Key 的每日配額皆已用盡
                if not active_keys:
                    next_reset = get_next_rpd_reset_timestamp()
                    sleep_time = max(10.0, next_reset - current_time)
                    sleep_time = min(sleep_time, 300.0)  # 每次最多休眠 5 分鐘後重新檢查，以利輸出進度
                    logging.warning(f"⚠️ [每日限額用盡] 偵測到所有 API 金鑰皆已達 RPD 上限。")
                    logging.warning(f"⏳ 系統將自動進入深度休眠 {sleep_time:.1f} 秒，等待每日額度重置（預計重置：台北時間下午 3:00）...")
                else:
                    # 狀況 B：檢查是否有符合當前 RPM/TPM 限額的可用 Key
                    for key in active_keys:
                        if key.can_request(current_time, estimated_tokens):
                            key.add_request(current_time, estimated_tokens)
                            res_model = chosen_model
                            self.model_idx = (self.model_idx + 1) % len(self.models)
                            return key.client, res_model, key
                    
                    # 狀況 C：所有 active key 都在冷卻中，計算最快能空出 RPM 額度的時間
                    active_request_times = [k.request_times[0] + 60.5 for k in active_keys if k.request_times]
                    if active_request_times:
                        earliest_wakeup = min(active_request_times)
                        sleep_time = max(0.5, earliest_wakeup - current_time)
                    else:
                        sleep_time = 1.0
                    logging.info(f"⏳ [限速調度] 所有可用金鑰皆在冷卻中。等待 {sleep_time:.1f} 秒...")
                    
            time.sleep(sleep_time)

    def handle_rate_limit_error(self, key: FreeTierKey, error_msg: str):
        with self.lock:
            msg = error_msg.lower()
            if "perday" in msg or "quota" in msg or "daily" in msg or "free_tier_requests" in msg:
                key.mark_rpd_exhausted()
                logging.warning(f"[額度報銷] 金鑰 {key.api_key[:8]}... 已達每日上限，已轉移至背景冷卻。")
            else:
                # 突發型 429 則加入冷卻隊伍
                key.request_times.extend([time.time()] * 3)

    def generate_with_retry(self, contents, response_schema, temperature=0.2, max_attempts=5, preferred_model: Optional[str] = None, enable_thinking: bool = True, task_desc: str = ""):
        estimated_tokens = self.estimate_tokens(contents)
        attempts = 0
        while attempts < max_attempts:
            # 取得可用資源（內建調度等待，保證不會回傳 None）
            client, model, key_obj = self.get_current_resource(preferred_model=preferred_model, estimated_tokens=estimated_tokens)
            self.last_model_used = model
            
            thinking_config = None
            if any(m in model for m in ["gemini-3.5", "gemini-2.5", "gemini-3"]):
                if enable_thinking:
                    thinking_config = types.ThinkingConfig(thinking_level='HIGH')
                else:
                    thinking_config = types.ThinkingConfig(thinking_level="MINIMAL")
                
            try:
                # 🚨 核心修改：引入隨機平滑抖動（Jitter）
                # 在每個線程發送請求前，隨機微調等待 0.2 到 0.7 秒。
                # 這能有效打破多線程高度同步的發送特徵（不會同時射出幾十個 API 請求），避開 Google 安全防護的自動化指紋偵測
                import random
                time.sleep(random.uniform(1.8, 6))

                desc_str = f" {task_desc}" if task_desc else ""
                print(f"🔹{desc_str} 嘗試使用模型 {model} 呼叫 API，思考: {thinking_config.thinking_level if thinking_config else '關閉'} / 溫度: {temperature} (第 {attempts + 1} 次嘗試)...", flush=True)
                # 每次執行時顯示金鑰狀態
                self.print_keys_status()

                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        max_output_tokens=65535,
                        response_schema=response_schema,
                        temperature=temperature,
                        thinking_config=thinking_config
                    ),
                )

                if not response:
                    logging.warning("⚠️ API 回傳完全為空 (None)，可能發生網路瞬斷，準備重試...")
                    attempts += 1
                    time.sleep(2)
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
                if not raw_text or raw_text.strip() == "":
                    logging.warning("⚠️ API 回傳了空字串，準備重試...")
                    attempts += 1
                    continue

                try:
                    import re
                    raw_text = response.text
                    
                    # 🚨 1. 修正 LaTeX 結尾反斜線誤轉義 JSON 邊界引號的問題
                    raw_text = re.sub(r'\\"(?=\s*[,}\]])', r'\\\\"', raw_text)
                    
                    # 🚨 2. 優先將 LaTeX 指令的單反斜線轉義（防止被誤讀為 JSON 換行符）
                    raw_text = self.escape_latex_backslashes(raw_text)
                    
                    # 🚨 3. 修正字串內不慎夾帶的「實體換行符」
                    def escape_literal_newlines(match):
                        return match.group(0).replace("\n", "\\n").replace("\r", "\\r")
                    raw_text = re.sub(r'"(?:[^"\\]|\\.)*"', escape_literal_newlines, raw_text, flags=re.DOTALL)
                    
                    # 🚨 3.5 [終極記憶體層淨化]：在字串階段直接進行全局繁簡轉換，徹底阻絕簡體字物件化！
                    # OpenCC 只會替換中文字元，完美避開 JSON 結構與英數 LaTeX 符號
                    if hasattr(self, '_convert_to_t'):
                        pass # 避免重複宣告
                    raw_text = s2t(raw_text)
                    
                    # 🚨 3.6 兩岸學術名詞精確性修正 (記憶體層級暴力替換)
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

                    # 🚨 4. 安全地執行一次解析
                    parsed_json = json.loads(raw_text)
                    
                    # 🚨 5. 還原控制字元
                    parsed_json = self.repair_dict_latex(parsed_json)
                    return parsed_json, None
                except json.JSONDecodeError as je:
                    raw_text = response.text
                    logging.warning(f"⚠️ [JSON 截斷] 偵測到模型輸出被截斷 ({je})。啟動緊急救援機制 (Salvage Mode)...")
                    
                    extracted_objects = []
                    depth = 0
                    obj_start = -1
                    in_string = False
                    escape = False
                    
                    for i, char in enumerate(raw_text):
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
                                if depth == 1: 
                                    obj_start = i
                                depth += 1
                            elif char == '}':
                                depth -= 1
                                if depth == 1 and obj_start != -1:
                                    try:
                                        obj_str = raw_text[obj_start:i+1]
                                        obj = json.loads(obj_str, strict=False)
                                        if any(k in obj for k in ["detailed_solution", "is_valid", "question_text"]):
                                            extracted_objects.append(obj)
                                    except:
                                        pass
                    
                    if extracted_objects:
                        logging.info(f"🦸‍♂️ [救援成功] 成功從損毀的 JSON 中救回 {len(extracted_objects)} 題的完整數據！")
                        if "solutions" in str(response_schema):
                            return {"solutions": extracted_objects}, "partial_success"
                        elif "validators" in str(response_schema):
                            return {"validators": extracted_objects}, "partial_success"
                    
                    return None, "json_decode_error"
                
            except APIError as e:
                err_str = str(e).lower()
                if e.code == 503 or "unavailable" in err_str or e.code == 504 or "gateway timeout" in err_str:
                    logging.warning(f"⚠️ [503 伺服器忙碌] 遇到臨時性服務過載，將於 3 秒後自動重試（不計入失敗次數）...")
                    time.sleep(3)
                    continue
                
                # 🚨 新增：401 Unauthorized 錯誤處理（無效/過期金鑰）
                if e.code == 401 or "unauthorized" in err_str or "api key not valid" in err_str:
                    with self.lock:
                        key_obj.is_disabled = True
                        key_obj.disable_reason = "401_Unauthorized"
                    self.log_key_error(key_obj.api_key, 401, "金鑰無效、過期或拼寫錯誤")
                    logging.warning(f"⚠️ [401 金鑰無效] 金鑰 {key_obj.api_key[:8]}... 被判定為無效，已將其永久停用，並記錄至 key_errors_summary.txt。")
                    # 不計入當前內容的嘗試次數，直接換下一組金鑰重試
                    continue

                # 🚨 新增：403 Forbidden 錯誤處理（地區拒絕、未開通 API 或帳單欠費）
                if e.code == 403 or "forbidden" in err_str or "permission denied" in err_str or "restricted" in err_str:
                    with self.lock:
                        key_obj.is_disabled = True
                        key_obj.disable_reason = "403_Forbidden"
                    self.log_key_error(key_obj.api_key, 403, "存取被拒(未開通API、地區限制或帳單餘額欠費)")
                    logging.warning(f"⚠️ [403 權限拒絕] 金鑰 {key_obj.api_key[:8]}... 存取被拒，已將其永久停用，並記錄至 key_errors_summary.txt。")
                    # 不計入當前內容的嘗試次數，直接換下一組金鑰重試
                    continue
                
                attempts += 1
                if e.code == 429 or "quota" in err_str or "exhausted" in err_str:
                    self.handle_rate_limit_error(key_obj, err_str)
                else:
                    time.sleep(2)
            except Exception as e:
                attempts += 1
                err_msg = str(e).lower()
                if "timeout" in err_msg or "time out" in err_msg:
                    logging.warning(f"⚠️ [API 呼叫超時] (嘗試第 {attempts} 次): {e}。進行金鑰降溫與平滑重試...")
                    with self.lock:
                        key_obj.request_times.extend([time.time()] * 5)
                else:
                    logging.exception(f"發生未預期錯誤（嘗試第 {attempts} 次）: {e}")
                time.sleep(2)
        return None, "請求失敗"

# ==========================================
# 3. 核心處理類別 (PDF 裁切、AI 呼叫)
# ==========================================


SUBJECT_FILE_LOCK = threading.Lock() # 在檔案頂端定義
def update_subject_taxonomy(normalized_subject: str, solution_data: dict):
    """動態比對 AI 新增的考點與解題技巧，即時更新並覆寫至 subject.json"""
    global SUBJECT_TAXONOMY
    with SUBJECT_FILE_LOCK: # 確保同一時間只有一個人能改檔案
        updated = False
        
        # 🚨 確保科目名稱本身為繁體
        normalized_subject = s2t(normalized_subject)
        
        if normalized_subject not in SUBJECT_TAXONOMY:
            SUBJECT_TAXONOMY[normalized_subject] = {"topics": [], "techniques": []}
            updated = True
            
        # 1. 檢查並更新主題 (Topic)
        new_topic = solution_data.get("topic_category")
        if new_topic:
            new_topic = s2t(new_topic) # 🚨 確保轉為繁體
            if new_topic not in SUBJECT_TAXONOMY[normalized_subject]["topics"]:
                SUBJECT_TAXONOMY[normalized_subject]["topics"].append(new_topic)
                logging.info(f"🆕 [動態考點] 已自動新增考點至 subject.json: {new_topic}")
                updated = True
            
        # 2. 檢查並更新解題技巧 (Technique)
        new_techs = solution_data.get("techniques_used", [])
        for tech in new_techs:
            if tech:
                tech = s2t(tech) # 🚨 確保轉為繁體
                if tech not in SUBJECT_TAXONOMY[normalized_subject]["techniques"]:
                    SUBJECT_TAXONOMY[normalized_subject]["techniques"].append(tech)
                    logging.info(f"🆕 [動態技巧] 已自動新增解題技巧至 subject.json: {tech}")
                    updated = True
                
        # 3. 存回檔案前，進行一次全局規範化合併與去重
        if updated:
            try:
                SUBJECT_TAXONOMY = normalize_and_merge_subject_taxonomy(SUBJECT_TAXONOMY)
                with open("subject.json", "w", encoding="utf-8") as f:
                    json.dump(SUBJECT_TAXONOMY, f, ensure_ascii=False, indent=4)
            except Exception as e:
                logging.error(f"無法寫入更新的 subject.json: {e}")

def safe_filename(name: str) -> str:
    """過濾掉 Windows 檔案系統不允許的特殊字元，確保存檔安全"""
    return re.sub(r'[\\/*?:"<>|]', "_", name)
    
class ExamParser:
    # 🚨 將提示詞作為類別屬性（縮排 4 格），確保 self.ANSWERS_OCR_PROMPT 的呼叫完全合規
    ANSWERS_OCR_PROMPT = """這是一份台灣大考的解答卷（選擇與選填題答案表）圖片。
    請你扮演最嚴謹、零失誤的官方數據核對專家，將「題號/列號」與「標準答案」精確萃取為 JSON 列表。

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
    1. **單選與多選題**：答案必須為純英文字母或數字。如果是多選，請將所有字母相連（如 "134" 或 "ACD"），嚴禁加逗號。
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

    def get_initial_batch_size(self, subject: str) -> int:
        """依照學科思考複雜度設定最佳批次大小"""
        sub = subject.lower()
        if any(x in sub for x in ["數學", "數a", "數b", "數甲", "數乙"]):
            return 1  # 數學計算極複雜，批次 1 題
        elif any(x in sub for x in ["物理", "化學", "生物", "地球科學", "自然"]):
            return 1  # 自然科涉及多圖表與實驗，批次 1 題
        else:
            return 2  # 社會文科，批次 2 題
    
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
                        preferred_model="gemini-3.1-flash-lite",
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
        max_workers = min(len(temp_images), len(self.ai_manager.keys) * 2, 16)
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
                    new_imgs = self.execute_crop(page, bboxes, img_dir, f"Rubric_{q_num}_p{page_num}")
                    
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

    def execute_crop(self, page, bboxes, img_dir, prefix) -> List[str]:
        saved_paths = []
        # 取得該頁面上所有實體文字的精確坐標 (x0, y0, x1, y1, "word", block_no, line_no, word_no)
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
                clip_pix = page.get_pixmap(clip=rect, dpi=300)
                clip_pix.save(img_filepath)
                if os.path.exists(img_filepath):
                    saved_paths.append(img_filepath)
            except Exception as e:
                logging.error(f"裁切失敗 {prefix}: {e}")
        return saved_paths

    # 請在 ExamParser 類別中，精確替換此方法（注意縮排為 8 個空格）：
    # 🚨 將解析答案卷的提示詞抽離為類別變數，以利多個模型共用，保持代碼簡潔
    
    def _run_single_ocr(self, a_pdf: str, model: str) -> dict:
        """為共識機制設計的單次獨立 OCR 執行器"""
        try:
            doc = fitz.open(a_pdf)
            cat = doc.pdf_catalog()
            if cat > 0: doc.xref_set_key(cat, "StructTreeRoot", "null")
        except Exception as e:
            logging.error(f"無法開啟解答 PDF {a_pdf}: {e}")
            return {}
            
        all_answers = {}
        task_id = uuid.uuid4().hex[:8]
        
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            img_path = os.path.abspath(f"temp_ans_{task_id}_p{page_num}_{model[:8]}.png")
            pix.save(img_path)

            if not os.path.exists(img_path):
                time.sleep(0.5) 
                if not os.path.exists(img_path):
                    continue
            
            # 【優化機制 1】提取該頁的 PDF 數位文字層（文字對照組）
            page_text_layer = page.get_text("text").strip()
            
            # 【優化機制 3】正則自動提取「預期題號與格位清單」作為提示詞地圖
            import re
            raw_keys = re.findall(r'\b\d+(?:-\d+)?\b|[A-G]\b', page_text_layer)
            hint_keys = sorted(list(set(raw_keys)), key=natural_sort_key)
            
            hint_str = ""
            if hint_keys:
                hint_str = f"\n\n💡【本頁偵測到的預期答案鍵值提示】：{hint_keys}\n請務必以此結構化對照表為地圖，將解析到的答案填入對應的鍵值中，嚴禁跳格或遺漏任何一格。"
            
            text_layer_str = ""
            if page_text_layer:
                text_layer_str = f"\n\n=== 該頁數位文字層對照 ===\n{page_text_layer}"
                
            # 縫合優化後的複合 Prompt
            combined_prompt = self.ANSWERS_OCR_PROMPT + text_layer_str + hint_str + "\n\n請務必結合提供的圖片排版與上述數位文字提示，進行交叉校驗，確保公式與數值完全一致。"
            
            try:
                with Image.open(img_path) as pil_img:
                    from PIL import ImageEnhance
                    enhancer_contrast = ImageEnhance.Contrast(pil_img)
                    pil_img_enhanced = enhancer_contrast.enhance(1.6)  # 提升 60% 對比度
                    enhancer_sharp = ImageEnhance.Sharpness(pil_img_enhanced)
                    pil_img_enhanced = enhancer_sharp.enhance(1.4)      # 提升 40% 銳利度
                    
                    res_dict, err = self.ai_manager.generate_with_retry(
                        contents=[combined_prompt, pil_img_enhanced],
                        response_schema=AnswerKey,
                        temperature=0.0,
                        preferred_model=model,
                        enable_thinking=True
                    )
                    if res_dict and 'answers' in res_dict:
                        for item in res_dict['answers']:
                            raw_ans = item['standard_answer']
                            # 🚨 清理多欄排版干擾下的格式污染與中文字洩漏
                            cleaned_ans = clean_ocr_answer_format(raw_ans)
                            all_answers[item['question_number']] = cleaned_ans
            except Exception as e:
                logging.error(f"模型 {model} 解析解答卷第 {page_num} 頁出錯: {e}")
            finally:
                if os.path.exists(img_path):
                    try: os.remove(img_path)
                    except Exception: pass
        doc.close()
        return all_answers

    def _resolve_ocr_conflict(self, a_pdf: str, dict_1: dict, dict_2: dict, mismatched_keys: list) -> dict:
        """
        對衝突的題號進行二次比對與終極仲裁
        """
        # 預設融合：先以更穩定的 dict_1 (3.5-flash) 為基準
        merged_dict = {**dict_2, **dict_1} 
        
        task_id = uuid.uuid4().hex[:8]
        try:
            # 💡 採用 Pythonic 的 with 語法，防止異常發生時 PDF 檔案描述符發生洩漏
            with fitz.open(a_pdf) as doc:
                pix = doc[0].get_pixmap(dpi=300)
                img_path = os.path.abspath(f"temp_arbitrate_{task_id}.png")
                pix.save(img_path)
        except Exception as e:
            logging.error(f"仲裁時無法讀取解答卷第一頁: {e}")
            return merged_dict

        class ArbitrationResult(BaseModel):
            resolutions: List[AnswerItem] = Field(description="對所有衝突題號的最終仲裁結果列表。")

        prompt = f"""
        這是一份大考解答卷。我們在使用兩個不同的 AI 模型解析以下題號的標準答案時發生了衝突：
        衝突題號：{mismatched_keys}
        模型 1 的解析：{ {k: dict_1.get(k) for k in mismatched_keys} }
        模型 2 的解析：{ {k: dict_2.get(k) for k in mismatched_keys} }

        請你扮演「最終仲裁官」：
        1. 仔細辨識原圖中，這些衝突題號對應的真實標準答案。
        2. 做出最準確的裁決，並將結果填入 Pydantic 結構中。
        """

        try:
            with Image.open(img_path) as pil_img:
                # 提升對比度與銳利度
                from PIL import ImageEnhance
                enhancer_contrast = ImageEnhance.Contrast(pil_img)
                pil_img_enhanced = enhancer_contrast.enhance(1.6)
                enhancer_sharp = ImageEnhance.Sharpness(pil_img_enhanced)
                pil_img_enhanced = enhancer_sharp.enhance(1.4)

                res, err = self.ai_manager.generate_with_retry(
                    contents=[prompt, pil_img_enhanced],
                    response_schema=ArbitrationResult,
                    temperature=0.0,
                    preferred_model="gemini-3.5-flash", # 使用具備最強視覺細節與思考能力的核心模型進行仲裁
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
            if os.path.exists(img_path):
                try: os.remove(img_path)
                except Exception: pass
                
        return merged_dict

    def extract_clean_answers(self, a_pdf: Optional[str]) -> str:
        """
        [方案 3]：雙重答案卷 OCR 投票機制 (Consensus Voting)
        使用兩個不同的模型分別獨立解析答案，並在 Python 中比對。
        若發現不一致，由第 3 個模型進行裁決，確保 100% 精確度。
        """
        if not a_pdf or not os.path.exists(a_pdf):
            return "無官方解答。"

        logging.info(f"🧠 [共識投票] 啟動解答卷雙模型雙重驗證: {os.path.basename(a_pdf)}")
        
        # 1. 第一輪：使用主力模型 gemini-3.5-flash
        ans_dict_1 = self._run_single_ocr(a_pdf, model="gemini-3.5-flash")
        
        # 2. 第二輪：使用輕量模型 gemini-3.1-flash-lite 進行盲測對比
        ans_dict_2 = self._run_single_ocr(a_pdf, model="gemini-3.1-flash-lite")
        
        if not ans_dict_1 and not ans_dict_2:
            return "無官方解答。"
        if not ans_dict_1: return json.dumps(ans_dict_2, ensure_ascii=False)
        if not ans_dict_2: return json.dumps(ans_dict_1, ensure_ascii=False)
        
        # 3. 在 Python 中進行精確的 Key-Value 比對，並在比對前統一空值、None、斜線與未作答標記為 "／"
        def sanitize_answer_value(val) -> str:
            if val is None:
                return "／"
            val_str = str(val).strip()
            if val_str in ["", "None", "null", "none", "Undefined", "undefined", "／", "/", "\\", "無", "無答案", "－", "[]", "{}"]:
                return "／"
            return val_str

        all_keys = set(list(ans_dict_1.keys()) + list(ans_dict_2.keys()))
        for k in all_keys:
            ans_dict_1[k] = sanitize_answer_value(ans_dict_1.get(k))
            ans_dict_2[k] = sanitize_answer_value(ans_dict_2.get(k))

        mismatched_keys = []
        for k in all_keys:
            if ans_dict_1[k] != ans_dict_2[k]:
                mismatched_keys.append(k)
                
        if not mismatched_keys:
            logging.info("🎉 [共識達成] 雙模型對位完全一致！答案卷數據 100% 信賴。")
            return json.dumps(ans_dict_1, ensure_ascii=False)
            
        logging.warning(f"⚠️ [共識衝突] 偵測到以下題號在雙模型解析中不一致: {mismatched_keys}")
        # 先在外面計算好子字典，再放入 f-string 輸出
        ans_1_sub = {k: ans_dict_1.get(k) for k in mismatched_keys}
        ans_2_sub = {k: ans_dict_2.get(k) for k in mismatched_keys}
        logging.warning(f"  -> Model 1 (3.5-flash): {ans_1_sub}")
        logging.warning(f"  -> Model 2 (3.1-lite): {ans_2_sub}")
        
        # 4. 啟動第三輪：終極仲裁
        resolved_dict = self._resolve_ocr_conflict(a_pdf, ans_dict_1, ans_dict_2, mismatched_keys)
        return json.dumps(resolved_dict, ensure_ascii=False)

    def extract_text_from_pdf(self, pdf_path: Optional[str]) -> str:
        """提取文字 (用於讀取解答與評分標準)"""
        if not pdf_path or not os.path.exists(pdf_path):
            return "無官方資料提供。"
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text("text") + "\n"
            doc.close()
        except Exception as e:
            logging.error(f"無法讀取 PDF {pdf_path}：{e}")
        return text
        
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
                preferred_model="gemini-3.5-flash",
                enable_thinking=True,
                task_desc="[終極學術仲裁]"
            )
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
            
            # 1. 【方案 4：硬性題型與選項清洗】
            # 若為非選擇題，硬性將 options 清空，防範大考答案卷數據被當作選項混入
            if q_type in ["選填題", "簡答題", "繪圖作圖題"]:
                if q.get("options"):
                    logging.warning(f"🧹 [自動清洗] 偵測到非選擇題型 {q_num} ({q_type}) 含有不合規的 options，已自動強制清除！")
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

    def process_exam_paper(self, subject: str, year: str, exam_type: str, mock_tag: str, q_pdf: str, a_pdf: Optional[str], rubric_pdf: Optional[str], output_dir: str, skip_cover: bool = False):
        safe_year = safe_filename(year)
        safe_subject = safe_filename(subject)
        paper_tag = f"[{year} {subject}]" # 🚨 新增：本份考卷的唯一日誌與 API 派發識別標籤
        
        subjects_stem = ["數學", "數A", "數B", "數學乙", "數學甲", "數甲", "數乙", "物理", "化學", "生物", "地球科學", "自然"]
        is_stem = any(t in subject for t in subjects_stem)
        
        # 規則：第一階段純掃描、文科詳解一律優先使用 Lite 節省額度；理科解題與審查無條件優先分配最強的 3.5-Flash
        stage_1_model = "gemini-3.1-flash-lite"
        stage_2_model = "gemini-3.1-flash-lite"
        validator_model = "gemini-3.1-flash-lite"
        # stage_2_model = "gemini-3.5-flash" if is_stem else "gemini-3.1-flash-lite"
        # validator_model = "gemini-3.5-flash" if is_stem else "gemini-3.1-flash-lite"
        
        # 1. 決定大類資料夾名稱 (學測 / 分科指考 / 模擬考)
        type_folder = "學測" if exam_type == "GSAT" else ("分科指考" if exam_type == "AST" else "模擬考")
        
        # 2. 決定這份試卷的唯一識別名稱
        if exam_type == "MOCK":
            spec_name = f"{safe_year}_模擬考{mock_tag}_{safe_subject}"
        else:
            spec_name = f"{safe_year}_{type_folder}_{safe_subject}"
            
        # 3. 決定學年度與考卷名稱之 100% 剛性標準化變數，徹底根除跨頁面、跨模型生出不一致考卷名稱與年份之 Bug
        year_digits = "".join(filter(str.isdigit, year))
        if not year_digits:
            year_digits = year
            
        if exam_type == "GSAT":
            standard_academic_year = f"{year_digits}學測"
            standard_exam_source = f"{year_digits}學年度學科能力測驗{subject}"
        elif exam_type == "AST":
            standard_academic_year = f"{year_digits}分科"
            standard_exam_source = f"{year_digits}學年度分科測驗{subject}"
        else:  # MOCK
            standard_academic_year = f"{year_digits}模考"
            standard_exam_source = f"{year_digits}學年度模擬考{mock_tag}_{subject}"

        # 4. 建立專屬的分類資料夾與 JSON 儲存路徑
        os.makedirs(os.path.join(output_dir, type_folder), exist_ok=True)
        json_path = os.path.join(output_dir, type_folder, f"{spec_name}_database.json")
        
        if os.path.exists(json_path):
            # 🚨 核心優化：在跳過已存在的 Database 之前，先加載並檢查該檔案中是否含有簡體字，若有則自動轉為繁體並覆寫
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                new_data = s2t_recursive(old_data)
                if old_data != new_data:
                    logging.info(f"🔄 [歷史資料庫轉換] 偵測到已存在資料庫 {json_path} 中含有簡體字，已自動對其進行繁體標準化轉換並覆寫！")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(new_data, f, ensure_ascii=False, indent=4)
            except Exception as e:
                logging.error(f"讀取、修復或轉換現有資料庫 {json_path} 時發生未預期錯誤: {e}")
                
            logging.info(f"⏭️  {json_path} 已存在，自動跳過。")
            return
            
        # 4. 建立這份試卷專屬的圖片資料夾
        img_dir = os.path.join(output_dir, type_folder, "images", spec_name)
        os.makedirs(img_dir, exist_ok=True)

        # 5. 一次性將原卷、解答、評分原則 PDF 轉換成高解析度整頁圖片
        logging.info("📸 正在將原卷、解答、評分原則 PDF 轉換成黃金 300 DPI 整頁圖片...")
        # 🚨 [核心優化] 全局提升為 300 DPI，確保微小的對數底數、上標、負號與緊湊解答表格不發生像素模糊，杜絕 OCR 誤讀
        q_image_paths = self.pdf_to_images(q_pdf, "q_full", img_dir, dpi=300) 
        a_image_paths = self.pdf_to_images(a_pdf, "a_full", img_dir, dpi=300) 
        rubric_image_paths = self.pdf_to_images(rubric_pdf, "rubric_full", img_dir, dpi=300)

        # 6. 抓取全卷答案與手寫評分標準
        ans_text = self.extract_clean_answers(a_pdf)
        rubric_text = self.extract_text_from_pdf(rubric_pdf)

        # ----------------- 【學科與數學類別標準化映射】 -----------------
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
        # ------------------------------------------------------------------

        doc = fitz.open(q_pdf)
        cat = doc.pdf_catalog()
        doc.xref_set_key(cat, "StructTreeRoot", "null")
        all_extracted_questions = []

        failed_questions = []  # 用來記錄這份試卷中「所有被跳過題目」的日誌列表
        
        # 🚨 新增：定義審查退件專屬的文字日誌路徑，每次執行前先初始化清空
        validation_records = []  
        val_log_lock = threading.Lock() # 新增鎖
        validation_log_path = json_path.replace("_database.json", "_validator_log.txt")
        if os.path.exists(validation_log_path):
            try:
                os.remove(validation_log_path)
            except Exception:
                pass

        rubric_visual_map = self.extract_rubric_visual(rubric_pdf, img_dir)

        # =========================================================
        # 【第一階段】：分頁批次掃描與附圖裁切 (4~8頁為一個 Batch)
        # =========================================================
        pages = []
        for page_num in range(len(doc)):
            # 若手動啟用 skip_cover，無條件跳過第一頁
            if skip_cover and page_num == 0:
                logging.info("  -> ⏭️ [封面跳過] 依據 skip_cover 參數，直接跳過第 1 頁分析。")
                continue
                
            page = doc[page_num]
            raw_page_text = page.get_text("text")
            
            # 🚨 新增：封面關鍵字過濾器（徹底解決第一、二題重複掃描、錯位裁切的問題）
            # 🚨 修正：封面與注意事項頁過濾器（擴大至前兩頁，並增加更多匹配詞彙與無題目結構校驗）
            # 🚨 修正：封面與注意事項頁過濾器（擴大至前兩頁，並增加更多匹配詞彙與無題目結構校驗，修正換行多行校驗）
            if page_num <= 1:
                cover_keywords = [
                    "作答注意事項", "考試時間", "作答方式", "選擇題範例", 
                    "答案卡", "畫卡樣式", "答題卷", "畫卡樣例", "注意事項", 
                    "作答說明", "答題說明", "答題卡", "考生姓名", "准考證號"
                ]
                has_cover_keyword = any(k in raw_page_text for k in cover_keywords)
                # 在保留原始分行 raw_page_text 上搜尋，確保換行起手式被精準識別 (防範跨頁空殼題目)
                has_real_question = re.search(r'^\s*(?:1|一|\(一\))\s*[.．、\s)]', raw_page_text, re.MULTILINE) is not None
                
                if has_cover_keyword and not has_real_question:
                    logging.info(f"  -> ⏭️ [封面/說明頁跳過] 偵測到第 {page_num + 1} 頁為封面或作答注意事項頁，自動跳過。")
                    continue
                
            pages.append(page_num)

        # 🚨 [核心修復] 解決高資訊密度試卷（如數乙）嚴重漏題（漏掉 Q2~Q9）的 Bug
        # 6 頁對輕量模型而言注意力負擔過重。改為理科每次掃描 2 頁，文科 3 頁，確保 100% 完整擷取
        page_batch_size = 2 if is_stem else 3

        for b_idx in range(0, len(pages), page_batch_size):
            batch_pages = pages[b_idx : b_idx + page_batch_size]
            logging.info(f"正在批次分析 [{year} {subject}] 試卷第 {batch_pages[0]+1} ~ {batch_pages[-1]+1} 頁...")

            # 蒐集本批次所有頁面的圖片與純文字
            batch_pil_imgs = []
            batch_raw_texts = []
            for p_num in batch_pages:
                full_page_filepath = q_image_paths[p_num]
                batch_pil_imgs.append(Image.open(full_page_filepath))
                batch_raw_texts.append(f"--- 第 {p_num+1} 頁純文字 ---\n" + s2t(doc[p_num].get_text("text")))
            
            try:
                batch_text_combined = "\n\n".join(batch_raw_texts)
                prompt_stage_1 = f"""
                🚨【範例過濾極度警告】🚨：
                - 本 PDF 的第一頁（或前幾頁）通常包含「作答範例」或「作答注意事項」。
                - **【嚴禁】** 擷取範例中的題目（如：範例第 1 題為單選題...）。
                - 你的擷取任務必須從真正的考題開始（通常在說明的橫線之後，或從第 1 題真正出現的地方開始）。
                - 確保 `question_number` 1 是考卷真正的第 1 題，而非範例題。

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
                - 英文大小寫、希臘字母（如 $\theta, \phi, \alpha$）必須完全以底層數位文字層的命名為最高準則，嚴禁因為視覺模糊而自行臆測或改寫變數名稱！

                【一、剛性文字與公式規格化】
                1. **題目文字完全對齊**：`question_text` 必須與圖片及底層純文字 100% 吻合。
                2. **LaTeX 規範**：所有數學公式、化學式、變數符號、數值等，**必須**使用標準 LaTeX 語法包裹，確保排版美觀。
                3. **LaTeX 嚴格符號規範**：所有數學公式、化學式、變數符號，【行內公式】必須嚴格使用 `$...$` 包裹；【獨立行公式】必須嚴格使用 `$$...$$` 包裹。絕對禁止使用 `\\( \\)` 或 `\\[ \\]` 作為公式標籤！
                4. **🚨【選項分離與題幹清洗硬性規定】**：對於選擇題（單選與多選），**必須且強制**將題幹文字（`question_text`）中所有包含選項描述的部分（例如：尾部的 `(A) xxx (B) yyy...` 或 `(1) xxx (2) yyy...` 或者是行內的選項敘述）**完全清除乾淨**！題幹 `question_text` 內只能保留最純粹的題目問句，絕對不能殘留任何選項的標籤與其文字內容！所有的選項標籤與文字必須且只能存在於 `options` 欄位中，嚴禁在題幹與選項列表間發生資料雙重冗餘與重複！
                - 🚨【選填題無選項剛性鐵律】：**選填題（填空題，即題號為大寫字母 A, B, C... 或帶有 [ 9 ] 挖空者）絕對沒有選項！** 它的 `options` 屬性必須 100% 為空列表 `[]`。**絕對禁止**將解答卷或答案卡上的欄位索引與答案對照（如：`8. 1. 9. 3.`）當作選項寫入 `options`！
                4-2. **表格強制轉換**：數據表格務必轉換為 Markdown Table 格式嵌入。
                5. **內文嵌入選項**：若遇上古文或國文科「克漏字」等選項內嵌在文章段落中的狀況，請在題幹中保留如 (A)、(B) 等引導標記以維持文章完整，並將對應的代號與內容整理到 `options` 欄位中。
                5. **【選填題挖空規則（極度重要）】**：大考選填題常使用圓圈數字（如 \u2468、\u2469 或 \u246c-\u2460）代表畫卡格。請將其轉換為 `[ 9 ]`、`[ 10 ]` 或 `[ 13-1 ]` 的格式。
                \U0001f6a8**警告**：有時候圓圈數字旁邊**真的有根號或其他數學符號**（例如 $\\frac{{\\u2468\\sqrt{{\\u2469}}}}{{32}}$），請務必精準轉換為 `\\frac{{ [ 9 ] \\sqrt{{ [ 10 ] }} }}{{ 32 }}`！絕對不可以把真正的根號吃掉，也絕對禁止擅自把 \u2468 和 \u2469 強行合併成 `[ 9-10 ]`！請忠實反映圖片上的數學結構。
                6. **表格強制轉換**：若題目中包含數據表格（如：表1、表2），請【務必】將表格內容完整轉換為 Markdown Table 格式，並嵌入到 `question_text` 中相應的位置。絕對不可省略表格內容！
                7. **選填題答案與畫卡格子對應規範（極度重要）**：
                   - 大考選填題的答案會對應至多個獨立的畫卡格子（例如：`[ 10-1 ]`、`[ 10-2 ]`、`[ 10-3 ]`）。
                   - **【強制規定】**：在 `answer` 欄位中，**必須**將每一個格子所對應的答案字元（包含數字、正負號或特定根號代號）依序填入，並以半形逗號 `,` 隔開！
                   - 例如：
                     * 若題目為 $a=[ 10-1 ][ 10-2 ]$, $b=[ 10-3 ]$，其中 $a = -4$, $b = 3$，其對應畫卡格答案分別為 `-`、`4`、`3`，則 `answer` 欄位必須寫成 `-,4,3`，絕對不可直接相連寫成 `-43`。
                     * 若選填題 A 答案為 $\frac{9}{10}$，畫卡格 9-10 為 9、1、0，則寫成 `9,1,0`，絕對不可寫成 `910`。
                8. **克漏字與文意選填特例（極度重要）**：克漏字的選項通常集中在文章下方。請你【主動去文章段落 (shared_context) 中尋找對應的題號】，將「包含該題號空格的那一整個完整句子」提取出來作為 `question_text`，並將題號替換為 `______`。絕對禁止將 (A) (B) (C) (D) 等選項文字當作題幹！
                9. **選填題挖空規則**：若遇到大考特有的圓圈畫卡題號（例如 ⑬-① ⑬-②），請統一轉換為標準挖空格式 `[ 13-1 ] [ 13-2 ]`，不要使用 LaTeX 的 \\bigcirc。這有利於系統自動生成填空輸入框。
                9-2. **🚨【選填題畫卡格圓圈數字不視為圖片】🚨**：
                   大考選填題中出現的帶圓圈數字（如 ⑧、⑨、⑩）是排版文字的一部分（請按規則 9 轉換為 `[ 8 ]`、`[ 9 ]`、`[ 10 ]`）。**【絕對禁止】**將這些圓圈數字、分數線或其相鄰的填空文字框選為 `image_bboxes`！只有當題目中出現真正的實體插圖、函數圖形、幾何圖形或大型數據表時，才將其框選為 `image_bboxes`。
                10. **【極度精準的數學 OCR】**：數學公式與不等式的辨識必須一字不差！例如 `x-y` 絕對不能看錯成 `2x-y` 或 `x+y`，大於小於符號絕對不能反！請反覆核對圖片中的方程式。
                10-2. **🚨【微細符號與循環小數極度預警】🚨**：
                   在數學科中，常常出現**循環小數**（例如 $1.\\bar{{5}}$，即數字 5 的上方有一條橫線，代表 $1.5555...$）。
                   - 這類微小的上標橫線極易被低解析度 OCR 遺漏並誤讀為普通小數（如 $1.5$）。
                   - 請你**仔細盯住原卷圖片上的每一個小數點與數字上方**！若看到數字上方有任何橫線、波浪號或圓點，**必須且強制**將其識別為標準 LaTeX 的循環小數格式，例如 `$1.\\bar{{5}}$`、`$7.\\bar{{7}}$` 或 `$1.\\dot{{5}}$`。
                   - 絕對禁止遺漏這些符號並將其簡化為普通小數，這會導致整道題目的數論邏輯與選項對照徹底崩潰！
                11. **【防錯位與防遺漏警告】**：大考的題目偶爾會分欄排版。請務必遵循正常的閱讀順序（先左後右，先上後下）完整提取 `question_text`。若題目包含附表，請確保 Markdown Table 欄位數與原圖完全一致，絕不可漏掉任何一行數據！
                12. **【防選項合併】**：請確保 `options` 欄位中，每個選項是獨立的物件，絕對不可以把選項 A 和選項 B 融合成一個選項輸出。
                13. **頁碼追蹤（極度重要）**：你必須在 `page_number` 欄位中，填入該題目在原卷 PDF 中的真實頁碼（從 1 開始計數）。這對於裁切考題附圖與表格至關重要。
                14. **【字母 y 與數字 3 的防混淆警告（化學與代數特防）】**：在化學分子式（例如 $CH_3(CH_2)_yCl$）中，**斜體的小寫字母 $y$ 極易被誤判為數字 $3$**！
                    如果題目中同時出現了 $x, z$ 作為待求變數（如 $CH_3SH_x$ 與 $CH_3NH_z$），夾在中間的變數絕對是小寫字母 $y$（即 $CH_3(CH_2)_yCl$），請務必精準辨識為小寫字母 $y$，絕對不可以看成數字 $3$！
                15. **【清洗圖形文字佔位符】**：大考 PDF 的純文字中常含有如 `[圖3 結構圖]`、`[圖形]`、`[圖片]` 或 `[圖 3]` 等無意義的純文字佔位符。請你在擷取 `question_text` 與 `shared_context` 時，**務必將這些無意義的圖形文字佔位符完全剔除**！因為我們後續會有實體的 `image_bboxes` 裁剪圖，不需要保留這些純文字垃圾。
                16. **【極度寬裕的 Bounding Box 標記】**：當你框選 `image_bboxes` 或選項附圖的 Bounding Box 時，**請務必畫得極度寬裕 (Very Generous)**！寧可多框 15% 的空白邊緣，也絕對不可以切到任何化學鍵、原子符號、反應箭頭、坐標軸文字、或選項字母 A, B, C 的邊角！
                17. **【題組判定極度嚴格警告】**：只有當試卷上明確印有『X-Y題為題組』時，才可將 X 到 Y 題歸為題組，並將共同引言寫入 `shared_context`。**絕對禁止**只因為多道題目印在同一頁、或者因為它們都是選擇題，就擅自編造『題組』將其歸類！非題組的題目，其 `shared_context` 必須為空！
                18. **科目精細分類**：請將考卷的原有名稱填入 'exam_source'，而 'sub_subject' 必須且只能從以下清單中挑選一項填寫：'物理', '化學', '生物', '地球科學', '歷史', '地理', '公民與社會', '數學', '英文', '國文', '國寫'。對於跨科考題，請強制選擇佔比最重的一科，絕對不可自創類別。

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
                5. **題號對位**：確保 `question_number` 欄位與你框選的 `image_bboxes` 屬於同一個邏輯區塊。絕對禁止將第 15 題的文字配上第 16 題的圖。


                【二、題組與附圖剛性規則】
                1. **題組共同題幹處理**：若本頁有「X-Y 為題組」（如閱讀測驗文章、實驗情境敘述），請務必將「共同引言/文章/數據表」**只填入 `shared_context` 欄位中**。`question_text` 絕對保持乾淨，只保留該單一子題的問句！
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
                        }},
                        {{
                            "academic_year": "114分科",
                            "exam_source": "化學",
                            "question_number": "22",
                            "page_number": 8,
                            "question_text": "根據此實驗結果，計算氫氧化鈣的溶解度（M）。",
                            "has_image": false,
                            "image_bboxes": [],
                            "options": [],
                            "answer": "0.0244(M)",
                            "question_type": "簡答題",
                            "scoring_criteria": "（一）列式正確得 1 分，答案正確再得 1 分。（二）列式正確，答案不正確只得 1 分。（三）只寫答案沒計算過程，則不給分。",
                            "full_page_image_path": "",
                            "shared_context": ""
                        }},
                        {{
                            "academic_year": "115學測",
                            "exam_source": "自然",
                            "question_number": "28",
                            "page_number": 6,
                            "question_text": "觀察洋蔥根尖切片，下列哪一進程最早形成2 倍量的DNA？（請參見附圖 A-E）",
                            "has_image": true,
                            "image_bboxes": [[150, 100, 300, 950]],  // 🚨 框選一整橫排 A~E 的大 Box
                            "options": [
                                {{"key": "A", "value": "【圖形選項，請參見題幹附圖】", "has_image": false, "image_bboxes": []}},
                                {{"key": "B", "value": "【圖形選項，請參見題幹附圖】", "has_image": false, "image_bboxes": []}}
                            ],
                            "answer": "A",
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
                
                batch_text_combined = "\n\n".join(batch_raw_texts)
                
                # 🚨 核心修復：在多張圖片輸入內容中，為每一張圖片顯式標註其在 PDF 中的絕對頁碼 (1-based)
                # 這能徹底解決 AI 混淆「批次相對頁碼」與「PDF 絕對頁碼」的錯位問題！
                batch_contents_with_labels = [prompt_stage_1]
                for idx, p_num in enumerate(batch_pages):
                    batch_contents_with_labels.append(f"=== 原卷 PDF 第 {p_num+1} 頁影像 ===")
                    batch_contents_with_labels.append(batch_pil_imgs[idx])
                
                # 加上純文字對照
                batch_contents_with_labels.append("\n\n=== 原始純文字對照 ===")
                batch_contents_with_labels.append(batch_text_combined)

                result_dict_1, error_1 = self.ai_manager.generate_with_retry(
                    contents=batch_contents_with_labels,
                    response_schema=PageExtraction,
                    temperature=0.0,
                    preferred_model=stage_1_model,
                    enable_thinking=False  # 🚨 關閉深度思考以節省額度與時間
                )

                # 🚨 核心優化：將 Stage 1 生成的所有結果遞迴且全面地轉為繁體中文，避免簡體字混入後續流程
                if result_dict_1:
                    result_dict_1 = s2t_recursive(result_dict_1)

                if error_1 or not result_dict_1:
                    logging.warning(f"Lite 模型掃描失敗，嘗試使用強力模型重試...")
                    result_dict_1, error_1 = self.ai_manager.generate_with_retry(
                        contents=[prompt_stage_1] + batch_pil_imgs + ["\n".join(batch_raw_texts)],
                        response_schema=PageExtraction,
                        preferred_model="gemini-3.5-flash" # 強力模型支援
                    )
                    if error_1:
                        logging.error(f"該頁掃描徹底失敗: {error_1}")
                        continue

                # ---------------------------------------------------------
                # 3. 處理圖片裁切（根據每題對應的 page_number 載入對應 page 物件）
                # ---------------------------------------------------------
                # 建立本批次的 Bounding Box 裁切快取，防止同頁、同題組的重疊圖重複裁切
                batch_crops_cache = {} # Key: page_number, Value: list of {"bbox": bbox, "path": path}

                # 快速計算兩個 Bounding Box 的重合比例 (IoU)
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
                    # 🚨 修正：防止最後一頁或其他頁面因異常預設或解析錯誤，而錯誤抓成第 0 頁（封面說明頁）
                    # 限制頁碼必須在當前批次處理的頁面列表 `batch_pages` 之中。
                    try:
                        parsed_page = int(q_data.get('page_number', 1))
                        # 🚨 關鍵優化：解除批次頁碼邊界限制，允許跨批次精確對位 PDF 內任意真實頁面進行裁切，防範錯位
                        target_page_idx = max(0, min(parsed_page - 1, len(doc) - 1))
                        q_data['page_number'] = target_page_idx + 1
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
                        # (A) 處理題幹主圖裁切
                        if q_data.get('has_image') and q_data.get('image_bboxes'):
                            bboxes = q_data['image_bboxes']
                            for b_idx, bbox in enumerate(bboxes, 1):
                                if len(bbox) == 4 and all(v is not None for v in bbox):
                                    ymin, xmin, ymax, xmax = bbox
                                    
                                    xmin_val = max(0, min(1000, min(xmin, xmax)))
                                    xmax_val = max(0, min(1000, max(xmin, xmax)))
                                    ymin_val = max(0, min(1000, min(ymin, ymax)))
                                    ymax_val = max(0, min(1000, max(ymin, ymax)))

                                    # 🚨 IoU 題組去重檢測：如果同頁面已經裁過極為接近的框，則直接共用，不重複生成圖片
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
                                    
                                    # 🚨 修正：防主圖切到頁眉頁尾與邊界裝飾垃圾
                                    x0_raw = (xmin_val / 1000.0) * page.rect.width
                                    y0_raw = (ymin_val / 1000.0) * page.rect.height
                                    x1_raw = (xmax_val / 1000.0) * page.rect.width
                                    y1_raw = (ymax_val / 1000.0) * page.rect.height

                                    x0 = max(page.rect.width * 0.03, min(x0_raw, page.rect.width * 0.97))
                                    x1 = max(page.rect.width * 0.03, min(x1_raw, page.rect.width * 0.97))
                                    y0 = max(page.rect.height * 0.065, min(y0_raw, page.rect.height * 0.935))
                                    y1 = max(page.rect.height * 0.065, min(y1_raw, page.rect.height * 0.935))

                                    width_pct = (xmax_val - xmin_val) / 10.0
                                    height_pct = (ymax_val - ymin_val) / 10.0
                                    
                                    if width_pct < 2.5 or height_pct < 2.5:
                                        continue
                                    
                                    # 💡 採用非對稱擴展安全區：向上與左右加寬，防止頂部公式切邊
                                    x_pad = page.rect.width * 0.08      # 左右擴展 8%
                                    y_pad_top = page.rect.height * 0.08  # 頂部向上多留 8% 緩衝區（大考圖號多在上方）
                                    y_pad_bot = page.rect.height * 0.04  # 底部留 4%
                                    
                                    rect = fitz.Rect(x0 - x_pad, y0 - y_pad_top, x1 + x_pad, y1 + y_pad_bot)
                                    rect = rect.intersect(page.rect)

                                    try:
                                        clip_pix = page.get_pixmap(clip=rect, dpi=300)
                                        safe_q_num = safe_filename(str(q_data.get('question_number', 'X')).replace(" ", ""))
                                        
                                        img_filename = f"Q{safe_q_num}_{b_idx}.png"
                                        img_filepath = os.path.join(img_dir, img_filename)
                                        clip_pix.save(img_filepath)
                                        
                                        normalized_path = img_filepath.replace("\\", "/")
                                        q_data['image_paths'].append(normalized_path)
                                        cropped_imgs.append(Image.open(img_filepath))

                                        # 寫入快取，供後續同題組題目共用
                                        batch_crops_cache[p_num].append({
                                            "bbox": bbox,
                                            "path": normalized_path
                                        })
                                    except Exception as e:
                                        logging.error(f"裁切題幹圖片失敗: {e}")

                        # (B) 處理「選項內」附圖裁切
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

                                        # 選項圖同樣執行 IoU 去重
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
                                        
                                        x0 = (xmin_val / 1000.0) * page.rect.width
                                        y0 = (ymin_val / 1000.0) * page.rect.height
                                        x1 = (xmax_val / 1000.0) * page.rect.width
                                        y1 = (ymax_val / 1000.0) * page.rect.height

                                        width_pct = (xmax_val - xmin_val) / 10.0
                                        height_pct = (ymax_val - ymin_val) / 10.0
                                        
                                        if width_pct < 2.0 or height_pct < 2.0:
                                            continue
                                            
                                        x_pad = page.rect.width * 0.07
                                        y_pad_top = page.rect.height * 0.06
                                        y_pad_bot = page.rect.height * 0.03
                                        
                                        rect = fitz.Rect(x0 - x_pad, y0 - y_pad_top, x1 + x_pad, y1 + y_pad_bot)
                                        rect = rect.intersect(page.rect)

                                        try:
                                            clip_pix = page.get_pixmap(clip=rect, dpi=300)
                                            safe_q_num = safe_filename(str(q_data.get('question_number', 'X')).replace(" ", ""))
                                            opt_key = safe_filename(str(opt.get('key', 'X')).replace(" ", ""))
                                            
                                            img_filename = f"Q{safe_q_num}_Opt{opt_key}_{b_idx}.png"
                                            img_filepath = os.path.join(img_dir, img_filename)
                                            clip_pix.save(img_filepath)
                                            
                                            normalized_path = img_filepath.replace("\\", "/")
                                            opt['image_paths'].append(normalized_path)
                                            cropped_imgs.append(Image.open(img_filepath))

                                            # 寫入快取
                                            batch_crops_cache[p_num].append({
                                                "bbox": bbox,
                                                "path": normalized_path
                                            })
                                        except Exception as e:
                                            logging.error(f"裁切選項 {opt_key} 圖片失敗: {e}")

                        # 圖片載入完成，將此題的基礎結構與裁剪圖片物件一併存回批次清單中
                        q_data['_cropped_pil_images'] = cropped_imgs
                        all_extracted_questions.append(q_data)

                    except Exception as e:
                        logging.error(f"準備多模態資訊失敗: {e}")

            finally:
                # 釋放階段一分頁內存
                for img in batch_pil_imgs:
                    try:
                        img.close()
                    except Exception:
                        pass

        
        
        # 🚨 [核心修復] 第一階段擷取完成，立即執行自動清洗與選填題長度驗證
        all_extracted_questions = self.clean_and_verify_questions(all_extracted_questions)
        # 🚨 [跨批次去重] 強制在進入解題前執行去重，過濾交界處重複讀取的題目
        all_extracted_questions = deduplicate_questions(all_extracted_questions)

        # 🚨 [防禦性機制 - 提案四：科目範疇與子學科安全邊界對齊器]
        # 避免大考綜合考科（如社會、自然）因題幹涉及交叉學科詞彙，導致 Stage 1 模型產生跨科分類污染
        # （例如：在社會科中將包含「生物多樣性」的地理題錯誤分類為「生物」）
        for q_data in all_extracted_questions:
            q_sub = q_data.get("sub_subject")
            
            if normalized_subject == "社會":
                if q_sub not in ["歷史", "地理", "公民與社會"]:
                    q_text = s2t(q_data.get("question_text", "") + q_data.get("shared_context", ""))
                    if any(k in q_text for k in ["憲法", "法律", "政府", "權利", "經濟", "市場", "社會", "勞工", "法規", "法治"]):
                        q_data["sub_subject"] = "公民與社會"
                    elif any(k in q_text for k in ["地圖", "氣候", "地形", "空間", "地理", "貿易", "生活圈", "自然環境", "沙丘", "生活圈", "位置"]):
                        q_data["sub_subject"] = "地理"
                    else:
                        q_data["sub_subject"] = "歷史"
            elif normalized_subject == "自然":
                if q_sub not in ["物理", "化學", "生物", "地球科學"]:
                    q_text = s2t(q_data.get("question_text", "") + q_data.get("shared_context", ""))
                    if any(k in q_text for k in ["力", "速度", "電", "磁", "能量", "波", "加速度", "力學"]):
                        q_data["sub_subject"] = "物理"
                    elif any(k in q_text for k in ["化學", "反應", "分子", "溶液", "元素", "原子", "化合物"]):
                        q_data["sub_subject"] = "化學"
                    elif any(k in q_text for k in ["細胞", "基因", "生態", "植物", "動物", "生物", "染色體", "群落"]):
                        q_data["sub_subject"] = "生物"
                    else:
                        q_data["sub_subject"] = "地球科學"
            else:
                # 單一學科考卷，強行對位，不允許任何分叉
                q_data["sub_subject"] = normalized_subject

        # 🚨 [防禦性機制 - 提案一：全卷題號覆蓋率主動對齊與精準補漏機制]
        try:
            expected_q_nums = []
            ans_map = json.loads(ans_text)
            if isinstance(ans_map, dict):
                expected_q_nums = sorted(list(ans_map.keys()), key=natural_sort_key)
                
            # 🚨 健全的子題/畫卡格號與範疇覆蓋檢查器，防止將 A-2, 9-2 等選填題畫卡格子誤判為獨立缺失題目
            def get_base_q_num(q_num_str: str) -> str:
                q_num_str = str(q_num_str).strip()
                match = re.match(r'^(\d+)-(\d+)$', q_num_str)
                if match:
                    return match.group(1)
                match_letter = re.match(r'^([A-Ga-g])-(\d+)$', q_num_str)
                if match_letter:
                    return match_letter.group(1)
                return q_num_str

            def is_question_covered(expected_base: str, extracted_questions: list) -> bool:
                expected_base = str(expected_base).strip()
                for q_item in extracted_questions:
                    q_num = str(q_item.get("question_number", "")).strip()
                    if q_num == expected_base:
                        return True
                    range_match = re.match(r'^(\d+)-(\d+)$', q_num)
                    if range_match:
                        try:
                            start = int(range_match.group(1))
                            end = int(range_match.group(2))
                            val = int(expected_base)
                            if start <= val <= end:
                                return True
                        except ValueError:
                            pass
                    if q_num.startswith(expected_base) and len(q_num) > len(expected_base):
                        next_char = q_num[len(expected_base)]
                        if not next_char.isalnum():
                            return True
                return False

            def get_official_answer_for_base(base_num: str, ans_dict: dict) -> str:
                if base_num in ans_dict:
                    return str(ans_dict[base_num])
                sub_keys = []
                for k in ans_dict.keys():
                    if k.startswith(f"{base_num}-"):
                        suffix = k[len(base_num)+1:]
                        if suffix.isdigit():
                            sub_keys.append((int(suffix), k))
                if sub_keys:
                    sub_keys.sort()
                    return ",".join(str(ans_dict[k]) for _, k in sub_keys)
                return ""

            gaps = []
            for num in expected_q_nums:
                base_num = get_base_q_num(num)
                if not is_question_covered(base_num, all_extracted_questions):
                    if base_num not in gaps:
                        gaps.append(base_num)
            
            if gaps:
                logging.warning(f"⚠️ [補漏機制啟動] 偵測到有 {len(gaps)} 道題目在第一階段漏抓：{gaps}")
                for gap_num in gaps:
                    # 1. 尋找缺失題號最可能存在的 PDF 頁面
                    target_page_num = 0
                    for page_idx in range(len(doc)):
                        page_text = doc[page_idx].get_text("text")
                        # 使用正則匹配，在頁面純文字中尋找該題號的物理邊界
                        if re.search(rf'\b{gap_num}\b', page_text) or f" {gap_num} " in page_text:
                            target_page_num = page_idx
                            break
                    
                    logging.info(f"🔍 題號 {gap_num} 最可能位於 PDF 第 {target_page_num+1} 頁，啟動單題高精度定向擷取...")
                    
                    gap_page = doc[target_page_num]
                    gap_image_path = q_image_paths[target_page_num]
                    
                    official_ans_val = get_official_answer_for_base(gap_num, ans_map)
                    ans_prompt_part = ""
                    if official_ans_val:
                        ans_prompt_part = f"\n🚨【本題官方標準答案】：{official_ans_val}\n請直接將此答案填入 `answer` 欄位中，絕對不可變更或縮水。"

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
                                    # 修正頁碼與基本路徑
                                    new_q['page_number'] = target_page_num + 1
                                    new_q['full_page_image_path'] = gap_image_path.replace("\\", "/")
                                    new_q['question_page_image_paths'] = q_image_paths
                                    new_q['answer_page_image_paths'] = a_image_paths
                                    new_q['rubric_page_image_paths'] = rubric_image_paths
                                    new_q['question_pdf_path'] = q_pdf.replace("\\", "/") if q_pdf else ""
                                    new_q['answer_pdf_path'] = a_pdf.replace("\\", "/") if a_pdf else ""
                                    new_q['rubric_pdf_path'] = rubric_pdf.replace("\\", "/") if rubric_pdf else ""
                                    
                                    # 執行補漏題目的主圖與選項圖裁切
                                    new_q['image_paths'] = []
                                    if new_q.get('has_image') and new_q.get('image_bboxes'):
                                        new_q['image_paths'] = self.execute_crop(gap_page, new_q['image_bboxes'], img_dir, f"Q{gap_num}_Gap")
                                    
                                    # 🚨 補件：同步載入並快取已裁剪的 PIL 影像，避免後續階段遺漏多模態資訊
                                    new_q['_cropped_pil_images'] = [Image.open(p) for p in new_q['image_paths'] if os.path.exists(p)]
                                    
                                    all_extracted_questions.append(new_q)
                                    logging.info(f"🎯 [補漏成功] 已成功補回第 {gap_num} 題，並完成影像高精裁切與 PIL 緩衝。")
                                    break
                
                # 重新排序與校驗
                all_extracted_questions = self.clean_and_verify_questions(all_extracted_questions)
                all_extracted_questions.sort(key=lambda x: natural_sort_key(x.get('question_number', '0')))
        except Exception as e:
            logging.error(f"❌ [補漏引擎異常] 執行補漏時發生未預期錯誤: {e}")

        # 🚨 [選填題格子對照驗證] 強制在進入解題前核對格子數量與答案長度
        mismatch_count = sum(1 for q in all_extracted_questions if q.get("_length_mismatch"))
        if mismatch_count > 0:
            logging.warning(f"⚠️ [選填題格子對照驗證] 偵測到 {mismatch_count} 題選填題存在挖空數量與答案長度不一致的問題！這將在解題階段向 AI 發出高度預警。")
        else:
            logging.info("🎯 [選填題格子對照驗證] 全數通過！所有選填題挖空數皆與官方答案長度吻合。")

        # 題目擷取完成後，進行對位補件

        # 🚨 第一階段所有的頁面 PDF 已經掃描與裁切完成，此時安全關閉 PDF 檔案
        # 🚨 第一階段所有的頁面 PDF 已經掃描與裁切完成，在此處安全關閉 PDF 檔

        # 題目擷取完成後，進行對位補件
        # 🚨 [防禦性機制 - 提案二：混合題型「選擇答案與手寫標準」跨 PDF 跨模態自動縫合器]
         for q_data in all_extracted_questions:
            q_num = q_data['question_number']
            q_type = q_data.get('question_type', '')
            has_options = len(q_data.get("options", [])) > 0
            
            # 1. 精確對位手寫評分圖文標準
            def is_precise_match(q_data: dict, r_k: str) -> bool:
                q_num = str(q_data.get('question_number', ''))
                q_digits = re.findall(r'\d+', q_num)
                r_digits = re.findall(r'\d+', r_k)
                
                cn_mapping = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
                q_cn = [cn_mapping[c] for c in q_num if c in cn_mapping]
                r_cn = [cn_mapping[c] for c in r_k if c in cn_mapping]
                
                if q_cn and r_cn and q_cn != r_cn:
                    return False
                if q_digits and r_digits:
                    return q_digits == r_digits
                return q_num.strip() == r_k.strip()

            matched_key = next((k for k in rubric_visual_map if is_precise_match(q_data, k)), None)
            
            if matched_key:
                q_data['scoring_criteria'] = rubric_visual_map[matched_key]['text']
                q_data['rubric_image_paths'] = rubric_visual_map[matched_key]['paths']
                logging.info(f"🔗 [評分補件] 題號 {q_num} 已成功掛載手寫圖文標準")
                
                # 2. 🚨 混合題型自動偵測：若一題同時有選項與手寫評分標準，定義其為「混合題」
                if has_options:
                    q_data['question_type'] = "混合題"
                    logging.info(f"💎 [混合題偵測] 題號 {q_num} 具備雙重屬性，已重塑其類型為: 混合題")
                    
                    # 3. 跨 PDF 選擇答案縫合：當答案卷答案為非選標記、空白或無答案時，從評分圖片中反向提取選擇題答案
                    current_ans = str(q_data.get('answer', '')).strip()
                    non_choice_indicators = ["見評分", "非選擇", "手寫", "無", "畫在", "作圖", "作答", ""]
                    
                    if any(ind in current_ans for ind in non_choice_indicators) or not current_ans:
                        rubric_paths = q_data.get('rubric_image_paths', [])
                        if rubric_paths and os.path.exists(rubric_paths[0]):
                            logging.info(f"⚖️ [跨PDF縫合] 題號 {q_num} (混合題) 無直接選擇答案，啟動跨模態答案反向提取...")
                            
                            class ExtractedAnswer(BaseModel):
                                answer: str = Field(description="從評分標準圖片中精確提取出的選擇題或勾選題部分的標準答案。必須且只能是選項字母或數字（如 '4'、'C' 或多選 'ACD'），嚴禁包含任何中文字或說明。")

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

                # 4. 🚨 [防禦性機制 - 提案三：混合題「勾選與選填選項」跨模態反向重建器]
                # 針對新課綱中常見的「勾選+簡答」混合題，其選項（如 ☑ 臺灣省戒嚴令）只印在答題卷上，
                # 導致從題目卷中擷取的 options 為空。我們在此主動利用已掛載的 評分標準文字/圖片 進行反向提取重建。
                sc_text = q_data.get('scoring_criteria', '')
                has_checkbox_clue = "勾選" in q_data.get('question_text', '') or any(sym in sc_text for sym in ["☑", "□", "■", "✔"])
                
                if has_checkbox_clue and not q_data.get('options'):
                    logging.info(f"⚖️ [選項重建] 偵測到題號 {q_num} 為「勾選混合題」且缺少選項，啟動跨模態選項提取...")
                    
                    class ExtractedCheckboxes(BaseModel):
                        options: List[OptionItem] = Field(description="從評分標準或答題卷中提取的所有勾選選項列表。按順序為 A, B, C, D...。")
                        correct_key: str = Field(description="被勾選（☑ 或 ■）的正確選項代號（如 'A', 'B' 等）。")

                    stitch_prompt = f"""
                    這是一份大考非選擇題的官方評分標準。
                    請你仔細閱讀以下評分文字與圖片，找出該題在答題卷上供學生「勾選」的所有選項內容（通常在『滿分參考答案』中以 ☑ 或 □ 標示）。
                    
                    任務：
                    1. 提取所有選項的文字，依序編號為 A, B, C, D...。
                    2. 找出被勾選（☑ 或帶有打勾、黑塊標記）的那個正確選項，將其 key（如 'A'）填入 `correct_key`。
                    
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
                            # 將重建的選項寫回題目的 options 欄位中
                            q_data['options'] = res_opts['options']
                            q_data['question_type'] = "混合題"
                            
                            # 如果原本答案為空、斜線或指示詞，則將 correct_key 作為答案
                            current_ans = str(q_data.get('answer', '')).strip()
                            if current_ans in ["", "／", "/", "\\", "無"]:
                                q_data['answer'] = res_opts.get('correct_key', '')
                                
                            logging.info(f"🎯 [重建成功] 題號 {q_num} 已成功恢復 {len(q_data['options'])} 個勾選選項，並更新答案為: {q_data['answer']}")
                    except Exception as e:
                        logging.error(f"混合題勾選選項重建失敗: {e}")

         # 🚨 新增：題組題圖片與資源自動傳播共享機制
        # 只要兩題以上的 shared_context 相同且不為空，即判定為同一題組。
        # 我們將它們的所有裁剪圖片、Bounding Boxes 進行合併，使題組內的每一題（如第7題）都能共享並看見題組原圖（如第6題的結構圖）！
        # 🚨 [題組圖片資源共享防污染機制] 嚴格驗證題號鄰近性與非通用說明的題組共享
        def is_generic_instruction(text):
            generic_keywords = ["注意事項", "答案卡", "畫記", "作答說明", "答題說明", "本部分", "單選題", "多選題", "選填題"]
            matches = sum(1 for kw in generic_keywords if kw in text)
            return matches >= 2 or len(text.strip()) < 20

        # 🚨 [題組圖片資源共享防污染機制] 嚴格驗證題號鄰近性與非通用說明的題組共享
        # 🚨 新增：題組題圖片與資源自動傳播共享機制
        # 只要兩題以上的 shared_context 相同且不為空，即判定為同一題組。
        # 我們將它們的所有裁剪圖片、Bounding Boxes 進行合併，使題組內的每一題（如第7題）都能共享並看見題組原圖（如第6題的結構圖）！
        
        # 🚨 [題組圖片資源共享防污染機制] 嚴格驗證題號鄰近性與非通用說明的題組共享
        def is_generic_instruction(text):
            generic_keywords = ["注意事項", "答案卡", "畫記", "作答說明", "答題說明", "本部分", "單選題", "多選題", "選填題"]
            matches = sum(1 for kw in generic_keywords if kw in text)
            return matches >= 2 or len(text.strip()) < 20

        def is_authentic_group_context(text):
            # 🚨 修正：題組共用背景必須包含明確的群組說明或圖表關鍵字，防止相似積木背景或模板文字引發錯誤分組
            group_keywords = ["題組", "閱讀", "共用", "下列各題", "圖", "表", "實驗", "情境", "背景"]
            return any(kw in text for kw in group_keywords)
        
        # 修正：利用字元交集比例，容忍 OCR 產生的微小標點符號或空格差異
        def get_context_similarity(s1, s2):
            set1 = set(s1.replace(" ", "").replace("\n", ""))
            set2 = set(s2.replace(" ", "").replace("\n", ""))
            if not set1 or not set2:
                return 0.0
            return len(set1.intersection(set2)) / float(len(set1.union(set2)))

        group_pools = {}
        for i, q in enumerate(all_extracted_questions):
            sc = q.get('shared_context', '').strip()
            if not sc or is_generic_instruction(sc) or not is_authentic_group_context(sc):
                continue
                
            matched_sc = None
            for existing_sc in group_pools.keys():
                prev_questions = group_pools[existing_sc]["questions"]
                is_nearby = (i - prev_questions[-1]["index"] <= 4)
                similarity = get_context_similarity(sc, existing_sc)
                
                # 🚨 跨頁題組相容性判定：若字元相似度 >= 0.92，或是題號鄰近且雙方皆含有較長（>100字）的實質背景描述，
                # 則視為同一個因跨頁而產生文本斷裂的題組，進行歸併與後續的文本縫合
                if similarity >= 0.92 or (is_nearby and len(sc) > 100 and len(existing_sc) > 100):
                    matched_sc = existing_sc
                    break
                        
            if matched_sc:
                group_pools[matched_sc]["questions"].append({"index": i, "data": q})
            else:
                group_pools[sc] = {
                    "questions": [{"index": i, "data": q}],
                    "image_paths": [],
                    "image_bboxes": [],
                    "_cropped_pil_images": []
                }

        # 僅針對真正含有 2 題或以上的題組進行資源共享，並對位去重與文本拼接
        for sc, pool in group_pools.items():
            if len(pool["questions"]) < 2:
                continue
            
            # 🚨 解決「跨頁文本斷裂 (Context-Splitting)」：提取並拼合題組中所有相異的 shared_context 文本片段
            combined_context_parts = []
            for item in pool["questions"]:
                part_sc = item["data"].get("shared_context", "").strip()
                if part_sc and part_sc not in combined_context_parts:
                    # 避免部分重合的子字串產生重複冗餘，僅加入相異部分
                    if not any(part_sc in existing or existing in part_sc for existing in combined_context_parts):
                        combined_context_parts.append(part_sc)
            
            merged_shared_context = "\n\n".join(combined_context_parts)
                
            for item in pool["questions"]:
                q = item["data"]
                # 🚨 修正：防選項附圖污染！僅共享題組題幹的非選項主圖，絕對排除含有 "_Opt" 字眼的選項專屬附圖
                pool["image_paths"].extend([p for p in q.get("image_paths", []) if "_Opt" not in p])
                pool["image_bboxes"].extend(q.get("image_bboxes", []))
                pool["_cropped_pil_images"].extend(q.get("_cropped_pil_images", []))
            
            # 資源去重
            unique_paths = list(dict.fromkeys(pool["image_paths"]))
            unique_bboxes = []
            for bbox in pool["image_bboxes"]:
                if bbox not in unique_bboxes:
                    unique_bboxes.append(bbox)
                    
            seen_ids = set()
            unique_pil_imgs = []
            for img in pool["_cropped_pil_images"]:
                if id(img) not in seen_ids:
                    seen_ids.add(id(img))
                    unique_pil_imgs.append(img)
                    
            for item in pool["questions"]:
                q = item["data"]
                q['shared_context'] = merged_shared_context # 🚨 核心修正：將縫合後最完整的題組背景同步寫回每一題，杜絕跨頁資訊遺漏！
                q['image_paths'] = unique_paths
                q['image_bboxes'] = unique_bboxes
                q['_cropped_pil_images'] = unique_pil_imgs
                q['has_image'] = len(unique_paths) > 0
                logging.info(f"  -> 🔗 [題組共享] 題號 {q['question_number']} 已自動共享並連結題組圖片與完整拼合上下文。")
                
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
        if normalized_subject in ["國文", "國寫"]:
            subject_specific_instruction = """
            【五、國文科專屬審查與解題規範】：
            1. **古文精準翻譯**：若涉及文言文，詳解必須給出關鍵字詞的「字義與詞性拆解」，並附上流暢的全文翻譯，嚴禁籠統帶過。
            2. **意象與修辭剖析**：分析選項時，需明確指出詩詞、現代文學中的核心意象（如：落葉象徵衰亡）與修辭手法（如：借代、雙關之隱含語意）。
            """
        elif normalized_subject == "英文":
            subject_specific_instruction = """
            【五、英文科專屬審查與解題規範】：
            1. **長難句骨架拆解**：遇到長難句，詳解必須拆解句子結構（主詞、動詞、關係子句分層說明），並詳細解析關鍵片語或搭配詞 (Collocations) 的語境。
            2. **閱讀定位句與同義代換 (Paraphrasing)**：閱讀題必須在詳解中指出文章中的「定位句」，並說明選項是如何進行同義字代換的。
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
        # 【第二、三階段】：滑動窗口批次詳解、審查與「指正題+新題湊10題」重試機制
        # =========================================================
        # 依照學科決定預設批次 (數學=2, 自然=3, 其他=4)
        active_batch_size = self.get_initial_batch_size(subject)
        max_single_attempts = 8
        max_rechecks = 8

        task_queue = []
        for q in all_extracted_questions:
            task_queue.append({"q_data": q, "critique": "", "retry_count": 0, "recheck_count": 0})
            
        all_final_questions = []
        queue_lock = threading.Lock() # 保護寫入共用陣列的鎖

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
                    
                    item_desc = f"=== 待解第 {idx} 題 ===\n題號：{q_data['question_number']}\n題型：{q_data['question_type']}\n具體科目分類：{q_sub}\n共同背景：{q_data.get('shared_context', '無')}\n題幹：{q_text_prompt}\n選項：\n{options_list_str}\n手寫評分標準：{q_data.get('scoring_criteria', '無')}\n官方答案：【{q_data['answer']}】\n"
                    
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

                # 修復原 Prompt 中的全域變數引用 (避免 q_data 被覆蓋)
                batch_prompt = batch_intro + PROMPT_STAGE_2_MAIN.format(
                    subject_rubric=q_rubric,
                    q_answer="各題上方所給定的官方答案", # 直接傳字串變數
                    topics=q_allowed.get('topics', []),
                    technique=q_allowed.get('techniques', []),
                    math_scope_instruction=math_scope_instruction,
                    subject_specific_instruction=subject_specific_instruction
                )
                batch_contents.insert(0, batch_prompt)

                try:
                    # 呼叫 Stage 2 API (加入標籤)
                    solutions_dict, s_err = self.ai_manager.generate_with_retry(
                        contents=batch_contents, response_schema=QuestionSolutionBatch,
                        temperature=0.1 if is_stem else 0.5, preferred_model=stage_2_model, enable_thinking=True,
                        task_desc=paper_tag
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
                    
                    # 處理失敗需重試的題目
                    for item in failed_batch:
                        item["retry_count"] += 1
                        if item["retry_count"] >= max_single_attempts:
                            item["q_data"]["detailed_solution"] = "詳解批次生成超時或失敗。"
                            with queue_lock: all_final_questions.append(item["q_data"])
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
                                # 🚨 繁簡雙向關鍵字容錯：支持多種肯定與否定語境的精確判定，防範 fallback 字典漏字造成的漏判
                                is_correct = any(w in exp for w in ["正確", "正确", "對", "对", "應選", "应选", "合適", "合适", "最適", "最适", "選", "选"])
                                is_incorrect = any(w in exp for w in ["錯誤", "错误", "不正確", "不正确", "不符", "不合"])
                                if is_correct and not is_incorrect:
                                    derived_correct_keys.append(opt.get("key", "").strip())
                                elif "正確" in exp and "錯誤" not in exp:
                                    derived_correct_keys.append(opt.get("key", "").strip())
                                    
                            derived_ans = "".join(sorted(derived_correct_keys))
                            official_ans = str(q_data.get('answer', '')).strip()
                            
                            # 雙向清理格式（去除所有逗號與空格），確保多選題 '3,4' 與 '34' 等價比對
                            derived_clean = derived_ans.replace(",", "").replace(" ", "")
                            official_clean = official_ans.replace(",", "").replace(" ", "")
                            
                            # 🚨 [大防禦機制修補]：
                            # 1. 正常情況下，若推導答案與官方答案不符，觸發衝突。
                            # 2. 空值/無解防禦：若該題為單選題，但 AI 的選項分析中竟然「沒有推導出任何一個正確選項」（derived_clean 為空），這在邏輯上本身就是嚴重漏洞，必須直接強制攔截並重試/仲裁！
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
                                
                    # 4. 執行原有的 Markdown 格式化轉換
                    # 🚨 [防禦性機制 - 提案二：選填題/非選題格式自動化預校驗器]
                    for idx, sol in enumerate(salvaged_solutions):
                        q_data = valid_batch[idx]["q_data"]
                        
                        # 呼叫預校驗器進行格式與類型清洗
                        pre_validate_format(q_data, sol)
                        
                        # 🚨 [防禦性機制 - 提案一：雙重不一致「學術裁決與防硬凹」仲裁機制 (第一階段：偵測)]
                        # 針對選擇題，提取 AI 真正判定為「正確」的選項
                        if q_data.get("question_type") in ["單選題", "多選題"] and isinstance(sol.get("options_analysis"), list):
                            derived_correct_keys = [opt.get("key", "").strip() for opt in sol.get("options_analysis", []) if "正確" in opt.get("explanation", "")]
                            derived_ans = "".join(sorted(derived_correct_keys))
                            official_ans = str(q_data.get('answer', '')).strip()
                            
                            # 雙向清理格式（去除所有逗號與空格），確保多選題 '3,4' 與 '34' 等價比對
                            derived_clean = derived_ans.replace(",", "").replace(" ", "")
                            official_clean = official_ans.replace(",", "").replace(" ", "")
                            
                            if derived_clean and official_clean and derived_clean != official_clean:
                                # 發現嚴重衝突！標記此題存在學術不對位，以便後續強制觸發學術仲裁
                                valid_batch[idx]["_discrepancy_detected"] = True
                                valid_batch[idx]["_derived_ans"] = derived_ans
                                logging.warning(f"⚖️ [不一致預警] 題號 {q_data['question_number']}：AI 實質推導出 '{derived_ans}'，但官方紀錄為 '{official_ans}'。已標記進行強制仲裁！")

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
                        validator_batch_intro += f"""=== 待審查第 {idx+1} 題 ===
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

                    validator_batch_prompt = PROMPT_STAGE_3_VALIDATOR.format(validator_batch_intro=validator_batch_intro)
                    val_dict, val_err = self.ai_manager.generate_with_retry(
                        contents=[validator_batch_prompt], response_schema=SolutionValidatorBatch,
                        temperature=0.0, preferred_model=validator_model, enable_thinking=True,
                        task_desc=f"{paper_tag} [審查]"
                    )

                    # 處理審查結果 (包含重掃描 OCR)
                    for idx, item in enumerate(valid_batch):
                        q_data = item["q_data"]
                        sol_data = salvaged_solutions[idx]
                        
                        # 🚨 核心優化：此處已在前面提速完成格式化，保留兜底檢查即可
                        if not sol_data.get("options_analysis"):
                            sol_data["options_analysis"] = "本題為非選擇題/選填題，無選項可供分析。"
                            
                        is_valid_passed = True
                        error_critique = ""
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
                            # 有任何一個環節判定不一致，即刻攔截
                            is_valid_passed = False 
                            
                            # 🚨 [防禦性機制 - 提案一：雙重不一致「學術裁決與防硬凹」仲裁機制 (第二階段：強制仲裁與覆寫)]
                            # 如果是「學術推導直接不一致」或「已經重新掃描過仍有衝突」，直接跳過二次重掃，交給學術仲裁器
                            if discrepancy_detected or item["recheck_count"] >= 1:
                                logging.warning(f"⚖️ {paper_tag} [學術衝突/重掃描上限] 題號 {q_data['question_number']} 啟動【學術仲裁器】進行最終裁決...")
                                
                                # 組合詳細的衝突上下文，供仲裁器精確研判
                                conflict_clue = error_critique if error_critique else f"AI 實質推導出正確選項為 '{item.get('_derived_ans')}'，但官方輸入答案為 '{q_data.get('answer')}'。"
                                arbitrated_data = self.run_academic_arbitration(q_data, sol_data, conflict_clue)
                                
                                if arbitrated_data:
                                    # 1. 自動覆寫官方答案欄位，修正為學術正確之答案
                                    q_data['answer'] = arbitrated_data.get('chosen_answer', q_data['answer'])
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
                                    if q_data.get('_cropped_pil_images'):
                                        recheck_contents_with_labels.append("=== 原始裁剪出的題目附圖 ===")
                                        recheck_contents_with_labels.extend(q_data['_cropped_pil_images'])
                                    
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
                                        # 更新數據
                                        q_data['shared_context'] = recheck_dict.get('corrected_shared_context', q_data['shared_context']) # 🆕 新增此行
                                        q_data['question_text'] = recheck_dict.get('corrected_question_text', q_data['question_text'])
                                        q_data['options'] = recheck_dict.get('corrected_options', q_data['options'])
                                        q_data['answer'] = recheck_dict.get('corrected_answer', q_data['answer'])

                                        # 處理補裁切
                                        if recheck_dict.get('found_new_images') and recheck_dict.get('new_image_bboxes'):
                                            logging.info(f"🔍 {paper_tag} [發現新圖] 第 {q_data['question_number']} 題正在補裁切...")
                                            try:
                                                with fitz.open(q_data['question_pdf_path']) as temp_doc:
                                                    cat = temp_doc.pdf_catalog()
                                                    if cat > 0: temp_doc.xref_set_key(cat, "StructTreeRoot", "null")
                                                    target_page = temp_doc[q_data['page_number'] - 1]
                                                    
                                                    new_img_paths = self.execute_crop(target_page, recheck_dict['new_image_bboxes'], img_dir, f"Q{q_data['question_number']}_Extra")
                                                    
                                                    for path in new_img_paths:
                                                        if path not in q_data['image_paths']:
                                                            q_data['image_paths'].append(path)
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
                            # 成功通過，加上總結句
                            ans = str(q_data.get('answer', '')).strip()
                            if ans:
                                # 🚨 核心修正：若官方答案為無答案或送分，則輸出優雅的中文總結，防止生成 (無)(答)(案)(全)(體)(給)(分) 的火星文
                                if any(k in ans for k in ["無答案", "全體給分", "送分", "不計分"]):
                                    q_data['detailed_solution'] += f"\n\n**綜上所述，本題官方公佈無答案，全體給分。**"
                                elif q_data.get('question_type', '') in ["單選題", "多選題"]:
                                    formatted_parts = [f"({char})" for char in ans if char.isalnum()]
                                    q_data['detailed_solution'] += f"\n\n**綜上所述，本題正確選項為：{''.join(formatted_parts)}**"
                                elif q_data.get('question_type', '') == "選填題" and "," in ans:
                                    formatted_parts = ans.split(",")
                                    blanks_desc = ", ".join([f"第 {idx+1} 空格為「{val}」" for idx, val in enumerate(formatted_parts)])
                                    q_data['detailed_solution'] += f"\n\n**綜上所述，本題選填題各畫卡格答案為：{ans}（即 {blanks_desc}）**"
                                else:
                                    q_data['detailed_solution'] += f"\n\n**綜上所述，本題正確答案為：{ans}**"
                            
                            with queue_lock:
                                if '_cropped_pil_images' in q_data:
                                    for pimg in q_data['_cropped_pil_images']:
                                        try: pimg.close()
                                        except: pass
                                    del q_data['_cropped_pil_images']
                                update_subject_taxonomy(q_data.get("sub_subject", subject), q_data)
                                all_final_questions.append(q_data)
                            completed_indices.add(idx) # 🆕 標記此題目在當前批次中已成功處理，避免重複寫入
                            logging.info(f"  -> 🎉 {paper_tag} 題號 {q_data['question_number']} 通過驗證，成功收錄。")
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
                                q_data['detailed_solution'] = "\n\n> **⚠️ [系統提示]：本題偵測到潛在瑕疵。**\n\n" + q_data.get('detailed_solution', '')
                                with queue_lock: all_final_questions.append(q_data)
                    
                    return retry_items, new_batch_size
                except Exception as inner_e:
                    # 🚨 修正：安全地閉合內層 try 區塊並向上拋出異常，避免引發 SyntaxError
                    raise inner_e
            except Exception as e:
                logging.exception(f"❌ [執行緒崩潰安全自癒] 處理批次時發生未預期異常: {e}")
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
        max_workers = min(len(API_KEYS), 8) # 根據 Key 數量決定並發量
        logging.info(f"🚀 開始並行詳解生成！啟動 {max_workers} 條執行緒...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = set()
            while True:
                with queue_lock:
                    # 分派任務直到線程池滿或佇列空
                    while len(task_queue) > 0 and len(futures) < max_workers:
                        current_batch = task_queue[:active_batch_size]
                        task_queue = task_queue[active_batch_size:]
                        futures.add(executor.submit(process_question_chunk, current_batch, active_batch_size))
                
                if not futures: break
                
                # 等待任意一個 Batch 完成
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                
                for fut in done:
                    try:
                        retry_items, new_batch_size = fut.result()
                        with queue_lock:
                            if new_batch_size < active_batch_size:
                                active_batch_size = new_batch_size
                            # 將重試題目插回佇列最前方
                            task_queue = retry_items + task_queue
                    except Exception as e:
                        logging.error(f"批次處理執行緒發生例外: {e}")

        # 最後排序題號以維持最終 JSON 整齊，防範字典序排序導致 10 排在 2 前面
        all_final_questions.sort(key=lambda x: natural_sort_key(x.get('question_number', '0')))

        if not all_final_questions:
            logging.error(f"❌ [任務中止] [{year} {subject}] 解析出的題目數量為 0！")
            logging.error("這可能是由於掃描階段失敗、無可用 Key 或 API 全程斷線所致。為避免生成無意義的空白資料庫檔案，我們將跳過寫入 JSON。請修復 API 金鑰或網路後再次執行！")
            return

        # =========================================================
        # 5. 儲存與寫入最終的 JSON 檔案
        # =========================================================
        # 🚨 寫入 JSON 前，徹底清除所有非 JSON 序列化的暫存屬性 (如 PngImageFile 影像對象) 🚨

        def normalize_latex_delimiters(text: str) -> str:
            """自動將 Rogue LaTeX 符號 \[ \] 轉換為標準 $$ $$，將 \( \) 轉換為 $ $，防範前端排版渲染崩潰"""
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
                # 統一 Windows 與網頁斜線路徑格式
                if "./" in obj or "gsat_" in obj or "ast_" in obj:
                    obj = obj.replace("\\", "/")
                # 執行全局 LaTeX 符號強制標準化
                obj = normalize_latex_delimiters(obj)
            return obj

        all_final_questions = clean_paths(all_final_questions)
        for q in all_final_questions:
            # 🚨 剛性規範：強制將學年度與試卷來源名稱標準化，徹底杜絕 AI 在不同批次中生出不一致之命名
            q["academic_year"] = standard_academic_year
            q["exam_source"] = standard_exam_source
            if '_cropped_pil_images' in q:
                for img_obj in q['_cropped_pil_images']:
                    try:
                        img_obj.close()
                    except Exception:
                        pass
                del q['_cropped_pil_images']
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_final_questions, f, ensure_ascii=False, indent=4)
        
        
        # 🚨 新增：若有審查未通過的歷史紀錄，自動儲存至專屬的詳細 JSON 日誌中
        if validation_records:
            # 日誌裡的路徑也清洗一下
            validation_records = clean_paths(validation_records)
            with open(validation_log_path, "w", encoding="utf-8") as f_val:
                json.dump(validation_records, f_val, ensure_ascii=False, indent=4)
            logging.warning(f"⚠️ 提示：本卷共有 {len(validation_records)} 次審查未通過紀錄。詳細偵錯日誌已寫入：{validation_log_path}")

        
        logging.info(f"✅ [{year} {subject}] 處理完成！共擷取 {len(all_final_questions)} 題，已儲存至 {json_path}")
        # 🚨 關閉 PDF 文件手把
        doc.close()

# 4. 自動尋找檔案腳本 (自動化遍歷)
# ==========================================
def auto_find_exam_sets(directories: List[str]) -> List[dict]:
    """
    🚨 [跨年度/學科精確對位匹配器] 重構：
    徹底摒棄依賴 loose prefix (f.split("_")[0]) 的分組邏輯。
    我們對每一個 PDF 文件名進行深度元數據 (學年度, 考試類型, 科目, 文件角色) 提取。
    只有當 Year, Exam Type, Subject 三者完全一致時，才將其歸為同一個 Exam Task。
    徹底根除 111 年考卷配到 107 年答案或 109 年評分標準的檔案錯配災難！
    """
    exam_sets = []
    
    SUBJECT_MAPPING = {
        "國文": ["國文", "國綜", "國語文", "國文考科", "國文考科國綜"],
        "國寫": ["國寫", "國文寫作"],
        "英文": ["英文", "英語"],
        "數學甲": ["數學甲", "數甲", "數學甲考科"],
        "數學乙": ["數學乙", "數乙", "數學乙考科"],
        "數學A": ["數學A", "數A", "數學A考科"],
        "數學B": ["數學B", "數B", "數學B考科"],
        "數學": ["數學", "數學考科"],
        "物理": ["物理"],
        "化學": ["化學"],
        "生物": ["生物"],
        "地球科學": ["地球科學", "地科"],
        "歷史": ["歷史"],
        "地理": ["地理"],
        "公民與社會": ["公民與社會", "公民"],
        "自然": ["自然"],
        "社會": ["社會"]
    }

    def parse_metadata(filename_clean: str) -> dict:
        # 1. 提取學年度 (例如 107學年度, 111學年度)
        year_match = re.search(r'(\d+)(學年度|年)', filename_clean)
        year = year_match.group(0) if year_match else "未知年份"
        
        # 2. 判斷考試類型
        exam_type = "MOCK"
        if any(k in filename_clean for k in ["指考", "指定科目", "分科"]):
            exam_type = "AST"
        elif any(k in filename_clean for k in ["學測", "學科能力"]):
            if not any(k in filename_clean for k in ["模擬", "模考"]):
                exam_type = "GSAT"
                
        # 3. 判斷科目 (優先採用 SUBJECT_MAPPING 進行對位)
        subject = "未知科目"
        for official_name, aliases in SUBJECT_MAPPING.items():
            if any(alias in filename_clean for alias in aliases):
                subject = official_name
                break
                
        # 4. 判斷文件角色
        role = "other"
        if any(k in filename_clean for k in ["非選", "評分", "原則", "標準"]):
            role = "rubric"
        elif any(k in filename_clean for k in ["答案", "解答"]):
            role = "answer"
        elif any(k in filename_clean for k in ["試卷", "題目", "試題"]):
            role = "question"
            
        return {
            "year": year,
            "exam_type": exam_type,
            "subject": subject,
            "role": role
        }

    for base_dir in directories:
        if not os.path.exists(base_dir):
            continue
            
        for root, dirs, files in os.walk(base_dir):
            pdf_files = [f for f in files if f.endswith(".pdf")]
            if not pdf_files:
                continue
            
            # 依據精確三元組 (Year, Exam Type, Subject) 進行分組
            task_groups = {}
            for f in pdf_files:
                meta = parse_metadata(f)
                # 只有具備有效年份與科目的檔案才參與精確分組，防範垃圾檔案干擾
                if meta["year"] == "未知年份" or meta["subject"] == "未知科目":
                    continue
                key = (meta["year"], meta["exam_type"], meta["subject"])
                if key not in task_groups:
                    task_groups[key] = []
                task_groups[key].append((f, meta))
            
            # 🚨 修正：統一調用 task_groups 變數，並精確拆分 tuple 進行角色分配，杜絕 NameError
            for key, grouped_files in task_groups.items():
                q_candidates = [filename for filename, meta in grouped_files if meta["role"] == "question"]
                if not q_candidates:
                    continue
                # 依據檔名長度挑選最主要的題目 PDF
                final_q_file = sorted(q_candidates, key=lambda x: len(x), reverse=True)[0]
                
                a_candidates = [filename for filename, meta in grouped_files if meta["role"] == "answer"]
                final_a_file = a_candidates[0] if a_candidates else None
                
                rubric_candidates = [filename for filename, meta in grouped_files if meta["role"] == "rubric"]
                final_rubric_file = rubric_candidates[0] if rubric_candidates else None
                
                exam_sets.append({
                    "year": key[0],
                    "exam_type": key[1],
                    "mock_tag": "",  # 模擬考特定標籤由下方獨立解析（若有）
                    "subject": key[2],
                    "q_pdf": os.path.join(root, final_q_file),
                    "a_pdf": os.path.join(root, final_a_file) if final_a_file else None,
                    "rubric_pdf": os.path.join(root, final_rubric_file) if final_rubric_file else None
                })
    return exam_sets
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
        import random
        # 🚨 核心修改：在程式啟動時，將金鑰列表順序完全隨機洗牌打散
        # 這會打亂相同帳號、相鄰金鑰的派發順序，降低單一專案在短時間內被集中高頻呼叫的風險
        random.shuffle(API_KEYS)
        print(f"✅ 成功載入並隨機打散 {len(API_KEYS)} 組金鑰。")
        print(f"打散後第一組: {API_KEYS[0][:15]}...")
        print(f"打散後最後一組: {API_KEYS[-1][:15]}...")
    # 推薦使用 flash 模型處理多模態 (Vision) 任務，速度快且便宜
    MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite-preview",
    ]
    
    manager = GeminiFreeTierManager(api_keys=API_KEYS, models=MODELS)
    parser = ExamParser(ai_manager=manager)

    # 1. 指定你的題庫根目錄
    TARGET_DIRECTORIES = ["ast_exam_papers_only", "gsat_exam_papers_only"]
    output_directory = "./exam_database_output"
    
    # 2. 自動尋找所有要處理的試卷
    exam_tasks = auto_find_exam_sets(TARGET_DIRECTORIES)
    logging.info(f"🔍 總共在目錄中找到了 {len(exam_tasks)} 份試卷需要處理。")  # ⚠️ 建議設 3 即可，因為單張考卷內部還有高達 25 個子執行緒
    max_exam_workers = 2
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
                output_dir=output_directory
            ))
            
        # 等待所有考卷處理完畢
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"❌ 處理考卷時發生嚴重錯誤: {e}")