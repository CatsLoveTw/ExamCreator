# =========================================================================
# 既有題庫資料庫一鍵離線無損修復與重漏題仲裁引擎
# =========================================================================
import os
import re
import json
import shutil
import logging
from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATABASE_DIR = "./exam_database_output"

# 幽靈題與無效題特徵關鍵字
GHOST_KEYWORDS = [
    "本題於原試卷中不存在", "全卷掃描漏失", "無完整題目文本", 
    "題目文字未在影像中提供", "此頁面為試卷封面", "本頁為「大學入學考試中心",
    "作答注意事項", "無實質試題文字", "本題不存在", "本頁為封面", "此頁面為試題封面",
    "題目內容缺失，無法進行題意分析", "題目闕如，無法判定"
]

def clean_latex_corruptions(text: str) -> str:
    """修復歷史 JSON 中因跳脫或控制字元造成的 LaTeX 殘損"""
    if not isinstance(text, str) or not text:
        return text
    
    # 1. 救回被吃掉的 LaTeX 指令
    text = re.sub(r'\\?fty\b', r'\\infty', text)
    text = re.sub(r'(?<!\\)\bimplies\b', r'\\implies', text)
    text = re.sub(r'(?<!\\)\biff\b', r'\\iff', text)
    text = re.sub(r'\\?bsim\b', r'\\approx', text)
    text = re.sub(r'\\bar\{log\}', r'\\log', text)
    
    # 2. 救回 Carriage Return 導致的括號錯位
    text = text.replace(r"\r\right", r"\right").replace(r"\r\left", r"\left")
    text = text.replace(r"\r", "")
    
    # 3. 救回被 form feed 破壞的 frac
    text = re.sub(r'[\x0c]rac(?![a-zA-Z])', r'\\frac', text)
    text = re.sub(r'(?<=\$)rac(?=\{)', r'\\frac', text)
    
    return text

def evaluate_solution_score(q: dict) -> int:
    """詳解深度評分器"""
    sol = str(q.get("detailed_solution", "")).strip()
    score = len(sol)
    
    # 命中無效字樣重罰
    for kw in GHOST_KEYWORDS:
        if kw in sol: score -= 80000
    if "超時或失敗" in sol or "崩潰" in sol: score -= 50000
    
    # 具備名師架構加分
    if "### 【標準解法】" in sol or "【標準解法】" in sol: score += 1000
    if "### 【另解" in sol or "【另解" in sol: score += 1200
    if "### 【速解" in sol or "【速解" in sol: score += 800
    if "![圖" in sol or "diagram_" in sol: score += 600
    
    opt_ana = str(q.get("options_analysis", ""))
    if opt_ana and "本題為非選擇題" not in opt_ana and len(opt_ana) > 50:
        score += 800
        
    return score

def get_text_similarity(s1: str, s2: str) -> float:
    def clean_set(s):
        return set(re.sub(r'[^\w\u4e00-\u9fa5]', '', str(s)))
    set1, set2 = clean_set(s1), clean_set(s2)
    if not set1 or not set2: return 0.0
    return len(set1.intersection(set2)) / float(len(set1.union(set2)))

def natural_sort_key(s):
    cn_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
    s_str = str(s).strip()
    for cn, num in cn_map.items():
        s_str = s_str.replace(cn, f"{num:02d}")
    parts = re.split(r'(\d+)', s_str)
    return [int(text) if text.isdigit() else text.lower() for text in parts]

def check_missing_questions(questions: List[dict], filepath: str) -> List[str]:
    """檢測考卷中可能漏掉的題號缺口 (Gaps)"""
    num_list = []
    letter_list = []
    
    for q in questions:
        q_num = str(q.get("question_number", "")).strip()
        if q_num.isdigit():
            num_list.append(int(q_num))
        elif len(q_num) == 1 and q_num.upper() in "ABCDEFGH":
            letter_list.append(q_num.upper())
            
    gaps = []
    # 檢查數字連續性
    if num_list:
        num_list = sorted(list(set(num_list)))
        min_n, max_n = min(num_list), max(num_list)
        # 只有在題數合理範圍內檢查連續性
        if max_n <= 60:
            for n in range(min_n, max_n + 1):
                if n not in num_list:
                    gaps.append(f"第 {n} 題")
                    
    # 檢查選填題字母連續性 (A, B, C...)
    if letter_list:
        letter_list = sorted(list(set(letter_list)))
        start_ord, end_ord = ord(letter_list[0]), ord(letter_list[-1])
        for o in range(start_ord, end_ord + 1):
            char = chr(o)
            if char not in letter_list:
                gaps.append(f"選填 {char} 題")
                
    return gaps

def repair_single_database(json_path: str) -> Dict[str, Any]:
    stats = {
        "file": os.path.basename(json_path),
        "original_count": 0,
        "cleaned_count": 0,
        "ghost_removed": 0,
        "duplicates_merged": 0,
        "missing_gaps": [],
        "repaired_fields": 0
    }
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"❌ 無法讀取 {json_path}: {e}")
        return stats
        
    if not isinstance(data, list):
        return stats
        
    stats["original_count"] = len(data)
    
    # 1. 備份原檔案
    bak_path = json_path + ".bak"
    if not os.path.exists(bak_path):
        shutil.copy2(json_path, bak_path)

    # 2. 剔除幽靈假題
    non_ghost_questions = []
    for q in data:
        q_text = str(q.get("question_text", "")).strip()
        ans_text = str(q.get("answer", "")).strip()
        
        # 判定是否為幽靈題特徵
        is_ghost = False
        if not q_text and not q.get("has_image"):
            is_ghost = True
        elif any(gk in q_text for gk in GHOST_KEYWORDS):
            is_ghost = True
        elif any(gk in ans_text for gk in ["本題不存在", "免予計分", "不適用（此頁為試題封面"]):
            is_ghost = True
            
        if is_ghost:
            stats["ghost_removed"] += 1
        else:
            non_ghost_questions.append(q)

    # 3. 智能重題去重與最佳解答仲裁
    deduped = []
    for q in non_ghost_questions:
        q_num = str(q.get("question_number", "")).strip()
        q_text = str(q.get("question_text", "")).strip()
        q_type = str(q.get("question_type", "")).strip()
        is_dup = False
        
        for ext in deduped:
            ext_num = str(ext.get("question_number", "")).strip()
            ext_text = str(ext.get("question_text", "")).strip()
            ext_type = str(ext.get("question_type", "")).strip()
            
            same_id = (q_num == ext_num and q_type == ext_type and q_num != "")
            sim = get_text_similarity(q_text, ext_text)
            
            if same_id or (sim > 0.85 and len(q_text) > 20):
                is_dup = True
                stats["duplicates_merged"] += 1
                # 仲裁誰的解答更好
                if evaluate_solution_score(q) > evaluate_solution_score(ext):
                    ext.clear()
                    ext.update(q)
                else:
                    # 保留原版本，但若新版本有圖則補圖
                    if not ext.get("image_paths") and q.get("image_paths"):
                        ext["image_paths"] = q["image_paths"]
                        ext["has_image"] = True
                break
                
        if not is_dup:
            deduped.append(q)

    # 4. 年代名詞校正、多知識點陣列升級、LaTeX 清洗與選項規範化
    for q in deduped:
        stats["repaired_fields"] += 1
        
        # 4-1. 年代名詞校正 (< 111 年改為指考)
        year_raw = str(q.get("academic_year", ""))
        digits = "".join(filter(str.isdigit, year_raw))
        if digits and digits.isdigit():
            y_int = int(digits)
            if y_int < 111:
                if "分科" in q.get("academic_year", ""):
                    q["academic_year"] = q["academic_year"].replace("分科", "指考")
                if "分科測驗" in q.get("exam_source", ""):
                    q["exam_source"] = q["exam_source"].replace("分科測驗", "指定科目考試")

        # 4-2. 多知識點升級
        main_cat = str(q.get("topic_category", "")).strip()
        raw_cats = q.get("topic_categories", [])
        if not isinstance(raw_cats, list) or not raw_cats:
            raw_cats = [main_cat] if main_cat else ["必修_綜合主題"]
        q["topic_categories"] = list(dict.fromkeys([c for c in raw_cats if c]))
        q["topic_category"] = q["topic_categories"][0]

        # 4-3. 清洗 LaTeX
        for field in ["question_text", "detailed_solution", "question_analysis", "solving_strategy", "scoring_rubric", "concept_review", "traps_and_warnings"]:
            if field in q and isinstance(q[field], str):
                q[field] = clean_latex_corruptions(q[field])
                
        # 4-4. 清洗選擇題誤植的手寫評分與答案標籤污染
        q_type = q.get("question_type", "")
        det_sol = q.get("detailed_solution", "")
        if q_type in ["單選題", "多選題"]:
            for h in ["### 【手寫題評分對照】", "【手寫題評分對照】", "### 【手寫題評分】"]:
                if h in det_sol:
                    det_sol = det_sol.split(h)[0].strip()
            q["detailed_solution"] = det_sol
            
            # 清理 (3)(5) [即 (C)(E)] 污染，還原為乾淨的選項代號
            raw_ans = str(q.get("answer", "")).strip()
            if "即" in raw_ans or "[" in raw_ans or "(" in raw_ans:
                tokens = re.findall(r'[A-Ga-g1-9]', raw_ans)
                if tokens:
                    q["answer"] = "".join(sorted(list(dict.fromkeys([t.upper() for t in tokens]))))

    # 5. 排序與缺口偵測
    def safe_sort(x):
        p_num = 1
        try: p_num = int(x.get("page_number", 1))
        except: pass
        return (p_num, natural_sort_key(x.get("question_number", "0")))
        
    deduped.sort(key=safe_sort)
    stats["missing_gaps"] = check_missing_questions(deduped, json_path)
    stats["cleaned_count"] = len(deduped)

    # 6. 安全覆寫儲存
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=4)

    return stats
TICKETS_FILE = "exam_issue_tickets.json"

def apply_user_feedback_tickets():
    """自動套用使用者提交的試題疑義修復工單"""
    if not os.path.exists(TICKETS_FILE):
        return 0
        
    try:
        with open(TICKETS_FILE, "r", encoding="utf-8") as f:
            tickets = json.load(f)
    except Exception:
        return 0
        
    if not isinstance(tickets, list) or not tickets:
        return 0
        
    resolved_count = 0
    pending_tickets = [t for t in tickets if t.get("status") == "pending"]
    if not pending_tickets:
        return 0
        
    logging.info(f"📬 [工單處理] 發現 {len(pending_tickets)} 筆使用者回報的試題疑義單，啟動自動校準...")
    
    # 建立 UID 對應工單字典
    ticket_map = {t.get("uid"): t for t in pending_tickets if t.get("uid")}
    
    for root, dirs, files in os.walk(DATABASE_DIR):
        for f in files:
            if f.endswith("_database.json") and not f.endswith("_partial_database.json"):
                full_p = os.path.join(root, f)
                try:
                    with open(full_p, "r", encoding="utf-8") as jf:
                        q_list = json.load(jf)
                except Exception:
                    continue
                    
                file_changed = False
                for q in q_list:
                    # 建立或比對 UID
                    q_uid = f"{q.get('academic_year','')}_{q.get('exam_source','')}_{q.get('question_number','')}"
                    if q_uid in ticket_map:
                        t = ticket_map[q_uid]
                        logging.info(f"  -> 🎯 命中工單 {t['ticket_id']}：針對 {q_uid} 套用修正建議...")
                        
                        # 1. 修正答案
                        if t.get("suggested_answer"):
                            q["answer"] = t["suggested_answer"]
                            q["official_answer"] = t["suggested_answer"]
                            q["has_answer_discrepancy"] = False
                            file_changed = True
                            
                        # 2. 修正題型
                        if t.get("suggested_type"):
                            q["question_type"] = t["suggested_type"]
                            file_changed = True
                            
                        # 3. 將使用者說明納入備註
                        append_note = f"\n\n**【使用者反饋學術修正備註】**：{t['description']}"
                        if append_note not in q.get("detailed_solution", ""):
                            q["detailed_solution"] = q.get("detailed_solution", "") + append_note
                            file_changed = True
                            
                        t["status"] = "resolved"
                        t["resolved_at"] = "auto_repaired"
                        resolved_count += 1
                        
                if file_changed:
                    with open(full_p, "w", encoding="utf-8") as jf:
                        json.dump(q_list, jf, ensure_ascii=False, indent=4)
                        
    # 寫回已完成工單
    with open(TICKETS_FILE, "w", encoding="utf-8") as f:
        json.dump(tickets, f, ensure_ascii=False, indent=4)
        
    logging.info(f"🎉 [工單完成] 已成功自動修復 {resolved_count} 筆使用者回報的題目問題！")
    return resolved_count
    
def run_database_repair_suite():
    if not os.path.exists(DATABASE_DIR):
        print(f"❌ 找不到題庫目錄: {DATABASE_DIR}")
        return

    # 先執行工單套用
    apply_user_feedback_tickets()

    print("=" * 70)
    print("🚀 啟動【100+ 份試卷資料庫離線無損修復引擎】...")
    print("=" * 70)

    total_files = 0
    total_ghosts = 0
    total_merged = 0
    all_gaps_report = {}

    for root, dirs, files in os.walk(DATABASE_DIR):
        for f in files:
            if f.endswith("_database.json") and not f.endswith("_partial_database.json"):
                full_p = os.path.join(root, f)
                total_files += 1
                res = repair_single_database(full_p)
                
                total_ghosts += res["ghost_removed"]
                total_merged += res["duplicates_merged"]
                
                if res["ghost_removed"] > 0 or res["duplicates_merged"] > 0 or res["missing_gaps"]:
                    print(f"📦 [{res['file']}]")
                    if res["ghost_removed"] > 0:
                        print(f"  -> 🧹 清除幽靈假題目: {res['ghost_removed']} 題")
                    if res["duplicates_merged"] > 0:
                        print(f"  -> 🔄 智能合併重複題目: {res['duplicates_merged']} 題")
                    if res["missing_gaps"]:
                        print(f"  -> ⚠️ 偵測到可能遺漏之題號缺口: {', '.join(res['missing_gaps'])}")
                        all_gaps_report[res['file']] = res["missing_gaps"]

    print("=" * 70)
    print("🎉 【資料庫修復圓滿完成】統計摘要：")
    print(f"  * 總掃描試卷數量: {total_files} 份")
    print(f"  * 拔除幽靈試題總數: {total_ghosts} 題")
    print(f"  * 仲裁重題合併總數: {total_merged} 題")
    if all_gaps_report:
        print(f"  * 共有 {len(all_gaps_report)} 份考卷存在題號缺口（可依缺口清單進行針對性定向補件）。")
    print("=" * 70)

if __name__ == "__main__":
    run_database_repair_suite()