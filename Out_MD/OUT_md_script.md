
---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 1 — Импорты и конфигурация
# ============================================================================
import os
import re
import json
import yaml
import time
import logging
import hashlib
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Подавление шумных логов
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("spacy").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("doc_pipeline")

try:
    import requests
    import spacy
    import ollama
except ImportError as e:
    logger.error(f"Необходимая библиотека не найдена: {e}")
    raise

@dataclass
class PipelineConfig:
    input_dir: Path = Path("InDOC")
    output_dir: Path = Path("outDOC")
    clean_text_dir: Path = Path("outDOC_clean")

    # Флаги этапов
    run_extraction: bool = True
    run_summarization: bool = False

    # Очистка md-тела при записи
    clean_md_text_on_write: bool = True

    # MinerU
    mineru_url: str = "http://172.18.12.45:8000/file_parse"

    # NLP
    spacy_model: str = "ru_core_news_lg"
    max_keywords: int = 25

    # Agno/Ollama для суммаризации
    ollama_summary_model: str = "qwen3:14b"
    ollama_host: str = "http://localhost:11434"
    summary_chunk_size: int = 20000
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 2 — Загрузка spaCy модели
# ============================================================================
nlp = None
try:
    nlp = spacy.load("ru_core_news_lg", disable=["parser", "lemmatizer"])
    logger.info("spaCy модель успешно загружена.")
except Exception as e:
    logger.error(f"Ошибка загрузки spaCy модели: {e}")
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 3 — Утилиты очистки текста
# ============================================================================
def remove_table_html(text: str) -> str:
    """Полное удаление HTML-таблиц и их содержимого из текста."""
    if not text:
        return ""
    text = re.sub(r"<table\b[^>]*>.*?</table>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<tr\b[^>]*>.*?</tr>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<td\b[^>]*>.*?</td>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<th\b[^>]*>.*?</th>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

def remove_md_images(text: str) -> str:
    """Удаляет markdown-картинки: ![alt](path)."""
    if not text:
        return ""
    return re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)

def remove_html_images(text: str) -> str:
    """Удаляет HTML-теги картинок: <img ...>."""
    if not text:
        return ""
    return re.sub(r"<img\b[^>]*?/?>", "", text, flags=re.IGNORECASE)

def fix_hyphenated_words(text: str) -> str:
    """Исправление переносов слов через дефис с пробелом."""
    if not text:
        return ""
    text = re.sub(r"([а-яёА-ЯЁ])- +([а-яёА-ЯЁ])", r"\1\2", text)
    text = re.sub(r"(\w)- +(\w)", r"\1\2", text)
    return text

def clean_document_text(text: str) -> str:
    """Очистка текста от колонтитулов, номеров страниц и шаблонного мусора."""
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    cleaned_lines = []
    page_num_pattern = re.compile(r"^\s*(?:лист|страница|стр\.|листок)?\s*\d+\s*$", re.IGNORECASE)
    change_pattern = re.compile(r"^изменение\s+№\s*\d+", re.IGNORECASE)

    header_footer_keywords = {
        "утверждаю", "согласовано", "принято", "проверил", "составил", "заместитель",
        "руководитель", "начальник", "директор", "генеральный", "зам", "гл", "инженер",
        "специалист", "отдел", "служба", "бюро", "управление", "дата", "код", "примечание",
        "разослать", "причина", "применяемость", "срок изм", "дата выпуска"
    }

    for ln in lines:
        ln_lower = ln.lower().strip()

        if page_num_pattern.match(ln_lower):
            continue
        if change_pattern.match(ln_lower):
            continue
        if any(trash in ln_lower for trash in ["rowspan", "colspan", "x6.", "x7.", "x8.", "x9.", "tr", "td", "div"]):
            continue
        if any(kw in ln_lower for kw in header_footer_keywords) and len(ln) < 80 and not re.search(r"[а-яё]{10,}", ln, re.IGNORECASE):
            continue
        if len(ln.strip()) < 3 and not ln.strip().isdigit():
            continue

        cleaned_lines.append(ln)

    cleaned_text = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines))
    return cleaned_text.strip()

def extract_clean_text_for_summary(text: str) -> str:
    """Извлечение чистого текста для суммаризации."""
    if not text:
        return ""

    text = remove_table_html(text)
    text = remove_html_images(text)
    text = remove_md_images(text)
    text = fix_hyphenated_words(text)

    lines = text.splitlines()
    content_lines = []

    for ln in lines:
        ln_strip = ln.strip()
        if not ln_strip:
            continue

        if "|" in ln_strip and ("---" in ln_strip or ln_strip.startswith("|") or ln_strip.endswith("|")):
            continue

        if re.match(r"^(Код|Дата выпуска|Срок изм|Применяемость|Разослать|Причина|Зам|Изм\.|Лист|Страница):\s*", ln_strip, re.IGNORECASE):
            continue

        if re.search(r"</?(tr|td|th|div|span|p|br)\b", ln_strip, re.IGNORECASE):
            continue

        if len(ln_strip) > 10 or re.search(r"[а-яё]{4,}", ln_strip, re.IGNORECASE):
            content_lines.append(ln_strip)

    return "\n".join(content_lines).strip()

def clean_metadata_values(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Очистка строковых значений метаданных."""
    if not meta:
        return {}
    cleaned = {}
    for key, value in meta.items():
        if isinstance(value, str):
            cleaned[key] = re.sub(r"\s+", " ", value.strip().replace("\uFEFF", "").replace("\ufeff", ""))
        else:
            cleaned[key] = value
    return cleaned
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 4 — Фильтрация ключевых слов
# ============================================================================
def normalize_keyword(keyword: str) -> str:
    """Нормализация ключевого слова через лемматизацию."""
    if not keyword:
        return ""
    if not nlp or not keyword.strip():
        return keyword.strip().lower()

    doc = nlp(keyword[:100])
    lemmas = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"NOUN", "ADJ", "PROPN"}
        and not token.is_stop
        and token.is_alpha
        and len(token.lemma_) > 2
    ]
    return " ".join(lemmas) if lemmas else keyword.strip().lower()

def filter_keywords(keywords: List[str]) -> List[str]:
    """Строгая фильтрация ключевых слов от мусора."""
    seen = set()
    filtered = []

    trash_patterns = [
        r"^(rowspan|colspan|table|tr|td|div|x\d+|x\d+\.\d+)$",
        r"^(лист|страница|стр\.|листок|зам|замечание|примечание|утверд|соглас|подп|дата|код|причина|примен|разослать)$",
        r"^(измен|изм|редакц|введ|действ|примеч|содерж|зам|провер|состав|руковод|началь|замест|генер|директор)$",
        r"^(отд|эхп|бнедр|тконтр|гл|инженер|специалист|ф[0-9]+)$",
        r"\d{2,}\.\d{2,}\.\d{2,}",
        r"[0-9]{4,}",
        r"^[a-z]+$",
        r"(тредован|избеш|укозан|содержон|бнедрени|согасоаон|коиизещ|разсать|дотительн|мучет|луст|лусм|лусмов)",
    ]

    for kw in keywords:
        kw_clean = kw.strip()
        if not kw_clean or len(kw_clean) < 4 or len(kw_clean) > 60:
            continue

        kw_lower = kw_clean.lower()

        if any(re.search(pat, kw_lower) for pat in trash_patterns):
            continue

        if re.search(r"\d{3,}", kw_clean) and not re.search(r"СТО|ГОСТ|ОСТ|ТУ", kw_clean, re.IGNORECASE):
            continue

        if re.search(r"[^а-яёА-ЯЁ\s\-\.]", kw_clean) and not re.search(r"СТО|ГОСТ|ОСТ|ТУ", kw_clean, re.IGNORECASE):
            continue

        norm = normalize_keyword(kw_clean)
        if not norm or norm in seen or len(norm) < 4:
            continue

        seen.add(norm)
        filtered.append(kw_clean)

    return filtered[:25]
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 5 — Извлечение ключевых слов
# ============================================================================
def extract_keywords_advanced(text: str, max_keywords: int = 25) -> List[str]:
    """Извлечение ключевых слов без noun_chunks."""
    if not text or not text.strip() or not nlp:
        return []

    clean_text = clean_document_text(text[:15000])
    if not clean_text:
        return []

    doc = nlp(clean_text[:10000])

    entities = [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ in {"ORG", "GPE", "PRODUCT", "LAW", "LOC"}
        and len(ent.text.strip()) > 4
        and not re.search(r"[0-9]{4,}", ent.text)
    ]

    terms = []
    current_term = []

    for token in doc:
        if token.pos_ in {"ADJ", "NOUN", "PROPN"} and not token.is_stop and token.is_alpha and len(token.text) > 2:
            current_term.append(token.text.strip())
        else:
            if current_term and 2 <= len(current_term) <= 5:
                term_text = " ".join(current_term)
                if 6 <= len(term_text) <= 60:
                    terms.append(term_text)
            current_term = []

    if current_term and 2 <= len(current_term) <= 5:
        term_text = " ".join(current_term)
        if 6 <= len(term_text) <= 60:
            terms.append(term_text)

    single_terms = [
        token.text.strip()
        for token in doc
        if token.pos_ in {"NOUN", "PROPN"}
        and not token.is_stop
        and token.is_alpha
        and len(token.text.strip()) > 4
        and (token.text.strip()[0].isupper() or len(token.text.strip()) > 6)
    ]

    combined = entities + terms + single_terms
    keywords = filter_keywords(combined)
    return keywords[:max_keywords]
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 6 — Именованные сущности + структура документа
# ============================================================================
def extract_named_entities_spacy(text: str) -> Dict[str, List[str]]:
    """Извлечение именованных сущностей из начала, середины и конца документа."""
    if not nlp or not text or not text.strip():
        return {}

    t = text.strip()
    n = len(t)

    if n <= 12000:
        windows = [t]
    else:
        chunk = 9000
        windows = [
            t[:chunk],
            t[max(0, n // 2 - chunk // 2): min(n, n // 2 + chunk // 2)],
            t[-chunk:]
        ]

    entities: Dict[str, List[str]] = {}
    for w in windows:
        doc = nlp(w)
        for ent in doc.ents:
            text_clean = ent.text.strip()
            if len(text_clean) < 3 or text_clean.isdigit():
                continue
            entities.setdefault(ent.label_, [])
            if text_clean not in entities[ent.label_]:
                entities[ent.label_].append(text_clean)

    return entities

def detect_document_structure(text: str) -> Dict[str, Any]:
    """Определение базовой структуры документа."""
    md = {
        "document_id": "",
        "document_type": "",
        "date": "",
    }

    id_pattern = r"(?:СТ[ОП]|ОСТ|ГОСТ)\s+Ж?\d+(?:\.\d+)?(?:-\d+)?(?:\s+№\s*\d+)?"
    m = re.search(id_pattern, text[:3000], re.IGNORECASE)
    if m:
        md["document_id"] = m.group(0).strip().replace("  ", " ")

    date_pattern = r"\d{2}\.\d{2}\.\d{4}"
    dates = re.findall(date_pattern, text[:5000])
    if dates:
        md["date"] = dates[0]

    t_upper = text[:800].upper()
    if "СТАНДАРТ" in t_upper:
        md["document_type"] = "стандарт"
    elif "ИНСТРУКЦИЯ" in t_upper:
        md["document_type"] = "инструкция"
    elif "ПРИКАЗ" in t_upper or "РАСПОРЯЖЕНИЕ" in t_upper:
        md["document_type"] = "приказ"
    elif "ПОЛОЖЕНИЕ" in t_upper:
        md["document_type"] = "положение"
    elif "АКТ" in t_upper and ("ПРОВЕРКИ" in t_upper or "ПРОВЕРКА" in t_upper):
        md["document_type"] = "акт проверки"
    elif "СВОДКА" in t_upper and "ОТЗЫВОВ" in t_upper:
        md["document_type"] = "сводка отзывов"
    else:
        md["document_type"] = "документ"

    return md

def generate_unique_document_id(meta: Dict[str, Any]) -> str:
    """Генерация уникального ID документа."""
    обозначение = meta.get("обозначение_документа", "").strip()
    наименование = meta.get("наименование_документа", "").strip()

    if обозначение:
        clean_id = re.sub(r"[^\w\-\.]", "_", обозначение)
        if наименование:
            name_hash = hashlib.md5(наименование.encode("utf-8")).hexdigest()[:8]
            return f"{clean_id}_{name_hash}"
        return clean_id

    if наименование:
        name_hash = hashlib.md5(наименование.encode("utf-8")).hexdigest()[:12]
        return f"DOC_{name_hash}"

    return f"DOC_{int(time.time())}"
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 7 — Клиент MinerU
# ============================================================================
class MinerUClient:
    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()

    def process(self, file_path: Path) -> Dict[str, Any]:
        try:
            with open(file_path, "rb") as f:
                files = {"files": (file_path.name, f, "application/octet-stream")}
                data = {
                    "output_dir": "./output",
                    "lang_list": "cyrillic",
                    "backend": "pipeline",
                    "parse_method": "auto",
                    "formula_enable": True,
                    "table_enable": True,
                    "return_md": True,
                    "return_images": True,
                    "response_format_zip": False,
                    "start_page_id": 0,
                    "end_page_id": 99999,
                }
                r = self.session.post(self.url, files=files, data=data, timeout=600)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            logger.error(f"MinerU ошибка для {file_path.name}: {e}")
            return {}

def mineru_extract_text(result: Dict[str, Any]) -> str:
    if "results" in result:
        for _, data in result["results"].items():
            if isinstance(data, dict) and data.get("md_content"):
                return data["md_content"]
    return ""

def mineru_extract_pages(result: Dict[str, Any]) -> int:
    if "results" in result:
        for _, data in result["results"].items():
            if isinstance(data, dict) and data.get("page_count"):
                return int(data["page_count"])
    return 0
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 8 — Извлечение текста
# ============================================================================
class TextExtractor:
    def __init__(self, cfg: PipelineConfig, mineru: Optional[MinerUClient] = None):
        self.cfg = cfg
        self.mineru = mineru

    def extract(self, file_path: Path) -> Tuple[str, int, Dict[str, Any]]:
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return self._extract_pdf(file_path)
        elif ext == ".docx":
            text = self._extract_docx(file_path)
            return text, 0, {"extractor": "docx"}
        elif ext in {".txt", ".md"}:
            text = self._read_text_file(file_path)
            return text, 0, {"extractor": "plain_text"}
        else:
            if self.mineru:
                res = self.mineru.process(file_path)
                text = mineru_extract_text(res)
                pages = mineru_extract_pages(res)
                return text, pages, {"extractor": "mineru_fallback"}
            return "", 0, {"extractor": "none", "error": f"unsupported ext={ext}"}

    def _extract_pdf(self, file_path: Path) -> Tuple[str, int, Dict[str, Any]]:
        """Обязательная обработка PDF через MinerU."""
        if not self.mineru:
            logger.error(f"MinerU не настроен, но требуется для обработки PDF: {file_path.name}")
            return "", 0, {"extractor": "none", "error": "mineru_required_for_pdf"}

        logger.info(f"Обработка PDF через MinerU: {file_path.name}")
        res = self.mineru.process(file_path)
        text = mineru_extract_text(res)
        pages = mineru_extract_pages(res)

        if text.strip():
            return text, pages, {"extractor": "mineru_pdf", "pages": pages}

        logger.warning(f"MinerU вернул пустой текст для: {file_path.name}")
        return "", pages, {"extractor": "mineru_pdf", "warning": "empty_text"}

    def _extract_docx(self, file_path: Path) -> str:
        try:
            import docx
            d = docx.Document(file_path)
            parts = [p.text.strip() for p in d.paragraphs if p.text.strip()]
            for table in d.tables:
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        parts.append(" | ".join(cells))
            return "\n".join(parts).strip()
        except Exception as e:
            logger.warning(f"Ошибка извлечения DOCX {file_path.name}: {e}")
            return ""

    def _read_text_file(self, file_path: Path) -> str:
        for enc in ("utf-8", "cp1251"):
            try:
                return file_path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return file_path.read_text(encoding="utf-8", errors="ignore")
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 9 — Чтение titul.md и объединение метаданных
# ============================================================================
def load_titul_metadata(dir_path: Path) -> Dict[str, Any]:
    """Чтение метаданных из titul.md."""
    titul_path = dir_path / "titul.md"
    if not titul_path.exists():
        return {}

    try:
        raw = titul_path.read_text(encoding="utf-8-sig", errors="ignore").strip()
    except Exception:
        raw = titul_path.read_text(encoding="utf-8", errors="ignore").strip()

    if not raw:
        return {}

    meta: Dict[str, Any] = {}

    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            try:
                data = yaml.safe_load(parts[1].strip())
                if isinstance(data, dict):
                    meta = data
            except Exception as e:
                logger.debug(f"Ошибка парсинга YAML в {titul_path}: {e}")

    if not meta:
        current_key = None
        current_value = []

        for line in raw.splitlines():
            line = line.rstrip()
            if not line.strip():
                continue

            if ":" in line and not line.startswith((" ", "\t")):
                if current_key is not None:
                    meta[current_key] = " ".join(current_value).strip()
                k, v = line.split(":", 1)
                current_key = k.strip()
                current_value = [v.strip()] if v.strip() else []
            elif current_key is not None:
                current_value.append(line.strip())

        if current_key is not None:
            meta[current_key] = " ".join(current_value).strip()

    return clean_metadata_values(meta)

def merge_metadata(base: Dict[str, Any], titul: Dict[str, Any]) -> Dict[str, Any]:
    """Объединение метаданных без дублирования title."""
    merged = {**base}
    for k, v in (titul or {}).items():
        if k != "title":
            merged[k] = v
    merged.pop("title", None)
    return merged
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 10 — Улучшение структуры Markdown
# ============================================================================
def is_probable_heading(line: str) -> bool:
    """Проверка, похожа ли строка на заголовок."""
    if not line:
        return False

    s = line.strip()
    if len(s) < 3 or len(s) > 180:
        return False

    if s.endswith(":") and len(s.split()) <= 8:
        return True

    heading_patterns = [
        r"^\d+\s+[А-ЯЁA-Z].{1,120}$",                  # 1 Область применения
        r"^\d+\.\d+\s+[А-ЯЁA-Z].{1,120}$",            # 1.1 Общие положения
        r"^\d+\.\d+\.\d+\s+[А-ЯЁA-Z].{1,120}$",       # 1.1.1 Порядок
        r"^\d+\.\s+[А-ЯЁA-Z].{1,120}$",               # 1. Наименование
        r"^(раздел|глава)\s+\d+\.?\s*.*$",            # Раздел 1
        r"^приложение\s+[А-ЯA-Z0-9]+\.?\s*.*$",       # Приложение А
        r"^(введение|заключение|общие положения|область применения|нормативные ссылки|термины и определения)$",
    ]

    s_lower = s.lower()
    return any(re.match(p, s_lower if "раздел" in p or "приложение" in p or "введение" in p else s) for p in heading_patterns)

def classify_heading(line: str) -> Optional[Tuple[int, str]]:
    """Определяет уровень заголовка и текст."""
    s = line.strip()
    if not s:
        return None

    patterns = [
        (r"^(\d+)\s+(.+)$", 2),
        (r"^(\d+)\.\s+(.+)$", 2),
        (r"^(\d+\.\d+)\s+(.+)$", 3),
        (r"^(\d+\.\d+\.\d+)\s+(.+)$", 4),
        (r"^(Раздел\s+\d+\.?\s*.*)$", 2),
        (r"^(Глава\s+\d+\.?\s*.*)$", 2),
        (r"^(Приложение\s+[А-ЯA-Z0-9]+\.?\s*.*)$", 2),
        (r"^(Введение|Заключение|Общие положения|Область применения|Нормативные ссылки|Термины и определения)$", 2),
    ]

    for pattern, level in patterns:
        m = re.match(pattern, s, flags=re.IGNORECASE)
        if m:
            if len(m.groups()) == 2:
                return level, f"{m.group(1)} {m.group(2)}".strip()
            return level, m.group(1).strip()

    return None

def improve_markdown_structure(text: str) -> str:
    """
    Улучшает структуру markdown-текста:
    - выделяет разделы, подразделы, подподразделы;
    - убирает шум;
    - сохраняет нумерацию заголовков;
    - формирует более читаемую структуру.
    """
    if not text:
        return ""

    text = clean_md_body_text_for_storage(text)
    text = fix_hyphenated_words(text)

    raw_lines = text.splitlines()
    result_lines = []

    for line in raw_lines:
        ln = line.strip()

        if not ln:
            if result_lines and result_lines[-1] != "":
                result_lines.append("")
            continue

        if any(x in ln.lower() for x in ["rowspan", "colspan", "x6.", "x7.", "x8.", "x9."]):
            continue

        heading = classify_heading(ln)
        if heading:
            level, heading_text = heading
            result_lines.append("")
            result_lines.append(f'{"#" * level} {heading_text}')
            result_lines.append("")
            continue

        # Определения вида "1.1.1 Термин: определение"
        term_match = re.match(r"^(\d+\.\d+\.\d+)\s+([А-Яа-яЁёA-Za-z][^:]{1,80}):\s*(.+)$", ln)
        if term_match:
            num = term_match.group(1).strip()
            term = term_match.group(2).strip()
            definition = term_match.group(3).strip()
            result_lines.append(f"- **{num} {term}**: {definition}")
            continue

        # Нумерованные пункты
        if re.match(r"^\d+\.\d+\.\d+\s+", ln):
            result_lines.append(f"- {ln}")
            continue

        if re.match(r"^\d+\.\d+\s+", ln) and not is_probable_heading(ln):
            result_lines.append(f"- {ln}")
            continue

        result_lines.append(ln)

    cleaned = []
    empty_count = 0
    for ln in result_lines:
        if not ln:
            empty_count += 1
            if empty_count <= 1:
                cleaned.append("")
        else:
            empty_count = 0
            cleaned.append(ln)

    return "\n".join(cleaned).strip()
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 11 — Обогащение + запись Markdown (Этап 1)
# ============================================================================
def enhance_metadata_nlp(text: str, meta: Dict[str, Any], cfg: PipelineConfig) -> Dict[str, Any]:
    """Обогащение метаданных ключевыми словами и сущностями."""
    meta["keywords"] = extract_keywords_advanced(text, max_keywords=cfg.max_keywords)
    meta["named_entities"] = extract_named_entities_spacy(text)

    struct_meta = detect_document_structure(text)
    meta.update(struct_meta)

    meta["document_id"] = generate_unique_document_id(meta)
    return meta

def filter_empty_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Фильтрация пустых и служебных метаданных."""
    excluded_fields = {
        "processed_at", "summary_status", "summary_model", "extractor", "summary_generated_at"
    }

    filtered = {}
    for k, v in meta.items():
        if k in excluded_fields:
            continue
        if v is None:
            continue
        if isinstance(v, str) and (not v.strip() or v.strip() in {"''", '""', "0"}):
            continue
        if isinstance(v, int) and v == 0 and k in {"pages", "page_count"}:
            continue
        if isinstance(v, (list, dict)) and not v:
            continue
        filtered[k] = v

    return filtered

def clean_md_body_text_for_storage(md_text: str) -> str:
    """Чистит markdown-текст для хранения в outDOC."""
    if not md_text:
        return ""

    t = md_text
    t = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", t)
    t = re.sub(r"<img\b[^>]*?/?>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<table\b[^>]*>.*?</table>", "", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"</?(tr|td|th)\b[^>]*>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"</?(div|span|p|br)\b[^>]*>", "", t, flags=re.IGNORECASE)

    bad_line_patterns = [
        r"^\s*Wlld\)\w*.*$",
        r"^\s*Bx\.\s*N\s*$",
        r"^\s*вх\.\s*№\s*.*$",
        r"^\s*_{3,}\s*$",
        r"^\s*[-–]{3,}\s*$",
    ]

    lines = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            lines.append("")
            continue
        if any(re.match(pat, s, flags=re.IGNORECASE) for pat in bad_line_patterns):
            continue
        lines.append(ln.rstrip())

    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

def write_markdown_stage1(out_path: Path, meta: Dict[str, Any], title: str, text: str, clean_text: str, cfg: PipelineConfig):
    """Запись markdown на этапе 1: YAML + структурированный текст."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_meta = filter_empty_metadata(meta)
    doc_id = filtered_meta.pop("document_id", "")
    fm_ordered = {"document_id": doc_id}
    fm_ordered.update(filtered_meta)

    body_text = (text or "").strip()
    if cfg.clean_md_text_on_write:
        body_text = clean_md_body_text_for_storage(body_text)

    structured_body = improve_markdown_structure(body_text)

    parts = [
        "---",
        yaml.dump(fm_ordered, allow_unicode=True, sort_keys=False).strip(),
        "---",
        "",
        f"# {title}",
        "",
        "## Текст",
        "",
        structured_body if structured_body else "_Текст отсутствует_",
        "",
    ]

    out_path.write_text("\n".join(parts), encoding="utf-8")

    clean_path = cfg.clean_text_dir / out_path.relative_to(cfg.output_dir).with_suffix(".txt")
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.write_text(clean_text or "", encoding="utf-8")

    logger.debug(f"Сохранён markdown: {out_path}")
    logger.debug(f"Сохранён clean-text: {clean_path}")
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 12 — Этап 1: Запуск извлечения
# ============================================================================
def process_one_file_stage1(file_path: Path, cfg: PipelineConfig, extractor: TextExtractor) -> Optional[Dict[str, Any]]:
    """Этап 1: извлечение текста и запись в outDOC / outDOC_clean."""
    titul = load_titul_metadata(file_path.parent)
    text, _, meta_info = extractor.extract(file_path)

    if not text.strip():
        logger.warning(f"ПУСТОЙ ТЕКСТ: {file_path.name} (извлечено через {meta_info.get('extractor', 'unknown')})")
        return None

    base_meta = {
        "source_path": str(file_path.resolve()),
        "original_filename": file_path.name,
        "pages": meta_info.get("pages", 0),
    }

    meta = merge_metadata(base_meta, titul)
    meta = enhance_metadata_nlp(text, meta, cfg)
    clean_text = extract_clean_text_for_summary(text)

    rel = file_path.relative_to(cfg.input_dir)
    md_path = cfg.output_dir / rel.with_suffix(".md")

    title = meta.get("наименование_документа") or meta.get("document_id") or file_path.stem
    write_markdown_stage1(md_path, meta, title=title, text=text, clean_text=clean_text, cfg=cfg)

    return {
        "file_path": str(file_path),
        "markdown_path": str(md_path),
        "clean_text_saved": True
    }

def run_stage1(cfg: PipelineConfig) -> List[Dict[str, Any]]:
    """Запуск этапа 1: извлечение текста."""
    logger.info("=== ЭТАП 1: ИЗВЛЕЧЕНИЕ ТЕКСТА ===")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.clean_text_dir.mkdir(parents=True, exist_ok=True)

    mineru = MinerUClient(cfg.mineru_url)
    extractor = TextExtractor(cfg, mineru=mineru)

    files_to_process = [
        p for p in cfg.input_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}
        and p.name != "titul.md"
    ]

    logger.info(f"Найдено файлов для обработки: {len(files_to_process)}")

    results = []
    for p in files_to_process:
        try:
            r = process_one_file_stage1(p, cfg, extractor)
            if r:
                results.append(r)
                logger.info(f"OK: {p.name} -> {Path(r['markdown_path']).name}")
        except Exception as e:
            logger.exception(f"ОШИБКА: {p.name} | {e}")

    logger.info(f"Этап 1 завершён. Обработано файлов: {len(results)}")
    return results
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 13 — Pydantic схемы для структурированного резюме
# ============================================================================
from pydantic import BaseModel, Field

class Entity(BaseModel):
    """Сущность, извлечённая из документа."""
    name: str = Field(description="Имя сущности на русском языке")
    type: str = Field(description="Тип сущности только на русском: персона, организация, дата, место, технология, стандарт, подразделение, прочее")
    context: Optional[str] = Field(default=None, description="Краткий контекст упоминания на русском языке")

class ActionItem(BaseModel):
    """Задача или действие, выявленное в документе."""
    task: str = Field(description="Действие, которое необходимо выполнить, на русском языке")
    owner: Optional[str] = Field(default=None, description="Исполнитель, если указан")
    deadline: Optional[str] = Field(default=None, description="Срок выполнения, если указан")
    priority: Optional[str] = Field(default=None, description="Приоритет только на русском: высокий, средний, низкий")

class DocumentSummary(BaseModel):
    """Структурированное резюме документа."""
    title: str = Field(description="Заголовок документа или выведенная тема на русском языке")
    document_type: str = Field(description="Тип документа только на русском: отчёт, инструкция, акт, стандарт, положение, письмо, протокол, прочее")
    summary: str = Field(description="Краткое резюме в 1-3 абзацах на русском языке")
    key_points: List[str] = Field(description="3-7 ключевых тезисов на русском языке")
    entities: List[Entity] = Field(default_factory=list, description="Ключевые сущности")
    action_items: List[ActionItem] = Field(default_factory=list, description="Выявленные задачи и следующие шаги")
    word_count: int = Field(description="Количество слов в исходном документе")
    confidence: float = Field(description="Уверенность в точности резюме (0.0-1.0)", ge=0.0, le=1.0)
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 14 — Инструмент чтения текстовых файлов
# ============================================================================
def read_text_file(file_path: str) -> str:
    """Чтение содержимого текстового файла для Agno Agent."""
    path = Path(file_path)

    if not path.exists():
        return f"Ошибка: файл не найден: {file_path}"

    supported_extensions = {".txt", ".md", ".markdown", ".rst", ".text"}
    if path.suffix.lower() not in supported_extensions:
        return f"Ошибка: неподдерживаемый тип файла: {path.suffix}"

    try:
        encodings = ["utf-8", "utf-8-sig", "cp1251", "latin-1", "cp1252"]
        content = None

        for encoding in encodings:
            try:
                content = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            return "Ошибка: не удалось декодировать файл поддерживаемыми кодировками"

        logger.info(f"Прочитан текстовый файл: {path.name} ({len(content)} символов)")

        word_count = len(content.split())
        metadata = (
            f"Файл: {path.name}\n"
            f"Тип: {path.suffix}\n"
            f"Количество слов: {word_count}\n"
            f"Содержимое:\n"
        )
        return metadata + content

    except Exception as e:
        logger.error(f"Ошибка чтения файла {file_path}: {e}")
        return f"Ошибка чтения файла: {e}"
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 15 — Настройка Agno Agent с усилением русского языка
# ============================================================================
import sys
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.ollama import Ollama
from agno.tools.reasoning import ReasoningTools
from agno.culture.manager import CultureManager
from agno.db.schemas.culture import CulturalKnowledge

agent_db = SqliteDb(db_file="tmp/agents.db")

culture_manager = CultureManager(db=agent_db)
document_search_culture = CulturalKnowledge(
    name="Инструкция для агента суммаризации промышленных документов",
    summary="Правила анализа и краткого изложения технической, отчётной и регламентной документации",
    categories=["summarization", "industrial_docs", "language", "ux"],
    content="""
Вы — эксперт по анализу производственной документации.

КРИТИЧЕСКОЕ ТРЕБОВАНИЕ:
- Всегда отвечайте только на русском языке.
- Запрещено использовать английский язык в полях структурированного ответа.
- Если в документе есть английские термины, допускается сохранить сам термин как цитату или обозначение, но объяснение и весь связный текст должны быть на русском.
- Если модель начала формировать ответ на английском, она должна немедленно переформулировать ответ полностью на русском.

ФОРМАТ:
1. Краткое резюме.
2. Ключевые тезисы.
3. Сущности.
4. Задачи и действия.

ОГРАНИЧЕНИЯ:
- Не выдумывать данные.
- Не добавлять факты, которых нет в документе.
- При нехватке информации прямо указывать это на русском языке.
""",
    notes=[
        "Строго русский язык",
        "Техническая точность выше стилистики",
        "Не допускать англоязычных полей в structured output"
    ],
    metadata={
        "source": "industrial_summarization_policy",
        "language": "ru",
        "version": 2,
    },
)
culture_manager.add_cultural_knowledge(document_search_culture)

SYSTEM_MESSAGE = """
Вы — эксперт по анализу производственной документации.

СТРОГОЕ СИСТЕМНОЕ ТРЕБОВАНИЕ:
1. Отвечайте исключительно на русском языке.
2. Любой английский связный текст в ответе запрещён.
3. Все значения полей output_schema должны быть только на русском языке.
4. Значения полей type, document_type, priority должны быть только на русском языке.
5. Если исходный документ частично на английском, всё равно формируйте резюме на русском.
6. Если вы случайно начали писать по-английски, немедленно перепишите весь ответ полностью на русском.

Обязанности:
- Сформировать краткое и точное резюме документа.
- Выделить ключевые тезисы.
- Извлечь сущности.
- Выявить задачи и действия.

Требования к качеству:
- Не придумывать факты.
- Не искажать термины.
- При сомнении указывать это явно.
- Использовать профессиональный, но естественный русский язык.
- Перед ответом мысленно спланировать структуру.

Допустимые значения:
- document_type: отчёт, инструкция, акт, стандарт, положение, письмо, протокол, прочее
- entity.type: персона, организация, дата, место, технология, стандарт, подразделение, прочее
- priority: высокий, средний, низкий
"""

summarizer_agent = Agent(
    name="Суммаризатор производственных документов",
    model=Ollama(id="qwen3:14b", host="http://172.18.12.45:11434"),
    system_message=SYSTEM_MESSAGE,
    add_culture_to_context=True,
    output_schema=DocumentSummary,
    tools=[
        ReasoningTools(add_instructions=True),
        read_text_file,
    ],
    add_datetime_to_context=True,
    add_history_to_context=True,
    num_history_runs=3,
    enable_agentic_memory=True,
    markdown=True,
    db=agent_db,
    telemetry=False,
    instructions=[
        "Всегда отвечай только на русском языке",
        "Запрещено использовать английский язык в summary, key_points, entities, action_items",
        "Если встречаются английские термины, поясняй их только по-русски",
        "Все значения полей схемы должны быть заполнены на русском языке",
        "Если ответ получился частично на английском, перепиши его полностью на русском"
    ]
)
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 16 — Функции контроля русского языка и суммаризации
# ============================================================================
def contains_too_much_latin(text: str, threshold: float = 0.15) -> bool:
    """Проверяет, не слишком ли много латиницы в тексте."""
    if not text:
        return False
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    if not letters:
        return False
    latin = re.findall(r"[A-Za-z]", text)
    return (len(latin) / max(len(letters), 1)) > threshold

def normalize_summary_to_russian(summary: DocumentSummary) -> DocumentSummary:
    """Нормализует отдельные значения к русским вариантам."""
    doc_type_map = {
        "report": "отчёт",
        "article": "статья",
        "meeting_notes": "протокол",
        "research_paper": "исследование",
        "email": "письмо",
        "other": "прочее",
        "instruction": "инструкция",
        "standard": "стандарт",
        "act": "акт",
    }
    entity_type_map = {
        "person": "персона",
        "organization": "организация",
        "date": "дата",
        "location": "место",
        "technology": "технология",
        "standard": "стандарт",
        "department": "подразделение",
        "other": "прочее",
    }
    priority_map = {
        "high": "высокий",
        "medium": "средний",
        "low": "низкий",
    }

    if summary.document_type:
        summary.document_type = doc_type_map.get(summary.document_type.lower(), summary.document_type)

    for ent in summary.entities:
        if ent.type:
            ent.type = entity_type_map.get(ent.type.lower(), ent.type)

    for item in summary.action_items:
        if item.priority:
            item.priority = priority_map.get(item.priority.lower(), item.priority)

    return summary

def summary_needs_russian_retry(summary: DocumentSummary) -> bool:
    """Проверка, нужен ли повторный запрос из-за ухода в английский."""
    texts = [summary.title, summary.document_type, summary.summary]
    texts.extend(summary.key_points)

    for ent in summary.entities:
        texts.append(ent.name or "")
        texts.append(ent.type or "")
        texts.append(ent.context or "")

    for item in summary.action_items:
        texts.append(item.task or "")
        texts.append(item.owner or "")
        texts.append(item.deadline or "")
        texts.append(item.priority or "")

    joined = "\n".join([t for t in texts if t])
    return contains_too_much_latin(joined, threshold=0.12)

def summarize_text_file(file_path: str) -> DocumentSummary:
    """Суммирование одного текстового файла через Agno Agent с повторной попыткой на русском."""
    content = read_text_file(file_path)
    if content.startswith("Ошибка:"):
        raise ValueError(content)

    prompt = (
        "Суммируй следующий документ.\n"
        "КРИТИЧЕСКОЕ ТРЕБОВАНИЕ: ответ и все поля схемы должны быть ТОЛЬКО на русском языке.\n"
        "Нельзя использовать английские названия типов, приоритетов и категорий.\n"
        "Если в исходном тексте есть английские фрагменты, перескажи их по-русски.\n\n"
        f"{content}"
    )

    response = summarizer_agent.run(prompt)

    if not response.content or not isinstance(response.content, DocumentSummary):
        raise ValueError("Не удалось сгенерировать структурированное резюме")

    summary = normalize_summary_to_russian(response.content)

    if summary_needs_russian_retry(summary):
        logger.warning(f"Обнаружен уход в английский, повторяем суммаризацию: {file_path}")

        retry_prompt = (
            "Переформулируй структурированное резюме заново.\n"
            "СТРОГОЕ ТРЕБОВАНИЕ: весь ответ и все значения полей — только на русском языке.\n"
            "Английские слова допускаются только как буквенные обозначения из исходного документа, "
            "но пояснение, формулировки и классификация должны быть на русском.\n"
            "Верни результат заново по той же схеме.\n\n"
            f"{content}"
        )

        retry_response = summarizer_agent.run(retry_prompt)
        if retry_response.content and isinstance(retry_response.content, DocumentSummary):
            summary = normalize_summary_to_russian(retry_response.content)

    return summary
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 17 — Обновление Markdown с структурированным резюме
# ============================================================================
def update_markdown_with_structured_summary(md_path: Path, summary: Optional[DocumentSummary], model: str):
    """Обновляет markdown, добавляя или заменяя секцию краткого содержания."""
    content = md_path.read_text(encoding="utf-8", errors="ignore")

    fm = {}
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_raw = parts[1].strip()
            body = parts[2].lstrip("\n")
            try:
                fm = yaml.safe_load(fm_raw) or {}
            except Exception:
                fm = {}

    doc_id = fm.pop("document_id", "")
    fm["summary_generated_at"] = datetime.now().isoformat()
    fm["summary_model"] = model

    fm_ordered = {"document_id": doc_id}
    fm_ordered.update(fm)

    if summary:
        summary_block = []
        summary_block.append(summary.summary.strip())
        summary_block.append("")
        summary_block.append("### Ключевые пункты")
        for i, point in enumerate(summary.key_points, 1):
            summary_block.append(f"{i}. {point}")

        if summary.entities:
            summary_block.append("")
            summary_block.append("### Сущности")
            for ent in summary.entities[:10]:
                line = f"- **{ent.name}** ({ent.type})"
                if ent.context:
                    line += f" — {ent.context}"
                summary_block.append(line)

        if summary.action_items:
            summary_block.append("")
            summary_block.append("### Задачи")
            for item in summary.action_items:
                line = f"- {item.task}"
                if item.owner:
                    line += f" (Ответственный: {item.owner})"
                if item.deadline:
                    line += f" [Срок: {item.deadline}]"
                if item.priority:
                    line += f" [Приоритет: {item.priority}]"
                summary_block.append(line)

        summary_block.append("")
        summary_block.append(f"_Уверенность: {summary.confidence}_")
        summary_block_text = "\n".join(summary_block).strip()
    else:
        summary_block_text = "_Суммаризация не удалась_"

    sec_re = re.compile(
        r"(^##\s*Краткое содержание\s*\n)(.*?)(?=^\s*##\s|\Z)",
        re.DOTALL | re.MULTILINE
    )

    if sec_re.search(body):
        body = sec_re.sub(rf"\1\n{summary_block_text}\n\n", body, count=1).strip() + "\n"
    else:
        lines = body.splitlines()
        out_lines = []
        inserted = False

        for line in lines:
            out_lines.append(line)
            if not inserted and line.startswith("# "):
                out_lines.append("")
                out_lines.append("## Краткое содержание")
                out_lines.append("")
                out_lines.append(summary_block_text)
                out_lines.append("")
                inserted = True

        if not inserted:
            body = f"## Краткое содержание\n\n{summary_block_text}\n\n" + body.strip() + "\n"
        else:
            body = "\n".join(out_lines).strip() + "\n"

    new_content_parts = [
        "---",
        yaml.dump(fm_ordered, allow_unicode=True, sort_keys=False).strip(),
        "---",
        "",
        body.strip(),
        "",
    ]

    md_path.write_text("\n".join(new_content_parts), encoding="utf-8")
    logger.info(f"Обновлён файл с кратким содержанием: {md_path.name}")
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 18 — Этап 2: Запуск суммаризации
# ============================================================================
def run_stage2(cfg: PipelineConfig) -> List[Dict[str, Any]]:
    """Запуск этапа 2: суммаризация через Agno Agent."""
    logger.info("=== ЭТАП 2: СУММАРИЗАЦИЯ ЧЕРЕЗ AGNO AGENT ===")

    try:
        models_response = ollama.list()

        if isinstance(models_response, dict) and "models" in models_response:
            model_names = [
                m.get("name", "").split(":")[0]
                for m in models_response["models"]
                if isinstance(m, dict)
            ]
        else:
            model_names = [str(m).split(":")[0] for m in models_response]

        target_model_base = cfg.ollama_summary_model.split(":")[0]
        if target_model_base not in model_names:
            logger.warning(f"Модель {cfg.ollama_summary_model} не найдена. Доступные модели: {model_names}")

        if not model_names:
            logger.error("Ollama запущен, но нет загруженных моделей. Загрузите модель: ollama pull qwen3:14b")
            return []

        logger.info(f"Ollama доступен. Используем модель: {cfg.ollama_summary_model}")

    except Exception as e:
        logger.error(f"Не удалось подключиться к Ollama: {e}")
        logger.error("Убедитесь, что Ollama запущен: ollama serve")
        return []

    md_files = list(cfg.output_dir.rglob("*.md"))
    logger.info(f"Найдено Markdown файлов для суммаризации: {len(md_files)}")

    results: List[Dict[str, Any]] = []

    for md_path in md_files:
        try:
            content = md_path.read_text(encoding="utf-8", errors="ignore")

            if '"summary_generated_at"' in content or "summary_generated_at:" in content:
                logger.debug(f"Пропускаем (уже обработан): {md_path.name}")
                continue

            rel = md_path.relative_to(cfg.output_dir)
            clean_path = (cfg.clean_text_dir / rel).with_suffix(".txt")

            if not clean_path.exists():
                logger.warning(f"Не найден clean-text файл: {clean_path}")
                update_markdown_with_structured_summary(md_path, None, cfg.ollama_summary_model)
                results.append({"file": md_path.name, "status": "failed", "error": "missing_clean_text"})
                continue

            doc_text_clean = clean_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not doc_text_clean:
                logger.warning(f"Пустой clean-text файл: {clean_path}")
                update_markdown_with_structured_summary(md_path, None, cfg.ollama_summary_model)
                results.append({"file": md_path.name, "status": "failed", "error": "empty_clean_text"})
                continue

            summary = summarize_text_file(str(clean_path))
            update_markdown_with_structured_summary(md_path, summary, cfg.ollama_summary_model)

            results.append({
                "file": md_path.name,
                "status": "completed",
                "confidence": summary.confidence
            })

            time.sleep(1.5)

        except Exception as e:
            logger.exception(f"Ошибка суммаризации файла {md_path.name}: {e}")
            results.append({"file": md_path.name, "status": "error", "error": str(e)})

    logger.info(f"Этап 2 завершён. Обработано файлов: {len(results)}")
    return results
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 19 — Запуск только оцифровки (Этап 1)
# ============================================================================
cfg = PipelineConfig(
    input_dir=Path("InDOC"),
    output_dir=Path("outDOC"),
    clean_text_dir=Path("outDOC_clean"),
    run_extraction=True,
    run_summarization=False,
    ollama_summary_model="qwen3:14b",
    ollama_host="http://localhost:11434",
)

results_stage1 = []

if cfg.run_extraction:
    logger.info("Запуск ЭТАПА 1: оцифровка и подготовка Markdown. Суммаризация НЕ выполняется.")
    results_stage1 = run_stage1(cfg)
    print(f"\nЭтап 1 завершён. Обработано файлов: {len(results_stage1)}")

    if results_stage1:
        sample_file = Path(results_stage1[0]["markdown_path"])
        if sample_file.exists():
            try:
                content = sample_file.read_text(encoding="utf-8")
                fm_part = content.split("---")[1].strip()
                fm = yaml.safe_load(fm_part)
                print("\nПример метаданных из первого файла:")
                print(f"  document_id: {fm.get('document_id', 'N/A')}")
                print("  Пример ключевых слов:")
                for i, kw in enumerate(fm.get("keywords", [])[:5], 1):
                    print(f"    {i}. {kw}")
            except Exception as e:
                logger.warning(f"Не удалось прочитать метаданные: {e}")
else:
    print("Этап 1 отключён.")
```

---

```python
# ============================================================================
# ✅ ЯЧЕЙКА 20 — Запуск только суммаризации (Этап 2)
# ============================================================================
cfg = PipelineConfig(
    input_dir=Path("InDOC"),
    output_dir=Path("outDOC"),
    clean_text_dir=Path("outDOC_clean"),
    run_extraction=False,
    run_summarization=True,
    ollama_summary_model="qwen3:14b",
    ollama_host="http://localhost:11434",
)

results_stage2 = []

if cfg.run_summarization:
    logger.info("Запуск ЭТАПА 2: суммаризация по уже подготовленным clean-text файлам.")
    results_stage2 = run_stage2(cfg)
    print(f"\nЭтап 2 завершён. Результаты суммаризации:")

    for r in results_stage2[:10]:
        status = "✅" if r["status"] == "completed" else "❌"
        conf = f"(conf: {r.get('confidence', 0):.2f})" if r.get("confidence") else ""
        err = f" ({r['error']})" if r.get("error") else ""
        print(f"  {status} {r['file']}: {r['status']} {conf}{err}")
else:
    print("Этап 2 отключён.")
```

---

