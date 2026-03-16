

---

## 📁 nlp_processing/README.md

```markdown
# NLP обработка текста

## 📋 Описание
Модуль предварительной обработки текста после OCR.

## 🔧 Функции очистки

### text_cleaning.py
| Функция | Назначение |
|---------|------------|
| `remove_table_html()` | Удаление HTML-таблиц |
| `remove_md_images()` | Удаление markdown-картинок |
| `remove_html_images()` | Удаление HTML-тегов img |
| `fix_hyphenated_words()` | Исправление переносов слов |
| `clean_document_text()` | Очистка от колонтитулов и мусора |
| `extract_clean_text_for_summary()` | Извлечение чистого текста для LLM |

### Фильтрация ключевых слов
- Нормализация через лемматизацию (spaCy)
- Отсев мусорных паттернов (даты, номера листов, служебные слова)
- Ограничение: 4-60 символов, только кириллица/ГОСТ/ТУ

## 🎯 Обработка сущностей
```python
extract_named_entities_spacy(text)
# Возвращает: ORG, GPE, PRODUCT, LAW, LOC
