# AI-система для анализа научных публикаций

Система для поиска и синтеза знаний из корпуса научных статей с использованием нескольких подходов RAG.

## Основные характеристики
- **Источник данных:** OpenAlex (open-access).
- **Архитектура:** Sentence Window Retrieval + дополнительные стратегии извлечения.
- **LLM:** локальный запуск через Ollama (по умолчанию `qwen2.5:0.5b`).
- **Интерфейс:** Streamlit.

## Подходы RAG
- **Baseline vector**: стандартный `VectorIndexRetriever`.
- **Query Fusion**: multi-query retrieval с объединением результатов (RRF/relative/simple).
- **LLM Rerank**: переранжирование контекста с помощью LLM.
- **Agent mode**: ReAct-агент с инструментами `vector_search` и `fusion_search`.
- **Multi-agent mode**: два ReAct-агента (retrieval и synthesis) работают последовательно.

## Метрики (оценка качества)
- **Semantic Recall**: близость ground truth к извлеченному контексту (cosine similarity).
- **BERTScore (F1)**: семантическое совпадение ответа с эталоном.
- **Answer relevancy**: близость вопроса и ответа (эмбеддинги).
- **Answer-context similarity**: близость ответа и контекста (эмбеддинги).

## Выбор статей
В интерфейсе есть фильтр статей по метаданным (title → file_name). Фильтрация применяется ко всем стратегиям.

## Запуск
1. Установите зависимости: `pip install -r requirements.txt`
2. Запустите Ollama и скачайте модель: `ollama pull qwen2.5:0.5b`
3. Скачайте статьи: `python src/collector.py --query "your topic" --limit 10`
4. Постройте индекс: `python src/indexer.py`
5. Запустите чат: `streamlit run src/app.py`

## Оценка качества
```
python src/evaluator.py \
  --question "your question" \
  --ground-truth "expected answer" \
  --strategy baseline \
  --fusion-queries 4 \
  --fusion-mode reciprocal_rerank \
  --rerank-top-n 3 \
  --files paper_1.pdf paper_3.pdf
```

## Примечания
- Индексация может занимать время (парсинг PDF + эмбеддинги).
- Для стабильной работы с Ollama используется `SafeOllama`, чтобы не зависеть от поля `usage` в ответах.

## Файлы
- `src/app.py` — Streamlit UI, выбор стратегии и фильтрация статей.
- `src/indexer.py` — индексация + подмешивание метаданных в узлы.
- `src/evaluator.py` — оценка качества по метрикам.
- `src/rag_strategies.py` — фабрики для query engine и agent.
- `src/llm_utils.py` — SafeOllama для совместимости с актуальным Ollama.
