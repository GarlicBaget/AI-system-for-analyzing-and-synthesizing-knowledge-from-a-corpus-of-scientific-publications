import os
import torch
import numpy as np
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from bert_score import score as bert_score_compute

# 1. Настройка моделей
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
# Используем твою микро-модель Qwen для генерации ответа
Settings.llm = Ollama(model="qwen2.5:0.5b", request_timeout=120.0)

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def run_evaluation():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    storage_dir = os.path.join(base_dir, "data", "storage")
    
    sc = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(sc)
    query_engine = index.as_query_engine(similarity_top_k=2)

    # ЭТАЛОН (Золотой вопрос)
    test_question = "What is the primary conclusion regarding AI in medical education?"
    ground_truth = "AI tools like ChatGPT show potential in enhancing medical learning but require oversight to ensure factual accuracy."

    print(f"--- ЗАПУСК АВТОНОМНОЙ ОЦЕНКИ ---")
    
    # 2. Получаем ответ системы
    print("1. Система генерирует ответ...")
    response = query_engine.query(test_question)
    generated_answer = response.response
    
    # Собираем весь найденный контекст в одну строку
    retrieved_context = " ".join([node.node.get_content() for node in response.source_nodes])

    # 3. Считаем Semantic Recall
    # Мы сравниваем смысл Ground Truth и найденного Контекста
    print("2. Считаем Semantic Recall...")
    gt_embedding = embed_model.get_text_embedding(ground_truth)
    context_embedding = embed_model.get_text_embedding(retrieved_context)
    recall_score = cosine_similarity(gt_embedding, context_embedding)

    # 4. Считаем BERTScore
    print("3. Считаем BERTScore...")
    # Сравниваем сгенерированный ответ с эталоном
    P, R, F1 = bert_score_compute([generated_answer], [ground_truth], lang="en", verbose=False)
    bs_f1 = F1.item()

    print("\n" + "="*40)
    print("ИТОГОВЫЕ МЕТРИКИ ДЛЯ КУРАТОРА:")
    print(f"Вопрос: {test_question}")
    print(f"Ответ ИИ: {generated_answer[:100]}...")
    print("-" * 40)
    print(f"1. Semantic Recall: {recall_score:.4f}")
    print(f"   (Насколько хорошо RAG нашел смысл эталона в статьях)")
    print(f"2. BERTScore (F1): {bs_f1:.4f}")
    print(f"   (Смысловое сходство ответа ИИ с эталоном)")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()