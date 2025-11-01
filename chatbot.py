# Arquivo: chatbot.py (Código Final Estável com Correção de API)

import os
from dotenv import load_dotenv

# Módulos LCEL (LangChain Expression Language)
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Garante que a GEMINI_API_KEY seja carregada
load_dotenv()

# --- Configurações ---
VECTOR_STORE_PATH = "vectorstore/faiss_index"

# 1. Função para carregar contexto do retriever


def format_docs(docs):
    """Formata os documentos recuperados em uma única string de contexto."""
    return "\n\n".join(doc.page_content for doc in docs)


# --- INÍCIO DA CORREÇÃO DA CHAVE DE API (LLM) ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # Se a chave não for carregada, o script para aqui.
    raise ValueError("ERRO CRÍTICO: GEMINI_API_KEY não encontrada no .env")

# 2. Configurar LLM e Embeddings (Injetando a chave)
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=api_key
)
# --- FIM DA CORREÇÃO ---


def run_chatbot():
    print("\n--- Carregando Sistema de Pesquisa (Estável) ---")

    if not os.path.exists(VECTOR_STORE_PATH):
        print("ERRO: O índice vetorial não foi encontrado. Execute 'python process_docs.py' primeiro.")
        return

    try:
        # Carrega o índice
        vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"ERRO CRÍTICO: Falha ao carregar o índice FAISS. Detalhe: {e}")
        return

    # 3. Criar o Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 4. Definir o Prompt e a Lógica RAG (Sintaxe Direta)

    system_prompt = (
        "Você é um assistente de pesquisa especializado em Engenharia de Software. Responda à pergunta do usuário "
        "APENAS com base no contexto fornecido nos documentos abaixo. "
        "Se a resposta não estiver no contexto, diga que não pode responder com base nas fontes. Contexto: {context}"
    )
    prompt = ChatPromptTemplate.from_template(
        system_prompt + "\n\nPergunta: {question}")

    # Lógica da Cadeia RAG (Combina busca, prompt e LLM)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n--- Chatbot de Pesquisa Ativo (Final) ---")
    print("Pronto para responder sobre seus documentos.")
    print("Digite 'sair' ou 'exit' para encerrar.")

    while True:
        question = input("\nSua Pergunta: ")
        if question.lower() in ["sair", "exit"]:
            print("Chat encerrado.")
            break

        if not question.strip():
            continue

        try:
            # Invocar a cadeia
            response = rag_chain.invoke(question)

            # Imprimir Resposta
            print("\n[RESPOSTA DA IA]:")
            print(response)

            # Imprimir Fontes (Estimativa)
            print("\n[FONTES UTILIZADAS - Estimativa]:")
            sources = set(doc.metadata.get('source')
                          for doc in retriever.get_relevant_documents(question))
            for source in sources:
                print(f"- {source}")

        except Exception as e:
            print(f"Ocorreu um erro na IA (Conexão ou API): {e}")


if __name__ == "__main__":
    run_chatbot()
