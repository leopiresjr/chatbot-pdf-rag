# Arquivo: process_docs.py (Código Completo e Corrigido)

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Garante que a GEMINI_API_KEY seja carregada do arquivo .env
load_dotenv()

# --- Configurações de Caminho ---
DOCS_DIR = "docs"
VECTOR_STORE_PATH = "vectorstore/faiss_index"


def process_documents():
    """Carrega PDFs, cria embeddings usando o Gemini e salva o índice vetorial FAISS."""

    print("--- 1. Carregando documentos ---")

    loader = PyPDFDirectoryLoader(DOCS_DIR)
    documents = loader.load()

    if not documents:
        print(
            f"ERRO: Nenhuma página de PDF encontrada na pasta '{DOCS_DIR}'. Verifique a pasta.")
        return  # <-- Válido porque está DENTRO da função

    print(f"Documentos carregados com sucesso: {len(documents)} páginas.")

    print("--- 2. Dividindo o texto (Chunking) ---")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    print(f"Total de fragmentos (chunks) criados: {len(texts)}")

    print("--- 3. Criando Embeddings e Índice Vetorial ---")

    # --- INÍCIO DA CORREÇÃO DA CHAVE DE API (DENTRO DA FUNÇÃO) ---

    # Carrega a chave do arquivo .env
    api_key = os.getenv("GEMINI_API_KEY")

    # Verificação de segurança
    if not api_key:
        print("ERRO CRÍTICO: GEMINI_API_KEY não encontrada.")
        print("Verifique se o arquivo .env existe e está formatado corretamente.")
        return  # <-- Válido porque está DENTRO da função

    print("Chave de API carregada. Inicializando embeddings...")

    # Injeta a chave de API diretamente no construtor
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=api_key
    )
    # --- FIM DA CORREÇÃO ---

    # Cria o índice vetorial FAISS
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Salva o índice no disco
    vectorstore.save_local(VECTOR_STORE_PATH)

    print(f"\nSucesso! Índice FAISS salvo em: {VECTOR_STORE_PATH}")


# Esta linha executa a função definida acima
if __name__ == "__main__":
    process_documents()
