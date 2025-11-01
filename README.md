# 🤖 Chatbot de TCC (Engenharia de Software) com RAG e IA Generativa

Este projeto é uma solução de **IA Generativa** desenvolvida para o desafio DIO, focada em criar um assistente virtual capaz de responder perguntas complexas com base em uma base de conhecimento privada (documentos PDF).

O cenário simula um estudante de Engenharia de Software preparando seu TCC, que utiliza este chatbot para revisar e correlacionar artigos científicos sobre DevOps, CI/CD e Microsserviços.

A arquitetura central utilizada é a **RAG (Retrieval-Augmented Generation)**, que combina busca vetorial com o poder de geração de linguagem dos LLMs (Google Gemini).

## 🚀 Tecnologias Utilizadas

* **Python:** Linguagem principal do projeto.
* **LangChain:** Framework para orquestrar a arquitetura RAG (LCEL).
* **Google Gemini (GenAI):** Utilizado para:
    1.  **Embeddings** (`text-embedding-004`): Transformar os textos dos PDFs em vetores.
    2.  **LLM** (`gemini-2.5-flash`): Gerar as respostas com base no contexto encontrado.
* **FAISS (Facebook AI Similarity Search):** Banco de dados vetorial local para armazenar e buscar rapidamente os *chunks* de texto relevantes.
* **PyPDF:** Biblioteca para carregar e extrair texto dos arquivos PDF.

## 🛠️ Fluxo de Execução (RAG)

O projeto é dividido em duas etapas principais, refletindo a arquitetura RAG:

### 1. Processamento e Indexação (`process_docs.py`)

Este script é responsável por construir a base de conhecimento (o "cérebro" do assistente):

1.  **Load:** Carrega todos os PDFs da pasta `/docs`.
2.  **Split:** Divide os textos em fragmentos menores (*chunks*) de 1000 caracteres, com sobreposição de 200, para manter o contexto.
3.  **Embed:** Converte cada *chunk* de texto em um vetor numérico usando o Gemini.
4.  **Store:** Armazena esses vetores em um índice `FAISS` local (na pasta `/vectorstore`) para busca rápida por similaridade.

**[INSERIR PRINT 1 AQUI: Captura de tela do terminal executando `python process_docs.py`, mostrando a contagem de chunks (ex: 1638) e a mensagem de sucesso.]**

### 2. Chat Interativo (`chatbot.py`)

Este script executa a interface de perguntas e respostas (o RAG em ação):

1.  **Load:** Carrega o índice FAISS salvo anteriormente.
2.  **Retrieve:** Quando o usuário faz uma pergunta, o sistema busca no FAISS os 4 *chunks* de texto mais relevantes (similares) à pergunta.
3.  **Augment (Prompting):** O sistema injeta os *chunks* encontrados (o "Contexto") em um *prompt* de sistema, instruindo o LLM a responder *apenas* com base nesse contexto.
4.  **Generate:** O LLM (Gemini) gera uma resposta em linguagem natural, fundamentada nos seus documentos.

## 📊 Resultados e Testes

O chatbot demonstrou alta fidelidade ao conteúdo dos documentos, respondendo perguntas complexas e recusando-se a responder perguntas fora do contexto dos PDFs.

**[INSERIR PRINT 2 AQUI: Captura de tela do terminal executando `python chatbot.py`, mostrando uma pergunta específica sobre Engenharia de Software e a [RESPOSTA DA IA] com as [FONTES UTILIZADAS].]**

## 💡 Insights e Aprendizados

* **O Poder do RAG:** A técnica RAG é fundamental para evitar que a IA "alucine" (invente fatos), forçando-a a usar apenas o conhecimento proprietário fornecido (os PDFs).
* **Importância do Chunking:** A estratégia de divisão dos textos (tamanho e sobreposição) é crucial para garantir que o contexto recuperado pela busca vetorial seja relevante.
* **Desafios de Dependência:** O desenvolvimento em IA (especialmente com LangChain) exige atenção constante às versões das bibliotecas, pois o ecossistema evolui muito rapidamente, causando frequentes erros de importação que exigem migração de sintaxe (como a mudança de `RetrievalQA` para LCEL).