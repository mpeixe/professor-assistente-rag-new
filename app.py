import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# --- Funções de processamento de RAG ---

def setup_rag_system():
    """
    Configura e retorna a cadeia de RAG com múltiplos índices.
    Esta função deve ser executada apenas uma vez.
    """
    # Usamos o cache do Streamlit para evitar reprocessar a cada interação
    # Cache do Streamlit: @st.cache_resource
    # Para o Streamlit Cloud, é recomendado usar variáveis de ambiente
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Chave de API da OpenAI não encontrada. Por favor, configure a variável de ambiente OPENAI_API_KEY.")
        st.stop()

    # Estrutura para os dados das disciplinas
    disciplinas_data = {
        "biologia": {
            "url": "https://pt.wikipedia.org/wiki/Biologia_celular",
        },
        "fisica": {
            "url": "https://pt.wikipedia.org/wiki/F%C3%ADsica_qu%C3%A2ntica",
        }
    }

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = OpenAIEmbeddings()

    # Cria e armazena os índices em cache
    for disciplina, data in disciplinas_data.items():
        index_name = f"faiss_index_{disciplina}"
        if not os.path.exists(index_name):
            with st.spinner(f"Criando índice de {disciplina.capitalize()}..."):
                loader = WebBaseLoader(data["url"])
                documents = loader.load()
                docs = text_splitter.split_documents(documents)
                vectorstore = FAISS.from_documents(docs, embeddings)
                vectorstore.save_local(index_name)
        else:
            with st.spinner(f"Carregando índice de {disciplina.capitalize()}..."):
                vectorstore = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
        data["vectorstore"] = vectorstore

    # --- Roteador para classificar a pergunta ---
    llm_classifier = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt_classificador = ChatPromptTemplate.from_messages([
        ("system", """Você é um assistente de roteamento que classifica perguntas sobre diferentes disciplinas.
        Sua tarefa é identificar a qual disciplina a pergunta pertence. As disciplinas são: biologia, fisica.
        Responda APENAS com o nome da disciplina, em letras minúsculas.
        Se não se encaixar, responda 'outros'."""),
        ("human", "{input}")
    ])
    classificacao_chain = prompt_classificador | llm_classifier

    # --- Cadeia de resposta (genérica) ---
    llm_responder = ChatOpenAI(temperature=0)
    prompt_resposta = ChatPromptTemplate.from_template("""
    Responda à pergunta do usuário usando apenas o contexto fornecido.
    Contexto: {context}
    Pergunta: {input}
    """)
    document_chain = create_stuff_documents_chain(llm_responder, prompt_resposta)

    return classificacao_chain, document_chain, disciplinas_data

def get_answer(question, classificacao_chain, document_chain, disciplinas_data):
    """
    Roteia a pergunta, busca no índice correto e gera a resposta.
    """
    # 1. Classificar a pergunta
    disciplina_content = classificacao_chain.invoke({"input": question}).content
    disciplina = disciplina_content.strip().lower()

    if disciplina in disciplinas_data:
        # 2. Obter o retriever correto
        vectorstore = disciplinas_data[disciplina]["vectorstore"]
        retriever = vectorstore.as_retriever()

        # 3. Criar a cadeia de RAG específica para a disciplina
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # 4. Invocar a cadeia e obter a resposta
        response = retrieval_chain.invoke({"input": question})
        return response["answer"], disciplina
    else:
        return "Desculpe, a pergunta não se encaixa em nenhuma das disciplinas que conheço (biologia, física).", None

# --- Interface do Streamlit ---

st.set_page_config(page_title="Professor Assistente RAG", layout="wide")

st.title("👨‍🏫 Professor Assistente RAG")
st.write("Pergunte sobre Biologia ou Física e obtenha respostas baseadas em conhecimento especializado.")

# Inicializar o sistema RAG uma única vez
if "rag_system" not in st.session_state:
    st.session_state.rag_system = setup_rag_system()

classificacao_chain, document_chain, disciplinas_data = st.session_state.rag_system

# Entrada do usuário
user_question = st.text_input("Digite sua pergunta:", placeholder="Ex: O que é uma célula eucariota?")

if user_question:
    with st.spinner("Gerando resposta..."):
        answer, disciplina = get_answer(user_question, classificacao_chain, document_chain, disciplinas_data)

    if disciplina:
        st.markdown(f"**Identifiquei que sua pergunta é sobre:** *{disciplina.capitalize()}*")
    st.success(answer)
