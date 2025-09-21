import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Funções de processamento de RAG ---

def setup_rag_system():
    """
    Configura e retorna a cadeia de RAG com múltiplos índices.
    Esta função deve ser executada apenas uma vez.
    """
    try:
        # 1. PEGA A CHAVE DO OPENROUTER DOS SECRETS DO STREAMLIT
        api_key = st.secrets.get("OPENROUTER_API_KEY")
        if not api_key:
            st.error("Chave de API do OpenRouter não encontrada. Por favor, configure-a nos 'Secrets' do Streamlit Cloud.")
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

        # Configuração para embeddings via OpenRouter
        try:
            # Primeira tentativa: OpenAI Embeddings via OpenRouter
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1"
            )
            st.info("✅ Usando embeddings via OpenRouter")
        except Exception as e:
            st.warning(f"⚠️ Falha com OpenRouter embeddings: {e}")
            # Fallback: usar OpenAI diretamente (se tiver chave OpenAI)
            openai_key = st.secrets.get("OPENAI_API_KEY")
            if openai_key:
                embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
                st.info("✅ Usando embeddings via OpenAI direto")
            else:
                st.error("Nem OpenRouter nem OpenAI API keys encontradas!")
                st.stop()

        # Cria e armazena os índices em cache
        for disciplina, data in disciplinas_data.items():
            index_name = f"faiss_index_{disciplina}"
            if not os.path.exists(index_name):
                with st.spinner(f"Criando índice de {disciplina.capitalize()}..."):
                    loader = WebBaseLoader(data["url"])
                    documents = loader.load()

                    if not documents:
                        st.error(f"Falha ao carregar documentos para {disciplina}")
                        continue

                    # Limpando o texto antes de dividir
                    for doc in documents:
                        doc.page_content = doc.page_content.strip().replace("\n", " ").replace("  ", " ")

                    docs = text_splitter.split_documents(documents)
                    if not docs:
                        st.error(f"Falha ao dividir documentos para {disciplina}")
                        continue
                        
                    vectorstore = FAISS.from_documents(docs, embeddings)
                    vectorstore.save_local(index_name)
                    st.success(f"✅ Índice criado para {disciplina}")
            else:
                with st.spinner(f"Carregando índice de {disciplina.capitalize()}..."):
                    vectorstore = FAISS.load_local(
                        index_name, 
                        embeddings, 
                        allow_dangerous_deserialization=True
                    )
                    st.success(f"✅ Índice carregado para {disciplina}")
            data["vectorstore"] = vectorstore

        # Configuração para ChatOpenAI via OpenRouter
        llm_classifier = ChatOpenAI(
            model="openai/gpt-3.5-turbo",
            temperature=0,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )
        
        prompt_classificador = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente de roteamento que classifica perguntas sobre diferentes disciplinas.
            Sua tarefa é identificar a qual disciplina a pergunta pertence. As disciplinas são: biologia, fisica.
            Responda APENAS com o nome da disciplina, em letras minúsculas.
            Se não se encaixar, responda 'outros'."""),
            ("human", "{input}")
        ])
        classificacao_chain = prompt_classificador | llm_classifier

        llm_responder = ChatOpenAI(
            model="openai/gpt-3.5-turbo",
            temperature=0,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )
        
        prompt_resposta = ChatPromptTemplate.from_template("""
        Responda à pergunta do usuário usando apenas o contexto fornecido.
        Se o contexto não contém informações suficientes, diga que não tem informação suficiente.
        
        Contexto: {context}
        Pergunta: {input}
        
        Resposta:
        """)
        document_chain = create_stuff_documents_chain(llm_responder, prompt_resposta)

        return classificacao_chain, document_chain, disciplinas_data
    
    except Exception as e:
        st.error(f"Erro ao configurar o sistema RAG: {str(e)}")
        st.error("Verifique se as dependências estão instaladas corretamente:")
        st.code("""
        pip install streamlit
        pip install langchain
        pip install langchain-openai
        pip install langchain-community
        pip install faiss-cpu
        pip install beautifulsoup4
        """)
        st.stop()

def get_answer(question, classificacao_chain, document_chain, disciplinas_data):
    """
    Roteia a pergunta, busca no índice correto e gera a resposta.
    """
    try:
        # Classificar a pergunta
        disciplina_response = classificacao_chain.invoke({"input": question})
        disciplina = disciplina_response.content.strip().lower()
        
        st.info(f"🔍 Pergunta classificada como: **{disciplina}**")

        if disciplina in disciplinas_data:
            vectorstore = disciplinas_data[disciplina]["vectorstore"]
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({"input": question})
            return response["answer"], disciplina
        else:
            return "Desculpe, a pergunta não se encaixa em nenhuma das disciplinas que conheço (biologia, física). Tente reformular sua pergunta.", None
    
    except Exception as e:
        return f"Erro ao processar a pergunta: {str(e)}", None

# --- Interface do Streamlit ---

st.set_page_config(
    page_title="Professor Assistente RAG",
    page_icon="👨‍🏫",
    layout="wide"
)

st.title("👨‍🏫 Professor Assistente RAG")
st.markdown("""
**Pergunte sobre Biologia ou Física e obtenha respostas baseadas em conhecimento especializado.**

📚 **Disciplinas disponíveis:**
- 🧬 **Biologia** (Biologia Celular)
- ⚛️ **Física** (Física Quântica)
""")

# Inicializar o sistema RAG uma única vez
if "rag_system" not in st.session_state:
    with st.spinner("🔧 Configurando sistema RAG..."):
        st.session_state.rag_system = setup_rag_system()

if st.session_state.rag_system:
    classificacao_chain, document_chain, disciplinas_data = st.session_state.rag_system

    # Entrada do usuário
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_question = st.text_input(
                "Digite sua pergunta:",
                placeholder="Ex: O que é uma célula eucariota? Como funciona o efeito fotoelétrico?"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Espaçamento
            clear_button = st.button("🗑️ Limpar", use_container_width=True)

    if clear_button:
        st.rerun()

    if user_question:
        with st.spinner("🤔 Gerando resposta..."):
            answer, disciplina = get_answer(user_question, classificacao_chain, document_chain, disciplinas_data)

        if disciplina:
            st.markdown(f"### 📖 Resposta - {disciplina.capitalize()}")
        
        if answer.startswith("Erro"):
            st.error(f"❌ {answer}")
        else:
            st.success(f"✅ {answer}")
            
        # Adicionar feedback do usuário
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("👍 Útil"):
                st.success("Obrigado pelo feedback!")
        with col2:
            if st.button("👎 Não útil"):
                st.info("Tente reformular sua pergunta para melhores resultados.")
        with col3:
            if st.button("🔄 Nova pergunta"):
                st.rerun()