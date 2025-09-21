import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

def debug_log(message):
    """Log de debug"""
    st.write(f"🔍 **DEBUG:** {message}")

def safe_extract_content(response, step="desconhecido"):
    """Extrai conteúdo de forma segura de qualquer tipo de resposta"""
    debug_log(f"Extraindo conteúdo na etapa: {step}")
    debug_log(f"Tipo da resposta: {type(response)}")
    
    try:
        # Se é AIMessage (LangChain)
        if hasattr(response, 'content'):
            content = response.content
            debug_log(f"✅ Extraído via .content: {content}")
            return content
        
        # Se é string
        elif isinstance(response, str):
            debug_log(f"✅ Já é string: {response}")
            return response
        
        # Se é dict
        elif isinstance(response, dict):
            if 'content' in response:
                content = response['content']
                debug_log(f"✅ Extraído via ['content']: {content}")
                return content
            elif 'answer' in response:
                content = response['answer']
                debug_log(f"✅ Extraído via ['answer']: {content}")
                return content
        
        # Fallback: converter para string
        content = str(response)
        debug_log(f"⚠️ Convertido para string: {content}")
        return content
        
    except Exception as e:
        debug_log(f"❌ Erro ao extrair conteúdo: {str(e)}")
        return f"Erro na extração: {str(e)}"

def setup_rag_system():
    """
    Configura e retorna a cadeia de RAG com debug detalhado.
    """
    try:
        debug_log("Iniciando configuração do sistema RAG...")
        
        # 1. VERIFICAR API KEY
        api_key = st.secrets.get("OPENROUTER_API_KEY")
        if not api_key:
            st.error("❌ Chave de API do OpenRouter não encontrada nos Secrets!")
            st.stop()
        debug_log("✅ API Key encontrada")

        # 2. TESTAR CONEXÃO BÁSICA
        debug_log("Testando conexão com OpenRouter...")
        try:
            test_llm = ChatOpenAI(
                model="openai/gpt-3.5-turbo",
                temperature=0,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                max_tokens=5
            )
            test_response = test_llm.invoke("teste")
            debug_log(f"✅ Teste de conexão bem-sucedido: {type(test_response)}")
        except Exception as e:
            st.error(f"❌ Falha no teste de conexão: {str(e)}")
            st.stop()

        # 3. CONFIGURAÇÃO DOS DADOS
        disciplinas_data = {
            "biologia": {
                "url": "https://pt.wikipedia.org/wiki/Biologia_celular",
            },
            "fisica": {
                "url": "https://pt.wikipedia.org/wiki/F%C3%ADsica_qu%C3%A2ntica",
            }
        }
        debug_log("✅ Dados das disciplinas configurados")

        # 4. CONFIGURAÇÃO DE EMBEDDINGS
        debug_log("Configurando embeddings...")
        try:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1"
            )
            # Teste dos embeddings
            test_embedding = embeddings.embed_query("teste")
            debug_log(f"✅ Embeddings funcionando - dimensão: {len(test_embedding)}")
        except Exception as e:
            st.error(f"❌ Erro nos embeddings: {str(e)}")
            st.stop()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # 5. CRIAR/CARREGAR ÍNDICES
        debug_log("Processando índices...")
        for disciplina, data in disciplinas_data.items():
            index_name = f"faiss_index_{disciplina}"
            
            try:
                if not os.path.exists(index_name):
                    debug_log(f"Criando índice para {disciplina}...")
                    with st.spinner(f"Criando índice de {disciplina.capitalize()}..."):
                        loader = WebBaseLoader(data["url"])
                        documents = loader.load()

                        if not documents:
                            st.warning(f"⚠️ Nenhum documento carregado para {disciplina}")
                            continue

                        # Limpeza
                        for doc in documents:
                            doc.page_content = doc.page_content.strip().replace("\n", " ").replace("  ", " ")

                        docs = text_splitter.split_documents(documents)
                        if not docs:
                            st.warning(f"⚠️ Nenhum chunk criado para {disciplina}")
                            continue

                        vectorstore = FAISS.from_documents(docs, embeddings)
                        vectorstore.save_local(index_name)
                        debug_log(f"✅ Índice criado para {disciplina}")
                else:
                    debug_log(f"Carregando índice existente para {disciplina}...")
                    with st.spinner(f"Carregando índice de {disciplina.capitalize()}..."):
                        vectorstore = FAISS.load_local(
                            index_name, 
                            embeddings, 
                            allow_dangerous_deserialization=True
                        )
                        debug_log(f"✅ Índice carregado para {disciplina}")
                
                data["vectorstore"] = vectorstore
                
            except Exception as e:
                st.error(f"❌ Erro ao processar índice de {disciplina}: {str(e)}")
                continue

        # 6. CONFIGURAR MODELOS
        debug_log("Configurando modelos LLM...")
        
        # Classificador
        llm_classifier = ChatOpenAI(
            model="openai/gpt-3.5-turbo",
            temperature=0,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )
        
        # Responder
        llm_responder = ChatOpenAI(
            model="openai/gpt-3.5-turbo",
            temperature=0,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )
        
        debug_log("✅ Modelos LLM configurados")

        # 7. CRIAR CHAINS (SEM USAR PIPES)
        debug_log("Criando chains...")
        
        prompt_resposta = ChatPromptTemplate.from_template("""
        Responda à pergunta do usuário usando apenas o contexto fornecido.
        Contexto: {context}
        Pergunta: {input}
        """)
        
        document_chain = create_stuff_documents_chain(llm_responder, prompt_resposta)
        debug_log("✅ Chains criadas")

        debug_log("🎉 Sistema RAG configurado com sucesso!")
        return llm_classifier, document_chain, disciplinas_data
    
    except Exception as e:
        st.error(f"❌ Erro geral na configuração: {str(e)}")
        import traceback
        st.error(f"**Traceback completo:** {traceback.format_exc()}")
        st.stop()

def classify_question_safe(question, llm_classifier):
    """Classificação com tratamento seguro"""
    debug_log(f"Classificando pergunta: {question}")
    
    try:
        prompt = f"""Classifique esta pergunta em uma das seguintes disciplinas:
        - biologia
        - fisica
        
        Responda APENAS com o nome da disciplina em letras minúsculas.
        Se não se encaixar, responda 'outros'.
        
        Pergunta: {question}"""
        
        response = llm_classifier.invoke(prompt)
        disciplina = safe_extract_content(response, "classificação")
        
        # Limpar e validar
        disciplina = disciplina.strip().lower()
        if disciplina not in ['biologia', 'fisica', 'outros']:
            debug_log(f"⚠️ Disciplina inesperada: {disciplina}, usando 'outros'")
            disciplina = 'outros'
        
        debug_log(f"✅ Disciplina classificada: {disciplina}")
        return disciplina
        
    except Exception as e:
        debug_log(f"❌ Erro na classificação: {str(e)}")
        return 'outros'

def get_answer_safe(question, llm_classifier, document_chain, disciplinas_data):
    """
    Processamento completo da pergunta com tratamento seguro
    """
    try:
        debug_log("=== INICIANDO PROCESSAMENTO ===")
        
        # 1. CLASSIFICAR
        disciplina = classify_question_safe(question, llm_classifier)
        
        if disciplina not in disciplinas_data:
            return "Desculpe, não consegui classificar sua pergunta adequadamente. Tente perguntar sobre biologia ou física.", None
        
        if 'vectorstore' not in disciplinas_data[disciplina]:
            return f"Índice de {disciplina} não disponível.", disciplina
        
        # 2. BUSCAR NO VECTORSTORE
        debug_log(f"Buscando no vectorstore de {disciplina}...")
        try:
            vectorstore = disciplinas_data[disciplina]["vectorstore"]
            retriever = vectorstore.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            debug_log("Executando retrieval chain...")
            response = retrieval_chain.invoke({"input": question})
            
            debug_log(f"Tipo da resposta do retrieval: {type(response)}")
            
            # Extrair resposta de forma segura
            if isinstance(response, dict) and 'answer' in response:
                answer = response['answer']
            else:
                answer = safe_extract_content(response, "resposta final")
            
            debug_log(f"✅ Resposta extraída: {answer[:100]}...")
            return answer, disciplina
            
        except Exception as e:
            debug_log(f"❌ Erro no retrieval: {str(e)}")
            return f"Erro ao buscar informações sobre {disciplina}: {str(e)}", disciplina
    
    except Exception as e:
        debug_log(f"❌ Erro geral no processamento: {str(e)}")
        return f"Erro ao processar pergunta: {str(e)}", None

# --- Interface do Streamlit ---

st.set_page_config(page_title="Professor Assistente RAG", layout="wide")

st.title("👨‍🏫 Professor Assistente RAG (Versão Debugada)")
st.write("Pergunte sobre Biologia ou Física e obtenha respostas baseadas em conhecimento especializado.")

# Debug toggle
show_debug = st.checkbox("🔧 Mostrar informações de debug", value=True)

if not show_debug:
    # Redirecionar debug_log para não aparecer
    def debug_log(message):
        pass

# Inicializar sistema
if "rag_system" not in st.session_state:
    with st.spinner("🔧 Configurando sistema RAG..."):
        st.session_state.rag_system = setup_rag_system()

if st.session_state.rag_system:
    llm_classifier, document_chain, disciplinas_data = st.session_state.rag_system

    # Interface principal
    user_question = st.text_input("Digite sua pergunta:", placeholder="Ex: O que é uma célula eucariota?")

    if user_question:
        with st.spinner("🤔 Processando pergunta..."):
            answer, disciplina = get_answer_safe(user_question, llm_classifier, document_chain, disciplinas_data)

        # Mostrar resultado
        if disciplina:
            st.markdown(f"**📖 Disciplina identificada:** *{disciplina.capitalize()}*")
        
        if answer.startswith("Erro"):
            st.error(f"❌ {answer}")
        else:
            st.success(f"✅ {answer}")

        # Botão para nova pergunta
        if st.button("🔄 Fazer nova pergunta"):
            st.rerun()