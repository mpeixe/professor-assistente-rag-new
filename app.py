import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# USAR EMBEDDINGS LOCAIS EM VEZ DE OPENAI/OPENROUTER
from langchain_community.embeddings import HuggingFaceEmbeddings

def setup_rag_system():
    """
    Configura sistema RAG com embeddings locais (sem OpenAI/OpenRouter para embeddings)
    """
    try:
        # 1. VERIFICAR API KEY
        api_key = st.secrets.get("OPENROUTER_API_KEY")
        if not api_key:
            st.error("❌ Chave de API do OpenRouter não encontrada nos Secrets!")
            st.stop()
        st.success("✅ API Key encontrada")

        # 2. TESTAR CONEXÃO COM OPENROUTER (APENAS PARA LLM)
        st.info("🔧 Testando conexão com OpenRouter...")
        try:
            test_llm = ChatOpenAI(
                model="openai/gpt-3.5-turbo",
                temperature=0,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                max_tokens=5
            )
            test_response = test_llm.invoke("teste")
            st.success(f"✅ OpenRouter funcionando: {type(test_response)}")
        except Exception as e:
            st.error(f"❌ Falha no teste de conexão: {str(e)}")
            st.stop()

        # 3. CONFIGURAR EMBEDDINGS LOCAIS (SEM OPENROUTER)
        st.info("🔧 Configurando embeddings locais...")
        try:
            # Usar modelo local em vez de OpenAI/OpenRouter
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Testar embeddings
            test_embedding = embeddings.embed_query("teste")
            st.success(f"✅ Embeddings locais funcionando - dimensão: {len(test_embedding)}")
            
        except Exception as e:
            st.error(f"❌ Erro nos embeddings locais: {str(e)}")
            st.info("💡 Instalando dependências necessárias...")
            try:
                # Fallback: usar embeddings ainda mais simples
                from langchain_community.embeddings import OpenAIEmbeddings
                # Usar OpenAI direto se tiver chave
                openai_key = st.secrets.get("OPENAI_API_KEY")
                if openai_key:
                    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
                    st.success("✅ Usando OpenAI embeddings direto")
                else:
                    st.error("❌ Instale sentence-transformers: pip install sentence-transformers")
                    st.stop()
            except:
                st.error("❌ Erro crítico nos embeddings")
                st.stop()

        # 4. CONFIGURAÇÃO DOS DADOS
        disciplinas_data = {
            "biologia": {
                "url": "https://pt.wikipedia.org/wiki/Biologia_celular",
            },
            "fisica": {
                "url": "https://pt.wikipedia.org/wiki/F%C3%ADsica_qu%C3%A2ntica",
            }
        }

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # 5. CRIAR/CARREGAR ÍNDICES COM EMBEDDINGS LOCAIS
        for disciplina, data in disciplinas_data.items():
            index_name = f"faiss_index_{disciplina}"
            
            if not os.path.exists(index_name):
                with st.spinner(f"📚 Criando índice de {disciplina.capitalize()}..."):
                    try:
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
                        st.success(f"✅ Índice criado para {disciplina}")
                        
                    except Exception as e:
                        st.error(f"❌ Erro ao criar índice de {disciplina}: {str(e)}")
                        continue
            else:
                with st.spinner(f"📖 Carregando índice de {disciplina.capitalize()}..."):
                    try:
                        vectorstore = FAISS.load_local(
                            index_name, 
                            embeddings, 
                            allow_dangerous_deserialization=True
                        )
                        st.success(f"✅ Índice carregado para {disciplina}")
                    except Exception as e:
                        st.error(f"❌ Erro ao carregar índice de {disciplina}: {str(e)}")
                        continue
            
            data["vectorstore"] = vectorstore

        # 6. CONFIGURAR LLMS (OPENROUTER FUNCIONA PARA CHAT)
        llm_classifier = ChatOpenAI(
            model="openai/gpt-3.5-turbo",
            temperature=0,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )
        
        llm_responder = ChatOpenAI(
            model="openai/gpt-3.5-turbo",
            temperature=0,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )

        # 7. CRIAR CHAINS
        prompt_resposta = ChatPromptTemplate.from_template("""
        Responda à pergunta do usuário usando apenas o contexto fornecido.
        Contexto: {context}
        Pergunta: {input}
        """)
        
        document_chain = create_stuff_documents_chain(llm_responder, prompt_resposta)

        st.success("🎉 Sistema RAG configurado com sucesso!")
        return llm_classifier, document_chain, disciplinas_data
    
    except Exception as e:
        st.error(f"❌ Erro geral na configuração: {str(e)}")
        import traceback
        st.error(f"**Traceback completo:** {traceback.format_exc()}")
        st.stop()

def classify_question(question, llm_classifier):
    """Classificação de pergunta"""
    try:
        prompt = f"""Classifique esta pergunta em uma das seguintes disciplinas:
        - biologia
        - fisica
        
        Responda APENAS com o nome da disciplina em letras minúsculas.
        Se não se encaixar, responda 'outros'.
        
        Pergunta: {question}"""
        
        response = llm_classifier.invoke(prompt)
        
        # Extração segura
        if hasattr(response, 'content'):
            disciplina = response.content.strip().lower()
        else:
            disciplina = str(response).strip().lower()
        
        return disciplina
        
    except Exception as e:
        st.error(f"❌ Erro na classificação: {str(e)}")
        return 'outros'

def get_answer(question, llm_classifier, document_chain, disciplinas_data):
    """
    Processamento completo da pergunta
    """
    try:
        # 1. CLASSIFICAR
        disciplina = classify_question(question, llm_classifier)
        
        if disciplina not in disciplinas_data:
            return "Desculpe, não consegui classificar sua pergunta adequadamente. Tente perguntar sobre biologia ou física.", None
        
        if 'vectorstore' not in disciplinas_data[disciplina]:
            return f"Índice de {disciplina} não disponível.", disciplina
        
        # 2. BUSCAR E RESPONDER
        vectorstore = disciplinas_data[disciplina]["vectorstore"]
        retriever = vectorstore.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({"input": question})
        
        # Extrair resposta
        if isinstance(response, dict) and 'answer' in response:
            answer = response['answer']
        else:
            answer = str(response)
        
        return answer, disciplina
            
    except Exception as e:
        return f"Erro ao processar pergunta: {str(e)}", None

# --- Interface do Streamlit ---

st.set_page_config(page_title="Professor Assistente RAG", layout="wide")

st.title("👨‍🏫 Professor Assistente RAG")
st.markdown("""
**Pergunte sobre Biologia ou Física!**

📚 **Disciplinas disponíveis:**
- 🧬 **Biologia** (Biologia Celular)
- ⚛️ **Física** (Física Quântica)

ℹ️ *Esta versão usa embeddings locais para maior confiabilidade.*
""")

# Inicializar sistema
if "rag_system" not in st.session_state:
    with st.spinner("🔧 Configurando sistema RAG..."):
        st.session_state.rag_system = setup_rag_system()

if st.session_state.rag_system:
    llm_classifier, document_chain, disciplinas_data = st.session_state.rag_system

    # Interface principal
    user_question = st.text_input(
        "💬 Digite sua pergunta:",
        placeholder="Ex: O que é uma célula eucariota? Como funciona o efeito fotoelétrico?"
    )

    if user_question:
        with st.spinner("🤔 Processando pergunta..."):
            answer, disciplina = get_answer(user_question, llm_classifier, document_chain, disciplinas_data)

        # Mostrar resultado
        if disciplina and disciplina in ['biologia', 'fisica']:
            st.info(f"🔍 **Disciplina identificada:** {disciplina.capitalize()}")
        
        if answer.startswith("Erro"):
            st.error(f"❌ {answer}")
        else:
            st.success(f"📖 {answer}")

        # Opções adicionais
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Resposta útil"):
                st.balloons()
                st.success("Obrigado pelo feedback!")
        with col2:
            if st.button("🔄 Nova pergunta"):
                st.rerun()

# Informações do sistema
with st.expander("ℹ️ Informações do Sistema"):
    if st.session_state.get("rag_system"):
        st.success("✅ Sistema inicializado")
        st.write("🤖 **LLM:** OpenRouter (gpt-3.5-turbo)")
        st.write("🔢 **Embeddings:** HuggingFace (local)")
        st.write("📚 **Vectorstore:** FAISS")
        st.write("🎯 **Disciplinas:** Biologia, Física")
    else:
        st.warning("⏳ Sistema não inicializado")