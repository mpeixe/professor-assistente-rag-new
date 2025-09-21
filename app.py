import streamlit as st
from langchain_openai import ChatOpenAI

st.title("🔍 Debug Minimal - OpenRouter")

# 1. TESTE BÁSICO DE CONEXÃO
api_key = st.secrets.get("OPENROUTER_API_KEY")

if not api_key:
    st.error("❌ OPENROUTER_API_KEY não encontrada")
    st.stop()

st.success("✅ API Key encontrada")

# 2. TESTE DIFERENTES CONFIGURAÇÕES
st.markdown("## Testando Configurações...")

configs = [
    {
        "name": "Config 1 - base_url",
        "config": {
            "model": "openai/gpt-3.5-turbo",
            "openai_api_key": api_key,
            "openai_api_base": "https://openrouter.ai/api/v1"
        }
    },
    {
        "name": "Config 2 - sem base_url",
        "config": {
            "model": "openai/gpt-3.5-turbo",
            "openai_api_key": api_key
        }
    }
]

for i, config_info in enumerate(configs):
    st.markdown(f"### {config_info['name']}")
    
    try:
        llm = ChatOpenAI(**config_info['config'])
        
        # Teste simples
        response = llm.invoke("Responda apenas: OK")
        
        st.success(f"✅ Sucesso!")
        st.write(f"**Tipo da resposta:** {type(response)}")
        st.write(f"**Conteúdo:** {response}")
        
        # Tentar acessar diferentes atributos
        if hasattr(response, 'content'):
            st.write(f"**response.content:** {response.content}")
        if hasattr(response, 'data'):
            st.write(f"**response.data:** {response.data}")
        
        # Mostrar todos os atributos
        st.write(f"**Atributos disponíveis:** {dir(response)}")
        
    except Exception as e:
        st.error(f"❌ Erro: {str(e)}")
        st.error(f"**Tipo do erro:** {type(e)}")

# 3. TESTE MANUAL COM INPUT
st.markdown("## Teste Manual")

if st.button("🧪 Testar Pergunta Manual"):
    try:
        # Usar a primeira config que funcionou
        llm = ChatOpenAI(
            model="openai/gpt-3.5-turbo",
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )
        
        response = llm.invoke("Classifique esta pergunta: 'O que é uma célula?' Responda apenas: biologia ou fisica")
        
        st.write(f"**Resposta completa:** {response}")
        st.write(f"**Tipo:** {type(response)}")
        
        # Diferentes formas de acessar o conteúdo
        try:
            content = response.content
            st.success(f"✅ response.content: {content}")
        except:
            st.error("❌ Não tem .content")
            
        try:
            data = response.data
            st.success(f"✅ response.data: {data}")
        except:
            st.error("❌ Não tem .data")
            
        # Como string
        st.write(f"**Como string:** {str(response)}")
        
    except Exception as e:
        st.error(f"❌ Erro no teste manual: {str(e)}")
        import traceback
        st.error(f"**Traceback:** {traceback.format_exc()}")