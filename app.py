import streamlit as st
import requests
import json
import re
from typing import List, Dict

# Configuração da página
st.set_page_config(
    page_title="Professor Assistente RAG",
    page_icon="👨‍🏫",
    layout="wide"
)

# Base de conhecimento simples (embarcada)
KNOWLEDGE_BASE = {
    "biologia": {
        "conteudo": [
            {
                "texto": "A célula eucariota é uma célula que possui núcleo definido, delimitado por uma membrana nuclear. Ela possui material genético (DNA) organizado dentro do núcleo, ao contrário das células procariontes.",
                "palavras_chave": ["célula eucariota", "núcleo", "membrana nuclear", "DNA", "material genético"]
            },
            {
                "texto": "As organelas celulares são estruturas especializadas presentes no citoplasma das células eucariotas. As principais incluem: mitocôndrias (respiração celular), retículo endoplasmático (síntese de proteínas), complexo de Golgi (processamento de proteínas), lisossomos (digestão celular).",
                "palavras_chave": ["organelas", "mitocôndrias", "retículo endoplasmático", "complexo de Golgi", "lisossomos"]
            },
            {
                "texto": "A diferença principal entre célula eucariota e procariota é a organização do material genético. Eucariotas têm núcleo organizado, enquanto procariotas têm nucleoide (região sem membrana). Exemplos de eucariotas: animais, plantas, fungos. Exemplos de procariotas: bactérias.",
                "palavras_chave": ["eucariota", "procariota", "nucleoide", "bactérias", "diferença"]
            },
            {
                "texto": "As células vegetais possuem características únicas: parede celular (celulose), cloroplastos (fotossíntese), vacúolo central grande. Diferem das células animais que não possuem parede celular nem cloroplastos.",
                "palavras_chave": ["células vegetais", "parede celular", "cloroplastos", "fotossíntese", "vacúolo"]
            },
            {
                "texto": "A divisão celular em eucariotos ocorre por dois processos: mitose (células somáticas, mantém número de cromossomos) e meiose (células reprodutivas, reduz pela metade os cromossomos).",
                "palavras_chave": ["divisão celular", "mitose", "meiose", "cromossomos", "células reprodutivas"]
            }
        ]
    },
    "fisica": {
        "conteudo": [
            {
                "texto": "O efeito fotoelétrico foi explicado por Albert Einstein em 1905, rendendo-lhe o Prêmio Nobel de Física em 1921. Consiste na emissão de elétrons quando luz incide sobre uma superfície metálica.",
                "palavras_chave": ["efeito fotoelétrico", "Einstein", "elétrons", "luz", "superfície metálica"]
            },
            {
                "texto": "No efeito fotoelétrico, elétrons são ejetados de um metal quando luz de frequência suficiente incide sobre ele. A energia dos fótons deve superar a função trabalho do material para que elétrons sejam liberados.",
                "palavras_chave": ["fótons", "frequência", "função trabalho", "energia", "liberados"]
            },
            {
                "texto": "A energia dos fótons é proporcional à frequência da luz, segundo a equação E = hf, onde h é a constante de Planck e f é a frequência. Esta relação demonstra a natureza quântica da luz.",
                "palavras_chave": ["energia dos fótons", "frequência", "constante de Planck", "natureza quântica", "E=hf"]
            },
            {
                "texto": "A mecânica quântica descreve o comportamento de partículas em escala atômica e subatômica. Introduz conceitos como quantização de energia, dualidade onda-partícula e probabilidade.",
                "palavras_chave": ["mecânica quântica", "escala atômica", "quantização", "dualidade onda-partícula", "probabilidade"]
            },
            {
                "texto": "O princípio da incerteza de Heisenberg estabelece que não é possível determinar simultaneamente com precisão a posição e o momento de uma partícula. Quanto mais precisa a medida de uma grandeza, menos precisa será a outra.",
                "palavras_chave": ["princípio da incerteza", "Heisenberg", "posição", "momento", "precisão"]
            }
        ]
    }
}

def call_openrouter_api(prompt: str, api_key: str, max_tokens: int = 150) -> str:
    """Chama a API do OpenRouter diretamente"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
        else:
            return f"Erro na API: {response.status_code}"
    except Exception as e:
        return f"Erro na conexão: {str(e)}"
    
    return "Não foi possível obter resposta."

def classify_question(question: str, api_key: str) -> str:
    """Classifica a pergunta usando OpenRouter"""
    prompt = f"""Classifique esta pergunta em uma das seguintes disciplinas:
    - biologia
    - fisica
    
    Responda APENAS com o nome da disciplina em letras minúsculas.
    Se não se encaixar, responda 'outros'.
    
    Pergunta: {question}"""
    
    result = call_openrouter_api(prompt, api_key, max_tokens=10)
    
    # Limpar e validar resultado
    disciplina = result.lower().strip()
    if disciplina not in ['biologia', 'fisica']:
        disciplina = 'outros'
    
    return disciplina

def find_relevant_content(question: str, disciplina: str) -> str:
    """Busca conteúdo relevante usando busca por palavras-chave simples"""
    if disciplina not in KNOWLEDGE_BASE:
        return ""
    
    # Converter pergunta para minúsculas para busca
    question_lower = question.lower()
    
    # Pontuar cada item de conteúdo
    scored_content = []
    
    for item in KNOWLEDGE_BASE[disciplina]["conteudo"]:
        score = 0
        
        # Verificar palavras-chave
        for palavra in item["palavras_chave"]:
            if palavra.lower() in question_lower:
                score += 2
        
        # Verificar palavras da pergunta no texto
        palavras_pergunta = re.findall(r'\b\w+\b', question_lower)
        for palavra in palavras_pergunta:
            if len(palavra) > 3 and palavra in item["texto"].lower():
                score += 1
        
        if score > 0:
            scored_content.append((score, item["texto"]))
    
    # Ordenar por pontuação e pegar os 2 melhores
    scored_content.sort(reverse=True, key=lambda x: x[0])
    
    # Retornar os textos mais relevantes
    relevant_texts = [item[1] for item in scored_content[:2]]
    return "\n\n".join(relevant_texts)

def generate_answer(question: str, context: str, api_key: str) -> str:
    """Gera resposta usando o contexto encontrado"""
    if not context:
        return "Não encontrei informações específicas sobre essa pergunta. Tente reformular ou perguntar sobre tópicos como células, organelas, efeito fotoelétrico ou mecânica quântica."
    
    prompt = f"""Responda à pergunta usando apenas o contexto fornecido.
    Seja claro e educativo. Se o contexto não tem informação suficiente, diga que precisa de mais detalhes.
    
    Contexto:
    {context}
    
    Pergunta: {question}
    
    Resposta:"""
    
    return call_openrouter_api(prompt, api_key, max_tokens=200)

def process_question(question: str, api_key: str) -> tuple:
    """Processa a pergunta completa"""
    try:
        # 1. Classificar pergunta
        disciplina = classify_question(question, api_key)
        
        if disciplina == 'outros':
            return "Esta pergunta não se encaixa nas disciplinas disponíveis (Biologia e Física). Tente perguntar sobre células, organelas, efeito fotoelétrico ou mecânica quântica.", None
        
        # 2. Buscar conteúdo relevante
        context = find_relevant_content(question, disciplina)
        
        # 3. Gerar resposta
        answer = generate_answer(question, context, api_key)
        
        return answer, disciplina
        
    except Exception as e:
        return f"Erro ao processar pergunta: {str(e)}", None

# --- Interface Streamlit ---

st.title("👨‍🏫 Professor Assistente RAG")
st.markdown("""
**Faça perguntas sobre Biologia ou Física!**

📚 **Tópicos disponíveis:**
- 🧬 **Biologia**: Células eucariotas, organelas, diferenças celulares, divisão celular
- ⚛️ **Física**: Efeito fotoelétrico, mecânica quântica, princípio da incerteza

💡 *Esta versão funciona inteiramente via API, sem dependências extras.*
""")

# Verificar API key
api_key = st.secrets.get("OPENROUTER_API_KEY")

if not api_key:
    st.error("❌ Configure OPENROUTER_API_KEY nos Secrets do Streamlit")
    st.stop()

# Teste de conexão inicial
if "connection_tested" not in st.session_state:
    with st.spinner("🔧 Testando conexão com OpenRouter..."):
        test_result = call_openrouter_api("Responda: OK", api_key, 5)
        if "OK" in test_result or "ok" in test_result.lower():
            st.success("✅ Conexão com OpenRouter funcionando!")
            st.session_state.connection_tested = True
        else:
            st.error(f"❌ Problema na conexão: {test_result}")
            st.stop()

# Interface principal
user_question = st.text_input(
    "💬 Digite sua pergunta:",
    placeholder="Ex: O que é uma célula eucariota? Como funciona o efeito fotoelétrico?"
)

if user_question:
    with st.spinner("🤔 Processando sua pergunta..."):
        answer, disciplina = process_question(user_question, api_key)
    
    # Exibir resultado
    if disciplina:
        st.info(f"🔍 **Disciplina identificada:** {disciplina.capitalize()}")
    
    if answer.startswith("Erro"):
        st.error(f"❌ {answer}")
    else:
        st.success(f"📖 {answer}")
    
    # Feedback e ações
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👍 Útil"):
            st.balloons()
            st.success("Obrigado!")
    
    with col2:
        if st.button("👎 Não útil"):
            st.info("Tente reformular sua pergunta para melhores resultados.")
    
    with col3:
        if st.button("🔄 Nova pergunta"):
            st.rerun()

# Exemplos de perguntas
with st.expander("💡 Exemplos de Perguntas"):
    st.markdown("""
    **🧬 Biologia:**
    - O que é uma célula eucariota?
    - Quais são as principais organelas celulares?
    - Qual a diferença entre célula vegetal e animal?
    - Como funciona a divisão celular?
    
    **⚛️ Física:**
    - O que é o efeito fotoelétrico?
    - Como Einstein explicou o efeito fotoelétrico?
    - O que é mecânica quântica?
    - Explique o princípio da incerteza de Heisenberg.
    """)

# Status do sistema
with st.expander("ℹ️ Status do Sistema"):
    st.success("✅ Sistema operacional")
    st.write("🤖 **LLM:** OpenRouter (gpt-3.5-turbo)")
    st.write("🔍 **Busca:** Palavras-chave + pontuação")
    st.write("📚 **Base:** Conhecimento embarcado")
    st.write("🎯 **Disciplinas:** Biologia, Física")