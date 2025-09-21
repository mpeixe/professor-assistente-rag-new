import streamlit as st
import requests
import json
import re

# Configuração da página
st.set_page_config(
    page_title="Professor Assistente RAG",
    page_icon="👨‍🏫",
    layout="wide"
)

# Base de conhecimento simples (embarcada)
KNOWLEDGE_BASE = {
    "biologia": [
        "A célula eucariota possui núcleo definido, delimitado por membrana nuclear. O material genético (DNA) fica organizado dentro do núcleo, diferente das células procariontes.",
        "As principais organelas celulares são: mitocôndrias (respiração), retículo endoplasmático (síntese de proteínas), complexo de Golgi (processamento), lisossomos (digestão celular).",
        "Diferença eucariota vs procariota: eucariotas têm núcleo organizado, procariotas têm nucleoide sem membrana. Exemplos: eucariotas (animais, plantas), procariotas (bactérias).",
        "Células vegetais têm parede celular, cloroplastos e vacúolo grande. Células animais não têm parede celular nem cloroplastos.",
        "Divisão celular: mitose (células somáticas, mantém cromossomos) e meiose (células reprodutivas, reduz cromossomos pela metade)."
    ],
    "fisica": [
        "O efeito fotoelétrico foi explicado por Einstein (1905, Nobel 1921). Elétrons são emitidos quando luz incide sobre metal com frequência suficiente.",
        "No efeito fotoelétrico, fótons devem ter energia suficiente para superar a função trabalho do material e liberar elétrons do metal.",
        "Energia dos fótons: E = hf (h = constante de Planck, f = frequência). Esta relação demonstra a natureza quântica da luz.",
        "Mecânica quântica descreve partículas em escala atômica. Conceitos: quantização de energia, dualidade onda-partícula, probabilidade.",
        "Princípio da incerteza de Heisenberg: impossível medir simultaneamente com precisão total a posição e momento de uma partícula."
    ]
}

def call_openrouter(prompt: str, api_key: str) -> str:
    """Chama OpenRouter de forma simples"""
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "openai/gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.3
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            return f"Erro API: {response.status_code}"
            
    except Exception as e:
        return f"Erro: {str(e)}"

def classify_question(question: str, api_key: str) -> str:
    """Classifica pergunta"""
    prompt = f"""Esta pergunta é sobre biologia ou física?
    
    Pergunta: {question}
    
    Responda apenas: biologia OU fisica OU outros"""
    
    result = call_openrouter(prompt, api_key)
    
    if "biologia" in result.lower():
        return "biologia"
    elif "fisica" in result.lower() or "física" in result.lower():
        return "fisica"
    else:
        return "outros"

def find_best_content(question: str, disciplina: str) -> str:
    """Busca melhor conteúdo para a pergunta"""
    if disciplina not in KNOWLEDGE_BASE:
        return ""
    
    question_lower = question.lower()
    best_content = []
    
    # Palavras-chave por disciplina
    keywords = {
        "biologia": ["célula", "eucariota", "organela", "núcleo", "mitocôndria", "vegetal", "animal", "divisão", "mitose", "meiose"],
        "fisica": ["efeito", "fotoelétrico", "einstein", "quantum", "quântica", "heisenberg", "incerteza", "fóton", "energia"]
    }
    
    for content in KNOWLEDGE_BASE[disciplina]:
        score = 0
        
        # Pontuar por palavras-chave
        for keyword in keywords[disciplina]:
            if keyword in question_lower:
                if keyword in content.lower():
                    score += 3
        
        # Pontuar por palavras da pergunta
        question_words = re.findall(r'\b\w{4,}\b', question_lower)
        for word in question_words:
            if word in content.lower():
                score += 1
        
        if score > 0:
            best_content.append((score, content))
    
    # Retornar os 2 melhores
    best_content.sort(reverse=True, key=lambda x: x[0])
    return "\n\n".join([item[1] for item in best_content[:2]])

def generate_answer(question: str, context: str, disciplina: str, api_key: str) -> str:
    """Gera resposta educativa"""
    if not context:
        return f"Não encontrei informações específicas sobre '{question}'. Tente perguntar sobre tópicos básicos de {disciplina}."
    
    prompt = f"""Você é um professor assistente. Responda a pergunta usando o contexto fornecido.
    Seja educativo, claro e didático.
    
    Contexto sobre {disciplina}:
    {context}
    
    Pergunta do aluno: {question}
    
    Resposta educativa:"""
    
    return call_openrouter(prompt, api_key)

# --- Interface Streamlit ---

st.title("👨‍🏫 Professor Assistente RAG")
st.markdown("""
**Faça perguntas sobre Biologia ou Física!**

📚 **Disponível:**
- 🧬 **Biologia**: Células, organelas, diferenças celulares
- ⚛️ **Física**: Efeito fotoelétrico, mecânica quântica
""")

# Verificar API key
api_key = st.secrets.get("OPENROUTER_API_KEY")
if not api_key:
    st.error("❌ Configure OPENROUTER_API_KEY nos Secrets")
    st.stop()

st.success("✅ Sistema pronto! (OpenRouter configurado)")

# Interface principal
user_question = st.text_input(
    "💬 Sua pergunta:",
    placeholder="Ex: O que é uma célula eucariota?"
)

if user_question:
    with st.spinner("🤔 Pensando..."):
        # 1. Classificar
        disciplina = classify_question(user_question, api_key)
        
        if disciplina == "outros":
            st.warning("⚠️ Pergunta não identificada como Biologia ou Física. Tente reformular.")
        else:
            st.info(f"🔍 **Disciplina:** {disciplina.capitalize()}")
            
            # 2. Buscar contexto
            context = find_best_content(user_question, disciplina)
            
            # 3. Gerar resposta
            answer = generate_answer(user_question, context, disciplina, api_key)
            
            # 4. Mostrar resultado
            if answer.startswith("Erro"):
                st.error(f"❌ {answer}")
            else:
                st.success(f"📖 {answer}")

# Exemplos rápidos
st.markdown("### 💡 Exemplos:")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🧬 Biologia:**")
    if st.button("O que é célula eucariota?"):
        st.rerun()
    if st.button("Diferença entre célula vegetal e animal?"):
        st.rerun()

with col2:
    st.markdown("**⚛️ Física:**")
    if st.button("O que é efeito fotoelétrico?"):
        st.rerun()
    if st.button("Explique mecânica quântica"):
        st.rerun()

# Status
with st.expander("ℹ️ Como funciona"):
    st.write("""
    1. 🤖 **Classificação**: OpenRouter identifica se é Biologia ou Física
    2. 🔍 **Busca**: Sistema encontra conteúdo relevante na base de dados
    3. 📖 **Resposta**: OpenRouter gera explicação educativa
    4. ✅ **Resultado**: Resposta personalizada para sua pergunta
    """)