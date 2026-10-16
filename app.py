"""
Analisador de Sentimentos Offline / Local com Streamlit e TextBlob.
Licença: MIT
"""

import streamlit as st
from textblob import TextBlob
from deep_translator import GoogleTranslator


def translate_if_needed(text: str) -> str:
    """
    Traduz o texto de entrada para o inglês via deep-translator caso necessário,
    garantindo máxima acurácia na análise de polaridade léxica do TextBlob.
    """
    try:
        # source='auto' detecta automaticamente o idioma (ex.: pt -> en)
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated if translated else text
    except Exception:
        # Fallback resiliente: caso ocorra falha de conexão na tradução, analisa o original
        return text


def classify_sentiment(polarity: float) -> tuple[str, str, str]:
    """
    Determina a classe de sentimento, mensagem e tipo de alerta baseado no score de polaridade.
    Polaridade do TextBlob varia de -1.0 (muito negativo) a +1.0 (muito positivo).
    """
    # Thresholds clássicos para TextBlob
    if polarity > 0.1:
        return "Positivo", "success", "😊 O texto transmite um sentimento predominantemente **Positivo**."
    elif polarity < -0.1:
        return "Negativo", "error", "😞 O texto transmite um sentimento predominantemente **Negativo**."
    else:
        return "Neutro", "warning", "😐 O texto transmite um sentimento predominantemente **Neutro** ou balanceado."


def main():
    # Configuração da página
    st.set_page_config(
        page_title="Analisador de Sentimentos Offline",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # Cabeçalho da aplicação
    st.title("🧠 Analisador de Sentimentos Offline")
    st.caption("Processamento de Linguagem Natural (PLN) com bibliotecas tradicionais de Python")

    st.markdown(
        """
        Esta aplicação realiza a análise de sentimentos **sem depender de modelos proprietários de LLM** 
        (como OpenAI ou Google Gemini). O pipeline utiliza a técnica de análise léxica baseada em regras 
        do **TextBlob**, com tradução automática transparente para assegurar alta precisão mesmo com entradas em português.
        """
    )
    st.divider()

    # Entrada do usuário
    user_input = st.text_area(
        label="Digite ou cole sua frase/parágrafo aqui:",
        placeholder="Ex.: O atendimento foi espetacular e a equipe super prestativa!",
        height=150,
    )

    # Botão de Ação
    if st.button("Analisar Sentimento", type="primary", use_container_width=True):
        clean_text = user_input.strip()

        if not clean_text:
            st.info("Por favor, insira algum texto antes de iniciar a análise.")
            return

        with st.spinner("Processando texto e calculando polaridade..."):
            # Tradução transparente de suporte
            processed_text = translate_if_needed(clean_text)

            # Análise léxica com TextBlob
            blob = TextBlob(processed_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            sentiment_label, status_type, status_msg = classify_sentiment(polarity)

        # Exibição dos resultados visuais
        st.subheader("Resultado da Análise")

        if status_type == "success":
            st.success(status_msg)
        elif status_type == "error":
            st.error(status_msg)
        else:
            st.warning(status_msg)

        # Métricas detalhadas em colunas
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Polaridade",
                value=f"{polarity:.2f}",
                help="Varia de -1.0 (extremamente negativo) a +1.0 (extremamente positivo).",
            )
        with col2:
            st.metric(
                label="Subjetividade",
                value=f"{subjectivity:.2f}",
                help="Varia de 0.0 (muito objetivo/factual) a 1.0 (muito subjetivo/opinativo).",
            )


if __name__ == "__main__":
    main()
