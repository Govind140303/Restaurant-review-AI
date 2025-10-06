import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
from textblob import TextBlob

# -----------------------------------
#  Define sentiment aspect keywords
# -----------------------------------
ASPECT_KEYWORDS = {
    "food": ["pizza", "taste", "flavor", "crust", "cheese", "menu", "pasta", "food", "dish", "portion"],
    "service": ["waiter", "staff", "service", "delivery", "attitude", "response", "speed"],
    "ambience": ["ambience", "atmosphere", "environment", "music", "decor", "vibe", "place"],
    "price": ["price", "cost", "expensive", "cheap", "value", "money", "bill"]
}

# -----------------------------------
#  Functions
# -----------------------------------
def analyze_sentiment(texts):
    """Compute overall and aspect-specific sentiment."""
    if not texts:
        return {"overall": 0, "food": 0, "service": 0, "ambience": 0, "price": 0}

    overall_scores = []
    aspect_scores = {key: [] for key in ASPECT_KEYWORDS}

    for text in texts:
        blob = TextBlob(text)
        overall_scores.append(blob.sentiment.polarity)
        lower_text = text.lower()
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(k in lower_text for k in keywords):
                aspect_scores[aspect].append(blob.sentiment.polarity)

    aspect_sentiment = {"overall": sum(overall_scores) / len(overall_scores) if overall_scores else 0}
    for aspect, scores in aspect_scores.items():
        aspect_sentiment[aspect] = sum(scores) / len(scores) if scores else 0
    return aspect_sentiment


# -----------------------------------
#  LLM + Prompt Setup
# -----------------------------------
model = OllamaLLM(model="llama3.2")

prompt_template = """
You are an expert in answering questions about a pizza restaurant.

Here are relevant customer reviews:
{reviews}

Average sentiment scores (from -1 to +1):
- Overall: {overall_sentiment:.2f}
- Food: {food_sentiment:.2f}
- Service: {service_sentiment:.2f}
- Ambience: {ambience_sentiment:.2f}
- Price: {price_sentiment:.2f}

User's question:
{question}

Give a clear, friendly, and data-based response.
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
chain = prompt | model


# -----------------------------------
# 🌐 Streamlit Web UI
# -----------------------------------
st.set_page_config(page_title=" Restaurant Review AI", page_icon="", layout="wide")

st.title(" AI Restaurant Review Assistant")
st.write("Ask any question about the restaurant — powered by **LangChain + Ollama + ChromaDB**")

question = st.text_input(" Ask your question:", placeholder="e.g., How is the pizza quality?")

if question:
    with st.spinner("Retrieving reviews and generating answer..."):
        retrieved_docs = retriever.invoke(question)
        reviews_text = [doc.page_content for doc in retrieved_docs]

        sentiment_data = analyze_sentiment(reviews_text)
        reviews_combined = "\n\n".join(reviews_text)

        result = chain.invoke({
            "reviews": reviews_combined,
            "overall_sentiment": sentiment_data["overall"],
            "food_sentiment": sentiment_data["food"],
            "service_sentiment": sentiment_data["service"],
            "ambience_sentiment": sentiment_data["ambience"],
            "price_sentiment": sentiment_data["price"],
            "question": question
        })

    st.subheader(" AI Response")
    st.write(result)

    st.subheader(" Sentiment Breakdown")
    cols = st.columns(5)
    for i, (aspect, score) in enumerate(sentiment_data.items()):
        emoji = "😃" if score > 0.3 else ("😐" if -0.3 <= score <= 0.3 else "☹️")
        cols[i].metric(label=aspect.capitalize(), value=f"{score:+.2f}", delta=emoji)

    st.markdown("---")
    with st.expander(" See the top reviews retrieved"):
        for doc in retrieved_docs:
            st.write(f"- {doc.page_content}")
