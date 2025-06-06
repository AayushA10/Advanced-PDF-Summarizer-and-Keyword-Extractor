import streamlit as st
import PyPDF2
import spacy
from transformers import pipeline
from keybert import KeyBERT
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load Models
nlp = spacy.load('en_core_web_sm')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
kw_model = KeyBERT()

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # Check if page_text is not None
            text += page_text
    return text

# Summarize text
def generate_summary(text):
    if len(text.split()) < 150:
        st.warning("Text is too short for summarization, showing original text.")
        return text
    else:
        summary = summarizer(text, max_length=200, min_length=30, do_sample=False)
        return summary[0]['summary_text']

# Extract keywords
def extract_keywords(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
    return [kw[0] for kw in keywords]

# Create wordcloud
def create_wordcloud(keywords):
    text = ' '.join(keywords)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Streamlit UI
st.title('📚 Advanced PDF Summarizer & Keyword Extractor')

uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)

    st.subheader('📖 Generated Summary')
    summary = generate_summary(text)
    st.write(summary)

    st.download_button("Download Summary", summary, file_name='summary.txt')

    st.subheader('🔑 Extracted Keywords')
    keywords = extract_keywords(text)
    st.write(keywords)

    st.download_button("Download Keywords", '\n'.join(keywords), file_name='keywords.txt')

    st.subheader('☁️ WordCloud of Keywords')
    create_wordcloud(keywords)
