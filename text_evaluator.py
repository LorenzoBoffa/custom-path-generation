import streamlit as st
import spacy
import pandas as pd
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stop_words = get_stop_words('italian')

col1, col2 = st.columns(2)

if 'df_sorted' not in st.session_state:
    st.session_state.df_sorted = pd.DataFrame(columns=['Keyword', 'Normalized Document Score'])

if 'df_similarity' not in st.session_state:
    st.session_state.df_similarity = pd.DataFrame(columns=['Query', 'Normalized Document Score'])

with col1: 
    text = st.text_area("Inserisci il testo da valutare:")
    
    if st.button("Calcola"):
        if text:
            st.session_state.df_similarity = pd.DataFrame(columns=['Query', 'Normalized Document Score'])
            
            tokens = text.split()
            keywords = [word for word in tokens if word.lower() not in stop_words]
            filtered_text = ' '.join(keywords)

            nlp = spacy.load("it_core_news_sm")
            doc = nlp(filtered_text)
            text_chunks = [sent.text for sent in doc.sents]

            st.session_state.text_chunks = text_chunks

            keyword_stats = []

            keyword_without_punctuation = list(set([keyword.lower().replace(",", "").replace(".", "").replace(":", "").replace(";", "").replace("'", "").replace('"', "") for keyword in keywords]))
            for keyword in keyword_without_punctuation:
                prompt = keyword
            
                vectorizer = TfidfVectorizer(stop_words=stop_words)
                tfidf_matrix = vectorizer.fit_transform(text_chunks)
                prompt_vector = vectorizer.transform([prompt])
            
                similarity_scores = cosine_similarity(prompt_vector, tfidf_matrix).flatten()
                document_score = similarity_scores.sum() 
                max_possible_score = len(text_chunks)  
                normalized_score = document_score / max_possible_score

                document_length = sum(len(chunk.split()) for chunk in text_chunks) 
                num_chunks = len(similarity_scores) 

                keyword_stats.append({
                    'Keyword': keyword,
                    'Normalized Document Score': normalized_score,
                })

            df = pd.DataFrame(keyword_stats)
            df_sorted = df.sort_values(by='Normalized Document Score', ascending=False)
            st.session_state.df_sorted = df_sorted.reset_index(drop=True) 

    if not st.session_state.df_sorted.empty:
        st.dataframe(st.session_state.df_sorted)

with col2:
    new_sentence = st.text_area("Inserisci un prompt:")
    
    if st.button("Calcola prompt"):
        text_chunks = st.session_state.get('text_chunks', [])
        
        if text_chunks and new_sentence:  
            vectorizer = TfidfVectorizer(stop_words=stop_words)
            tfidf_matrix = vectorizer.fit_transform(text_chunks)
            new_sentence_vector = vectorizer.transform([new_sentence])
            similarity_scores = cosine_similarity(new_sentence_vector, tfidf_matrix).flatten()
            document_score = similarity_scores.sum()
            max_possible_score = len(text_chunks)
            normalized_score = document_score / max_possible_score

            new_entry = pd.DataFrame({'Query': [new_sentence], 'Normalized Document Score': [normalized_score]})
            
            st.session_state.df_similarity = pd.concat([st.session_state.df_similarity, new_entry], ignore_index=True)
            st.session_state.df_similarity = st.session_state.df_similarity.sort_values(by='Normalized Document Score', ascending=False)
            st.session_state.df_similarity = st.session_state.df_similarity.reset_index(drop=True)
        else:
            st.write("Processa prima il documento principale nel box di sinistra")

    if not st.session_state.df_similarity.empty:
        st.dataframe(st.session_state.df_similarity)
