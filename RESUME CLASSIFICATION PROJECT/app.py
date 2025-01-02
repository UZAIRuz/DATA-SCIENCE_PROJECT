# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 00:18:45 2024

@author: viraj
"""
import pickle
import streamlit as st
import pandas as pd
from tika import parser
import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nlp = spacy.load("en_core_web_sm")

# download additional data resources for nltk
nltk.download('punkt')
nltk.download('stopwords')


# Reading the file using tika
def extract_text(file_path):
    parsed = parser.from_file(file_path)
    return parsed["content"]

# Cleaning the extracted text
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def clean_text(text):
    cleaned_text = str(text)
    # Remove email-like addresses
    cleaned_text = re.sub(r'\S+@\S+', '', cleaned_text)
    # Remove links
    cleaned_text = re.sub(r'\S+\.com\S*', '', cleaned_text)
    # Remove URLS
    cleaned_text = re.sub(r'http\S+', '', cleaned_text)
    # Remove Emojis
    cleaned_text = deEmojify(cleaned_text)
    # Remove images
    cleaned_text = re.sub(r'\b\w+\.(png|jpg|jpeg)\b', '', text)
    # Removing the escape characters
    cleaned_text = re.sub(r'\\.', '', cleaned_text)
    # Removing bullets
    cleaned_text = re.sub(r' · ', '', cleaned_text)
    # Remove all the non-alpha symbols
    cleaned_text = re.sub(r'[^a-zA-Z]',' ',cleaned_text)
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # Removing 2 character word
    cleaned_text = re.sub(r'\b([a-zA-Z])\1\b', "", cleaned_text)
    # Removing single character word
    cleaned_text = re.sub(r'\b[a-zA-Z]\b', "", cleaned_text)
    # Converting to the lowercase
    cleaned_text = cleaned_text.lower()

    return cleaned_text.strip()

# Function to do text normalization
def lemmatization(text):
    # Process the text 
    doc = nlp(text) 
    # Extract and lemmatize tokens 
    lemmatized_tokens = [token.lemma_.strip() for token in doc]
    return ' '.join(lemmatized_tokens)

# Function removes stopwords from the text
def remove_stopwords(text):
    # Define stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text) # Tokenize the text
    filtered_text = [word.strip() for word in word_tokens if word.lower() not in stop_words] # Filter out stopwords
    
    return ' '.join(filtered_text)

# Removing the follwing NER from the dataset

ner_categories =  ['PERSON','GPE','LOC','NORP', 'FAC','PRODUCT','EVENT','WORK_OF_ART','DATE','TIME','LANGUAGE','MONEY']   # https://dataknowsall.com/blog/ner.html
def remove_NER_categories(text):
    doc = nlp(text)
    # Identify named entities and remove names
    cleaned_text = " ".join([token.text for token in doc if not token.ent_type_ in ner_categories])
    return cleaned_text

# function to Use regular expressions to find and remove words with less than 3 characters
def remove_short_words(text):
    cleaned_text = re.sub(r'\b\w{1,2}\b', '', text)
    # Remove any extra spaces created by the removal
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

file_type_df = pd.DataFrame([], columns=['Uploaded File',  'Predicted Profile'])
filename = []
predicted = []

# Open model in read binary mode
rf_model = pickle.load(open(r'rf_clf.pkl','rb'))
vectorizer = pickle.load(open(r'vector.pkl','rb'))

def main():
    st.markdown("<h1 style='text-align: center;'>Resume Classification</h1>", unsafe_allow_html=True)
    st.image('Resume_img.jpg', width=700, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
    st.markdown("<h2 style='text-align: center;'>Upload the resumes you want to classify</h1>", unsafe_allow_html=True)
    #st.title('Resume Classification')
    #st.subheader('Upload the resumes you want to classify')
    uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=True)
    
    for doc_file in uploaded_file:
        if doc_file is not None:
            filename.append(doc_file.name)
            text = extract_text(doc_file)
            processed_text = remove_short_words(remove_NER_categories(remove_stopwords(lemmatization(clean_text(text)))))
            rol_dict = {0: 'Peoplesoft', 1: 'React Developer', 2: 'SQL Developer', 3: 'workday'}
            result = rf_model.predict(vectorizer.transform([processed_text]))[0]
            predicted.append(rol_dict[result])
            
    if st.button('Classify'):
        if len(predicted) > 0:
            file_type_df['Uploaded File'] = filename
            file_type_df['Predicted Profile'] = predicted
            st.table(file_type_df.style.format())
    
    
    #if uploaded_file is not None:
        # To read file as string:
        #text = extract_text(uploaded_file)
#        processed_text = remove_short_words(remove_NER_categories(remove_stopwords(lemmatization(clean_text(text)))))
  #      
  #      if st.button('Read'):
  #          st.write(processed_text)
  #         
  #      if st.button('Classify'):
  #        rol_dict = {0: 'Peoplesoft', 1: 'React Developer', 2: 'SQL Developer', 3: 'workday'}
  #         result = rf_model.predict(vectorizer.transform([processed_text]))[0]
   #         st.write('Job Roll classification is : '+rol_dict[result])
        
if __name__ == '__main__':
    main()