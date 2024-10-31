import re
import string
punctuation = string.punctuation
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import (
    PorterStemmer,
    LancasterStemmer,
    SnowballStemmer,
    RegexpStemmer
)
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.cistem import Cistem

def basic_downloads():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))
lemmatizer = wnl = WordNetLemmatizer()
# Initialize the stemmers
ps = PorterStemmer()
ls = LancasterStemmer()
ss = SnowballStemmer("english")
rs = RegexpStemmer('ing$|s$|e$', min=4) # Fixed the syntax error in RegexpStemmer
wordnet_map={"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
def get_wordnet_pos(tag):
    if pd.isnull(tag):  
        return np.nan
    if tag.startswith('J'):
        return wordnet.ADJ  # adjective
    elif tag.startswith('V'):
        return wordnet.VERB  # verb
    elif tag.startswith('N'):
        return wordnet.NOUN  # noun
    elif tag.startswith('R'):
        return wordnet.ADV  # adverb
    else:
        return wordnet.NOUN  # default to noun

def remove_stopwords(text):
    if pd.isnull(text):  
        return np.nan
    tokens = text.split()
    filtered_words = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_words)


def clean_text_list(text):
    if pd.isnull(text):  
        return np.nan
    text = "".join([word for word in text if word not in punctuation])
    tokens = re.split(r'\W+', text)
    text = [word for word in tokens if word not in stopwords.words('english')]  # Use stopwords.words('english') directly
    return text
def clean_text(text):
    if pd.isnull(text):  
        return np.nan
    text = "".join([word for word in text if word not in punctuation])
    tokens = re.split(r'\W+', text)
    text = [word for word in tokens if word not in stopwords.words('english')]  # Use stopwords.words('english') directly
    return ' '.join(text)

def stem_words(text):
    if pd.isnull(text):  
        return np.nan
    return ' '.join([ps.stem(word) for word in text.split()])

def tokenized_text(text):
    if pd.isnull(text):  
        return np.nan
    return nltk.word_tokenize(text)
# Define stemming functions for each stemmer
def porter_stemming(tokenised_text):
    if isinstance(tokenised_text, list):  # Check if it's a list
        return [ps.stem(word) for word in tokenised_text]  # Apply stemming word by word
    else:
        return np.nan 

def lancaster_stemming(tokenised_text):
    if isinstance(tokenised_text, list):  # Check if it's a list
        return [ls.stem(word) for word in tokenised_text]  # Apply stemming word by word
    else:
        return np.nan 
def snowball_stemming(tokenised_text):
    if isinstance(tokenised_text, list):  # Check if it's a list
        return [ss.stem(word) for word in tokenised_text]  # Apply stemming word by word
    else:
        return np.nan 

def regexp_stemming(tokenised_text):
    """if pd.isnull(tokenized_text):  
        return np.nan
    return [rs.stem(word) for word in tokenized_text]"""
    if isinstance(tokenised_text, list):  # Check if it's a list
        return [rs.stem(word) for word in tokenised_text]  # Apply stemming word by word
    else:
        return np.nan 

# Define lemmatization function using WordNetLemmatizer
def wordnet_lemmatizing(tokenised_text):
    if isinstance(tokenised_text, list):  
        return [lemmatizer.lemmatize(word) for word in tokenised_text]  
    else:
        return np.nan 


def perform_lemmatization_with_pos(text):
    if pd.isnull(text):  
        return np.nan
    tokens = nltk.word_tokenize(text)  # Tokenize the tweet
    pos_tags = nltk.pos_tag(tokens)  # Get POS tags for each token
    lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]
    return ' '.join(lemmatized_words)

def word_pos_tag(text):
    if pd.isnull(text):  
        return np.nan
    # Tokenize the text
    tokens = re.split(r'\W+', text)
    # Remove stopwords
    text = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatize the text
    text_lemmatized = [wnl.lemmatize(word) for word in text]
    # Perform POS tagging
    pos_tags = nltk.pos_tag(text_lemmatized)
    return pos_tags 
    # return text  # This line is unreachable and can be removed

