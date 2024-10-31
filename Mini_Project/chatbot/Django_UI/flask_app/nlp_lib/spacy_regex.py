import re
import spacy
from spacy import displacy
import pandas as pd
import numpy as np
nlp=spacy.load("en_core_web_sm")
options={'distance':110,"compact":True,"color":"black","bg":"white","font":"time"}

def pos_tag(text):
    if pd.isnull(text):  
        return np.nan
    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    return pos_tags


def lower_case(body_text):
    if pd.isnull(body_text):  
        return np.nan
    # Call the pos_tag function to get the pos_tags for the current body_text
    pos_tags = pos_tag(body_text)
    return [(word.lower(), tag) for word, tag in pos_tags]


def displacy_render(text):
    if pd.isnull(text):  
        return np.nan
    doc = nlp(text)
    try:
        displacy.render(doc, style="dep", jupyter=True, options=options)
    except :
        pass
    finally :
        print("use jupyter for this")
    return None

def stopwordremwithlema(text):
    if pd.isnull(text):  
        return np.nan
    doc = nlp(text)

    filtered_tokens = []
    for token in doc:
        if token.is_stop:
            continue
        filtered_tokens.append(token.lemma_)

    return " ".join(filtered_tokens)

def puncremwithlema(text):
    if pd.isnull(text):  
        return np.nan
    doc = nlp(text)

    filtered_tokens = []
    for token in doc:
        if token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)

    return " ".join(filtered_tokens)
def lema(text):
    if pd.isnull(text):  
        return np.nan
    doc = nlp(text)

    filtered_tokens = []
    for token in doc:

        filtered_tokens.append(token.lemma_)

    return " ".join(filtered_tokens)
def preprocesswithlema(text):
    if pd.isnull(text):  
        return np.nan
    doc = nlp(text)

    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)

    return " ".join(filtered_tokens)

def stopwordrem(text):
    if pd.isnull(text):  
        return np.nan
    doc = nlp(text)

    filtered_tokens = []
    for token in doc:
        if token.is_stop:
            continue
        filtered_tokens.append(token.text)

    return " ".join(filtered_tokens)

def puncrem(text):
    if pd.isnull(text):  
        return np.nan
    doc = nlp(text)

    filtered_tokens = []
    for token in doc:
        if token.is_punct:
            continue
        filtered_tokens.append(token.text)

    return " ".join(filtered_tokens)

def preprocess(text):
    if pd.isnull(text):  
        return np.nan
    doc = nlp(text)

    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.text)

    return " ".join(filtered_tokens)
