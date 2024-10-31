#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string 
import re
import  pandas as pd
import numpy as np

# In[2]:

def list_joint(text):
    # Check if the input is a list of characters
    if isinstance(text, list):
        return ' '.join(text)  # Join the list of characters into a string
    
    # If it's NaN, return an empty string
    if pd.isnull(text):
        return ""
    
    return str(text)  # Otherwise, just convert to a string
  
def remove_punctuations(text):
  if pd.isnull(text):  
        return np.nan
  text = text.translate(str.maketrans('', '', string.punctuation))
  return text.strip()


# In[3]:


def removes_specials(text):
  tmp=text
  if pd.isnull(tmp):  
        return np.nan
  pattern = re.compile(r'[A-Za-z0-9\s]')
  specials = re.sub(pattern, "", tmp)
  return text.translate(str.maketrans('', '', specials)).strip()


# In[4]:


def removes_non_printables(text):
  tmp=text
  if pd.isnull(tmp):  
        return np.nan
  specials = tmp.translate(str.maketrans('', '', string.printable)).strip()
  return text.translate(str.maketrans('', '', specials)).strip()


# In[5]:


def remove_whitespace(text):
  if pd.isnull(text):  
        return np.nan
  text = text.translate(str.maketrans('', '', string.whitespace))
  return text.strip()


# In[6]:


def remove_hexdigits(text):
  if pd.isnull(text):  
        return np.nan
  text = text.translate(str.maketrans('', '', string.hexdigits))
  return text.strip()


# In[7]:


def remove_octdigits(text):
  if pd.isnull(text):  
        return np.nan
  text = text.translate(str.maketrans('', '', string.octdigits))
  return text.strip()


# In[8]:



def remove_html_tags(text):
    if isinstance(text, str):  # Check if the input is a string
        reg = re.compile(r"<[^>]*>")  # Regular expression to match HTML tags
        return reg.sub('', text)  # Remove HTML tags
    else:
        return text


# In[9]:


def remove_url(text):
  if isinstance(text, str):
    url_pattern = re.compile(r"(http?|ftp|https?|file)://\S+ ")
    return re.sub(url_pattern, "", text)
  else:
    return text


# In[10]:


'''use on df by df[].apply(function_name)
remove_punctuations
remove_url
remove_html_tags
removes_specials'''

