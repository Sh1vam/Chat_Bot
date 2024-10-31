import nltk
def basic_downloads():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
try :
    from nltk.corpus import stopwords
except:
    basic_downloads()

try:
    from string_regex import *
    from nltk_regex import *
    from spacy_regex import *
except:
    from nlp_lib.string_regex import *
    from nlp_lib.nltk_regex import *
    from nlp_lib.spacy_regex import *
else :
    from . import *
'''    
try :  
    __all__ = (
        string_regex.__all__ +
        nltk_regex.__all__ +
        spacy_regex.__all__
    )
except :
    __all__ = ( nlp_lib.__all__)
else:
    print("Start")'''
