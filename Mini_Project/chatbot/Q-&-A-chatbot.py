#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nlp_lib import * # this will directly load pandas,numpy as pd and np : https://github.com/Sh1vam/nlp_lib


# In[2]:


from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import joblib
import yaml
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import chain
import random
from tensorflow.keras.models import load_model


# In[3]:


stopwords = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
df=pd.read_csv("myds.csv")
df.drop(columns="Intent",inplace=True)
df['Question']=df['Question'].str.lower()
df['Answer']=df['Answer'].str.lower()
df['Question']=df['Question'].apply(removes_non_printables)
df['Answer']=df['Answer'].apply(removes_non_printables)


# In[4]:


# Define a basic synonym replacement function with built-in methods
def basic_synonym_replacement(sentence, limit=100):
    tokens = sentence.split()  # Use basic split to tokenize
    augmented_sentences = []
    
    # A simple manual synonym dictionary as a placeholder for WordNet
    synonym_dict = {
        "is": ["exists", "constitutes"],
        "belt": ["region", "area"],
        "largest": ["biggest", "most substantial","huge"],
        "smallest":["small","tiny"],
        "object": ["entity", "thing"],
        "asteroid": ["space rock", "minor planet"],
        "what": ["define what","which"],
        "how many":["what is","what are","how much"],
        "how much":["what is","what are","how many"],
        "what is": ["define","give","what are", "give me","tell about","tell me about"],
        "which":["what"],
        "which is":["what","give","which are", "give me","tell about","tell me about"],
        "exists":["where exists","what exists"],
        "mars":["red planet","redplanet","red-planet"],
        "much":["many"],
        "what is the distance":["how much distance","how much is the distance"],
        "solar system":["solarsystem","solar-system"],
        "solarsystem":["solar system","solar-system"],
        "between":["from"],
        "to":["and"],
        "do":["does"],
        "does":["do"],
        "the":["a","an",""],
        'a':["the","an",""],
        "an":["the","a",""],
        "number":["no"]
    }
    
    # Iterate through each token in the sentence
    for i, token in enumerate(tokens):
        synonyms = synonym_dict.get(token.lower(), [])  # Get synonyms if available
        
        if synonyms:
            num_augmentations = min(limit, len(synonyms))
            sampled_synonyms = random.sample(synonyms, num_augmentations)
            
            # Create new augmented sentences by replacing the word with each synonym
            for synonym in sampled_synonyms:
                augmented_tokens = tokens[:i] + [synonym] + tokens[i+1:]
                augmented_sentences.append(' '.join(augmented_tokens))
    
    return augmented_sentences

def augment_dataset(df, limit=1111):
    augmented_data = []

    for _, row in df.iterrows():
        question = row['Question']
        answer = row['Answer']

        # Generate augmented sentences for the question
        augmented_sentences = basic_synonym_replacement(question, limit)
        
        # Store the original question with the corresponding answer
        augmented_data.append([question, answer])  # Original question
        
        # Store each augmented question with the same original answer
        for augmented_sentence in augmented_sentences:
            augmented_data.append([augmented_sentence, answer])  # Augmented question with original answer

    # Return the augmented data as a DataFrame with both Question and Answer columns
    return pd.DataFrame(augmented_data, columns=['Question', 'Answer'])

# Apply the function to the dataset
augmented_df1 = augment_dataset(df)
#df=pd.concat([df, augmented_df1], ignore_index=True)
df=augmented_df1


# In[5]:


# Synonym replacement function with caching
synonym_cache = {}

def synonym_replacement(tokens, limit=100):
    augmented_sentences = []
    
    for i in range(len(tokens)):
        token = tokens[i]
        
        # Check if we've already cached the synonyms for this token
        if token in synonym_cache:
            synonyms = synonym_cache[token]
        else:
            synonyms = set()
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != token:
                        synonyms.add(synonym)
            synonyms = list(synonyms)
            synonym_cache[token] = synonyms  # Cache the result
        
        # If synonyms were found, create augmented sentences
        if synonyms:
            num_augmentations = min(limit, len(synonyms))
            sampled_synonyms = random.sample(synonyms, num_augmentations)
            for synonym in sampled_synonyms:
                augmented_tokens = tokens[:i] + [synonym] + tokens[i+1:]
                augmented_sentences.append(' '.join(augmented_tokens))
    
    return augmented_sentences

# Define limit for augmentation per tag
limit_per_tag = 1111

# Process each row in the dataset and collect augmented data
text_data = []
answers = []

for _, row in df.iterrows():
    example = row['Question']
    answer = row['Answer']
    
    # Tokenize and preprocess
    tokens = word_tokenize(example.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    
    # Only augment if the filtered tokens are valid
    if filtered_tokens:
        # Add the original data
        text_data.append(' '.join(filtered_tokens))
        answers.append(answer)
        
        # Generate augmented sentences
        augmented_sentences = synonym_replacement(filtered_tokens, limit=limit_per_tag)
        
        # Limit the number of augmentations per tag
        for idx, augmented_sentence in enumerate(augmented_sentences):
            if idx >= limit_per_tag:
                break
            text_data.append(augmented_sentence)
            answers.append(answer)

# Convert augmented data into a new DataFrame
augmented_df2 = pd.DataFrame({
    'Question': text_data,
    'Answer': answers
})
#df=pd.concat([df, augmented_df2], ignore_index=True)
df = augmented_df2


# In[6]:


df['Question'] = df['Question'].apply(removes_specials).str.lower()
df['Answer'] = df['Answer'].apply(removes_specials).str.lower()


# In[7]:


df.shape


# In[8]:


df.to_csv("qa.csv",index=False)


# In[3]:


df=pd.read_csv("qa.csv")


# In[ ]:


# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load model parameters from config
lstm_units = config['model_params']['lstm_units']
dropout_rate = config['model_params']['dropout_rate']
recurrent_dropout = config['model_params']['recurrent_dropout']
embedding_dim = config['model_params']['embedding_dim']
max_epochs = config['model_params']['max_epochs']
batch_size = config['model_params']['batch_size']

# Load training parameters from config
validation_split = config['training_params']['validation_split']
early_stopping_patience = config['training_params']['early_stopping_patience']
monitor_metric = config['training_params']['monitor_metric']

# Prepare the data for training
texts = df['Question'].tolist()
answers = df['Answer'].tolist()

# Tokenize the questions
tokenizer = Tokenizer(oov_token=config['tokenizer_params']['oov_token'])
tokenizer.fit_on_texts(texts)

# Save the tokenizer
joblib.dump(tokenizer, 'tokenizer.joblib')

# Convert texts to sequences and pad them
encoded_texts = tokenizer.texts_to_sequences(texts)
max_len = max([len(x) for x in encoded_texts])

# Save the max_len for future use
joblib.dump(max_len, 'max_len.joblib')

padded_texts = pad_sequences(encoded_texts, maxlen=max_len, padding=config['tokenizer_params']['padding_type'])

# Encode the answers using LabelEncoder
le = LabelEncoder()
encoded_answers = le.fit_transform(answers)

# Save the LabelEncoder for future use
joblib.dump(le, 'label_encoder.joblib')

# Get the number of unique answers
num_answers = len(le.classes_)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_texts, encoded_answers, 
                                                  test_size=validation_split, random_state=42)

# Define model architecture using Bidirectional LSTM
input_layer = Input(shape=(max_len,))
embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim)(input_layer)  # Remove input_length
bilstm_layer = Bidirectional(LSTM(lstm_units, return_sequences=False, recurrent_dropout=recurrent_dropout))(embedding_layer)
dropout_layer = Dropout(dropout_rate)(bilstm_layer)
output_layer = Dense(num_answers, activation='softmax')(dropout_layer)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping and model checkpointing
early_stopping = EarlyStopping(monitor=monitor_metric, patience=early_stopping_patience, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor=monitor_metric, mode='min')

# Train the model with explicit validation data
history = model.fit(X_train, y_train, 
                    epochs=max_epochs, 
                    batch_size=batch_size, 
                    validation_data=(X_val, y_val), 
                    callbacks=[early_stopping, model_checkpoint])

# Save the final trained model
model.save('chatbot_bilstm_model.keras')

# Plot the accuracy and loss metrics
plt.figure(figsize=(12, 4))

# Plot training accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Print final training accuracy/loss
print("Final training accuracy:", history.history['accuracy'][-1])
print("Final validation accuracy:", history.history['val_accuracy'][-1])
print("Final training loss:", history.history['loss'][-1])
print("Final validation loss:", history.history['val_loss'][-1])


# In[ ]:


from spellchecker import SpellChecker
spell = SpellChecker()

def correct_spelling(text):
    words = text.split()
    corrected_words = [spell.correction(word) if spell.correction(word) else word for word in words]
    return ' '.join(corrected_words)
# Load the trained model
model = load_model('chatbot_bilstm_model.h5')

# Compile the model to avoid the warning (optional)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load max_len

max_len = joblib.load("max_len.joblib")
    
# Load the tokenizer

tokenizer = joblib.load('tokenizer.joblib')

# Load the LabelEncoder for answers

le = joblib.load("label_encoder.joblib")


# Function to predict the answer to a user question
def get_answer(question):
    # Tokenize and pad the user input
    encoded_input = tokenizer.texts_to_sequences([question])
    padded_input = pad_sequences(encoded_input, maxlen=max_len, padding='post')

    # Predict the answer
    answer_prob = model.predict(padded_input)
    answer_idx = np.argmax(answer_prob, axis=-1)[0]  # Get the index of the predicted answer
    predicted_answer = le.inverse_transform([answer_idx])[0]  # Decode to get the actual answer

    return predicted_answer

# Example interaction loop
print('Welcome to the chatbot! Type "quit" to exit.')
while True:
    user_question = input('You: ').lower().strip()  # Get user input
    
    if user_question == 'quit':
        break
        
    user_question = correct_spelling(remove_punctuations(user_question))
    print(user_question)
    # Get the predicted answer
    predicted_answer = get_answer(user_question)
    print("Chatbot:", predicted_answer)


# In[ ]:


import nlpaug.augmenter.word as naw

# Initialize the synonym augmenter
aug = naw.SynonymAug(aug_p=0.1)  # Change the probability as needed

# Original question
original_question = "Give me the name of largest planet in solar system"

# Generate augmented questions
augmented_questions = [aug.augment(original_question) for _ in range(100)]
print(augmented_questions)

