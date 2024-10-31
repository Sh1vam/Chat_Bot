import yaml
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import joblib
from nlp_lib import *
# Load config.yaml for default parameters
def load_config(config_path='model_config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
import matplotlib.pyplot as plt

# Plot accuracy and loss after training
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot to a file
    plt.savefig('graph/training_plot.png')
    plt.show()

# Function to load CSV and train the model using your provided code
def train_model(file_path, user_params):
    config = load_config()  # Load default parameters from YAML
    
    # Override default parameters with user-supplied params
    params = {**config['model_params'], **config['training_params'], **user_params}

    # Load dataset
    df = pd.read_csv(file_path)
    df.drop(columns="Intent", inplace=True)
    df['Question'] = df['Question'].str.lower()
    df['Question'] = df['Question'].apply(removes_specials)
    df['Answer'] = df['Answer'].str.lower()
    df['Answer'] = df['Answer'].apply(removes_specials)

    # Tokenize the questions
    tokenizer = Tokenizer(oov_token=params['tokenizer_params']['oov_token'])
    tokenizer.fit_on_texts(df['Question'].tolist())

    # Convert texts to sequences and pad them
    encoded_texts = tokenizer.texts_to_sequences(df['Question'].tolist())
    max_len = max([len(x) for x in encoded_texts])
    padded_texts = pad_sequences(encoded_texts, maxlen=max_len, padding='post')

    # Encode the answers
    le = LabelEncoder()
    encoded_answers = le.fit_transform(df['Answer'].tolist())

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(padded_texts, encoded_answers, 
                                                      test_size=params['validation_split'], random_state=42)

    # Define model architecture
    input_layer = Input(shape=(max_len,))
    embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=params['embedding_dim'])(input_layer)
    bilstm_layer = Bidirectional(LSTM(params['lstm_units'], return_sequences=False, 
                                      recurrent_dropout=params['recurrent_dropout']))(embedding_layer)
    dropout_layer = Dropout(params['dropout_rate'])(bilstm_layer)
    output_layer = Dense(len(le.classes_), activation='softmax')(dropout_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    early_stopping = EarlyStopping(monitor=params['monitor_metric'], patience=params['early_stopping_patience'], 
                                   restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('models/best_model.keras', save_best_only=True, monitor=params['monitor_metric'], mode='min')

    history = model.fit(X_train, y_train, epochs=params['max_epochs'], batch_size=params['batch_size'],
              validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

    # Save the final trained model and other files
    model.save('models/chatbot_bilstm_model.keras')
    joblib.dump(tokenizer, 'models/tokenizer.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    joblib.dump(max_len, 'models/max_len.joblib')
    final_train_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    plot_training_history(history)
    print(f"Final Training Accuracy: {final_train_accuracy}")
    print(f"Final Validation Accuracy: {final_val_accuracy}")
    print(f"Final Training Loss: {final_train_loss}")
    print(f"Final Validation Loss: {final_val_loss}")
    return {
        'train_accuracy': final_train_accuracy,
        'val_accuracy': final_val_accuracy,
        'train_loss': final_train_loss,
        'val_loss': final_val_loss,
        'model_path': ['models/chatbot_bilstm_model.keras','models/label_encoder.joblib','models/tokenizer.joblib','models/max_len.joblib']
        'plot_path': 'graph/training_plot.png'  # Add plot path to the return data
    }


# Function to load and use the trained model for prediction
def load_trained_model():
    model = load_model('models/chatbot_bilstm_model.keras')
    tokenizer = joblib.load('models/tokenizer.joblib')
    le = joblib.load('models/label_encoder.joblib')
    max_len = joblib.load('models/max_len.joblib')

    return model, tokenizer, le, max_len
