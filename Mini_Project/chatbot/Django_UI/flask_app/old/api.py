import os
import shutil
import yaml
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from spellchecker import SpellChecker
from zipfile import ZipFile
from nlp_lib import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MODEL_FOLDER'] = './models'
app.config['MODEL_CONFIG_FOLDER'] = './model_config'
app.config['GRAPH_FOLDER'] = './graph'

spell = SpellChecker()

def correct_spelling(text):
    words = text.split()
    corrected_words = [spell.correction(word) if spell.correction(word) else word for word in words]
    return ' '.join(corrected_words)

# Load config.yaml for default parameters
def load_config(config_path='model_config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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
    if not os.path.exists(app.config['GRAPH_FOLDER']):
        os.makedirs(app.config['GRAPH_FOLDER'])
    plt.savefig(f"{app.config['GRAPH_FOLDER']}/training_plot.png")
    plt.show()

# Function to load CSV and train the model
def train_model(file_path, user_params):
    config = load_config()  # Load default parameters from YAML

    # Override default parameters with user-supplied params
    params = {**config['model_params'], **config['training_params'], **user_params}

    # Load dataset
    df = pd.read_csv(file_path)
    #df.drop(columns="Intent", inplace=True)
    df['Question'] = df['Question'].str.lower()
    df['Question'] = df['Question'].apply(removes_specials)
    df['Answer'] = df['Answer'].str.lower()
    df['Answer'] = df['Answer'].apply(removes_specials))

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
    bilstm_layer = Bidirectional(LSTM(params['lstm_units'], return_sequences=False, recurrent_dropout=params['recurrent_dropout']))(embedding_layer)
    dropout_layer = Dropout(params['dropout_rate'])(bilstm_layer)
    output_layer = Dense(len(le.classes_), activation='softmax')(dropout_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    early_stopping = EarlyStopping(monitor=params['monitor_metric'], patience=params['early_stopping_patience'], restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(f"{app.config['MODEL_FOLDER']}/best_model.keras", save_best_only=True, monitor=params['monitor_metric'], mode='min')

    history = model.fit(X_train, y_train, epochs=params['max_epochs'], batch_size=params['batch_size'],
                        validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

    # Save the final trained model and other files
    model.save(f"{app.config['MODEL_FOLDER']}/chatbot_bilstm_model.keras")
    joblib.dump(tokenizer, f"{app.config['MODEL_FOLDER']}/tokenizer.joblib")
    joblib.dump(le, f"{app.config['MODEL_FOLDER']}/label_encoder.joblib")
    joblib.dump(max_len, f"{app.config['MODEL_FOLDER']}/max_len.joblib")

    # Plot the training history
    plot_training_history(history)

    # Prepare for download by zipping the model files
    zip_file = f"{app.config['MODEL_FOLDER']}/model_files.zip"
    with ZipFile(zip_file, 'w') as zipf:
        zipf.write(f"{app.config['MODEL_FOLDER']}/chatbot_bilstm_model.keras", arcname="chatbot_bilstm_model.keras")
        zipf.write(f"{app.config['MODEL_FOLDER']}/tokenizer.joblib", arcname="tokenizer.joblib")
        zipf.write(f"{app.config['MODEL_FOLDER']}/label_encoder.joblib", arcname="label_encoder.joblib")
        zipf.write(f"{app.config['MODEL_FOLDER']}/max_len.joblib", arcname="max_len.joblib")
        zipf.write(f"{app.config['GRAPH_FOLDER']}/training_plot.png", arcname="training_plot.png")

    # Return training results and the zip file path
    final_train_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    return {
        'train_accuracy': final_train_accuracy,
        'val_accuracy': final_val_accuracy,
        'train_loss': final_train_loss,
        'val_loss': final_val_loss,
        'zip_file': zip_file
    }

# Route to upload CSV, train the model, and allow users to set parameters
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract training parameters from request or use defaults from config.yaml
        """user_params = {
            'batch_size': request.form.get('batch_size', None),
            'epochs': request.form.get('epochs', None),
            'lstm_units': request.form.get('lstm_units', None),
            'embedding_dim': request.form.get('embedding_dim', None),
            'dropout_rate': request.form.get('dropout_rate', None),
            'recurrent_dropout': request.form.get('recurrent_dropout', None),
            'validation_split': request.form.get('validation_split', None),
            'monitor_metric': request.form.get('monitor_metric', None),
            'early_stopping_patience': request.form.get('early_stopping_patience', None),
        }"""
        # Ensure that parameters are integers or floats as necessary
        try:
            user_params['batch_size'] = int(user_params['batch_size']) if user_params['batch_size'] else config['model_params']['batch_size']
            user_params['epochs'] = int(user_params['epochs']) if user_params['epochs'] else config['model_params']['max_epochs']
            user_params['lstm_units'] = int(user_params['lstm_units']) if user_params['lstm_units'] else config['model_params']['lstm_units']
            user_params['embedding_dim'] = int(user_params['embedding_dim']) if user_params['embedding_dim'] else config['model_params']['embedding_dim']
            user_params['dropout_rate'] = float(user_params['dropout_rate']) if user_params['dropout_rate'] else config['model_params']['dropout_rate']
            user_params['recurrent_dropout'] = float(user_params['recurrent_dropout']) if user_params['recurrent_dropout'] else config['model_params']['recurrent_dropout']
            user_params['validation_split'] = float(user_params['validation_split']) if user_params['validation_split'] else config['training_params']['validation_split']
            user_params['early_stopping_patience'] = int(user_params['early_stopping_patience']) if user_params['early_stopping_patience'] else config['training_params']['early_stopping_patience']
        except ValueError as e:
            return jsonify({"error": f"Invalid parameter value: {str(e)}"}), 400


        # Train model with the uploaded CSV and provided parameters
        model_data = train_model(file_path, user_params)

        return jsonify({
            "train_accuracy": model_data['train_accuracy'],
            "val_accuracy": model_data['val_accuracy'],
            "train_loss": model_data['train_loss'],
            "val_loss": model_data['val_loss'],
            "zip_file": model_data['zip_file']
        })

# Route to download the trained model zip
@app.route('/download_model', methods=['GET'])
def download_model():
    zip_file_path = request.args.get('zip_file')
    return send_file(zip_file_path, as_attachment=True)

# Route to test chatbot prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    question = correct_spelling(remove_punctuations((data['question']).lower()))
    model, tokenizer, le, max_len = load_trained_model()

    # Tokenize and pad the user input
    encoded_input = tokenizer.texts_to_sequences([question])
    padded_input = pad_sequences(encoded_input, maxlen=max_len, padding='post')

    # Predict the answer
    answer_prob = model.predict(padded_input)
    answer_idx = np.argmax(answer_prob, axis=-1)[0]  # Get the index of the predicted answer
    predicted_answer = le.inverse_transform([answer_idx])[0]  # Decode to get the actual answer

    return jsonify({"answer": predicted_answer})

if __name__ == '__main__':
    app.run(debug=True)
