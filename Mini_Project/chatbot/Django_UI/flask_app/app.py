import os
import shutil
import yaml
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from spellchecker import SpellChecker
from zipfile import ZipFile
from nlp_lib import *
from werkzeug.utils import secure_filename
from flask import send_from_directory
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MODEL_FOLDER'] = './models'
app.config['MODEL_CONFIG_FOLDER'] = './model_config'
app.config['GRAPH_FOLDER'] = './graph'
app.config['ZIP_FOLDER'] = './zips'

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
    plt.close()  # Close the figure to prevent display

# Function to load CSV and train the model
def train_model(file_path, user_params):
    config = load_config()  # Load default parameters from YAML

    # Override default parameters with user-supplied params
    params = {**config['model_params'], **config['training_params'],**config['tokenizer_params'],**config['optimizer_params'], **user_params}

    # Load dataset
    df = pd.read_csv(file_path)
    df['Question'] = df['Question'].str.lower().apply(removes_specials)
    df['Answer'] = df['Answer'].str.lower().apply(removes_specials)

    # Tokenize the questions
    tokenizer = Tokenizer(oov_token=params.get('oov_token', '<OOV>'))
    tokenizer.fit_on_texts(df['Question'].tolist())

    # Convert texts to sequences and pad them
    encoded_texts = tokenizer.texts_to_sequences(df['Question'].tolist())
    max_len = max([len(x) for x in encoded_texts])
    padded_texts = pad_sequences(encoded_texts, maxlen=max_len, padding=params.get('padding_type', 'post'))

    # Encode the answers
    le = LabelEncoder()
    encoded_answers = le.fit_transform(df['Answer'].tolist())
    # Get the number of unique answers
    num_answers = len(le.classes_)

    # Calculate class weights for imbalanced classes
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(encoded_answers), y=encoded_answers)
    class_weights = dict(enumerate(class_weights))

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(padded_texts, encoded_answers, 
                                                      test_size=params.get('validation_split',0.2), random_state=42)

    # Define model architecture
    input_layer = Input(shape=(max_len,))
    embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=params.get('embedding_dim', 128))(input_layer)
    bilstm_layer = Bidirectional(LSTM(params.get('lstm_units', 128), return_sequences=False, recurrent_dropout=params.get('recurrent_dropout', 0.3)))(embedding_layer)
    dropout_layer = Dropout(params.get('dropout_rate', 0.3))(bilstm_layer)
    output_layer = Dense(len(le.classes_), activation='softmax')(dropout_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    # Defining Optimizer
    #learning_rate = #config['optimizer_params']['learning_rate']  # Retrieve learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.get('learning_rate',0.0001))
    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Train the model
    early_stopping = EarlyStopping(monitor=params.get('monitor_metric', 'val_loss'), patience=params.get('early_stopping_patience', 5), restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(f"{app.config['MODEL_FOLDER']}/best_model.keras", save_best_only=True, monitor=params.get('monitor_metric', 'val_loss'), mode='min')

    lr_scheduler = ReduceLROnPlateau(monitor=config['training_params']['monitor_metric'], factor=0.2, patience=3, min_lr=params.get('min_learning_rate',0.0001))

    
    history = model.fit(X_train, y_train, epochs=params.get('max_epochs', 50), batch_size=params.get('batch_size', 16),
                        validation_data=(X_val, y_val),class_weight=class_weights,  # Apply class weights
                        callbacks=[early_stopping, model_checkpoint, lr_scheduler])

    # Save the final trained model and other files
    model.save(f"{app.config['MODEL_FOLDER']}/chatbot_bilstm_model.keras")
    joblib.dump(tokenizer, f"{app.config['MODEL_FOLDER']}/tokenizer.joblib")
    joblib.dump(le, f"{app.config['MODEL_FOLDER']}/label_encoder.joblib")
    joblib.dump(max_len, f"{app.config['MODEL_FOLDER']}/max_len.joblib")

    # Plot the training history
    plot_training_history(history)

    # Prepare for download by zipping the model files
    zip_file = f"{app.config['ZIP_FOLDER']}/model_files.zip"
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
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract training parameters from request or use defaults from config.yaml
        user_params = {
            'batch_size': int(request.form.get('batch_size', 16)),
            'max_epochs': int(request.form.get('max_epochs', 50)),
            'lstm_units': int(request.form.get('lstm_units', 128)),
            'embedding_dim': int(request.form.get('embedding_dim', 128)),
            'dropout_rate': float(request.form.get('dropout_rate', 0.3)),
            'recurrent_dropout': float(request.form.get('recurrent_dropout', 0.3)),
            'validation_split': float(request.form.get('validation_split', 0.2)),
            'monitor_metric': request.form.get('monitor_metric', 'val_loss'),
            'early_stopping_patience': int(request.form.get('early_stopping_patience', 5)),
            'oov_token': request.form.get('oov_token', '<OOV>'),
            'padding_type': request.form.get('padding_type', 'post'),
            'learning_rate': request.form.get('learning_rate', 0.0001),
            'min_learning_rate': request.form.get('learning_rate', 0.0001),
        }

        # Train model with the uploaded CSV and provided parameters
        model_data = train_model(file_path, user_params)

        return jsonify({
            "train_accuracy": model_data['train_accuracy'],
            "val_accuracy": model_data['val_accuracy'],
            "train_loss": model_data['train_loss'],
            "val_loss": model_data['val_loss'],
            "zip_file": model_data['zip_file']
        })

# Route to download a zip of the models folder
@app.route('/download_model', methods=['GET'])
def download_model():
    zip_file_path = f"{app.config['ZIP_FOLDER']}/model_files.zip"
    return send_file(zip_file_path, as_attachment=True)
    
@app.route('/graph/<path:filename>', methods=['GET'])
def send_graph(filename):
    return send_from_directory(app.config['GRAPH_FOLDER'], filename)

def load_trained_model():
    model = load_model(f"{app.config['MODEL_FOLDER']}/chatbot_bilstm_model.keras")
    tokenizer = joblib.load(f"{app.config['MODEL_FOLDER']}/tokenizer.joblib")
    label_encoder = joblib.load(f"{app.config['MODEL_FOLDER']}/label_encoder.joblib")
    max_len = joblib.load(f"{app.config['MODEL_FOLDER']}/max_len.joblib")
    return model, tokenizer, label_encoder, max_len

# Route to test chatbot prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    question = data.get('question', '').strip()

    if not question:
        return jsonify({"error": "Question is missing or empty."}), 400

    question = correct_spelling(remove_punctuations(question.lower()))
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
    # Ensure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    os.makedirs(app.config['GRAPH_FOLDER'], exist_ok=True)
    
    app.run(debug=True)
