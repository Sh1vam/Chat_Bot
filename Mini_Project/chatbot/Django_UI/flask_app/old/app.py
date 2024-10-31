from flask import Flask, request, jsonify, send_file
from train_chatbot import *
import os
import pandas as pd
from nlp_lib import *
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MODEL_FOLDER'] = './models'
app.config['MODEL_CONFIG_FOLDER'] = './model_config'
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
        user_params = {
            'batch_size': request.form.get('batch_size', None),
            'epochs': request.form.get('epochs', None),
            'lstm_units': request.form.get('lstm_units', None),
            'embedding_dim': request.form.get('embedding_dim', None),
            'dropout_rate': request.form.get('dropout_rate', None),
            'recurrent_dropout': request.form.get('recurrent_dropout', None),
            'validation_split': request.form.get('validation_split', None),
            'monitor_metric': request.form.get('monitor_metric', None),
            'early_stopping_patience': request.form.get('early_stopping_patience', None),
        }

        # Train model with the uploaded CSV and provided parameters
        model_path = train_model(file_path, user_params)
        return jsonify({"model_path": model_path})

# Route to download the trained model
@app.route('/download_model', methods=['GET'])
def download_model():
    model_name = request.args.get('model_name')
    return send_file(os.path.join(app.config['MODEL_FOLDER'], model_name), as_attachment=True)

from spellchecker import SpellChecker
spell = SpellChecker()
def correct_spelling(text):
    words = text.split()
    corrected_words = [spell.correction(word) if spell.correction(word) else word for word in words]
    return ' '.join(corrected_words)
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
