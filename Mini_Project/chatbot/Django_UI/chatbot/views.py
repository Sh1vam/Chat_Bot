from django.shortcuts import render
import requests
import os
from django.http import HttpResponse

# Define the path where zip files are stored
ZIP_FOLDER = './models'



def download_model_view(request):
    # Construct the URL for the Flask API download endpoint
    zip_file_url = 'http://127.0.0.1:5000/download_model?zip_file=model_files.zip'

    # Use requests to fetch the zip file from Flask
    try:
        response = requests.get(zip_file_url)

        if response.status_code == 200:
            # Create the HTTP response with the content of the zip file
            http_response = HttpResponse(response.content)
            http_response['Content-Disposition'] = 'attachment; filename="model_files.zip"'
            return http_response
        else:
            return render(request, 'error.html', {'message': 'Failed to download model files.'})

    except requests.exceptions.RequestException as e:
        return render(request, 'error.html', {'message': f'Error downloading model files: {str(e)}'})

# Function to delete the old zip file if it exists
def delete_old_zip(zip_filename):
    zip_path = os.path.join(ZIP_FOLDER, zip_filename)
    if os.path.exists(zip_path):
        os.remove(zip_path)

# View to handle model training
def train_chatbot(request):
    if request.method == 'POST':
        # File and parameters sent to Flask API
        file = request.FILES['csv_file']
        data = {
            'batch_size': int(request.POST.get('batch_size', 16)),
            'max_epochs': int(request.POST.get('max_epochs', 50)),
            'lstm_units': int(request.POST.get('lstm_units', 128)),
            'dropout_rate': float(request.POST.get('dropout_rate', 0.3)),
            'recurrent_dropout': float(request.POST.get('recurrent_dropout', 0.3)),
            'embedding_dim': int(request.POST.get('embedding_dim', 128)),
            'validation_split': float(request.POST.get('validation_split', 0.2)),
            'early_stopping_patience': int(request.POST.get('early_stopping_patience', 5)),
            'monitor_metric': request.POST.get('monitor_metric', 'val_loss'),
            'oov_token': request.POST.get('oov_token', '<OOV>'),
            'padding_type': request.POST.get('padding_type', 'post'),
            'learning_rate': float(request.POST.get('learning_rate', 0.0001)),
            'min_learning_rate': float(request.POST.get('min_learning_rate', 0.0001)),
        }
        files = {'file': file}

        # Delete old zip file if it exists
        delete_old_zip('model_files.zip')  # Corrected the filename

        # Send POST request to Flask API
        response = requests.post('http://127.0.0.1:5000/upload', data=data, files=files)
        
        if response.status_code == 200:
            response_data = response.json()
            return render(request, 'train.html', {'response': response_data})
        else:
            return render(request, 'error.html', {'message': 'Failed to train the model.'})

    return render(request, 'train.html')

# View to handle chatbot testing
def test_chatbot(request):
    if request.method == 'POST':
        question = request.POST['question']
        response = requests.post('http://127.0.0.1:5000/predict', json={'question': question})
        response_data = response.json()

        return render(request, 'test.html', {'response': response_data})

    return render(request, 'test.html')

# View for the index page
def index(request):
    return render(request, 'index.html')
