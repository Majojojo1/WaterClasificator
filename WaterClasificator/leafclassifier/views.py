import pandas as pd
from django.shortcuts import render, redirect
from .forms import FileUploadForm, FilePredictForm
from .models import UploadedFile
from .predictor import predict_with_model
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from django.shortcuts import render

def index(request):
    return render(request, 'index.html')


def upload_file(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save()
            return redirect('predict', file_id=uploaded_file.id)
    else:
        form = FileUploadForm()

    return render(request, 'upload_file.html', {'form': form})

def predict_file(request, file_id):
    uploaded_file = UploadedFile.objects.get(id=file_id)

    if request.method == 'POST':
        predict_form = FilePredictForm(request.POST, request.FILES)
        if predict_form.is_valid():
            file_path = handle_uploaded_file(predict_form.cleaned_data['file'])
            predictions = predict_from_file(file_path)
            return render(request, 'result.html', {'predictions': predictions})
    else:
        predict_form = FilePredictForm()

    return render(request, 'leafclassifier/predict_file.html', {'predict_form': predict_form, 'uploaded_file': uploaded_file})

def handle_uploaded_file(file):
    with open('temp_file', 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return 'temp_file'

def read_file(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError('Formato de archivo no compatible')

def predict_with_model(features, target):

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    clfHGBC = HistGradientBoostingClassifier(max_iter=100)
    clfHGBC.fit(X_train, y_train)

    predictions = clfHGBC.predict(features)
    return predictions

def predict_from_file(file_path):
    data = read_file(file_path)
    features = data.drop(columns=['target_column'])
    predictions = predict_with_model(features)
    return predictions
