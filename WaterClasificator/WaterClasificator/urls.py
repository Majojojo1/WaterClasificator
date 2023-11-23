from django.contrib import admin
from django.urls import path
from leafclassifier.views import upload_file, predict_file, index

urlpatterns = [
    path('admin/', admin.site.urls),
    path('upload/', upload_file, name='upload_file'),
    path('predict/<int:file_id>/', predict_file, name='predict'),
    path('', index, name='index'),  # Ruta para la vista index
]
