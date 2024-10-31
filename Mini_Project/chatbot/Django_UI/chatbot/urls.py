from django.urls import path
from . import views

"""urlpatterns = [
    path('train/', views.train_chatbot, name='train_chatbot'),
    path('test/', views.test_chatbot, name='test_chatbot'),
]
from django.urls import path"""
from .views import train_chatbot, test_chatbot, index, download_model_view

urlpatterns = [
    path('', index, name='index'),  # Route for the index page
    path('train/', train_chatbot, name='train_chatbot'),
    path('test/', test_chatbot, name='test_chatbot'),
    path('download_model/', download_model_view, name='download_model')
]
