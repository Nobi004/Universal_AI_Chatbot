# ai_chatbot/urls.py

from django.urls import path
from . import views

app_name = 'ai_chatbot'

urlpatterns = [
    # Session management
    path('', views.ChatSessionListView.as_view(), name='session_list'),
    path('new/', views.new_chat, name='new_chat'),
    path('session/create/', views.create_chat_session, name='create_session'),
    path('session/<uuid:pk>/', views.ChatView.as_view(), name='chat'),
    path('session/<uuid:session_id>/delete/', views.delete_session, name='delete_session'),
    path('session/<uuid:session_id>/archive/', views.archive_session, name='archive_session'),
    
    # AJAX endpoints
    path('session/<uuid:session_id>/send/', views.send_message_ajax, name='send_message'),
    
    # Configuration
    path('config/', views.LLMConfigUpdateView.as_view(), name='config'),
]