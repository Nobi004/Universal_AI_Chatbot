from django.urls import path
from . import views
from django.contrib import admin
from django.contrib.auth import views as auth_views
from ai_chatbot.views import home_view
from .views import RegisterView


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
    path('models/add-huggingface/', views.add_huggingface_model, name='add_hf_model'),

    path('', home_view, name='home'),  # Home page
    path('admin/', admin.site.urls),

    # Add authentication URLs
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('register/', RegisterView.as_view(), name='register'),
]