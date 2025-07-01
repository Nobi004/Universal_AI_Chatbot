# ai_chatbot/api/urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    ChatSessionViewSet, ChatMessageViewSet, SendMessageView,
    LLMModelViewSet, LLMConfigView, UserAPIKeyViewSet,
    UsageStatsView, ProfileView
)

router = DefaultRouter()
router.register(r'sessions', ChatSessionViewSet, basename='session')
router.register(r'models', LLMModelViewSet, basename='model')
router.register(r'api-keys', UserAPIKeyViewSet, basename='api-key')

app_name = 'ai_chatbot_api'

urlpatterns = [
    path('', include(router.urls)),
    path('send/', SendMessageView.as_view(), name='send-message'),
    path('config/', LLMConfigView.as_view(), name='config'),
    path('usage/', UsageStatsView.as_view(), name='usage-stats'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('sessions/<uuid:session_id>/messages/', 
         ChatMessageViewSet.as_view({'get': 'list'}), 
         name='session-messages'),
]