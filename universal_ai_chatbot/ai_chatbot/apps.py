# ai_chatbot/apps.py

from django.apps import AppConfig


class AiChatbotConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ai_chatbot'
    verbose_name = 'AI Chatbot'
    
    def ready(self):
        """Initialize app when Django starts."""
        # Import signal handlers if any
        try:
            from . import signals
        except ImportError:
            pass