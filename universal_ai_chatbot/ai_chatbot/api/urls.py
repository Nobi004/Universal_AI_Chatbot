from django.urls import path, include
from rest_framework.routers import DefaultRouter

# For now, just create an empty router
router = DefaultRouter()

app_name = 'ai_chatbot_api'

urlpatterns = [
    path('', include(router.urls)),
]
