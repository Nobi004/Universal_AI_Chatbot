# ai_chatbot/serializers.py

from rest_framework import serializers
from django.contrib.auth.models import User
from .models import (
    ChatSession, ChatMessage, LLMModel, LLMProvider,
    LLMConfig, UserAPIKey, UsageLog
)


class LLMProviderSerializer(serializers.ModelSerializer):
    """Serializer for LLM providers."""
    models_count = serializers.IntegerField(source='models.count', read_only=True)
    
    class Meta:
        model = LLMProvider
        fields = ['id', 'name', 'provider_type', 'is_active', 'models_count']
        read_only_fields = ['id', 'models_count']


class LLMModelSerializer(serializers.ModelSerializer):
    """Serializer for LLM models."""
    provider_name = serializers.CharField(source='provider.name', read_only=True)
    provider_type = serializers.CharField(source='provider.provider_type', read_only=True)
    
    class Meta:
        model = LLMModel
        fields = [
            'id', 'name', 'display_name', 'description', 
            'provider_name', 'provider_type', 'max_tokens',
            'supports_streaming', 'supports_functions',
            'cost_per_1k_input_tokens', 'cost_per_1k_output_tokens',
            'is_active'
        ]
        read_only_fields = ['id', 'provider_name', 'provider_type']


class ChatMessageSerializer(serializers.ModelSerializer):
    """Serializer for chat messages."""
    timestamp = serializers.DateTimeField(source='created_at', read_only=True)
    
    class Meta:
        model = ChatMessage
        fields = [
            'id', 'role', 'content', 'timestamp', 
            'input_tokens', 'output_tokens', 'response_time_ms',
            'is_error', 'error_message', 'metadata'
        ]
        read_only_fields = [
            'id', 'timestamp', 'input_tokens', 'output_tokens', 
            'response_time_ms', 'is_error', 'error_message', 'metadata'
        ]


class ChatSessionSerializer(serializers.ModelSerializer):
    """Serializer for chat sessions."""
    messages = ChatMessageSerializer(many=True, read_only=True)
    messages_count = serializers.IntegerField(source='messages.count', read_only=True)
    model_name = serializers.CharField(source='model.display_name', read_only=True)
    last_message = serializers.SerializerMethodField()
    
    class Meta:
        model = ChatSession
        fields = [
            'id', 'title', 'model', 'model_name', 'messages',
            'messages_count', 'last_message', 'system_prompt',
            'temperature', 'max_tokens', 'total_tokens_used',
            'total_cost', 'is_archived', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'messages', 'messages_count', 'last_message',
            'total_tokens_used', 'total_cost', 'created_at', 'updated_at'
        ]
    
    def get_last_message(self, obj):
        """Get the last message preview."""
        last_msg = obj.messages.last()
        if last_msg:
            return {
                'role': last_msg.role,
                'content': last_msg.content[:100] + '...' if len(last_msg.content) > 100 else last_msg.content,
                'timestamp': last_msg.created_at
            }
        return None


class ChatSessionListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for session list views."""
    model_name = serializers.CharField(source='model.display_name', read_only=True)
    messages_count = serializers.IntegerField(source='messages.count', read_only=True)
    last_message = serializers.SerializerMethodField()
    
    class Meta:
        model = ChatSession
        fields = [
            'id', 'title', 'model_name', 'messages_count',
            'last_message', 'is_archived', 'created_at', 'updated_at'
        ]
        read_only_fields = fields
    
    def get_last_message(self, obj):
        """Get the last message preview."""
        last_msg = obj.messages.last()
        if last_msg:
            return {
                'role': last_msg.role,
                'content': last_msg.content[:50] + '...' if len(last_msg.content) > 50 else last_msg.content,
                'timestamp': last_msg.created_at
            }
        return None


class CreateChatSessionSerializer(serializers.ModelSerializer):
    """Serializer for creating new chat sessions."""
    
    class Meta:
        model = ChatSession
        fields = ['title', 'model', 'system_prompt', 'temperature', 'max_tokens']
        
    def validate_model(self, value):
        """Ensure model is active."""
        if value and not value.is_active:
            raise serializers.ValidationError("Selected model is not active.")
        return value


class SendMessageSerializer(serializers.Serializer):
    """Serializer for sending messages."""
    session_id = serializers.UUIDField(required=False, allow_null=True)
    content = serializers.CharField(min_length=1, max_length=32000)
    model_id = serializers.PrimaryKeyRelatedField(
        queryset=LLMModel.objects.filter(is_active=True),
        required=False,
        allow_null=True
    )
    temperature = serializers.FloatField(min_value=0, max_value=2, required=False)
    max_tokens = serializers.IntegerField(min_value=1, max_value=32000, required=False)
    stream = serializers.BooleanField(default=False)
    
    def validate_session_id(self, value):
        """Validate session belongs to the user."""
        if value:
            request = self.context.get('request')
            try:
                session = ChatSession.objects.get(id=value, user=request.user)
            except ChatSession.DoesNotExist:
                raise serializers.ValidationError("Session not found or access denied.")
        return value


class LLMConfigSerializer(serializers.ModelSerializer):
    """Serializer for LLM configuration."""
    default_model_name = serializers.CharField(source='default_model.display_name', read_only=True)
    
    class Meta:
        model = LLMConfig
        fields = [
            'default_model', 'default_model_name', 'temperature',
            'max_tokens', 'system_prompt', 'rate_limit_per_day',
            'enable_streaming'
        ]
        
    def validate_default_model(self, value):
        """Ensure model is active."""
        if value and not value.is_active:
            raise serializers.ValidationError("Selected model is not active.")
        return value


class UserAPIKeySerializer(serializers.ModelSerializer):
    """Serializer for user API keys."""
    provider_name = serializers.CharField(source='provider.name', read_only=True)
    masked_key = serializers.SerializerMethodField()
    
    class Meta:
        model = UserAPIKey
        fields = ['id', 'provider', 'provider_name', 'masked_key', 'is_active', 'created_at']
        read_only_fields = ['id', 'masked_key', 'created_at']
        
    def get_masked_key(self, obj):
        """Return masked API key for security."""
        if obj.api_key:
            return f"{obj.api_key[:8]}...{obj.api_key[-4:]}"
        return None
    
    def create(self, validated_data):
        """Create user API key."""
        validated_data['user'] = self.context['request'].user
        return super().create(validated_data)


class UsageStatsSerializer(serializers.Serializer):
    """Serializer for usage statistics."""
    date = serializers.DateField()
    message_count = serializers.IntegerField()
    total_tokens = serializers.IntegerField()
    total_cost = serializers.DecimalField(max_digits=10, decimal_places=6)
    remaining_messages = serializers.IntegerField()
    
    
class UserSerializer(serializers.ModelSerializer):
    """Basic user serializer."""
    
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']
        read_only_fields = fields