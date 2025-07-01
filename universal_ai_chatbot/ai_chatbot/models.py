# ai_chatbot/models.py

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone
import uuid


class LLMProvider(models.Model):
    """Represents an LLM provider like OpenAI, Anthropic, etc."""
    PROVIDER_CHOICES = [
        ('openai', 'OpenAI'),
        ('anthropic', 'Anthropic'),
        ('mistral', 'Mistral'),
        ('huggingface', 'HuggingFace'),
        ('cohere', 'Cohere'),
        ('custom', 'Custom'),
    ]
    
    name = models.CharField(max_length=50, unique=True)
    provider_type = models.CharField(max_length=20, choices=PROVIDER_CHOICES)
    base_url = models.URLField(help_text="Base API URL for the provider")
    api_key = models.CharField(max_length=255, help_text="API key for authentication")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
        verbose_name = "LLM Provider"
        verbose_name_plural = "LLM Providers"
    
    def __str__(self):
        return f"{self.name} ({self.get_provider_type_display()})"


class LLMModel(models.Model):
    """Represents a specific model from an LLM provider."""
    provider = models.ForeignKey(LLMProvider, on_delete=models.CASCADE, related_name='models')
    name = models.CharField(max_length=100, help_text="Model identifier (e.g., gpt-3.5-turbo)")
    display_name = models.CharField(max_length=100, help_text="User-friendly name")
    description = models.TextField(blank=True)
    max_tokens = models.IntegerField(default=4096, validators=[MinValueValidator(1)])
    supports_streaming = models.BooleanField(default=True)
    supports_functions = models.BooleanField(default=False)
    cost_per_1k_input_tokens = models.DecimalField(
        max_digits=10, decimal_places=6, null=True, blank=True,
        help_text="Cost per 1000 input tokens in USD"
    )
    cost_per_1k_output_tokens = models.DecimalField(
        max_digits=10, decimal_places=6, null=True, blank=True,
        help_text="Cost per 1000 output tokens in USD"
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['provider', 'name']
        unique_together = ['provider', 'name']
        verbose_name = "LLM Model"
        verbose_name_plural = "LLM Models"
    
    def __str__(self):
        return f"{self.display_name} ({self.provider.name})"


class LLMConfig(models.Model):
    """Global configuration for LLM behavior."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='llm_config')
    default_model = models.ForeignKey(LLMModel, on_delete=models.SET_NULL, null=True, blank=True)
    temperature = models.FloatField(
        default=0.7,
        validators=[MinValueValidator(0.0), MaxValueValidator(2.0)],
        help_text="Controls randomness in responses (0-2)"
    )
    max_tokens = models.IntegerField(
        default=2000,
        validators=[MinValueValidator(1), MaxValueValidator(32000)],
        help_text="Maximum tokens per response"
    )
    system_prompt = models.TextField(
        blank=True,
        help_text="Default system prompt for all conversations"
    )
    rate_limit_per_day = models.IntegerField(
        default=100,
        validators=[MinValueValidator(0)],
        help_text="Maximum messages per day (0 for unlimited)"
    )
    enable_streaming = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "LLM Configuration"
        verbose_name_plural = "LLM Configurations"
    
    def __str__(self):
        return f"Config for {self.user.username}"


class ChatSession(models.Model):
    """Represents a chat conversation session."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_sessions')
    title = models.CharField(max_length=200, blank=True)
    model = models.ForeignKey(LLMModel, on_delete=models.SET_NULL, null=True)
    system_prompt = models.TextField(blank=True)
    temperature = models.FloatField(
        default=0.7,
        validators=[MinValueValidator(0.0), MaxValueValidator(2.0)]
    )
    max_tokens = models.IntegerField(
        default=2000,
        validators=[MinValueValidator(1)]
    )
    total_tokens_used = models.IntegerField(default=0)
    total_cost = models.DecimalField(max_digits=10, decimal_places=6, default=0)
    is_archived = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-updated_at']
        verbose_name = "Chat Session"
        verbose_name_plural = "Chat Sessions"
        indexes = [
            models.Index(fields=['user', '-updated_at']),
            models.Index(fields=['is_archived']),
        ]
    
    def __str__(self):
        return f"{self.title or 'Untitled'} - {self.user.username}"
    
    def save(self, *args, **kwargs):
        # Auto-generate title from first message if not set
        if not self.title and self.messages.exists():
            first_message = self.messages.first()
            self.title = first_message.content[:50] + "..." if len(first_message.content) > 50 else first_message.content
        super().save(*args, **kwargs)
    
    def calculate_cost(self):
        """Calculate total cost based on token usage."""
        if not self.model:
            return 0
        
        input_tokens = sum(msg.input_tokens or 0 for msg in self.messages.all())
        output_tokens = sum(msg.output_tokens or 0 for msg in self.messages.all())
        
        input_cost = (input_tokens / 1000) * (self.model.cost_per_1k_input_tokens or 0)
        output_cost = (output_tokens / 1000) * (self.model.cost_per_1k_output_tokens or 0)
        
        return input_cost + output_cost


class ChatMessage(models.Model):
    """Represents a single message in a chat session."""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
        ('function', 'Function'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)
    input_tokens = models.IntegerField(null=True, blank=True)
    output_tokens = models.IntegerField(null=True, blank=True)
    response_time_ms = models.IntegerField(null=True, blank=True, help_text="Response time in milliseconds")
    is_error = models.BooleanField(default=False)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
        verbose_name = "Chat Message"
        verbose_name_plural = "Chat Messages"
        indexes = [
            models.Index(fields=['session', 'created_at']),
            models.Index(fields=['role']),
        ]
    
    def __str__(self):
        return f"{self.get_role_display()}: {self.content[:50]}..."


class UserAPIKey(models.Model):
    """Allow users to use their own API keys."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='api_keys')
    provider = models.ForeignKey(LLMProvider, on_delete=models.CASCADE)
    api_key = models.CharField(max_length=255)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['user', 'provider']
        verbose_name = "User API Key"
        verbose_name_plural = "User API Keys"
    
    def __str__(self):
        return f"{self.user.username} - {self.provider.name}"


class UsageLog(models.Model):
    """Track usage for analytics and rate limiting."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='usage_logs')
    date = models.DateField(default=timezone.now)
    message_count = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)
    total_cost = models.DecimalField(max_digits=10, decimal_places=6, default=0)
    
    class Meta:
        unique_together = ['user', 'date']
        verbose_name = "Usage Log"
        verbose_name_plural = "Usage Logs"
        indexes = [
            models.Index(fields=['user', 'date']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.date}"