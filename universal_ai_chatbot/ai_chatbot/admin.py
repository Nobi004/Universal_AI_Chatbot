# ai_chatbot/admin.py

from django.contrib import admin
from django.utils.html import format_html
from django.db.models import Count, Sum
from .models import (
    LLMProvider, LLMModel, LLMConfig, ChatSession, 
    ChatMessage, UserAPIKey, UsageLog
)


@admin.register(LLMProvider)
class LLMProviderAdmin(admin.ModelAdmin):
    """Admin for LLM providers."""
    list_display = ['name', 'provider_type', 'is_active', 'models_count', 'created_at']
    list_filter = ['provider_type', 'is_active', 'created_at']
    search_fields = ['name', 'base_url']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        (None, {
            'fields': ('name', 'provider_type', 'is_active')
        }),
        ('API Configuration', {
            'fields': ('base_url', 'api_key'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def models_count(self, obj):
        """Count of models for this provider."""
        return obj.models.count()
    models_count.short_description = 'Models'


@admin.register(LLMModel)
class LLMModelAdmin(admin.ModelAdmin):
    """Admin for LLM models."""
    list_display = [
        'display_name', 'name', 'provider', 'max_tokens', 
        'supports_streaming', 'is_active', 'formatted_costs'
    ]
    list_filter = ['provider', 'supports_streaming', 'supports_functions', 'is_active']
    search_fields = ['name', 'display_name', 'description']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        (None, {
            'fields': ('provider', 'name', 'display_name', 'description', 'is_active')
        }),
        ('Capabilities', {
            'fields': ('max_tokens', 'supports_streaming', 'supports_functions')
        }),
        ('Pricing', {
            'fields': ('cost_per_1k_input_tokens', 'cost_per_1k_output_tokens'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def formatted_costs(self, obj):
        """Display formatted costs."""
        if obj.cost_per_1k_input_tokens and obj.cost_per_1k_output_tokens:
            return format_html(
                '<span title="Input cost">${}</span> / <span title="Output cost">${}</span>',
                obj.cost_per_1k_input_tokens,
                obj.cost_per_1k_output_tokens
            )
        return '-'
    formatted_costs.short_description = 'Cost per 1K tokens (In/Out)'


@admin.register(LLMConfig)
class LLMConfigAdmin(admin.ModelAdmin):
    """Admin for user LLM configurations."""
    list_display = [
        'user', 'default_model', 'temperature', 'max_tokens', 
        'rate_limit_per_day', 'enable_streaming'
    ]
    list_filter = ['enable_streaming', 'created_at']
    search_fields = ['user__username', 'user__email']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        (None, {
            'fields': ('user', 'default_model')
        }),
        ('Parameters', {
            'fields': ('temperature', 'max_tokens', 'system_prompt')
        }),
        ('Limits & Features', {
            'fields': ('rate_limit_per_day', 'enable_streaming')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )


class ChatMessageInline(admin.TabularInline):
    """Inline admin for chat messages."""
    model = ChatMessage
    extra = 0
    readonly_fields = ['created_at', 'response_time_ms']
    fields = ['role', 'content', 'input_tokens', 'output_tokens', 'response_time_ms', 'created_at']
    can_delete = False
    
    def has_add_permission(self, request, obj=None):
        return False


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    """Admin for chat sessions."""
    list_display = [
        'title_preview', 'user', 'model', 'messages_count', 
        'total_tokens_used', 'formatted_cost', 'is_archived', 'updated_at'
    ]
    list_filter = ['is_archived', 'model', 'created_at', 'updated_at']
    search_fields = ['title', 'user__username', 'user__email']
    readonly_fields = ['id', 'total_tokens_used', 'total_cost', 'created_at', 'updated_at']
    inlines = [ChatMessageInline]
    
    fieldsets = (
        (None, {
            'fields': ('id', 'user', 'title', 'model', 'is_archived')
        }),
        ('Configuration', {
            'fields': ('system_prompt', 'temperature', 'max_tokens')
        }),
        ('Statistics', {
            'fields': ('total_tokens_used', 'total_cost'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def title_preview(self, obj):
        """Show truncated title."""
        title = obj.title or 'Untitled'
        return title[:50] + '...' if len(title) > 50 else title
    title_preview.short_description = 'Title'
    
    def messages_count(self, obj):
        """Count of messages in session."""
        return obj.messages.count()
    messages_count.short_description = 'Messages'
    
    def formatted_cost(self, obj):
        """Display formatted cost."""
        if obj.total_cost:
            return f'${obj.total_cost:.4f}'
        return '$0.00'
    formatted_cost.short_description = 'Total Cost'
    
    def get_queryset(self, request):
        """Optimize queryset with related data."""
        return super().get_queryset(request).select_related(
            'user', 'model', 'model__provider'
        ).annotate(
            message_count=Count('messages')
        )


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    """Admin for chat messages."""
    list_display = [
        'content_preview', 'role', 'session', 'tokens_info', 
        'response_time_ms', 'is_error', 'created_at'
    ]
    list_filter = ['role', 'is_error', 'created_at']
    search_fields = ['content', 'session__title']
    readonly_fields = ['id', 'created_at', 'metadata']
    
    fieldsets = (
        (None, {
            'fields': ('id', 'session', 'role', 'content')
        }),
        ('Metrics', {
            'fields': ('input_tokens', 'output_tokens', 'response_time_ms')
        }),
        ('Error Information', {
            'fields': ('is_error', 'error_message'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('metadata', 'created_at'),
            'classes': ('collapse',)
        })
    )
    
    def content_preview(self, obj):
        """Show truncated content."""
        return obj.content[:100] + '...' if len(obj.content) > 100 else obj.content
    content_preview.short_description = 'Content'
    
    def tokens_info(self, obj):
        """Display token counts."""
        if obj.input_tokens or obj.output_tokens:
            return format_html(
                '<span title="Input tokens">{}</span> / <span title="Output tokens">{}</span>',
                obj.input_tokens or 0,
                obj.output_tokens or 0
            )
        return '-'
    tokens_info.short_description = 'Tokens (In/Out)'


@admin.register(UserAPIKey)
class UserAPIKeyAdmin(admin.ModelAdmin):
    """Admin for user API keys."""
    list_display = ['user', 'provider', 'masked_key', 'is_active', 'created_at']
    list_filter = ['provider', 'is_active', 'created_at']
    search_fields = ['user__username', 'user__email']
    readonly_fields = ['created_at', 'masked_key']
    
    fieldsets = (
        (None, {
            'fields': ('user', 'provider', 'api_key', 'is_active')
        }),
        ('Information', {
            'fields': ('masked_key', 'created_at'),
            'classes': ('collapse',)
        })
    )
    
    def masked_key(self, obj):
        """Return masked API key for security."""
        if obj.api_key:
            return f"{obj.api_key[:8]}...{obj.api_key[-4:]}"
        return None
    masked_key.short_description = 'Masked Key'


@admin.register(UsageLog)
class UsageLogAdmin(admin.ModelAdmin):
    """Admin for usage logs."""
    list_display = ['user', 'date', 'message_count', 'total_tokens', 'formatted_cost']
    list_filter = ['date']
    search_fields = ['user__username', 'user__email']
    date_hierarchy = 'date'
    
    fieldsets = (
        (None, {
            'fields': ('user', 'date')
        }),
        ('Usage Statistics', {
            'fields': ('message_count', 'total_tokens', 'total_cost')
        })
    )
    
    def formatted_cost(self, obj):
        """Display formatted cost."""
        if obj.total_cost:
            return f'${obj.total_cost:.4f}'
        return '$0.00'
    formatted_cost.short_description = 'Total Cost'
    
    def get_queryset(self, request):
        """Optimize queryset."""
        return super().get_queryset(request).select_related('user')
    
    def has_add_permission(self, request):
        """Prevent manual creation of usage logs."""
        return False


# Admin site customization
admin.site.site_header = "AI Chatbot Admin"
admin.site.site_title = "AI Chatbot"
admin.site.index_title = "Welcome to AI Chatbot Administration"