# ai_chatbot/config.py

from django.conf import settings

# Default settings that can be overridden in Django settings
DEFAULT_SETTINGS = {
    # Provider defaults
    'DEFAULT_PROVIDER': 'openai',
    'DEFAULT_MODEL': 'gpt-3.5-turbo',
    
    # Generation parameters
    'TEMPERATURE': 0.7,
    'MAX_TOKENS': 2000,
    'TOP_P': 1.0,
    'FREQUENCY_PENALTY': 0.0,
    'PRESENCE_PENALTY': 0.0,
    
    # System behavior
    'STREAMING_ENABLED': True,
    'RATE_LIMIT_PER_DAY': 100,
    'MAX_SESSIONS_PER_USER': 50,
    'MAX_MESSAGES_PER_SESSION': 1000,
    
    # UI Settings
    'THEME': 'light',  # 'light', 'dark', or 'auto'
    'SHOW_TOKEN_USAGE': True,
    'SHOW_COST_ESTIMATE': True,
    'ENABLE_MARKDOWN': True,
    'ENABLE_CODE_HIGHLIGHTING': True,
    'ENABLE_LATEX': False,
    
    # Security
    'ALLOW_USER_API_KEYS': True,
    'REQUIRE_EMAIL_VERIFICATION': False,
    'SESSION_TIMEOUT_MINUTES': 1440,  # 24 hours
    
    # Caching
    'CACHE_RESPONSES': False,
    'CACHE_TIMEOUT_SECONDS': 3600,
    
    # Export options
    'ALLOW_EXPORT': True,
    'EXPORT_FORMATS': ['json', 'markdown', 'pdf'],
    
    # System prompts
    'DEFAULT_SYSTEM_PROMPT': """You are a helpful AI assistant. You provide clear, accurate, and helpful responses while being concise and friendly.""",
    
    # Provider-specific settings
    'OPENAI': {
        'BASE_URL': 'https://api.openai.com/v1',
        'DEFAULT_MODEL': 'gpt-3.5-turbo',
        'TIMEOUT': 60,
    },
    'ANTHROPIC': {
        'BASE_URL': 'https://api.anthropic.com',
        'DEFAULT_MODEL': 'claude-3-haiku-20240307',
        'TIMEOUT': 60,
    },
    'MISTRAL': {
        'BASE_URL': 'https://api.mistral.ai/v1',
        'DEFAULT_MODEL': 'mistral-tiny',
        'TIMEOUT': 60,
    }
}


def get_setting(name, default=None):
    """Get a setting from Django settings or use default."""
    ai_settings = getattr(settings, 'AI_CHATBOT', {})
    
    # Check if it's in Django settings
    if name in ai_settings:
        return ai_settings[name]
    
    # Check if it's in default settings
    if name in DEFAULT_SETTINGS:
        return DEFAULT_SETTINGS[name]
    
    # Return provided default
    return default


def get_provider_setting(provider, setting, default=None):
    """Get a provider-specific setting."""
    ai_settings = getattr(settings, 'AI_CHATBOT', {})
    
    # Check Django settings first
    if provider.upper() in ai_settings and setting in ai_settings[provider.upper()]:
        return ai_settings[provider.upper()][setting]
    
    # Check default settings
    if provider.upper() in DEFAULT_SETTINGS and setting in DEFAULT_SETTINGS[provider.upper()]:
        return DEFAULT_SETTINGS[provider.upper()][setting]
    
    return default


# Convenience functions
def is_streaming_enabled():
    """Check if streaming is enabled."""
    return get_setting('STREAMING_ENABLED', True)


def get_rate_limit():
    """Get daily rate limit."""
    return get_setting('RATE_LIMIT_PER_DAY', 100)


def get_default_model():
    """Get default model name."""
    return get_setting('DEFAULT_MODEL', 'gpt-3.5-turbo')


def get_default_provider():
    """Get default provider."""
    return get_setting('DEFAULT_PROVIDER', 'openai')


def allow_user_api_keys():
    """Check if users can provide their own API keys."""
    return get_setting('ALLOW_USER_API_KEYS', True)


def get_export_formats():
    """Get allowed export formats."""
    if not get_setting('ALLOW_EXPORT', True):
        return []
    return get_setting('EXPORT_FORMATS', ['json', 'markdown'])