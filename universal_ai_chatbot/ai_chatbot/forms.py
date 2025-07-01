# ai_chatbot/forms.py

from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator
from .models import ChatMessage, ChatSession, LLMConfig, LLMModel


class ChatMessageForm(forms.Form):
    """Form for sending chat messages."""
    content = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Type your message here...',
            'autofocus': True,
            'maxlength': 32000,
        }),
        max_length=32000,
        label='',
        required=True
    )


class ChatSessionForm(forms.ModelForm):
    """Form for creating/editing chat sessions."""
    
    class Meta:
        model = ChatSession
        fields = ['title', 'model', 'system_prompt', 'temperature', 'max_tokens']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter a title for this chat (optional)'
            }),
            'model': forms.Select(attrs={
                'class': 'form-select'
            }),
            'system_prompt': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Enter a system prompt to set the AI\'s behavior (optional)'
            }),
            'temperature': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.1',
                'min': '0',
                'max': '2'
            }),
            'max_tokens': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '1',
                'max': '32000'
            })
        }
        help_texts = {
            'temperature': 'Controls randomness: 0 = focused, 2 = very random',
            'max_tokens': 'Maximum length of the AI response',
            'system_prompt': 'Sets the behavior and context for the AI assistant'
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only show active models
        self.fields['model'].queryset = LLMModel.objects.filter(is_active=True).select_related('provider')
        
        # Set default values
        if not self.instance.pk:
            self.fields['temperature'].initial = 0.7
            self.fields['max_tokens'].initial = 2000


class LLMConfigForm(forms.ModelForm):
    """Form for user's LLM configuration."""
    
    class Meta:
        model = LLMConfig
        fields = [
            'default_model', 'temperature', 'max_tokens', 
            'system_prompt', 'rate_limit_per_day', 'enable_streaming'
        ]
        widgets = {
            'default_model': forms.Select(attrs={
                'class': 'form-select'
            }),
            'temperature': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.1',
                'min': '0',
                'max': '2'
            }),
            'max_tokens': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '1',
                'max': '32000'
            }),
            'system_prompt': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Default system prompt for all conversations'
            }),
            'rate_limit_per_day': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '0',
                'placeholder': '0 for unlimited'
            }),
            'enable_streaming': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            })
        }
        help_texts = {
            'default_model': 'Default AI model for new conversations',
            'temperature': 'Default randomness setting (0-2)',
            'max_tokens': 'Default maximum response length',
            'system_prompt': 'Applied to all new conversations',
            'rate_limit_per_day': 'Maximum messages per day (0 = unlimited)',
            'enable_streaming': 'Stream responses in real-time'
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only show active models
        self.fields['default_model'].queryset = LLMModel.objects.filter(
            is_active=True
        ).select_related('provider')


class ModelSelectionForm(forms.Form):
    """Form for quick model selection in chat."""
    model = forms.ModelChoiceField(
        queryset=LLMModel.objects.filter(is_active=True),
        widget=forms.Select(attrs={
            'class': 'form-select form-select-sm',
            'onchange': 'this.form.submit()'
        }),
        required=False
    )
    
    def __init__(self, *args, **kwargs):
        initial_model = kwargs.pop('initial_model', None)
        super().__init__(*args, **kwargs)
        if initial_model:
            self.fields['model'].initial = initial_model