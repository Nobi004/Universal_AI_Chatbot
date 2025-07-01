# ai_chatbot/management/commands/setup_providers.py

import os
from django.core.management.base import BaseCommand
from django.conf import settings
from dotenv import load_dotenv

from ai_chatbot.models import LLMProvider, LLMModel

# Load environment variables
load_dotenv()


class Command(BaseCommand):
    help = 'Set up default LLM providers and models'
    
    def handle(self, *args, **options):
        self.stdout.write('Setting up LLM providers and models...')
        
        # OpenAI
        if os.getenv('OPENAI_API_KEY'):
            provider, created = LLMProvider.objects.get_or_create(
                name='OpenAI',
                defaults={
                    'provider_type': 'openai',
                    'base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                    'api_key': os.getenv('OPENAI_API_KEY'),
                    'is_active': True
                }
            )
            
            if created:
                self.stdout.write(self.style.SUCCESS(f'Created provider: {provider.name}'))
                
                # Create OpenAI models
                models = [
                    {
                        'name': 'gpt-3.5-turbo',
                        'display_name': 'GPT-3.5 Turbo',
                        'description': 'Fast and efficient model for most tasks',
                        'max_tokens': 4096,
                        'supports_streaming': True,
                        'supports_functions': True,
                        'cost_per_1k_input_tokens': 0.0015,
                        'cost_per_1k_output_tokens': 0.002
                    },
                    {
                        'name': 'gpt-4',
                        'display_name': 'GPT-4',
                        'description': 'Most capable model for complex tasks',
                        'max_tokens': 8192,
                        'supports_streaming': True,
                        'supports_functions': True,
                        'cost_per_1k_input_tokens': 0.03,
                        'cost_per_1k_output_tokens': 0.06
                    },
                    {
                        'name': 'gpt-4-turbo-preview',
                        'display_name': 'GPT-4 Turbo',
                        'description': 'Latest GPT-4 with improved performance',
                        'max_tokens': 128000,
                        'supports_streaming': True,
                        'supports_functions': True,
                        'cost_per_1k_input_tokens': 0.01,
                        'cost_per_1k_output_tokens': 0.03
                    }
                ]
                
                for model_data in models:
                    model, model_created = LLMModel.objects.get_or_create(
                        provider=provider,
                        name=model_data['name'],
                        defaults=model_data
                    )
                    if model_created:
                        self.stdout.write(f'  - Created model: {model.display_name}')
        
        # Anthropic
        if os.getenv('ANTHROPIC_API_KEY'):
            provider, created = LLMProvider.objects.get_or_create(
                name='Anthropic',
                defaults={
                    'provider_type': 'anthropic',
                    'base_url': os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com'),
                    'api_key': os.getenv('ANTHROPIC_API_KEY'),
                    'is_active': True
                }
            )
            
            if created:
                self.stdout.write(self.style.SUCCESS(f'Created provider: {provider.name}'))
                
                # Create Anthropic models
                models = [
                    {
                        'name': 'claude-3-haiku-20240307',
                        'display_name': 'Claude 3 Haiku',
                        'description': 'Fast and affordable model',
                        'max_tokens': 4096,
                        'supports_streaming': True,
                        'supports_functions': False,
                        'cost_per_1k_input_tokens': 0.00025,
                        'cost_per_1k_output_tokens': 0.00125
                    },
                    {
                        'name': 'claude-3-sonnet-20240229',
                        'display_name': 'Claude 3 Sonnet',
                        'description': 'Balanced performance and cost',
                        'max_tokens': 4096,
                        'supports_streaming': True,
                        'supports_functions': False,
                        'cost_per_1k_input_tokens': 0.003,
                        'cost_per_1k_output_tokens': 0.015
                    },
                    {
                        'name': 'claude-3-opus-20240229',
                        'display_name': 'Claude 3 Opus',
                        'description': 'Most capable Claude model',
                        'max_tokens': 4096,
                        'supports_streaming': True,
                        'supports_functions': False,
                        'cost_per_1k_input_tokens': 0.015,
                        'cost_per_1k_output_tokens': 0.075
                    }
                ]
                
                for model_data in models:
                    model, model_created = LLMModel.objects.get_or_create(
                        provider=provider,
                        name=model_data['name'],
                        defaults=model_data
                    )
                    if model_created:
                        self.stdout.write(f'  - Created model: {model.display_name}')
        
        # Mistral
        if os.getenv('MISTRAL_API_KEY'):
            provider, created = LLMProvider.objects.get_or_create(
                name='Mistral',
                defaults={
                    'provider_type': 'mistral',
                    'base_url': os.getenv('MISTRAL_BASE_URL', 'https://api.mistral.ai/v1'),
                    'api_key': os.getenv('MISTRAL_API_KEY'),
                    'is_active': True
                }
            )
            
            if created:
                self.stdout.write(self.style.SUCCESS(f'Created provider: {provider.name}'))
                
                # Create Mistral models
                models = [
                    {
                        'name': 'mistral-tiny',
                        'display_name': 'Mistral Tiny',
                        'description': 'Smallest and fastest Mistral model',
                        'max_tokens': 8192,
                        'supports_streaming': True,
                        'supports_functions': False,
                        'cost_per_1k_input_tokens': 0.00025,
                        'cost_per_1k_output_tokens': 0.00025
                    },
                    {
                        'name': 'mistral-small',
                        'display_name': 'Mistral Small',
                        'description': 'Balanced Mistral model',
                        'max_tokens': 8192,
                        'supports_streaming': True,
                        'supports_functions': False,
                        'cost_per_1k_input_tokens': 0.002,
                        'cost_per_1k_output_tokens': 0.006
                    },
                    {
                        'name': 'mistral-medium',
                        'display_name': 'Mistral Medium',
                        'description': 'Most capable Mistral model',
                        'max_tokens': 8192,
                        'supports_streaming': True,
                        'supports_functions': False,
                        'cost_per_1k_input_tokens': 0.0027,
                        'cost_per_1k_output_tokens': 0.0081
                    }
                ]
                
                for model_data in models:
                    model, model_created = LLMModel.objects.get_or_create(
                        provider=provider,
                        name=model_data['name'],
                        defaults=model_data
                    )
                    if model_created:
                        self.stdout.write(f'  - Created model: {model.display_name}')
        
        self.stdout.write(self.style.SUCCESS('Setup complete!'))