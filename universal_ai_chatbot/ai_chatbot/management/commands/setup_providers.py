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
        
        # HuggingFace
        if os.getenv('HUGGINGFACE_API_KEY'):
            provider, created = LLMProvider.objects.get_or_create(
                name='HuggingFace',
                defaults={
                    'provider_type': 'huggingface',
                    'base_url': os.getenv('HUGGINGFACE_BASE_URL', 'https://api-inference.huggingface.co/models/'),
                    'api_key': os.getenv('HUGGINGFACE_API_KEY'),
                    'is_active': True
                }
            )
            
            if created:
                self.stdout.write(self.style.SUCCESS(f'Created provider: {provider.name}'))
                
                # Create HuggingFace models
                models = [
                    # Open Source Chat Models
                    {
                        'name': 'meta-llama/Llama-2-7b-chat-hf',
                        'display_name': 'Llama 2 7B Chat',
                        'description': 'Meta\'s Llama 2 7B model fine-tuned for chat',
                        'max_tokens': 4096,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'meta-llama/Llama-2-13b-chat-hf',
                        'display_name': 'Llama 2 13B Chat',
                        'description': 'Meta\'s Llama 2 13B model fine-tuned for chat',
                        'max_tokens': 4096,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'meta-llama/Llama-2-70b-chat-hf',
                        'display_name': 'Llama 2 70B Chat',
                        'description': 'Meta\'s largest Llama 2 model for chat',
                        'max_tokens': 4096,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                        'display_name': 'Mixtral 8x7B Instruct',
                        'description': 'Mistral\'s MoE model with 8x7B parameters',
                        'max_tokens': 32768,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'mistralai/Mistral-7B-Instruct-v0.2',
                        'display_name': 'Mistral 7B Instruct v0.2',
                        'description': 'Mistral\'s efficient 7B instruction model',
                        'max_tokens': 8192,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'google/gemma-7b-it',
                        'display_name': 'Gemma 7B Instruct',
                        'description': 'Google\'s Gemma 7B instruction-tuned model',
                        'max_tokens': 8192,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'google/gemma-2b-it',
                        'display_name': 'Gemma 2B Instruct',
                        'description': 'Google\'s lightweight Gemma 2B model',
                        'max_tokens': 8192,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'HuggingFaceH4/zephyr-7b-beta',
                        'display_name': 'Zephyr 7B Beta',
                        'description': 'HuggingFace\'s Zephyr model based on Mistral',
                        'max_tokens': 8192,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'microsoft/phi-2',
                        'display_name': 'Phi-2',
                        'description': 'Microsoft\'s small but capable 2.7B model',
                        'max_tokens': 2048,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'tiiuae/falcon-7b-instruct',
                        'display_name': 'Falcon 7B Instruct',
                        'description': 'TII\'s Falcon 7B instruction model',
                        'max_tokens': 2048,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'codellama/CodeLlama-7b-Instruct-hf',
                        'display_name': 'Code Llama 7B Instruct',
                        'description': 'Meta\'s Code Llama for programming tasks',
                        'max_tokens': 16384,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'Qwen/Qwen1.5-7B-Chat',
                        'display_name': 'Qwen 1.5 7B Chat',
                        'description': 'Alibaba\'s Qwen 1.5 chat model',
                        'max_tokens': 32768,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'nvidia/Llama3-ChatQA-1.5-8B',
                        'display_name': 'Llama 3 ChatQA 8B',
                        'description': 'NVIDIA\'s conversational QA model',
                        'max_tokens': 8192,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'upstage/SOLAR-10.7B-Instruct-v1.0',
                        'display_name': 'SOLAR 10.7B Instruct',
                        'description': 'Upstage\'s SOLAR instruction model',
                        'max_tokens': 4096,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    # Text Generation Models
                    {
                        'name': 'bigscience/bloom',
                        'display_name': 'BLOOM 176B',
                        'description': 'BigScience\'s multilingual model',
                        'max_tokens': 2048,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'EleutherAI/gpt-j-6b',
                        'display_name': 'GPT-J 6B',
                        'description': 'EleutherAI\'s GPT-J 6B model',
                        'max_tokens': 2048,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'databricks/dolly-v2-3b',
                        'display_name': 'Dolly v2 3B',
                        'description': 'Databricks\' instruction-following model',
                        'max_tokens': 2048,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'stabilityai/stablelm-tuned-alpha-7b',
                        'display_name': 'StableLM Alpha 7B',
                        'description': 'Stability AI\'s language model',
                        'max_tokens': 4096,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'microsoft/DialoGPT-medium',
                        'display_name': 'DialoGPT Medium',
                        'description': 'Microsoft\'s conversational AI model',
                        'max_tokens': 1024,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
                    {
                        'name': 'facebook/blenderbot-400M-distill',
                        'display_name': 'BlenderBot 400M',
                        'description': 'Facebook\'s conversational model',
                        'max_tokens': 1024,
                        'supports_streaming': True,
                        'supports_functions': False,
                    },
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