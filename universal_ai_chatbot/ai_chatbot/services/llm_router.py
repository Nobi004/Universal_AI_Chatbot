# ai_chatbot/services/llm_router.py

import os
import time
import json
import logging
from typing import Dict, List, Optional, Generator, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

import openai
import anthropic
from mistralai import Mistral
from mistralai.models import ChatCompletionRequest, UserMessage, SystemMessage, AssistantMessage
from huggingface_hub import InferenceClient

from django.conf import settings
from ..models import LLMProvider, LLMModel, ChatMessage

logger = logging.getLogger('ai_chatbot')


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    response_time_ms: int
    metadata: Dict = None
    is_streaming: bool = False
    
    def to_dict(self):
        return {
            'content': self.content,
            'model': self.model,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'response_time_ms': self.response_time_ms,
            'metadata': self.metadata or {}
        }


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: str, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url
    
    @abstractmethod
    def get_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Generator[str, None, None]]:
        """Get response from the LLM provider."""
        pass
    
    def format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        """Convert Django ChatMessage objects to provider format."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count."""
        return len(text) // 4


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider implementation."""
    
    def __init__(self, api_key: str, base_url: str = None):
        super().__init__(api_key, base_url)
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1"
        )
    
    def get_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Generator[str, None, None]]:
        
        start_time = time.time()
        
        try:
            if stream:
                return self._stream_response(messages, model, temperature, max_tokens, **kwargs)
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                response_time_ms=response_time_ms,
                metadata={
                    'finish_reason': response.choices[0].finish_reason,
                    'system_fingerprint': getattr(response, 'system_fingerprint', None)
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def _stream_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream response from OpenAI."""
        
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""
    
    def __init__(self, api_key: str, base_url: str = None):
        super().__init__(api_key, base_url)
        self.client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url
        )
    
    def get_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Generator[str, None, None]]:
        
        start_time = time.time()
        
        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                anthropic_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        try:
            if stream:
                return self._stream_response(
                    anthropic_messages, model, temperature, max_tokens, 
                    system_message, **kwargs
                )
            
            response = self.client.messages.create(
                model=model,
                messages=anthropic_messages,
                system=system_message,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Calculate tokens (Anthropic doesn't provide exact counts)
            input_tokens = len(str(messages)) // 4  # Rough estimate
            output_tokens = len(response.content[0].text) // 4
            
            return LLMResponse(
                content=response.content[0].text,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                response_time_ms=response_time_ms,
                metadata={
                    'stop_reason': response.stop_reason,
                    'id': response.id
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise
    
    def _stream_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        system_message: str,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream response from Anthropic."""
        
        stream = self.client.messages.create(
            model=model,
            messages=messages,
            system=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                yield chunk.delta.text


class MistralProvider(BaseLLMProvider):
    """Mistral AI provider implementation."""
    
    def __init__(self, api_key: str, base_url: str = None):
        super().__init__(api_key, base_url)
        self.client = Mistral(api_key=api_key)
    
    def get_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Generator[str, None, None]]:
        
        start_time = time.time()
        
        # Convert to Mistral message format
        mistral_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                mistral_messages.append(SystemMessage(content=msg['content']))
            elif msg['role'] == 'user':
                mistral_messages.append(UserMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                mistral_messages.append(AssistantMessage(content=msg['content']))
        
        try:
            if stream:
                return self._stream_response(
                    mistral_messages, model, temperature, max_tokens, **kwargs
                )
            
            response = self.client.chat.complete(
                model=model,
                messages=mistral_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                response_time_ms=response_time_ms,
                metadata={
                    'finish_reason': response.choices[0].finish_reason
                }
            )
            
        except Exception as e:
            logger.error(f"Mistral API error: {str(e)}")
            raise
    
    def _stream_response(
        self,
        messages: List,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream response from Mistral."""
        
        stream = self.client.chat.stream(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        for chunk in stream:
            if chunk.data.choices[0].delta.content:
                yield chunk.data.choices[0].delta.content


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Inference API provider implementation."""
    
    def __init__(self, api_key: str, base_url: str = None):
        super().__init__(api_key, base_url)
        self.client = InferenceClient(token=api_key)
        self.base_url = base_url or "https://api-inference.huggingface.co/models/"
    
    def get_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Generator[str, None, None]]:
        
        start_time = time.time()
        
        # Store original messages for streaming
        kwargs['original_messages'] = messages
        
        # Convert messages to HuggingFace format
        # HF expects a single prompt, so we need to format the conversation
        prompt = self._format_conversation(messages)
        
        try:
            if stream:
                return self._stream_response(prompt, model, temperature, max_tokens, **kwargs)
            
            # For chat models
            if self._is_chat_model(model):
                response = self.client.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                content = response.choices[0].message.content
            else:
                # For text generation models
                response = self.client.text_generation(
                    prompt,
                    model=model,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    return_full_text=False,
                    **kwargs
                )
                content = response
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Estimate tokens (HF doesn't always provide token counts)
            input_tokens = self.estimate_tokens(prompt)
            output_tokens = self.estimate_tokens(content)
            
            return LLMResponse(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                response_time_ms=response_time_ms,
                metadata={
                    'provider': 'huggingface',
                    'model_id': model
                }
            )
            
        except Exception as e:
            logger.error(f"HuggingFace API error: {str(e)}")
            raise
    
    def _stream_response(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream response from HuggingFace."""
        
        try:
            # For chat models with streaming
            if self._is_chat_model(model):
                messages = kwargs.get('original_messages', [])
                stream = self.client.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    **kwargs
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                # For text generation models with streaming
                stream = self.client.text_generation(
                    prompt,
                    model=model,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    return_full_text=False,
                    stream=True,
                    **kwargs
                )
                
                for chunk in stream:
                    if hasattr(chunk, 'token') and hasattr(chunk.token, 'text'):
                        yield chunk.token.text
                    elif isinstance(chunk, str):
                        yield chunk
                        
        except Exception as e:
            logger.error(f"HuggingFace streaming error: {str(e)}")
            yield f"\n\nError: {str(e)}"
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation for non-chat models."""
        formatted = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                formatted.append(f"System: {content}")
            elif role == 'user':
                formatted.append(f"Human: {content}")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}")
        
        # Add prompt for assistant response
        formatted.append("Assistant:")
        
        return "\n\n".join(formatted)
    
    def _is_chat_model(self, model: str) -> bool:
        """Check if the model supports chat format."""
        chat_models = [
            'microsoft/DialoGPT',
            'facebook/blenderbot',
            'meta-llama/Llama-2',
            'mistralai/Mixtral',
            'HuggingFaceH4/zephyr',
            'tiiuae/falcon',
            'Qwen/Qwen',
            'nvidia/Llama',
            'google/gemma',
            'upstage/SOLAR',
            'codellama/CodeLlama',
            'teknium/OpenHermes',
            'TheBloke/',
            'NousResearch/',
            'togethercomputer/',
        ]
        
        return any(model.lower().startswith(cm.lower()) for cm in chat_models)


class LLMRouter:
    """Routes requests to appropriate LLM providers."""
    
    _providers = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'mistral': MistralProvider,
        'huggingface': HuggingFaceProvider,
    }
    
    def __init__(self):
        self._provider_instances = {}
    
    def get_provider(self, provider_type: str, api_key: str, base_url: str = None) -> BaseLLMProvider:
        """Get or create a provider instance."""
        cache_key = f"{provider_type}:{api_key[:8]}"
        
        if cache_key not in self._provider_instances:
            if provider_type not in self._providers:
                raise ValueError(f"Unknown provider type: {provider_type}")
            
            provider_class = self._providers[provider_type]
            self._provider_instances[cache_key] = provider_class(api_key, base_url)
        
        return self._provider_instances[cache_key]
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a new provider type."""
        if not issubclass(provider_class, BaseLLMProvider):
            raise ValueError("Provider class must inherit from BaseLLMProvider")
        cls._providers[name] = provider_class
    
    def get_response(
        self,
        provider: LLMProvider,
        model: LLMModel,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False,
        user_api_key: str = None,
        **kwargs
    ) -> Union[LLMResponse, Generator[str, None, None]]:
        """
        Get response from the appropriate LLM provider.
        
        Args:
            provider: LLMProvider instance
            model: LLMModel instance
            messages: List of ChatMessage objects
            temperature: Temperature setting
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            user_api_key: Optional user-specific API key
            **kwargs: Additional provider-specific parameters
        
        Returns:
            LLMResponse object or generator for streaming
        """
        
        # Use user's API key if provided, otherwise use system key
        api_key = user_api_key or provider.api_key
        
        # Get the provider instance
        provider_instance = self.get_provider(
            provider.provider_type,
            api_key,
            provider.base_url
        )
        
        # Format messages for the provider
        formatted_messages = provider_instance.format_messages(messages)
        
        # Get response from provider
        return provider_instance.get_response(
            messages=formatted_messages,
            model=model.name,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough estimate of token count."""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    @staticmethod
    def validate_provider_config(provider_type: str, api_key: str, base_url: str = None) -> bool:
        """Validate provider configuration by making a test request."""
        try:
            router = LLMRouter()
            provider = router.get_provider(provider_type, api_key, base_url)
            
            # Make a simple test request
            test_messages = [{"role": "user", "content": "Hello"}]
            
            if provider_type == 'openai':
                response = provider.get_response(
                    messages=test_messages,
                    model='gpt-3.5-turbo',
                    max_tokens=10
                )
            elif provider_type == 'anthropic':
                response = provider.get_response(
                    messages=test_messages,
                    model='claude-3-haiku-20240307',
                    max_tokens=10
                )
            elif provider_type == 'mistral':
                response = provider.get_response(
                    messages=test_messages,
                    model='mistral-tiny',
                    max_tokens=10
                )
            elif provider_type == 'huggingface':
                response = provider.get_response(
                    messages=test_messages,
                    model='microsoft/DialoGPT-small',
                    max_tokens=10
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Provider validation failed: {str(e)}")
            return False


# Singleton instance
llm_router = LLMRouter()