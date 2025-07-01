# ai_chatbot/views.py

import json
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.db import transaction
from django.db.models import F
from datetime import date

from .models import ChatSession, ChatMessage, LLMModel, LLMConfig, UsageLog
from .forms import ChatMessageForm, ChatSessionForm, LLMConfigForm
from .services.llm_router import llm_router


class ChatSessionListView(LoginRequiredMixin, ListView):
    """List all chat sessions for the current user."""
    model = ChatSession
    template_name = 'ai_chatbot/session_list.html'
    context_object_name = 'sessions'
    paginate_by = 20
    
    def get_queryset(self):
        """Filter sessions by current user and archive status."""
        queryset = ChatSession.objects.filter(user=self.request.user)
        
        # Filter by archive status
        show_archived = self.request.GET.get('archived', 'false') == 'true'
        queryset = queryset.filter(is_archived=show_archived)
        
        # Search functionality
        search_query = self.request.GET.get('q', '')
        if search_query:
            queryset = queryset.filter(title__icontains=search_query)
        
        return queryset.select_related('model').prefetch_related('messages')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['search_query'] = self.request.GET.get('q', '')
        context['show_archived'] = self.request.GET.get('archived', 'false') == 'true'
        context['active_sessions_count'] = ChatSession.objects.filter(
            user=self.request.user, is_archived=False
        ).count()
        return context


class ChatView(LoginRequiredMixin, DetailView):
    """Main chat interface."""
    model = ChatSession
    template_name = 'ai_chatbot/chat.html'
    context_object_name = 'session'
    
    def get_queryset(self):
        """Ensure user can only access their own sessions."""
        return ChatSession.objects.filter(user=self.request.user)
    
    def get_object(self, queryset=None):
        """Get session or create new one if needed."""
        session_id = self.kwargs.get('pk')
        
        if session_id:
            return super().get_object(queryset)
        else:
            # Create new session
            config = self._get_user_config()
            session = ChatSession.objects.create(
                user=self.request.user,
                model=config.default_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                system_prompt=config.system_prompt
            )
            return session
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Add messages
        context['messages'] = self.object.messages.all().order_by('created_at')
        
        # Add available models
        context['available_models'] = LLMModel.objects.filter(
            is_active=True
        ).select_related('provider')
        
        # Add user config
        context['user_config'] = self._get_user_config()
        
        # Add usage stats
        context['usage_stats'] = self._get_usage_stats()
        
        # Add form
        context['message_form'] = ChatMessageForm()
        
        return context
    
    def _get_user_config(self):
        """Get or create user's LLM config."""
        config, _ = LLMConfig.objects.get_or_create(
            user=self.request.user,
            defaults={
                'default_model': LLMModel.objects.filter(is_active=True).first()
            }
        )
        return config
    
    def _get_usage_stats(self):
        """Get today's usage statistics."""
        usage, _ = UsageLog.objects.get_or_create(
            user=self.request.user,
            date=date.today()
        )
        
        config = self._get_user_config()
        remaining = config.rate_limit_per_day - usage.message_count if config.rate_limit_per_day > 0 else -1
        
        return {
            'messages_today': usage.message_count,
            'tokens_today': usage.total_tokens,
            'remaining_messages': remaining,
            'has_limit': config.rate_limit_per_day > 0
        }


@login_required
def create_chat_session(request):
    """Create a new chat session."""
    if request.method == 'POST':
        form = ChatSessionForm(request.POST)
        if form.is_valid():
            session = form.save(commit=False)
            session.user = request.user
            session.save()
            return redirect('ai_chatbot:chat', pk=session.id)
    else:
        form = ChatSessionForm()
    
    return render(request, 'ai_chatbot/create_session.html', {'form': form})


@login_required
@csrf_exempt
def send_message_ajax(request, session_id):
    """AJAX endpoint for sending messages."""
    if request.method != 'POST':
        return HttpResponseBadRequest('Only POST allowed')
    
    try:
        data = json.loads(request.body)
        content = data.get('content', '').strip()
        
        if not content:
            return JsonResponse({'error': 'Message content is required'}, status=400)
        
        # Get session
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        
        # Check rate limit
        config, _ = LLMConfig.objects.get_or_create(user=request.user)
        usage, _ = UsageLog.objects.get_or_create(
            user=request.user,
            date=date.today()
        )
        
        if config.rate_limit_per_day > 0 and usage.message_count >= config.rate_limit_per_day:
            return JsonResponse({
                'error': 'Daily message limit exceeded',
                'limit': config.rate_limit_per_day
            }, status=429)
        
        # Create user message
        user_message = ChatMessage.objects.create(
            session=session,
            role='user',
            content=content
        )
        
        # Get conversation history
        messages = list(session.messages.filter(is_error=False).order_by('created_at'))
        
        # Add system prompt if exists
        if session.system_prompt:
            messages.insert(0, ChatMessage(
                role='system',
                content=session.system_prompt
            ))
        
        # Get response from LLM
        try:
            response = llm_router.get_response(
                provider=session.model.provider,
                model=session.model,
                messages=messages,
                temperature=session.temperature,
                max_tokens=session.max_tokens,
                stream=False
            )
            
            # Create assistant message
            assistant_message = ChatMessage.objects.create(
                session=session,
                role='assistant',
                content=response.content,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                response_time_ms=response.response_time_ms,
                metadata=response.metadata
            )
            
            # Update session stats and usage
            with transaction.atomic():
                session.total_tokens_used = F('total_tokens_used') + response.input_tokens + response.output_tokens
                session.save()
                
                usage.message_count = F('message_count') + 1
                usage.total_tokens = F('total_tokens') + response.input_tokens + response.output_tokens
                usage.save()
            
            # Auto-generate title if needed
            if not session.title:
                session.title = content[:50] + '...' if len(content) > 50 else content
                session.save()
            
            return JsonResponse({
                'success': True,
                'message': {
                    'id': str(assistant_message.id),
                    'role': 'assistant',
                    'content': assistant_message.content,
                    'timestamp': assistant_message.created_at.isoformat(),
                    'tokens': {
                        'input': response.input_tokens,
                        'output': response.output_tokens
                    }
                },
                'usage': {
                    'remaining': config.rate_limit_per_day - (usage.message_count + 1) if config.rate_limit_per_day > 0 else -1
                }
            })
            
        except Exception as e:
            # Create error message
            error_message = ChatMessage.objects.create(
                session=session,
                role='assistant',
                content="I apologize, but I encountered an error processing your request. Please try again.",
                is_error=True,
                error_message=str(e)
            )
            
            return JsonResponse({
                'success': False,
                'error': 'Failed to get AI response',
                'message': {
                    'id': str(error_message.id),
                    'role': 'assistant',
                    'content': error_message.content,
                    'timestamp': error_message.created_at.isoformat(),
                    'is_error': True
                }
            }, status=500)
            
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def delete_session(request, session_id):
    """Delete a chat session."""
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    
    if request.method == 'POST':
        session.delete()
        return redirect('ai_chatbot:session_list')
    
    return render(request, 'ai_chatbot/confirm_delete.html', {'session': session})


@login_required
def archive_session(request, session_id):
    """Archive/unarchive a session."""
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    session.is_archived = not session.is_archived
    session.save()
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'is_archived': session.is_archived})
    
    return redirect('ai_chatbot:session_list')


class LLMConfigUpdateView(LoginRequiredMixin, UpdateView):
    """Update user's LLM configuration."""
    model = LLMConfig
    form_class = LLMConfigForm
    template_name = 'ai_chatbot/config.html'
    success_url = reverse_lazy('ai_chatbot:session_list')
    
    def get_object(self, queryset=None):
        """Get or create config for current user."""
        config, _ = LLMConfig.objects.get_or_create(
            user=self.request.user,
            defaults={
                'default_model': LLMModel.objects.filter(is_active=True).first()
            }
        )
        return config


@login_required
def new_chat(request):
    """Create a new chat session and redirect to it."""
    config, _ = LLMConfig.objects.get_or_create(
        user=request.user,
        defaults={
            'default_model': LLMModel.objects.filter(is_active=True).first()
        }
    )
    
    session = ChatSession.objects.create(
        user=request.user,
        model=config.default_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        system_prompt=config.system_prompt
    )
    
    return redirect('ai_chatbot:chat', pk=session.id)