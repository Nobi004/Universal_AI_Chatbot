# ai_chatbot/api/views.py

import json
import logging
from datetime import date
from typing import Optional

from django.db import transaction
from django.db.models import F, Q
from django.utils import timezone
from django.shortcuts import get_object_or_404
from django.http import StreamingHttpResponse

from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.throttling import UserRateThrottle

from ..models import (
    ChatSession, ChatMessage, LLMModel, LLMProvider,
    LLMConfig, UserAPIKey, UsageLog
)
from ..serializers import (
    ChatSessionSerializer, ChatSessionListSerializer, CreateChatSessionSerializer,
    ChatMessageSerializer, SendMessageSerializer, LLMModelSerializer,
    LLMConfigSerializer, UserAPIKeySerializer, UsageStatsSerializer,
    UserSerializer
)
from ..services.llm_router import llm_router, LLMResponse

logger = logging.getLogger('ai_chatbot')


class ChatRateThrottle(UserRateThrottle):
    """Custom rate throttle for chat messages."""
    scope = 'chat_messages'
    
    def get_rate(self):
        """Get rate from user's config or default."""
        try:
            user = self.request.user
            if hasattr(user, 'llm_config'):
                limit = user.llm_config.rate_limit_per_day
                if limit > 0:
                    return f'{limit}/day'
        except:
            pass
        return super().get_rate()


class ChatSessionViewSet(viewsets.ModelViewSet):
    """ViewSet for managing chat sessions."""
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Filter sessions by current user."""
        queryset = ChatSession.objects.filter(user=self.request.user)
        
        # Apply filters
        is_archived = self.request.query_params.get('archived', None)
        if is_archived is not None:
            queryset = queryset.filter(is_archived=is_archived.lower() == 'true')
        
        # Search by title
        search = self.request.query_params.get('search', None)
        if search:
            queryset = queryset.filter(title__icontains=search)
        
        return queryset.select_related('model', 'model__provider')
    
    def get_serializer_class(self):
        """Use different serializers for different actions."""
        if self.action == 'list':
            return ChatSessionListSerializer
        elif self.action == 'create':
            return CreateChatSessionSerializer
        return ChatSessionSerializer
    
    def perform_create(self, serializer):
        """Create session for current user."""
        serializer.save(user=self.request.user)
    
    @action(detail=True, methods=['post'])
    def archive(self, request, pk=None):
        """Archive/unarchive a session."""
        session = self.get_object()
        session.is_archived = not session.is_archived
        session.save()
        return Response({'is_archived': session.is_archived})
    
    @action(detail=True, methods=['post'])
    def clear(self, request, pk=None):
        """Clear all messages in a session."""
        session = self.get_object()
        session.messages.all().delete()
        session.total_tokens_used = 0
        session.total_cost = 0
        session.save()
        return Response({'message': 'Session cleared'})
    
    @action(detail=True, methods=['get'])
    def export(self, request, pk=None):
        """Export session as JSON or Markdown."""
        session = self.get_object()
        format_type = request.query_params.get('format', 'json')
        
        if format_type == 'markdown':
            content = self._export_as_markdown(session)
            response = Response(content, content_type='text/markdown')
            response['Content-Disposition'] = f'attachment; filename="{session.title}.md"'
        else:
            serializer = ChatSessionSerializer(session)
            response = Response(serializer.data)
            response['Content-Disposition'] = f'attachment; filename="{session.title}.json"'
        
        return response
    
    def _export_as_markdown(self, session):
        """Export session as markdown."""
        lines = [
            f"# {session.title}",
            f"**Model**: {session.model.display_name if session.model else 'Unknown'}",
            f"**Created**: {session.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Messages**: {session.messages.count()}",
            "",
            "---",
            ""
        ]
        
        for msg in session.messages.all():
            role = msg.get_role_display()
            timestamp = msg.created_at.strftime('%H:%M')
            lines.extend([
                f"### {role} ({timestamp})",
                msg.content,
                ""
            ])
        
        return '\n'.join(lines)


class ChatMessageViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing chat messages."""
    serializer_class = ChatMessageSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Filter messages by session and user."""
        session_id = self.kwargs.get('session_id')
        return ChatMessage.objects.filter(
            session_id=session_id,
            session__user=self.request.user
        ).order_by('created_at')


class SendMessageView(APIView):
    """API endpoint for sending messages and getting AI responses."""
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [ChatRateThrottle]
    
    def post(self, request):
        """Send a message and get AI response."""
        serializer = SendMessageSerializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        
        # Extract validated data
        content = serializer.validated_data['content']
        session_id = serializer.validated_data.get('session_id')
        model = serializer.validated_data.get('model_id')
        temperature = serializer.validated_data.get('temperature')
        max_tokens = serializer.validated_data.get('max_tokens')
        stream = serializer.validated_data.get('stream', False)
        
        # Get or create session
        if session_id:
            session = ChatSession.objects.get(id=session_id, user=request.user)
        else:
            # Create new session
            config = self._get_user_config(request.user)
            session = ChatSession.objects.create(
                user=request.user,
                model=model or config.default_model,
                temperature=temperature or config.temperature,
                max_tokens=max_tokens or config.max_tokens,
                system_prompt=config.system_prompt
            )
        
        # Check rate limit
        if not self._check_rate_limit(request.user):
            return Response(
                {'error': 'Daily message limit exceeded'},
                status=status.HTTP_429_TOO_MANY_REQUESTS
            )
        
        # Create user message
        user_message = ChatMessage.objects.create(
            session=session,
            role='user',
            content=content
        )
        
        # Get AI response
        try:
            if stream:
                return self._stream_response(session, user_message)
            else:
                return self._get_response(session, user_message)
                
        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            # Create error message
            ChatMessage.objects.create(
                session=session,
                role='assistant',
                content="I'm sorry, an error occurred while processing your request.",
                is_error=True,
                error_message=str(e)
            )
            return Response(
                {'error': 'Failed to get AI response'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _get_user_config(self, user) -> LLMConfig:
        """Get or create user's LLM config."""
        config, created = LLMConfig.objects.get_or_create(
            user=user,
            defaults={
                'default_model': LLMModel.objects.filter(is_active=True).first(),
                'temperature': 0.7,
                'max_tokens': 2000
            }
        )
        return config
    
    def _check_rate_limit(self, user) -> bool:
        """Check if user has exceeded rate limit."""
        config = self._get_user_config(user)
        if config.rate_limit_per_day == 0:
            return True  # No limit
        
        usage, _ = UsageLog.objects.get_or_create(
            user=user,
            date=date.today()
        )
        
        return usage.message_count < config.rate_limit_per_day
    
    def _get_response(self, session: ChatSession, user_message: ChatMessage) -> Response:
        """Get non-streaming response from LLM."""
        # Get conversation history
        messages = list(session.messages.filter(is_error=False).order_by('created_at'))
        
        # Add system prompt if exists
        if session.system_prompt:
            messages.insert(0, ChatMessage(
                role='system',
                content=session.system_prompt
            ))
        
        # Get user's API key if exists
        user_api_key = None
        if hasattr(session.user, 'api_keys'):
            user_key = session.user.api_keys.filter(
                provider=session.model.provider,
                is_active=True
            ).first()
            if user_key:
                user_api_key = user_key.api_key
        
        # Get response from LLM
        response: LLMResponse = llm_router.get_response(
            provider=session.model.provider,
            model=session.model,
            messages=messages,
            temperature=session.temperature,
            max_tokens=session.max_tokens,
            stream=False,
            user_api_key=user_api_key
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
        
        # Update session stats
        with transaction.atomic():
            session.total_tokens_used = F('total_tokens_used') + response.input_tokens + response.output_tokens
            session.save()
            
            # Update usage log
            usage, _ = UsageLog.objects.get_or_create(
                user=session.user,
                date=date.today()
            )
            usage.message_count = F('message_count') + 1
            usage.total_tokens = F('total_tokens') + response.input_tokens + response.output_tokens
            usage.save()
        
        # Return response
        return Response({
            'session_id': str(session.id),
            'message': ChatMessageSerializer(assistant_message).data
        })
    
    def _stream_response(self, session: ChatSession, user_message: ChatMessage) -> StreamingHttpResponse:
        """Stream response from LLM."""
        def generate():
            # Send initial data
            yield json.dumps({
                'type': 'session',
                'session_id': str(session.id)
            }) + '\n'
            
            # Get conversation history
            messages = list(session.messages.filter(is_error=False).order_by('created_at'))
            
            # Add system prompt
            if session.system_prompt:
                messages.insert(0, ChatMessage(
                    role='system',
                    content=session.system_prompt
                ))
            
            # Stream response
            try:
                stream = llm_router.get_response(
                    provider=session.model.provider,
                    model=session.model,
                    messages=messages,
                    temperature=session.temperature,
                    max_tokens=session.max_tokens,
                    stream=True
                )
                
                full_response = ""
                for chunk in stream:
                    full_response += chunk
                    yield json.dumps({
                        'type': 'chunk',
                        'content': chunk
                    }) + '\n'
                
                # Create assistant message
                assistant_message = ChatMessage.objects.create(
                    session=session,
                    role='assistant',
                    content=full_response
                )
                
                yield json.dumps({
                    'type': 'complete',
                    'message_id': str(assistant_message.id)
                }) + '\n'
                
            except Exception as e:
                yield json.dumps({
                    'type': 'error',
                    'error': str(e)
                }) + '\n'
        
        response = StreamingHttpResponse(
            generate(),
            content_type='text/event-stream'
        )
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'
        return response


class LLMModelViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for listing available LLM models."""
    queryset = LLMModel.objects.filter(is_active=True).select_related('provider')
    serializer_class = LLMModelSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Filter by provider if specified."""
        queryset = super().get_queryset()
        provider = self.request.query_params.get('provider', None)
        if provider:
            queryset = queryset.filter(provider__provider_type=provider)
        return queryset


class LLMConfigView(APIView):
    """View for managing user's LLM configuration."""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        """Get user's LLM configuration."""
        config, _ = LLMConfig.objects.get_or_create(
            user=request.user,
            defaults={
                'default_model': LLMModel.objects.filter(is_active=True).first()
            }
        )
        serializer = LLMConfigSerializer(config)
        return Response(serializer.data)
    
    def put(self, request):
        """Update user's LLM configuration."""
        config, _ = LLMConfig.objects.get_or_create(user=request.user)
        serializer = LLMConfigSerializer(config, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)


class UserAPIKeyViewSet(viewsets.ModelViewSet):
    """ViewSet for managing user API keys."""
    serializer_class = UserAPIKeySerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Filter API keys by current user."""
        return UserAPIKey.objects.filter(user=self.request.user).select_related('provider')
    
    def perform_create(self, serializer):
        """Create API key for current user."""
        serializer.save(user=self.request.user)


class UsageStatsView(APIView):
    """View for getting usage statistics."""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        """Get usage stats for current user."""
        today = date.today()
        usage, _ = UsageLog.objects.get_or_create(
            user=request.user,
            date=today
        )
        
        config = LLMConfig.objects.filter(user=request.user).first()
        remaining = config.rate_limit_per_day - usage.message_count if config else 100
        
        data = {
            'date': today,
            'message_count': usage.message_count,
            'total_tokens': usage.total_tokens,
            'total_cost': usage.total_cost,
            'remaining_messages': max(0, remaining)
        }
        
        serializer = UsageStatsSerializer(data)
        return Response(serializer.data)


class ProfileView(APIView):
    """View for user profile information."""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        """Get current user's profile."""
        serializer = UserSerializer(request.user)
        return Response(serializer.data)