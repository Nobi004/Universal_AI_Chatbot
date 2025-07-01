# Django AI Chatbot

A production-ready Django app that provides a ChatGPT-style interface with full REST API support for mobile integration and multi-LLM provider support.

## Features

- 🤖 **Multi-LLM Support**: OpenAI, Anthropic, Mistral, and more
- 💬 **ChatGPT-style Interface**: Modern, responsive web UI
- 📱 **REST API**: Full API for Android/iOS/React Native integration
- 🔐 **User Authentication**: Per-user chat sessions and history
- 🎨 **Customizable UI**: Tailwind-based, dark/light themes
- 🔄 **Real-time Streaming**: Support for streaming responses
- 💾 **Export Options**: JSON and Markdown export
- 📊 **Usage Tracking**: Token usage and cost tracking
- 🔑 **User API Keys**: Users can provide their own API keys
- ⚙️ **Admin Panel**: Full Django admin integration

## Installation

### 1. Install the app

```bash
# Copy the ai_chatbot folder to your Django project
cp -r ai_chatbot /path/to/your/django/project/
```

### 2. Add to INSTALLED_APPS

```python
INSTALLED_APPS = [
    # ... existing apps
    'ai_chatbot',
    'rest_framework',
    'rest_framework.authtoken',
    'corsheaders',
]
```

### 3. Add middleware

```python
MIDDLEWARE = [
    # ... existing middleware
    'corsheaders.middleware.CorsMiddleware',
]
```

### 4. Configure settings

Add to your `settings.py`:

```python
# REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
}

# CORS settings for mobile apps
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    # Add your mobile app origins
]

# AI Chatbot settings (optional)
AI_CHATBOT = {
    'DEFAULT_PROVIDER': 'openai',
    'DEFAULT_MODEL': 'gpt-3.5-turbo',
    'MAX_TOKENS': 2000,
    'TEMPERATURE': 0.7,
    'RATE_LIMIT_PER_DAY': 100,
}
```

### 5. Environment variables

Create a `.env` file in your project root:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Mistral
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 6. URL configuration

Add to your main `urls.py`:

```python
from django.urls import path, include

urlpatterns = [
    # ... existing patterns
    path('chat/', include('ai_chatbot.urls')),
    path('api/chat/', include('ai_chatbot.api.urls')),
]
```

### 7. Run migrations

```bash
python manage.py makemigrations ai_chatbot
python manage.py migrate
python manage.py setup_providers  # Initialize LLM providers
```

### 8. Create auth tokens for users

```bash
python manage.py shell
```

```python
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token

# Create tokens for all users
for user in User.objects.all():
    Token.objects.get_or_create(user=user)
```

## Usage

### Web Interface

1. Navigate to `http://localhost:8000/chat/`
2. Start a new chat or continue existing sessions
3. Configure settings at `http://localhost:8000/chat/config/`

### REST API

#### Authentication

Include the token in your requests:

```bash
Authorization: Token your_auth_token_here
```

#### Endpoints

- `POST /api/chat/send/` - Send a message
- `GET /api/chat/sessions/` - List sessions
- `GET /api/chat/sessions/{id}/` - Get session details
- `POST /api/chat/sessions/` - Create new session
- `DELETE /api/chat/sessions/{id}/` - Delete session
- `GET /api/chat/models/` - List available models
- `GET /api/chat/config/` - Get user config
- `PUT /api/chat/config/` - Update user config

#### Example: Send a message

```bash
curl -X POST http://localhost:8000/api/chat/send/ \
  -H "Authorization: Token your_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Hello, how are you?",
    "session_id": "optional_session_uuid"
  }'
```

### Mobile Integration

#### Android (Retrofit)

```kotlin
interface ChatAPI {
    @POST("api/chat/send/")
    suspend fun sendMessage(
        @Header("Authorization") token: String,
        @Body message: MessageRequest
    ): MessageResponse
}
```

#### iOS (Swift)

```swift
func sendMessage(content: String, sessionId: String?) async throws -> Message {
    let url = URL(string: "\(baseURL)/api/chat/send/")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("Token \(authToken)", forHTTPHeaderField: "Authorization")
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    
    let body = ["content": content, "session_id": sessionId]
    request.httpBody = try JSONEncoder().encode(body)
    
    let (data, _) = try await URLSession.shared.data(for: request)
    return try JSONDecoder().decode(Message.self, from: data)
}
```

#### React Native

```javascript
const sendMessage = async (content, sessionId = null) => {
    const response = await fetch(`${API_BASE}/api/chat/send/`, {
        method: 'POST',
        headers: {
            'Authorization': `Token ${authToken}`,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content, session_id: sessionId }),
    });
    return response.json();
};
```

## Customization

### UI Customization

1. Override templates by creating `templates/ai_chatbot/` in your project
2. Copy the original template and modify as needed
3. Custom CSS can be added to `static/ai_chatbot/css/custom.css`

### Adding New LLM Providers

1. Create a new provider class in `services/llm_router.py`:

```python
class CustomProvider(BaseLLMProvider):
    def get_response(self, messages, model, **kwargs):
        # Your implementation
        pass
```

2. Register the provider:

```python
LLMRouter.register_provider('custom', CustomProvider)
```

### System Prompts

Configure default system prompts in settings:

```python
AI_CHATBOT = {
    'DEFAULT_SYSTEM_PROMPT': """You are a helpful assistant specialized in..."""
}
```

## Admin Panel

Access the Django admin at `/admin/` to:

- Manage LLM providers and models
- View chat sessions and messages
- Configure user settings
- Monitor usage statistics
- Manage API keys

## Troubleshooting

### Common Issues

1. **"No API key found"**
   - Check your `.env` file
   - Ensure environment variables are loaded
   - Run `python manage.py setup_providers`

2. **CORS errors in mobile app**
   - Add your app's domain to `CORS_ALLOWED_ORIGINS`
   - Check CORS middleware is installed

3. **"Rate limit exceeded"**
   - Adjust `RATE_LIMIT_PER_DAY` in settings
   - Check user's usage in admin panel

4. **Streaming not working**
   - Ensure your server supports streaming responses
   - Check `STREAMING_ENABLED` setting

## Production Deployment

### Using Gunicorn

```bash
gunicorn myproject.wsgi:application --bind 0.0.0.0:8000 --workers 4
```

### Using Uvicorn (for async support)

```bash
uvicorn myproject.asgi:application --host 0.0.0.0 --port 8000
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location /static/ {
        alias /path/to/staticfiles/;
    }
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # For streaming responses
        proxy_buffering off;
        proxy_cache off;
    }
}
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.