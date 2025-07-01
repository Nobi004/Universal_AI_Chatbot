// ai_chatbot/static/ai_chatbot/js/chat.js

class ChatInterface {
    constructor(sessionId) {
        this.sessionId = sessionId;
        this.messagesContainer = document.getElementById('messages-container');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-button');
        this.typingIndicator = document.getElementById('typing-indicator');
        this.charCount = document.getElementById('char-count');
        this.isStreaming = false;
        
        this.initializeEventListeners();
        this.initializeMarkdown();
        this.scrollToBottom();
    }
    
    initializeEventListeners() {
        // Message input events
        this.messageInput.addEventListener('input', () => this.updateCharCount());
        this.messageInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        
        // Form submission
        const chatForm = document.getElementById('chat-form');
        if (chatForm) {
            chatForm.addEventListener('submit', (e) => this.sendMessage(e));
        }
        
        // Copy code buttons
        this.initializeCodeCopyButtons();
        
        // Window resize
        window.addEventListener('resize', () => this.scrollToBottom());
    }
    
    initializeMarkdown() {
        // Configure marked.js
        marked.setOptions({
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    try {
                        return hljs.highlight(code, { language: lang }).value;
                    } catch (e) {
                        console.error('Highlight error:', e);
                    }
                }
                return hljs.highlightAuto(code).value;
            },
            breaks: true,
            gfm: true,
            tables: true,
            sanitize: false
        });
        
        // Process existing markdown content
        document.querySelectorAll('.markdown-content').forEach(element => {
            this.renderMarkdown(element);
        });
    }
    
    renderMarkdown(element) {
        const text = element.textContent || element.innerText;
        element.innerHTML = marked.parse(text);
        
        // Add copy buttons to code blocks
        element.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
            this.addCopyButton(block.parentElement);
        });
    }
    
    addCopyButton(preElement) {
        // Check if button already exists
        if (preElement.querySelector('.copy-code-btn')) return;
        
        const button = document.createElement('button');
        button.className = 'copy-code-btn';
        button.innerHTML = '<i class="bi bi-clipboard"></i> Copy';
        button.onclick = () => this.copyCode(preElement, button);
        
        preElement.style.position = 'relative';
        preElement.appendChild(button);
    }
    
    copyCode(preElement, button) {
        const code = preElement.querySelector('code').textContent;
        
        navigator.clipboard.writeText(code).then(() => {
            button.innerHTML = '<i class="bi bi-check"></i> Copied!';
            setTimeout(() => {
                button.innerHTML = '<i class="bi bi-clipboard"></i> Copy';
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy:', err);
            this.showToast('Failed to copy code', 'error');
        });
    }
    
    initializeCodeCopyButtons() {
        document.querySelectorAll('pre code').forEach(block => {
            const pre = block.parentElement;
            if (!pre.querySelector('.copy-code-btn')) {
                this.addCopyButton(pre);
            }
        });
    }
    
    updateCharCount() {
        const length = this.messageInput.value.length;
        this.charCount.textContent = `${length}/32000`;
        
        // Change color when approaching limit
        if (length > 30000) {
            this.charCount.style.color = 'var(--bs-danger)';
        } else if (length > 25000) {
            this.charCount.style.color = 'var(--bs-warning)';
        } else {
            this.charCount.style.color = 'var(--text-secondary)';
        }
    }
    
    handleKeyDown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage(event);
        }
    }
    
    async sendMessage(event) {
        event.preventDefault();
        
        const content = this.messageInput.value.trim();
        if (!content || this.isStreaming) return;
        
        // Disable input
        this.setInputState(false);
        
        // Add user message
        this.addMessage('user', content);
        
        // Clear input
        this.messageInput.value = '';
        this.updateCharCount();
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            // Check if streaming is enabled
            const streamingEnabled = document.body.dataset.streamingEnabled === 'true';
            
            if (streamingEnabled) {
                await this.sendStreamingMessage(content);
            } else {
                await this.sendStandardMessage(content);
            }
        } catch (error) {
            console.error('Send message error:', error);
            this.hideTypingIndicator();
            this.showToast('Failed to send message. Please try again.', 'error');
        } finally {
            this.setInputState(true);
            this.messageInput.focus();
        }
    }
    
    async sendStandardMessage(content) {
        try {
            const response = await fetch(`/chat/session/${this.sessionId}/send/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCsrfToken()
                },
                body: JSON.stringify({ content: content })
            });
            
            const data = await response.json();
            
            this.hideTypingIndicator();
            
            if (data.success) {
                this.addMessage('assistant', data.message.content, data.message);
                
                if (data.usage && data.usage.remaining >= 0) {
                    this.updateUsageStats(data.usage.remaining);
                }
            } else {
                this.handleError(data.error || 'Failed to get response');
            }
        } catch (error) {
            this.hideTypingIndicator();
            throw error;
        }
    }
    
    async sendStreamingMessage(content) {
        this.isStreaming = true;
        const messageId = this.generateTempId();
        
        try {
            const response = await fetch('/api/chat/send/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Token ${this.getAuthToken()}`,
                    'X-CSRFToken': this.getCsrfToken()
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    content: content,
                    stream: true
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            this.hideTypingIndicator();
            
            // Create empty assistant message
            const messageElement = this.addMessage('assistant', '', { id: messageId });
            const contentElement = messageElement.querySelector('.markdown-content');
            
            // Read the stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let fullContent = '';
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            const data = JSON.parse(line);
                            
                            if (data.type === 'chunk') {
                                fullContent += data.content;
                                contentElement.innerHTML = marked.parse(fullContent);
                                this.scrollToBottom();
                            } else if (data.type === 'complete') {
                                // Update with final message data
                                this.updateMessageMetadata(messageElement, data);
                            } else if (data.type === 'error') {
                                this.handleError(data.error);
                            }
                        } catch (e) {
                            console.error('Parse error:', e);
                        }
                    }
                }
            }
            
            // Process final code blocks
            contentElement.querySelectorAll('pre code').forEach(block => {
                hljs.highlightElement(block);
                this.addCopyButton(block.parentElement);
            });
            
        } finally {
            this.isStreaming = false;
        }
    }
    
    addMessage(role, content, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${role}`;
        messageDiv.dataset.messageId = metadata.id || this.generateTempId();
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (role === 'user') {
            // Simple text for user messages
            contentDiv.textContent = content;
        } else {
            // Markdown for assistant messages
            contentDiv.className = 'message-content markdown-content';
            if (content) {
                contentDiv.innerHTML = marked.parse(content);
                
                // Highlight code blocks
                contentDiv.querySelectorAll('pre code').forEach(block => {
                    hljs.highlightElement(block);
                    this.addCopyButton(block.parentElement);
                });
            }
        }
        
        messageDiv.appendChild(contentDiv);
        
        // Add metadata
        const metaDiv = this.createMessageMetadata(metadata);
        messageDiv.appendChild(metaDiv);
        
        // Insert before typing indicator
        this.messagesContainer.insertBefore(messageDiv, this.typingIndicator);
        this.scrollToBottom();
        
        return messageDiv;
    }
    
    createMessageMetadata(metadata) {
        const metaDiv = document.createElement('small');
        metaDiv.className = 'text-muted message-metadata';
        
        const time = metadata.timestamp 
            ? new Date(metadata.timestamp).toLocaleTimeString('en-US', { 
                hour: 'numeric', 
                minute: '2-digit' 
              })
            : new Date().toLocaleTimeString('en-US', { 
                hour: 'numeric', 
                minute: '2-digit' 
              });
        
        let metaText = time;
        
        if (metadata.input_tokens || metadata.output_tokens) {
            metaText += ` | Tokens: ${metadata.input_tokens || 0}/${metadata.output_tokens || 0}`;
        }
        
        if (metadata.response_time_ms) {
            metaText += ` | ${metadata.response_time_ms}ms`;
        }
        
        metaDiv.textContent = metaText;
        return metaDiv;
    }
    
    updateMessageMetadata(messageElement, data) {
        const metaDiv = messageElement.querySelector('.message-metadata');
        if (metaDiv && data.message) {
            const newMeta = this.createMessageMetadata(data.message);
            metaDiv.textContent = newMeta.textContent;
        }
    }
    
    showTypingIndicator() {
        this.typingIndicator.style.display = 'block';
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }
    
    setInputState(enabled) {
        this.messageInput.disabled = !enabled;
        this.sendButton.disabled = !enabled;
        
        if (enabled) {
            this.sendButton.innerHTML = '<i class="bi bi-send"></i> Send';
        } else {
            this.sendButton.innerHTML = '<span class="loading-spinner"></span> Sending...';
        }
    }
    
    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    updateUsageStats(remaining) {
        const statsElements = document.querySelectorAll('.usage-stats');
        statsElements.forEach(element => {
            if (remaining >= 0) {
                element.textContent = `Messages remaining: ${remaining}`;
            }
        });
    }
    
    handleError(error) {
        let errorMessage = 'An error occurred';
        
        if (typeof error === 'string') {
            errorMessage = error;
        } else if (error.message) {
            errorMessage = error.message;
        }
        
        // Add error message to chat
        this.addMessage('assistant', `I apologize, but I encountered an error: ${errorMessage}`, {
            is_error: true
        });
        
        this.showToast(errorMessage, 'error');
    }
    
    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toast-container') || this.createToastContainer();
        
        const toast = document.createElement('div');
        toast.className = `toast show align-items-center text-white bg-${type === 'error' ? 'danger' : 'primary'} border-0`;
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        toastContainer.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            toast.remove();
        }, 5000);
        
        // Initialize Bootstrap toast
        new bootstrap.Toast(toast).show();
    }
    
    createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(container);
        return container;
    }
    
    getCsrfToken() {
        return document.querySelector('[name=csrfmiddlewaretoken]')?.value || 
               this.getCookie('csrftoken');
    }
    
    getAuthToken() {
        return document.body.dataset.authToken || '';
    }
    
    getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    
    generateTempId() {
        return 'temp-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    const sessionId = document.body.dataset.sessionId;
    if (sessionId) {
        window.chatInterface = new ChatInterface(sessionId);
    }
});