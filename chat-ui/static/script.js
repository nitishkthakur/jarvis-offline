// ===========================================
// GLOBAL VARIABLES AND STATE
// ===========================================

class ChatApp {
    constructor() {
        this.isStreaming = false;
        this.messageCount = 0;
        this.tokenCount = 0;
        this.sessionStartTime = Date.now();
        this.currentAgent = 'deep-research';
        this.isProcessing = false;
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.updateSessionTimer();
        this.autoResizeTextarea();
    }

    // ===========================================
    // EVENT BINDINGS
    // ===========================================

    bindEvents() {
        // Input handling
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        const streamingToggle = document.getElementById('streamingMode');

        messageInput.addEventListener('input', this.handleInputChange.bind(this));
        messageInput.addEventListener('keydown', this.handleKeyDown.bind(this));
        sendBtn.addEventListener('click', this.sendMessage.bind(this));
        streamingToggle.addEventListener('change', this.toggleStreamingMode.bind(this));

        // Agent selection
        document.querySelectorAll('.agent-item').forEach(item => {
            item.addEventListener('click', this.selectAgent.bind(this));
        });

        // New chat button
        document.getElementById('newChatBtn').addEventListener('click', this.startNewChat.bind(this));

        // Quick actions
        document.querySelectorAll('.action-btn').forEach(btn => {
            btn.addEventListener('click', this.handleQuickAction.bind(this));
        });
    }

    // ===========================================
    // INPUT HANDLING
    // ===========================================

    handleInputChange(e) {
        const value = e.target.value;
        const charCount = document.getElementById('charCount');
        const sendBtn = document.getElementById('sendBtn');
        
        charCount.textContent = value.length;
        sendBtn.disabled = value.trim().length === 0 || this.isProcessing;
        
        this.autoResizeTextarea();
    }

    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }

    autoResizeTextarea() {
        const textarea = document.getElementById('messageInput');
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }

    // ===========================================
    // MESSAGE HANDLING
    // ===========================================

    async sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        
        if (!message || this.isProcessing) return;

        this.isProcessing = true;
        this.updateProcessingStatus('processing');
        
        // Clear input and add user message
        messageInput.value = '';
        this.handleInputChange({ target: messageInput });
        this.addMessage(message, 'user');

        try {
            if (this.isStreaming) {
                await this.sendStreamingMessage(message);
            } else {
                await this.sendNonStreamingMessage(message);
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
        } finally {
            this.isProcessing = false;
            this.updateProcessingStatus('ready');
        }
    }

    async sendStreamingMessage(message) {
        const messagesContainer = document.getElementById('messagesContainer');
        
        // Add typing indicator
        this.addTypingIndicator();

        try {
            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    agent: this.currentAgent
                })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            // Remove typing indicator
            this.removeTypingIndicator();

            // Create assistant message container
            const assistantMessage = this.createMessageElement('', 'assistant');
            const messageContent = assistantMessage.querySelector('.message-content');
            messagesContainer.appendChild(assistantMessage);
            this.scrollToBottom();

            // Handle streaming response
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let accumulatedText = '';

            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.substring(6);
                        if (data === '[DONE]') {
                            break;
                        }
                        
                        try {
                            const parsed = JSON.parse(data);
                            if (parsed.content) {
                                accumulatedText += parsed.content;
                                // Render accumulated text as markdown for streaming
                                const markdownHtml = marked.parse(accumulatedText);
                                const sanitizedHtml = DOMPurify.sanitize(markdownHtml);
                                messageContent.innerHTML = sanitizedHtml;
                                this.scrollToBottom();
                            }
                        } catch (e) {
                            // Ignore parsing errors for malformed chunks
                        }
                    }
                }
            }

            // Add copy buttons to code blocks after streaming is complete
            this.addCopyButtonsToCodeBlocks(assistantMessage);
            
            this.messageCount++;
            this.tokenCount += this.estimateTokens(message + accumulatedText);
            this.updateStats();

        } catch (error) {
            this.removeTypingIndicator();
            throw error;
        }
    }

    async sendNonStreamingMessage(message) {
        this.addTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    agent: this.currentAgent
                })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            this.removeTypingIndicator();
            this.addMessage(data.response, 'assistant');
            
            this.messageCount++;
            this.tokenCount += this.estimateTokens(message + data.response);
            this.updateStats();

        } catch (error) {
            this.removeTypingIndicator();
            throw error;
        }
    }

    // ===========================================
    // MESSAGE UI
    // ===========================================

    addMessage(content, sender) {
        const messagesContainer = document.getElementById('messagesContainer');
        
        // Remove welcome message if it exists
        const welcomeMessage = messagesContainer.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }

        const messageElement = this.createMessageElement(content, sender);
        messagesContainer.appendChild(messageElement);
        
        // Add fade-in animation
        messageElement.classList.add('fade-in');
        
        this.scrollToBottom();
    }

    addCopyButtonsToCodeBlocks(container) {
        const codeBlocks = container.querySelectorAll('pre');
        codeBlocks.forEach(pre => {
            // Create copy button
            const copyButton = document.createElement('button');
            copyButton.className = 'copy-button';
            copyButton.innerHTML = '<i class="fas fa-copy"></i>';
            copyButton.title = 'Copy code';
            
            // Add click handler
            copyButton.addEventListener('click', async () => {
                const code = pre.querySelector('code');
                const text = code ? code.textContent : pre.textContent;
                
                try {
                    await navigator.clipboard.writeText(text);
                    copyButton.innerHTML = '<i class="fas fa-check"></i>';
                    copyButton.style.background = 'var(--accent-success)';
                    
                    setTimeout(() => {
                        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                        copyButton.style.background = '';
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy text: ', err);
                    // Fallback for older browsers
                    const textArea = document.createElement('textarea');
                    textArea.value = text;
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textArea);
                    
                    copyButton.innerHTML = '<i class="fas fa-check"></i>';
                    setTimeout(() => {
                        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                    }, 2000);
                }
            });
            
            // Add button to pre element
            pre.style.position = 'relative';
            pre.appendChild(copyButton);
        });
    }

    createMessageElement(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        
        if (sender === 'user') {
            avatar.innerHTML = '<i class="fas fa-user"></i>';
        } else {
            avatar.innerHTML = '<i class="fas fa-robot"></i>';
        }

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Render content based on sender
        if (sender === 'assistant') {
            // Render markdown for assistant messages
            const markdownHtml = marked.parse(content);
            const sanitizedHtml = DOMPurify.sanitize(markdownHtml);
            messageContent.innerHTML = sanitizedHtml;
            
            // Add copy buttons to code blocks
            this.addCopyButtonsToCodeBlocks(messageContent);
        } else {
            // Plain text for user messages
            messageContent.textContent = content;
        }

        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = new Date().toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
        });

        messageContent.appendChild(messageTime);
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);

        return messageDiv;
    }

    addTypingIndicator() {
        const messagesContainer = document.getElementById('messagesContainer');
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typingIndicator';
        
        typingDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="typing-text">
                <span>Assistant is typing</span>
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        
        messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }

    removeTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    scrollToBottom() {
        const messagesContainer = document.getElementById('messagesContainer');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // ===========================================
    // AGENT MANAGEMENT
    // ===========================================

    selectAgent(e) {
        const agentItem = e.currentTarget;
        const agentType = agentItem.dataset.agent;
        
        // Update active agent
        document.querySelectorAll('.agent-item').forEach(item => {
            item.classList.remove('active');
        });
        agentItem.classList.add('active');
        
        this.currentAgent = agentType;
        
        // Update header
        this.updateAgentHeader(agentType);
    }

    updateAgentHeader(agentType) {
        const agentNames = {
            general: 'General Assistant',
            'deep-research': 'Deep Research',
            coding: 'Code Assistant'
        };
        
        const agentDescriptions = {
            general: 'Ready to help with any questions',
            'deep-research': 'Specialized in deep research and analysis',
            coding: 'Expert in programming and development'
        };
        
        const agentIcons = {
            general: 'fas fa-brain',
            'deep-research': 'fas fa-search',
            coding: 'fas fa-code'
        };
        
        const headerAgent = document.querySelector('.current-agent');
        const agentAvatar = headerAgent.querySelector('.agent-avatar i');
        const agentName = headerAgent.querySelector('.agent-name');
        const agentDescription = headerAgent.querySelector('.agent-description');
        
        agentAvatar.className = agentIcons[agentType];
        agentName.textContent = agentNames[agentType];
        agentDescription.textContent = agentDescriptions[agentType];
    }

    // ===========================================
    // UI UTILITIES
    // ===========================================

    toggleStreamingMode(e) {
        this.isStreaming = e.target.checked;
        console.log('Streaming mode:', this.isStreaming ? 'enabled' : 'disabled');
    }

    startNewChat() {
        const messagesContainer = document.getElementById('messagesContainer');
        messagesContainer.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">
                    <i class="fas fa-robot"></i>
                </div>
                <h3>Welcome to Jarvis AI</h3>
                <p>I'm here to help you with any questions or tasks. How can I assist you today?</p>
            </div>
        `;
        
        this.messageCount = 0;
        this.tokenCount = 0;
        this.sessionStartTime = Date.now();
        this.updateStats();
        
        console.log('Started new chat');
    }

    handleQuickAction(e) {
        const btn = e.currentTarget;
        const icon = btn.querySelector('i');
        const action = icon.className;
        
        if (action.includes('fa-save')) {
            this.saveChat();
        } else if (action.includes('fa-download')) {
            this.exportChat();
        } else if (action.includes('fa-trash')) {
            this.clearHistory();
        }
    }

    saveChat() {
        // Placeholder for save functionality
        console.log('Chat saved');
        this.showNotification('Chat saved successfully', 'success');
    }

    exportChat() {
        // Placeholder for export functionality
        console.log('Chat exported');
        this.showNotification('Chat exported successfully', 'success');
    }

    clearHistory() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            this.startNewChat();
            this.showNotification('Chat history cleared', 'info');
        }
    }

    showNotification(message, type = 'info') {
        // Create a simple notification (could be enhanced with a proper notification system)
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--accent-primary);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            z-index: 1000;
            animation: slideIn 0.3s ease-out;
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    // ===========================================
    // STATS AND TIMERS
    // ===========================================

    updateStats() {
        document.getElementById('messageCount').textContent = this.messageCount;
        document.getElementById('tokenCount').textContent = this.tokenCount;
    }

    updateSessionTimer() {
        setInterval(() => {
            const elapsed = Date.now() - this.sessionStartTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            document.getElementById('sessionDuration').textContent = 
                `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
    }

    updateProcessingStatus(status) {
        const statusElement = document.getElementById('processingStatus');
        const statusText = statusElement.parentElement.querySelector('span');
        
        if (status === 'processing') {
            statusElement.className = 'status-dot processing';
            statusText.textContent = 'Processing...';
        } else {
            statusElement.className = 'status-dot active';
            statusText.textContent = 'Ready';
        }
    }

    estimateTokens(text) {
        // Simple token estimation (roughly 4 characters per token)
        return Math.ceil(text.length / 4);
    }
}

// ===========================================
// KEYBOARD SHORTCUTS
// ===========================================

document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to send message
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        document.getElementById('sendBtn').click();
    }
    
    // Escape to focus on input
    if (e.key === 'Escape') {
        document.getElementById('messageInput').focus();
    }
    
    // Ctrl/Cmd + K for new chat
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        document.getElementById('newChatBtn').click();
    }
});

// ===========================================
// MOBILE RESPONSIVENESS
// ===========================================

function handleMobileMenu() {
    const menuBtn = document.createElement('button');
    menuBtn.className = 'mobile-menu-btn';
    menuBtn.innerHTML = '<i class="fas fa-bars"></i>';
    menuBtn.style.cssText = `
        display: none;
        position: fixed;
        top: 20px;
        left: 20px;
        z-index: 300;
        background: var(--accent-primary);
        color: white;
        border: none;
        border-radius: 8px;
        width: 44px;
        height: 44px;
        cursor: pointer;
    `;
    
    document.body.appendChild(menuBtn);
    
    menuBtn.addEventListener('click', () => {
        const sidebar = document.querySelector('.left-sidebar');
        sidebar.classList.toggle('open');
    });
    
    // Show menu button on mobile
    if (window.innerWidth <= 768) {
        menuBtn.style.display = 'flex';
        menuBtn.style.alignItems = 'center';
        menuBtn.style.justifyContent = 'center';
    }
    
    window.addEventListener('resize', () => {
        if (window.innerWidth <= 768) {
            menuBtn.style.display = 'flex';
            menuBtn.style.alignItems = 'center';
            menuBtn.style.justifyContent = 'center';
        } else {
            menuBtn.style.display = 'none';
            document.querySelector('.left-sidebar').classList.remove('open');
        }
    });
}

// ===========================================
// INITIALIZATION
// ===========================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize the chat application
    window.chatApp = new ChatApp();
    
    // Setup mobile menu
    handleMobileMenu();
    
    // Focus on input
    document.getElementById('messageInput').focus();
    
    console.log('Jarvis AI Chat Interface initialized');
});
