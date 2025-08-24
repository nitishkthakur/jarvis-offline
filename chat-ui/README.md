# 🤖 Jarvis AI Chat Interface

A professional ChatGPT-like web interface with FastAPI backend, featuring streaming responses, multiple AI agents, and a beautiful dark theme.

## ✨ Features

### 🎨 Frontend
- **Professional Dark Theme**: Minimalist design with custom color system
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Real-time Streaming**: Live streaming responses with typing indicators
- **Multiple Agents**: General, Research, and Coding specialists
- **Modern UI**: Clean interface with smooth animations
- **Keyboard Shortcuts**: Productivity-focused hotkeys

### 🚀 Backend
- **FastAPI Framework**: High-performance async API
- **Streaming & Non-streaming**: Both response modes supported
- **Agent Management**: Configurable AI agents with specialized prompts
- **Error Handling**: Robust error handling and recovery
- **CORS Support**: Full cross-origin resource sharing
- **Health Monitoring**: Built-in health checks and status endpoints

### 🔧 Technical Stack
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Backend**: FastAPI, Python 3.8+
- **AI Integration**: OpenAI GPT-4 API
- **Styling**: Custom CSS with design system
- **Icons**: Font Awesome 6
- **Fonts**: Inter (Google Fonts)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd jarvis-offline/chat-ui
   ```

2. **Set up environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your OpenAI API key**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

5. **Start the server**
   ```bash
   # Option 1: Use the startup script
   ./start.sh
   
   # Option 2: Run directly
   python main.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8000`

## 📁 Project Structure

```
chat-ui/
├── main.py                 # FastAPI backend server
├── requirements.txt        # Python dependencies
├── start.sh               # Startup script
├── README.md              # This file
├── static/
│   ├── styles.css         # Complete CSS styling
│   └── script.js          # Frontend JavaScript
└── templates/
    └── index.html         # Main HTML template
```

## 🎛️ API Endpoints

### Chat Endpoints
- `POST /api/chat` - Non-streaming chat completion
- `POST /api/chat/stream` - Streaming chat completion

### Agent Management
- `GET /api/agents` - List available agents
- `GET /api/agents/{agent_type}` - Get agent details

### System
- `GET /health` - Health check
- `GET /api/status` - System status
- `GET /api/models` - Available models

## 🤖 Available Agents

### General Assistant
- **Purpose**: General knowledge and conversation
- **Model**: GPT-4
- **Best for**: General questions, casual conversation

### Research Specialist
- **Purpose**: In-depth research and analysis
- **Model**: GPT-4
- **Best for**: Academic research, data analysis, detailed explanations

### Code Assistant
- **Purpose**: Programming and development help
- **Model**: GPT-4
- **Best for**: Code review, debugging, architecture advice

## ⌨️ Keyboard Shortcuts

- `Enter` - Send message
- `Shift + Enter` - New line in message
- `Ctrl/Cmd + Enter` - Force send message
- `Escape` - Focus on input field
- `Ctrl/Cmd + K` - Start new chat

## 🎨 Customization

### Theme Colors
The interface uses CSS custom properties for easy theming:

```css
--primary-bg: #0f0f0f        /* Main background */
--secondary-bg: #1a1a1a      /* Sidebar background */
--accent-primary: #00d4ff    /* Primary accent */
--accent-secondary: #7c3aed  /* Secondary accent */
--text-primary: #ffffff      /* Main text */
--text-secondary: #a1a1aa    /* Secondary text */
```

### Adding New Agents
Edit the `AGENT_CONFIGS` in `main.py`:

```python
AGENT_CONFIGS = {
    "your_agent": {
        "name": "Your Agent Name",
        "system_prompt": "Your system prompt here...",
        "model": "gpt-4"
    }
}
```

## 📱 Mobile Support

The interface is fully responsive with:
- Collapsible sidebar navigation
- Touch-friendly controls
- Optimized typography for mobile
- Gesture-friendly interactions

## 🔧 Configuration

### Environment Variables
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8000)
- `DEBUG` - Debug mode (default: true)
- `OPENAI_API_KEY` - Your OpenAI API key (required)

### Model Configuration
Models can be configured per agent in the `AGENT_CONFIGS` dictionary. Supported models:
- `gpt-4`
- `gpt-4-turbo`
- `gpt-3.5-turbo`
- `gpt-3.5-turbo-16k`

## 🐛 Troubleshooting

### Common Issues

**ImportError: No module named 'fastapi'**
```bash
pip install -r requirements.txt
```

**OpenAI API Key Error**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Port Already in Use**
```bash
export PORT=8001  # Use different port
python main.py
```

**Frontend Not Loading**
- Check if the server is running on the correct port
- Verify the static files are being served correctly
- Check browser console for JavaScript errors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for the GPT-4 API
- FastAPI for the excellent framework
- Font Awesome for the icons
- Google Fonts for the Inter typeface

---

**Made with ❤️ by the Jarvis AI team**
