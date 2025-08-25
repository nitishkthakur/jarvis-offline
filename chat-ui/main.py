# ===========================================
# JARVIS AI - FASTAPI BACKEND
# ===========================================

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncGenerator
import json
import asyncio
import time
import uvicorn
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import our OpenAI client
sys.path.append(str(Path(__file__).parent.parent))
from openai_client import OpenAIClient 
from ollama_client import OllamaClient
from deep_research_agents import deep_research_clarification_agent

provider = 'ollama'
if provider == "openai":
    Client  = OpenAIClient
    model1 = 'gpt-5-nano'
    model2 = 'gpt-5-mini'

else:
    Client = OllamaClient
    model1 = "qwen3:8b"
    model2 = "mistral-small3.2:latest"

# ===========================================
# FASTAPI APP INITIALIZATION
# ===========================================

app = FastAPI(
    title="Jarvis AI Chat API",
    description="A ChatGPT-like interface with streaming and non-streaming endpoints",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
static_path = Path(__file__).parent / "static"
templates_path = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=templates_path)

# Initialize the selected client
if provider == "openai":
    client = Client()
else:
    client = Client(role="chat_assistant")



# ===========================================
# PYDANTIC MODELS
# ===========================================

class ChatRequest(BaseModel):
    message: str
    agent: Optional[str] = "general"
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 2000
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    agent: str
    timestamp: float
    token_count: Optional[int] = None
    processing_time: Optional[float] = None

class StreamChunk(BaseModel):
    content: str
    timestamp: float
    done: bool = False

# ===========================================
# AGENT CONFIGURATIONS
# ===========================================

AGENT_CONFIGS = {
    "general": {
        "name": "General Assistant",
        "system_prompt": """You are Jarvis, a highly intelligent and helpful AI assistant. 
        You are knowledgeable, professional, and always aim to provide accurate and useful information. 
        Be conversational but precise in your responses.""",
        "model": model1
    },
    "deep-research": {
        "name": "Deep Research", 
        "system_prompt": """You are Jarvis, a specialized research assistant with expertise in 
        academic research, data analysis, and information synthesis. Provide detailed, well-sourced 
        responses with critical analysis and multiple perspectives when appropriate.""",
        "model": model2
    },
    "coding": {
        "name": "Code Assistant",
        "system_prompt": """You are Jarvis, a senior software engineer and coding assistant. 
        You excel at explaining programming concepts, debugging code, suggesting best practices, 
        and helping with software architecture decisions. Provide clear, practical coding solutions.""",
        "model": model1
    }
}

# ===========================================
# UTILITY FUNCTIONS
# ===========================================

def get_agent_config(agent_type: str) -> Dict[str, Any]:
    """Get configuration for a specific agent type."""
    return AGENT_CONFIGS.get(agent_type, AGENT_CONFIGS["general"])

def estimate_tokens(text: str) -> int:
    """Estimate token count for a given text."""
    return int(len(text.split()) * 1.3)  # Rough estimation

def format_agent_prompt(agent_config: Dict[str, Any], user_message: str) -> str:
    """Format the complete prompt for an agent."""
    return f"{agent_config['system_prompt']}\n\nUser: {user_message}\n\nAssistant:"

# ===========================================
# MAIN ROUTES
# ===========================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

# ===========================================
# CHAT ENDPOINTS
# ===========================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat_non_streaming(request: ChatRequest):
    """Non-streaming chat endpoint."""
    start_time = time.time()
    
    try:
        # Get agent configuration
        print(request.agent)
        if request.agent == "deep-research":
            print("here")
            print(request.message)
            clarifying_questions = deep_research_clarification_agent(request.message, model=model2, client=OllamaClient)
            processing_time = time.time() - start_time
            token_count = estimate_tokens(clarifying_questions)
            print("tokens: ", token_count)
            return ChatResponse(
                response=clarifying_questions,
                agent=request.agent,
                timestamp=time.time(),
                token_count=token_count,
                processing_time=processing_time
            )
        print(request)
        agent_config = get_agent_config(request.agent)
        print(agent_config)
        # Prepare the prompt
        prompt = format_agent_prompt(agent_config, request.message)
        print(prompt)

        # Call API (non-streaming)
        response_data = client.invoke(
            query=prompt,
            model_name=model2,
        )
        
        # Extract the text response from the Ollama client response
        response = response_data.get("text", "") if isinstance(response_data, dict) else str(response_data)
        print(response, type(response))
        processing_time = time.time() - start_time
        token_count = estimate_tokens(response)
        print("tokens: ", token_count)
        return ChatResponse(
            response=response,
            agent=request.agent,
            timestamp=time.time(),
            token_count=token_count,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/api/chat/stream")
async def chat_streaming(request: ChatRequest):
    """Streaming chat endpoint."""
    try:
        # Get agent configuration
        agent_config = get_agent_config(request.agent)
        print(agent_config)

        # Configure the deep research behaviour
        if agent_config['name'] == 'Deep Research':
            # Specific configuration for deep research
            pass
            
        # Prepare the prompt
        prompt = format_agent_prompt(agent_config, request.message)
        
        async def generate_stream():
            """Generate streaming response."""
            try:
                client = Client(role="ClarifyingQuestions")
                # Call API (streaming)
                stream = client.invoke_streaming(
                    query=prompt,
                    model_name=agent_config["model"],
                )
                
                # Process streaming response
                for chunk in stream:
                    if chunk:
                        chunk_data = {
                            "content": chunk,
                            "timestamp": time.time(),
                            "done": False
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                        await asyncio.sleep(0.01)  # Small delay for better UX
                
                # Send completion signal
                final_data = {
                    "content": "",
                    "timestamp": time.time(),
                    "done": True
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                error_data = {
                    "error": str(e),
                    "timestamp": time.time(),
                    "done": True
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing streaming request: {str(e)}")

# ===========================================
# AGENT MANAGEMENT ENDPOINTS
# ===========================================

@app.get("/api/agents")
async def get_agents():
    """Get available agents and their configurations."""
    agents = {}
    for agent_type, config in AGENT_CONFIGS.items():
        agents[agent_type] = {
            "name": config["name"],
            "model": config["model"],
            "description": config["system_prompt"][:100] + "..."  # Truncated description
        }
    return {"agents": agents}

@app.get("/api/agents/{agent_type}")
async def get_agent_details(agent_type: str):
    """Get detailed information about a specific agent."""
    if agent_type not in AGENT_CONFIGS:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    config = AGENT_CONFIGS[agent_type]
    return {
        "agent_type": agent_type,
        "name": config["name"],
        "model": config["model"],
        "system_prompt": config["system_prompt"]
    }

# ===========================================
# CONVERSATION MANAGEMENT ENDPOINTS
# ===========================================

@app.post("/api/conversation/save")
async def save_conversation(conversation_data: Dict[str, Any]):
    """Save conversation data (placeholder implementation)."""
    # In a real implementation, you would save to a database
    # For now, we'll just return a success response
    return {
        "status": "saved",
        "conversation_id": f"conv_{int(time.time())}",
        "timestamp": time.time()
    }

@app.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Retrieve a saved conversation (placeholder implementation)."""
    # In a real implementation, you would fetch from a database
    return {
        "conversation_id": conversation_id,
        "messages": [],
        "created_at": time.time(),
        "updated_at": time.time()
    }

@app.delete("/api/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a saved conversation (placeholder implementation)."""
    return {
        "status": "deleted",
        "conversation_id": conversation_id,
        "timestamp": time.time()
    }

# ===========================================
# SYSTEM STATUS ENDPOINTS
# ===========================================

@app.get("/api/status")
async def get_system_status():
    """Get system status and statistics."""
    return {
        "status": "operational",
        "openai_client": "connected",
        "agents_available": len(AGENT_CONFIGS),
        "uptime": time.time(),
        "version": "1.0.0"
    }

@app.get("/api/models")
async def get_available_models():
    """Get available OpenAI models."""
    # This would typically call the OpenAI API to get available models
    # For now, we'll return the models we know are available
    return {
        "models": [
            "gpt-5-nano",
            "gpt-5-mini",
        ],
        "default_model": "gpt-5-nano"
    }

# ===========================================
# ERROR HANDLERS
# ===========================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper error responses."""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": time.time()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "status_code": 500,
        "timestamp": time.time()
    }

# ===========================================
# MIDDLEWARE
# ===========================================

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# ===========================================
# MAIN APPLICATION ENTRY POINT
# ===========================================

if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    print(f"""
    ===============================================
    🤖 JARVIS AI CHAT SERVER
    ===============================================
    
    Server starting on: http://{host}:{port}
    Debug mode: {debug}
    Provider: {provider}
    Available Agents: {len(AGENT_CONFIGS)}
    
    API Endpoints:
    • GET  /                    - Chat Interface
    • POST /api/chat            - Non-streaming Chat
    • POST /api/chat/stream     - Streaming Chat
    • GET  /api/agents          - Available Agents
    • GET  /api/status          - System Status
    • GET  /health              - Health Check
    
    ===============================================
    """)
    
    # Run the application
    if debug:
        # In debug mode, use reload which requires import string
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=True,
            log_level="info"
        )
    else:
        # In production mode, use app object directly
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,
            log_level="warning"
        )
