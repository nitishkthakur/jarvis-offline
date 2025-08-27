import json
from pydantic import BaseModel

def deep_research_clarification_agent(deep_research_request, model, client):
    
    task = """You are an expert on asking clarifying Questions which will let you perform deep research on the user question. 

    1. Identify 3 different questions to ask the user the answers to which will help you answer the user question.
    2. keep the questions short and concise.
    3. The questions should clarify what the user wants to research / learn / understand / plan out.
    4. Once the user answers the questions, use their responses to guide your research and provide more tailored information.

    For example, if the user asks: 
    Example 1:
    User Question: Help me prepare an itinerary for my trip to Japan next month.
    Clarifying questions: 
    1. What specific cities in Japan are you planning to visit?
    2. How many days do you have for your trip?
    3. What type of activities are you interested in (e.g., cultural, outdoor, culinary)?
    4. What is your budget for the trip?

    Example 2:
    User Question: Which city are you moving to?
    Clarifying Questions:
    1. Which city are you moving to?
    2. What’s your reason for the move (e.g., job, studies, lifestyle change)?
    3. What’s your estimated move-in date?
    4. Do you need help finding housing, setting up utilities, understanding the local job market, or anything else?
    5. What is your budget for the move and for monthly living expenses?


    Example 3:
    User Question:
    I want to understand Autoencoders. Help me understand the key architectures, starting from basics to good models

    Clarifying Questions:
    1. What is your primary purpose for learning about autoencoders? (e.g., anomaly detection, dimensionality reduction, data generation, etc.)
    2. What level of detail are you looking for in the architecture explanations? Do you want theoretical grounding, mathematical formulation, and implementation-level details (e.g., PyTorch or TensorFlow)?
    3. Are you interested only in standard autoencoders or also in variants like variational autoencoders (VAEs), denoising autoencoders, sparse autoencoders, etc.?
    4. What is your level of current understanding? (e.g., beginner, intermediate, advanced)
    """

    clarifying_agent = client(role = task, model_name=model, agent_name="ClarifyingAgent")

    class ClarifyingQuestions(BaseModel):
        question_1: str
        question_2: str
        question_3: str


    #clarifying_questions = re_phrase_agent.invoke(query="""Help me plan a birthday party""", json_schema=ClarifyingQuestions)
    clarifying_questions = clarifying_agent.invoke(query=deep_research_request, 
                                                json_schema=ClarifyingQuestions)

    pre_r = json.loads(clarifying_questions['text'])
    r = "To help me answer your query, pleae answer the following questions:\n"

    for i in range(len(pre_r)):
        r += f"{i+1}. {pre_r[f'question_{i+1}']}\n"

    # Context till this node to be passed to other agents
    deep_research_agent_context = clarifying_agent.get_agent_context()
    return deep_research_agent_context, r



