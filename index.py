from dotenv import load_dotenv
import os
import google.generativeai as genai
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Load environment variables and configure API
load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

def create_system_instructions():
    return """You are BookGuide, an expert literary advisor. You provide helpful guidance about:
    - Book recommendations based on interests and reading level
    - Author backgrounds and writing styles
    - Genre exploration and literary analysis
    - Reading tips and comprehension strategies
    - Library resources and book-finding help
    
    Always strive to:
    - Provide specific, actionable recommendations
    - Include brief context about suggested books/authors
    - Respect reading levels and content sensitivities
    - Encourage a love of reading and literary exploration
    
    If unsure about a recommendation or detail,
    acknowledge uncertainty and suggest consulting a librarian
    or verified book database.
    
    Keep responses engaging, informative, and focused on fostering 
    a positive reading experience.
    
    Do not diverge from the topic of books or reading."""

def get_few_shot_examples():
    return [
        {"role": "user", "parts": "Can you recommend a good mystery book?"},
        {"role": "model", "parts": """Here are two excellent mystery recommendations:

1. 'The Thursday Murder Club' by Richard Osman
- A charming yet clever mystery featuring four retirees who solve cold cases
- Perfect for readers who enjoy wit mixed with their mysteries
- Suitable for adult readers, especially those who appreciate British humor

2. 'The 7½ Deaths of Evelyn Hardcastle' by Stuart Turton
- An innovative mystery with a time-loop twist
- Ideal for readers who enjoy complex plots and unique storytelling
- Best for experienced mystery readers

Would you like more details about either of these books?"""},
    ]

def classify_topic(user_input: str) -> str:
    """Classifies user input into specific book-related topics"""
    keywords = {
        'recommend': 'recommendations',
        'suggest': 'recommendations',
        'author': 'author_info',
        'write': 'author_info',
        'genre': 'genre_exploration',
        'type': 'genre_exploration',
        'understand': 'comprehension',
        'meaning': 'comprehension',
        'library': 'resources',
        'find': 'resources'
    }
    
    user_input = user_input.lower()
    for key, topic in keywords.items():
        if key in user_input:
            return topic
    return 'general'

def load_conversation_history() -> Dict:
    """Loads previous conversations from storage"""
    try:
        with open('conversation_history.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return empty dict if file doesn't exist or is corrupted
        return {}

def save_conversation(session_id: str, history: List) -> None:
    """Saves conversation history to storage"""
    conversations = load_conversation_history()
    
    # Convert history to serializable format
    serializable_history = []
    for message in history:
        # Handle both dictionary and Content object formats
        if isinstance(message, dict):
            role = message['role']
            parts = message['parts']
        else:
            # For Content objects from Gemini API
            role = 'model' if hasattr(message, 'role') else 'user'
            parts = str(message)
            
        serializable_message = {
            'role': role,
            'parts': parts
        }
        serializable_history.append(serializable_message)
    
    conversations[session_id] = {
        'timestamp': datetime.now().isoformat(),
        'history': serializable_history
    }
    
    with open('conversation_history.json', 'w') as f:
        json.dump(conversations, f, indent=2)

def summarize_conversation(history: List) -> str:
    """Creates a summary of the conversation"""
    # Extract user questions and key points
    questions = [msg['parts'] for msg in history if msg['role'] == 'user'][1:]  # Skip system prompt
    return f"Discussion covered: {', '.join(questions[:3])}..."

def gemini_api_query(user_input: str, chat_history: Optional[List] = None) -> Tuple[str, List]:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        if not chat_history:
            # Start new chat with system instructions and topic classification
            topic = classify_topic(user_input)
            system_prompt = create_system_instructions() + f"\nFocus on {topic} aspects in your response."
            
            chat = model.start_chat(history=[
                {"role": "user", "parts": system_prompt},
                {"role": "model", "parts": "I understand my role as BookGuide. How can I help with your questions about books or reading?"},
            ] + get_few_shot_examples())
        else:
            chat = model.start_chat(history=chat_history)
        
        response = chat.send_message(user_input)
        return response.text, chat.history
        
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again.", chat_history

def main():
    print("📚 Welcome to BookGuide! I'm here to help with all your book-related questions.")
    print("Type 'exit' to end our conversation.\n")
    print("Commands: 'history' to view conversation summary, 'save' to store conversation\n")
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    chat_history = None
    
    while True:
        user_input = input("\nYour book or reading related question (Type 'exit' to end): ").strip()
        
        if user_input.lower() == 'exit':
            if chat_history:
                save_conversation(session_id, chat_history)
            print("\nThank you for chatting with BookGuide! Happy reading! 📚")
            break
            
        if user_input.lower() == 'history':
            if chat_history:
                print("\nConversation Summary:", summarize_conversation(chat_history))
            continue
            
        if user_input.lower() == 'save':
            if chat_history:
                save_conversation(session_id, chat_history)
                print("\nConversation saved!")
            continue
            
        if not user_input:
            print("Please ask a book-related question!: ")
            continue
            
        response, chat_history = gemini_api_query(user_input, chat_history)
        print("\nBookGuide:", response)

if __name__ == "__main__":
    main()