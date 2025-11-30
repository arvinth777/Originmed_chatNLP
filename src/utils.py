import os
import re
import json
import logging
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def call_llm(prompt: str, system_message: str = "", temperature: float = 0.0, max_tokens: int = 800) -> Optional[str]:
    """
    Calls the Google Gemini LLM with the given prompt and system message.
    """
    try:
        # Using gemini-2.0-flash-lite for better rate limits (30 RPM vs 10 RPM)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        messages = []
        if system_message:
            messages.append(("system", system_message))
        messages.append(("human", prompt))
        
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None

def emergency_pii_removal(text: str) -> str:
    """
    Fallback regex-based PII removal.
    """
    # Simple regex patterns for common PII
    # Phone numbers (US format mostly)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[CONTACT_INFO]', text)
    # Emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Dates (MM/DD/YYYY or YYYY-MM-DD)
    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[DATE]', text)
    text = re.sub(r'\b\d{4}-\d{1,2}-\d{1,2}\b', '[DATE]', text)
    
    # Names are difficult to do reliably with regex without NLP libraries like Spacy
    # We will assume the LLM does the heavy lifting, this is just a safety net for obvious patterns
    return text

def append_jsonl(filepath: str, data: dict):
    """
    Appends a dictionary as a JSON line to the specified file.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "a") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        logger.error(f"Failed to write to log file {filepath}: {e}")
