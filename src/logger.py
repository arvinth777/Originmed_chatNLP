import logging
import os
from datetime import datetime

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logger
logger = logging.getLogger("ClinicalPipeline")
logger.setLevel(logging.INFO)

# File Handler
log_file = f"logs/api_calls_{datetime.now().strftime('%Y%m%d')}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler
if not logger.handlers:
    logger.addHandler(file_handler)

def log_api_call(agent_name: str, input_data, output_data):
    """
    Logs the details of an LLM API call.
    """
    logger.info(f"--- AGENT: {agent_name} ---")
    input_str = str(input_data) if not isinstance(input_data, str) else input_data
    output_str = str(output_data) if not isinstance(output_data, str) else output_data
    logger.info(f"INPUT: {input_str[:500]}...") # Truncate for readability
    logger.info(f"OUTPUT: {output_str}")
    logger.info("-" * 50)
