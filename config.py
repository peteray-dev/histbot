# no database yet, once there is database, one can extract the chatbot name using data ingestion from the database instead
# of saving it in this side 
import os

CHATBOT_NAME_FILE = "chatbot_name.txt"

def get_chatbot_name():
    """Retrieve the chatbot name from a file."""
    if os.path.exists(CHATBOT_NAME_FILE):
        with open(CHATBOT_NAME_FILE, "r", encoding="utf-8") as file:
            return file.read().strip()
    return "DefaultBot"

def set_chatbot_name(name):
    """Store the chatbot name in a file."""
    with open(CHATBOT_NAME_FILE, "w", encoding="utf-8") as file:
        file.write(name)

# Initialize the chatbot name
CHATBOT_NAME = get_chatbot_name()
