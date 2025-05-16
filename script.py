import os
import logging
import requests
import time
import random
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

# Hardcoded API tokens and IDs
WASENDER_API_TOKEN = "84fc8b481b469bea9e321894b5acc3fe9efdd14ad3ab858c152a811559d46938"
OPENAI_API_KEY = "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"
ASSISTANT_ID = "asst_wBPPsAtj5GX3tWmaMKZ1JTQl"

# Still load any other environment variables that might be needed
load_dotenv()

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory for storing conversations
CONVERSATIONS_DIR = 'conversations'
if not os.path.exists(CONVERSATIONS_DIR):
    os.makedirs(CONVERSATIONS_DIR)
    logging.info(f"Created conversations directory at {CONVERSATIONS_DIR}")

# API endpoint for sending WhatsApp messages
WASENDER_API_URL = "https://wasenderapi.com/api/send-message"

# Log that we're using hardcoded values
logging.info("Using hardcoded credentials:")
logging.info(f"WASENDER_API_TOKEN: {WASENDER_API_TOKEN[:4]}...{WASENDER_API_TOKEN[-4:]}")
logging.info(f"OPENAI_API_KEY: {OPENAI_API_KEY[:4]}...{OPENAI_API_KEY[-4:]}")
logging.info(f"ASSISTANT_ID: {ASSISTANT_ID}")

# Initialize OpenAI client with hardcoded API key
client = OpenAI(api_key=OPENAI_API_KEY)

# Verify the assistant actually exists
try:
    assistant = client.beta.assistants.retrieve(assistant_id=ASSISTANT_ID)
    logging.info(f"Successfully connected to Assistant: {assistant.name}")
except Exception as e:
    logging.error(f"Error retrieving assistant with ID {ASSISTANT_ID}: {e}")
    logging.error("The application might not work correctly with this assistant ID.")

@app.errorhandler(Exception)
def handle_global_exception(e):
    """Global handler for unhandled exceptions."""
    logging.error(f"Unhandled Exception: {e}", exc_info=True)
    return jsonify(status='error', message='An internal server error occurred.'), 500

# --- Load Persona ---
PERSONA_FILE_PATH = 'persona.json'
PERSONA_DESCRIPTION = "You are a helpful assistant." # Default persona
PERSONA_NAME = "Assistant"
BASE_PROMPT = "You are a helpful and concise AI assistant replying in a WhatsApp chat. Do not use Markdown formatting. Keep your answers short, friendly, and easy to read. If your response is longer than 3 lines, split it into multiple messages using \n every 3 lines. Each \n means a new WhatsApp message. Avoid long paragraphs or unnecessary explanations."

try:
    with open(PERSONA_FILE_PATH, 'r') as f:
        persona_data = json.load(f)
        custom_description = persona_data.get('description', PERSONA_DESCRIPTION)
        base_prompt = persona_data.get('base_prompt', BASE_PROMPT)
        PERSONA_DESCRIPTION = f"{base_prompt}\n\n{custom_description}"
        PERSONA_NAME = persona_data.get('name', PERSONA_NAME)
    logging.info(f"Successfully loaded persona: {PERSONA_NAME}")
except FileNotFoundError:
    logging.warning(f"Persona file not found at {PERSONA_FILE_PATH}. Using default persona.")
except json.JSONDecodeError:
    logging.error(f"Error decoding JSON from {PERSONA_FILE_PATH}. Using default persona.")
except Exception as e:
    logging.error(f"An unexpected error occurred while loading persona: {e}. Using default persona.")
# --- End Load Persona ---

def load_conversation_thread_id(user_id):
    """Loads the OpenAI thread ID for a given user_id."""
    file_path = os.path.join(CONVERSATIONS_DIR, f"{user_id}_thread.json")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data.get('thread_id')
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {file_path}.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error loading thread ID from {file_path}: {e}")
        return None

def save_conversation_thread_id(user_id, thread_id):
    """Saves the OpenAI thread ID for a given user_id."""
    file_path = os.path.join(CONVERSATIONS_DIR, f"{user_id}_thread.json")
    try:
        with open(file_path, 'w') as f:
            json.dump({'thread_id': thread_id}, f)
    except Exception as e:
        logging.error(f"Error saving thread ID to {file_path}: {e}")

def split_message(text, max_lines=3, max_chars_per_line=100):
    """Split a long message into smaller chunks for better WhatsApp readability."""
    # First split by existing newlines
    paragraphs = text.split('\\n')
    chunks = []
    current_chunk = []
    current_line_count = 0
    
    for paragraph in paragraphs:
        # Split long paragraphs into smaller lines
        if len(paragraph) > max_chars_per_line:
            words = paragraph.split()
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= max_chars_per_line:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    if current_line_count >= max_lines:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                        current_line_count = 0
                    current_chunk.append(' '.join(current_line))
                    current_line_count += 1
                    current_line = [word]
                    current_length = len(word)
            
            if current_line:
                if current_line_count >= max_lines:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_line_count = 0
                current_chunk.append(' '.join(current_line))
                current_line_count += 1
        else:
            if current_line_count >= max_lines:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_line_count = 0
            current_chunk.append(paragraph)
            current_line_count += 1
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def get_openai_assistant_response(message_text, user_id):
    """Generates a response from OpenAI Assistant API."""
    try:
        logging.info(f"Getting response from OpenAI Assistant for user {user_id}")
        
        # Get existing thread ID or create a new one
        thread_id = load_conversation_thread_id(user_id)
        
        if not thread_id:
            # Create a new thread
            logging.info(f"Creating new thread for user {user_id}")
            thread = client.beta.threads.create()
            thread_id = thread.id
            save_conversation_thread_id(user_id, thread_id)
            logging.info(f"Created new thread {thread_id} for user {user_id}")
        else:
            # Verify the thread exists and is accessible
            try:
                # Try to retrieve the thread to verify it exists
                client.beta.threads.retrieve(thread_id=thread_id)
                logging.info(f"Using existing thread {thread_id} for user {user_id}")
            except Exception as e:
                logging.warning(f"Thread {thread_id} for user {user_id} not found or inaccessible: {e}")
                # Create a new thread since the existing one couldn't be accessed
                thread = client.beta.threads.create()
                thread_id = thread.id
                save_conversation_thread_id(user_id, thread_id)
                logging.info(f"Created replacement thread {thread_id} for user {user_id}")
        
        # Add the user message to the thread
        logging.info(f"Adding message to thread {thread_id}: {message_text[:50]}...")
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message_text
        )
        logging.info(f"Added message {message.id} to thread {thread_id}")
        
        # Run the assistant on the thread
        logging.info(f"Running assistant {ASSISTANT_ID} on thread {thread_id}")
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID,
            instructions=PERSONA_DESCRIPTION
        )
        logging.info(f"Created run {run.id} for thread {thread_id}")
        
        # Poll for the run to complete
        max_retries = 30  # Maximum number of retries (30 seconds)
        retries = 0
        while retries < max_retries:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            logging.info(f"Run {run.id} status: {run_status.status}")
            
            if run_status.status == 'completed':
                logging.info(f"Run {run.id} completed successfully")
                break
            elif run_status.status in ['failed', 'expired', 'cancelled']:
                logging.error(f"Assistant run {run.id} failed with status: {run_status.status}")
                if hasattr(run_status, 'last_error'):
                    logging.error(f"Error details: {run_status.last_error}")
                return "Sorry, I encountered an issue while processing your message. Please try again."
            
            retries += 1
            time.sleep(1)  # Wait before polling again
        
        if retries >= max_retries:
            logging.warning(f"Run {run.id} did not complete within the timeout period")
            return "I'm still thinking about your message. Please try again in a moment."
        
        # Get the assistant's response
        logging.info(f"Retrieving messages from thread {thread_id}")
        messages = client.beta.threads.messages.list(
            thread_id=thread_id
        )
        
        # Get the most recent assistant message
        for message in messages.data:
            if message.role == "assistant":
                # Extract the text content from the message
                message_content = ""
                for content_part in message.content:
                    if content_part.type == "text":
                        message_content += content_part.text.value
                
                logging.info(f"Retrieved assistant response: {message_content[:50]}...")
                return message_content.strip()
        
        logging.warning(f"No assistant message found in thread {thread_id}")
        return "I'm not sure how to respond to that."

    except Exception as e:
        logging.error(f"Error calling OpenAI Assistant API: {e}", exc_info=True)
        return "I'm having trouble processing your request. Please try again later."

def send_whatsapp_message(recipient_number, message_content, message_type='text', media_url=None):
    """Sends a message via WaSenderAPI. Supports text and media messages."""
    # Use the hardcoded token directly
    token = WASENDER_API_TOKEN
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    # Log the token being used (partially masked for security)
    token_preview = token[:4] + "..." + token[-4:] if len(token) > 8 else "***" 
    logging.info(f"Using WaSender API token: {token_preview}")
    
    # Sanitize recipient_number to remove "@s.whatsapp.net"
    if recipient_number and "@s.whatsapp.net" in recipient_number:
        formatted_recipient_number = recipient_number.split('@')[0]
    else:
        formatted_recipient_number = recipient_number

    payload = {
        'to': formatted_recipient_number
    }

    if message_type == 'text':
        payload['text'] = message_content
    elif message_type == 'image' and media_url:
        payload['imageUrl'] = media_url
        if message_content:
            payload['text'] = message_content 
    elif message_type == 'video' and media_url:
        payload['videoUrl'] = media_url
        if message_content:
            payload['text'] = message_content
    elif message_type == 'audio' and media_url:
        payload['audioUrl'] = media_url
    elif message_type == 'document' and media_url:
        payload['documentUrl'] = media_url
        if message_content:
            payload['text'] = message_content
    else:
        if message_type != 'text':
             logging.error(f"Media URL is required for message type '{message_type}'.")
             return False
        logging.error(f"Unsupported message type or missing content/media_url: {message_type}")
        return False
    
    logging.debug(f"Attempting to send WhatsApp message. Payload: {payload}")

    try:
        response = requests.post(WASENDER_API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        logging.info(f"Message sent to {recipient_number}. Response: {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        status_code = getattr(e.response, 'status_code', 'N/A')
        response_text = getattr(e.response, 'text', 'N/A')
        logging.error(f"Error sending WhatsApp message to {recipient_number} (Status: {status_code}): {e}. Response: {response_text}")
        if status_code == 422:
            logging.error("WaSenderAPI 422 Error: This often means an issue with the payload (e.g., device_id, 'to' format, or message content/URL). Check the payload logged above and WaSenderAPI docs.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while sending WhatsApp message: {e}")
        return False

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handles incoming WhatsApp messages via webhook."""
    data = request.json
    logging.info(f"Received webhook data (first 200 chars): {str(data)[:200]}")

    try:
        if data.get('event') == 'messages.upsert' and data.get('data') and data['data'].get('messages'):
            message_info = data['data']['messages']
            
            # Check if it's a message sent by the bot itself
            if message_info.get('key', {}).get('fromMe'):
                logging.info(f"Ignoring self-sent message: {message_info.get('key', {}).get('id')}")
                return jsonify({'status': 'success', 'message': 'Self-sent message ignored'}), 200

            sender_number = message_info.get('key', {}).get('remoteJid')
            
            incoming_message_text = None
            message_type = 'unknown'

            # Extract message content based on message structure
            if message_info.get('message'):
                msg_content_obj = message_info['message']
                if 'conversation' in msg_content_obj:
                    incoming_message_text = msg_content_obj['conversation']
                    message_type = 'text'
                elif 'extendedTextMessage' in msg_content_obj and 'text' in msg_content_obj['extendedTextMessage']:
                    incoming_message_text = msg_content_obj['extendedTextMessage']['text']
                    message_type = 'text'

            if message_info.get('messageStubType'):
                stub_params = message_info.get('messageStubParameters', [])
                logging.info(f"Received system message of type {message_info['messageStubType']} from {sender_number}. Stub params: {stub_params}")
                return jsonify({'status': 'success', 'message': 'System message processed'}), 200

            if not sender_number:
                logging.warning("Webhook received message without sender information.")
                return jsonify({'status': 'error', 'message': 'Incomplete sender data'}), 400

            # Sanitize sender_number to use as a filename/user_id
            safe_sender_id = "".join(c if c.isalnum() else '_' for c in sender_number)

            if message_type == 'text' and incoming_message_text:
                logging.info(f"Processing text message from {sender_number} ({safe_sender_id}): {incoming_message_text}")
                
                # Get OpenAI Assistant's reply
                assistant_reply = get_openai_assistant_response(incoming_message_text, safe_sender_id)
                
                if assistant_reply:
                    # Split the response into chunks and send them sequentially
                    message_chunks = split_message(assistant_reply)
                    for i, chunk in enumerate(message_chunks):
                        if not send_whatsapp_message(sender_number, chunk, message_type='text'):
                            logging.error(f"Failed to send message chunk to {sender_number}")
                            break
                        # Delay between messages
                        if i < len(message_chunks) - 1:
                            delay = random.uniform(0.55, 1.5)
                            time.sleep(delay)
            elif incoming_message_text:
                logging.info(f"Received '{message_type}' message from {sender_number}. No text content. Full data: {message_info}")
            elif message_type != 'unknown':
                 logging.info(f"Received '{message_type}' message from {sender_number}. No text content. Full data: {message_info}")
            else:
                logging.warning(f"Received unhandled or incomplete message from {sender_number}. Data: {message_info}")
        elif data.get('event'):
            logging.info(f"Received event '{data.get('event')}' which is not 'messages.upsert'. Data: {str(data)[:200]}")

        return jsonify({'status': 'success'}), 200
    except Exception as e:
        logging.error(f"Error processing webhook: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    # For development with webhook testing via ngrok
    app.run(debug=True, port=5001, host='0.0.0.0')