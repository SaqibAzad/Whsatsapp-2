import os
import logging
import requests
import time
import random
import json
import base64
import hashlib
import hmac
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
import io
from PIL import Image

# Hardcoded API tokens and IDs
WASENDER_API_TOKEN = "84fc8b481b469bea9e321894b5acc3fe9efdd14ad3ab858c152a811559d46938"
OPENAI_API_KEY = ""
ASSISTANT_ID = "asst_wBPPsAtj5GX3tWmaMKZ1JTQl"

# Still load any other environment variables that might be needed
load_dotenv()

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory for storing conversations and media
CONVERSATIONS_DIR = 'conversations'
MEDIA_DIR = 'media'

# Create directories if they don't exist
for directory in [CONVERSATIONS_DIR, MEDIA_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory at {directory}")

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

def base64url_decode(data):
    """Decodes a base64url encoded string"""
    return base64.b64decode(data.translate(str.maketrans('-_', '+/')))

def download_file(url):
    """Downloads a file from a URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logging.error(f"Failed to download file from {url}: {e}")
        return None

def get_decryption_keys(media_key, media_type='image', length=112):
    """Generates decryption keys for WhatsApp media"""
    media_key_decoded = base64.b64decode(media_key)
    
    # Map media types to their corresponding info strings
    info_map = {
        'image': 'WhatsApp Image Keys',
        'video': 'WhatsApp Video Keys',
        'audio': 'WhatsApp Audio Keys',
        'document': 'WhatsApp Document Keys',
    }
    
    if media_type not in info_map:
        raise ValueError(f"Invalid media type: {media_type}")
    
    info = info_map[media_type]
    
    # Implement HKDF key derivation
    def hmac_sha256(key, msg):
        return hmac.new(key, msg, hashlib.sha256).digest()
    
    # Extract step
    prk = hmac_sha256(b'\x00' * 32, media_key_decoded)
    
    # Expand step
    t = b""
    okm = b""
    for i in range(1, (length // 32) + 2):
        t = hmac_sha256(prk, t + info.encode('utf-8') + bytes([i]))
        okm += t
    
    return okm[:length]

def decrypt_whatsapp_media(media_key, enc_file, media_type='image'):
    """Decrypts a WhatsApp media file"""
    try:
        # Get decryption keys based on media type
        keys = get_decryption_keys(media_key, media_type)
        iv = keys[:16]
        cipher_key = keys[16:48]
        
        # Remove the last 10 bytes (MAC) from the encrypted file
        ciphertext = enc_file[:-10]
        
        # Decrypt using AES-256-CBC
        cipher = Cipher(algorithms.AES(cipher_key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext
    except Exception as e:
        logging.error(f"Failed to decrypt media: {e}")
        return None

def handle_media_message(message_info):
    """Processes media messages from WhatsApp webhook"""
    message = message_info.get('message', {})
    
    # Determine the media type and extract media information
    media_info = None
    media_type = None
    extension = None
    
    if 'imageMessage' in message:
        media_info = message['imageMessage']
        media_type = 'image'
        extension = '.jpg'
    elif 'videoMessage' in message:
        media_info = message['videoMessage']
        media_type = 'video'
        extension = '.mp4'
    elif 'audioMessage' in message:
        media_info = message['audioMessage']
        media_type = 'audio'
        extension = '.ogg'
    elif 'documentMessage' in message:
        media_info = message['documentMessage']
        media_type = 'document'
        extension = ''  # Use the original extension if available
    else:
        # Not a media message
        return None, None
    
    # Extract required information
    media_key = media_info.get('mediaKey')
    url = media_info.get('url')
    message_id = message_info.get('key', {}).get('id', 'unknown')
    caption = media_info.get('caption', '')
    
    if not media_key or not url:
        logging.error(f"Missing media key or URL for message {message_id}")
        return None, caption
    
    # Create a unique filename
    output_path = os.path.join(MEDIA_DIR, f"{message_id}{extension}")
    
    try:
        # Download the encrypted file
        enc_file = download_file(url)
        if not enc_file:
            return None, caption
        
        # Decrypt the file
        decrypted_file = decrypt_whatsapp_media(media_key, enc_file, media_type)
        if not decrypted_file:
            return None, caption
        
        # Save the decrypted file
        with open(output_path, 'wb') as f:
            f.write(decrypted_file)
        
        logging.info(f"Successfully saved decrypted {media_type} to {output_path}")
        return output_path, caption
    except Exception as e:
        logging.error(f"Error handling media message: {e}")
        return None, caption

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

def get_openai_assistant_response(message_text, user_id, image_path=None):
    """Generates a response from OpenAI Assistant API with optional image."""
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
        if image_path:
            # For image messages, send both the image and any caption
            logging.info(f"Adding message with image to thread {thread_id}: {image_path}")
            
            # Prepare the image file
            with open(image_path, "rb") as image_file:
                file = client.files.create(
                    file=image_file,
                    purpose="assistants"
                )
                
            # Add message with image attachment - FIXED CODE HERE
            # Create message content with both text and image file reference
            message_content = []
            
            # Add text content if there's a message/caption
            if message_text:
                message_content.append({
                    "type": "text", 
                    "text": message_text
                })
            
            # Add image file reference
            message_content.append({
                "type": "image_file",
                "image_file": {
                    "file_id": file.id
                }
            })
                
            # Create the message with the content array
            message = client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=message_content
            )
            logging.info(f"Added message with image {file.id} to thread {thread_id}")
        else:
            # For text-only messages
            logging.info(f"Adding text message to thread {thread_id}: {message_text[:50]}...")
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

def send_whatsapp_message(recipient_number, message_content, message_type='text'):
    """Sends a message via WaSenderAPI. Supports text messages only."""
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

    # Only support text messages
    if message_type != 'text':
        logging.error(f"Unsupported message type: {message_type}. Only text messages are supported.")
        return False

    payload = {
        'to': formatted_recipient_number,
        'text': message_content
    }
    
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

# Authorized phone number for testing
AUTHORIZED_NUMBER = "923165893850"

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
            message_id = message_info.get('key', {}).get('id', f"msg_{int(time.time())}")
            
            # Check if the sender is the authorized number
            # Extract the phone number from the JID format (may be number@s.whatsapp.net)
            if sender_number:
                phone_number = sender_number.split('@')[0] if '@' in sender_number else sender_number
                
                # Log all incoming messages but only process authorized number
                if phone_number != AUTHORIZED_NUMBER:
                    logging.info(f"Ignoring message from unauthorized number: {phone_number}")
                    return jsonify({'status': 'success', 'message': 'Unauthorized sender ignored'}), 200
                else:
                    logging.info(f"Processing message from authorized number: {phone_number}")
            else:
                logging.warning("Webhook received message without sender information.")
                return jsonify({'status': 'error', 'message': 'Incomplete sender data'}), 400
            
            # Create a safe version of the sender ID for thread management
            safe_sender_id = phone_number.replace('+', '').replace(' ', '')
            
            # Check if this is a media message
            media_path = None
            caption = None
            incoming_message_text = None
            
            if message_info.get('message'):
                msg_content_obj = message_info['message']
                
                # First try to handle as media message
                media_path, caption = handle_media_message(message_info)
                
                # If it's not a media message, handle as text
                if not media_path:
                    # Handle text messages
                    if 'conversation' in msg_content_obj:
                        incoming_message_text = msg_content_obj['conversation']
                    elif 'extendedTextMessage' in msg_content_obj and 'text' in msg_content_obj['extendedTextMessage']:
                        incoming_message_text = msg_content_obj['extendedTextMessage']['text']
                    else:
                        # Unsupported message type
                        logging.info("Received unsupported message type. Ignoring.")
                        send_whatsapp_message(
                            sender_number,
                            "I can only process text and media messages at this time.",
                            message_type='text'
                        )
                        return jsonify({'status': 'success', 'message': 'Unsupported message type ignored'}), 200
                else:
                    # If we have media but no caption, set empty text
                    if not caption:
                        incoming_message_text = "Please describe this image."
                    else:
                        incoming_message_text = caption
                
                # Process with OpenAI
                if media_path or incoming_message_text:
                    logging.info(f"Processing {'media with caption' if media_path else 'text message'}: {incoming_message_text[:50] if incoming_message_text else ''}")
                    
                    # Get AI response
                    assistant_reply = get_openai_assistant_response(
                        incoming_message_text, 
                        safe_sender_id,
                        image_path=media_path
                    )
                    
                    if assistant_reply:
                        # Split and send the response
                        message_chunks = split_message(assistant_reply)
                        for i, chunk in enumerate(message_chunks):
                            send_whatsapp_message(sender_number, chunk, message_type='text')
                            if i < len(message_chunks) - 1:
                                time.sleep(random.uniform(0.5, 1))
                    else:
                        send_whatsapp_message(
                            sender_number, 
                            "I'm not sure how to respond to that. Could you please rephrase?", 
                            message_type='text'
                        )
            
            return jsonify({'status': 'success', 'message': 'Message processed successfully'}), 200
        
        return jsonify({'status': 'success', 'message': 'Non-message event ignored'}), 200
    
    except Exception as e:
        logging.error(f"Error processing webhook: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'Error processing webhook'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
