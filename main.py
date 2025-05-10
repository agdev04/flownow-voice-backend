from fastapi import FastAPI, WebSocket, Depends, HTTPException, status
from starlette.websockets import WebSocketState
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import base64
import json
import asyncio
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from asyncio import Semaphore
import firebase_admin
from firebase_admin import auth, credentials

# Load environment variables from .env file
load_dotenv()

# Get the base64-encoded Firebase Admin key from environment
firebase_base64 = os.getenv('FIREBASE_ADMIN_KEY_B64')

# Decode the base64 string to JSON string
decoded_json = base64.b64decode(firebase_base64).decode('utf-8')

# Parse the JSON string
service_account_info = json.loads(decoded_json)

# Initialize Firebase Admin with the credentials
cred = credentials.Certificate(service_account_info)
firebase_admin.initialize_app(cred)

# Security bearer token
security = HTTPBearer()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = "gemini-2.0-flash-live-001"

config = {
    "system_instruction": types.Content(
        parts=[
            types.Part(
                text=(
                    "You are a friendly, casual meditation guide who helps users with meditation inspired by the Sedona Method.\n"
                    "Speak slowly and gently, as if guiding someone through a peaceful meditation. Your tone should be soothing and calming.\n"
                    "If a user asks something unrelated to emotional well-being, meditation, or life guidance, kindly decline and gently remind them that your purpose is to support emotional release and inner peace.\n"
                    "Your responses should feel like a soft, flowing meditation session—calm, grounded, and supportive."
                )
            )
        ]
    ),
    "response_modalities": ["AUDIO"],
    "context_window_compression" : types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow(),
    ),
    "output_audio_transcription": {}
}

async def async_enumerate(aiterable):
    i = 0
    async for item in aiterable:
        yield i, item
        i += 1

def create_wav_header(pcm_data):
    """Generate WAV header for 16-bit mono PCM at 24kHz"""
    data_size = len(pcm_data)
    header = bytearray(44)
    header[0:4] = b'RIFF'
    header[4:8] = (data_size + 36).to_bytes(4, byteorder='little')
    header[8:12] = b'WAVE'
    header[12:16] = b'fmt '
    header[16:20] = (16).to_bytes(4, byteorder='little')
    header[20:22] = (1).to_bytes(2, byteorder='little')  # PCM format
    header[22:24] = (1).to_bytes(2, byteorder='little')  # Mono
    header[24:28] = (24000).to_bytes(4, byteorder='little')  # Sample rate
    header[28:32] = (48000).to_bytes(4, byteorder='little')  # Byte rate
    header[32:34] = (2).to_bytes(2, byteorder='little')  # Block align
    header[34:36] = (16).to_bytes(2, byteorder='little')  # Bits per sample
    header[36:40] = b'data'
    header[40:44] = data_size.to_bytes(4, byteorder='little')
    return header

app = FastAPI()

# Create a semaphore to limit concurrent sessions
session_semaphore = Semaphore(3)

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>WebSocket Test</title>
        <style>
            .loader {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                animation: spin 2s linear infinite;
                display: none;
                margin-left: 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            #connection-status {
                padding: 8px;
                margin: 10px 0;
                border-radius: 4px;
                font-weight: bold;
                text-align: center;
                background-color: #f8f9fa;
            }
        </style>
    </head>
    <body>
        <h1>WebSocket Test</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off" placeholder="Type your message..."/>
            <button>Send</button>
            <div id="loader" class="loader"></div>
        </form>
        <ul id='messages'>
        </ul>
        <audio id="audioPlayer" controls style="margin-top: 20px;"></audio>
        <script>
            // Get the Firebase token from localStorage
            var token = localStorage.getItem('firebaseToken');
            if (!token) {
                alert('Please sign in first');
                return;
            }

            // Create WebSocket connection with token
            var ws = new WebSocket(`ws://localhost:8000/ws?token=${token}`);

            // Connection status indicators
            var statusDiv = document.createElement('div');
            statusDiv.id = 'connection-status';
            document.body.insertBefore(statusDiv, document.getElementById('messages'));

            ws.onopen = function() {
                statusDiv.textContent = 'Connected';
                statusDiv.style.color = 'green';
            };

            ws.onclose = function(event) {
                statusDiv.textContent = 'Disconnected: ' + event.reason;
                statusDiv.style.color = 'red';
                // Attempt to reconnect after 5 seconds
                setTimeout(function() {
                    location.reload();
                }, 5000);
            };
            var audioContext = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 24000});
            var audioBufferQueue = [];
            var isPlaying = false;
            
            function playAudioBuffer() {
                if (audioBufferQueue.length === 0) {
                    isPlaying = false;
                    return;
                }
                
                isPlaying = true;
                var buffer = audioBufferQueue.shift();
                var source = audioContext.createBufferSource();
                source.buffer = buffer;
                source.connect(audioContext.destination);
                source.start();
                source.onended = playAudioBuffer;
            }
            
            ws.onmessage = function(event) {
                document.getElementById('loader').style.display = 'none';
                if (event.data instanceof Blob) {
                    var reader = new FileReader();
                    reader.onload = function(e) {
                        audioContext.decodeAudioData(e.target.result)
                            .then(function(buffer) {
                                audioBufferQueue.push(buffer);
                                if (!isPlaying) {
                                    playAudioBuffer();
                                }
                            });
                    };
                    reader.readAsArrayBuffer(event.data);
                } else {
                    var messages = document.getElementById('messages')
                    var message = document.createElement('li')
                    var content = document.createTextNode(event.data)
                    message.appendChild(content)
                    messages.appendChild(message)
                }
            };
            
            function sendMessage(event) {
                document.getElementById('loader').style.display = 'inline-block';
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.post("/verify-token")
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        return JSONResponse({"uid": uid, "status": "success"})
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Get the token from the connection query parameters
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="No authentication token provided")
        return
        
    try:
        # Verify the Firebase token
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        # Token is valid, accept the connection
        await websocket.accept()
        print(f"WebSocket connection accepted for user: {uid}")
    except Exception as e:
        error_reason = "Authentication failed"
        if isinstance(e, auth.ExpiredIdTokenError):
            error_reason = "Token has expired"
        elif isinstance(e, auth.RevokedIdTokenError):
            error_reason = "Token has been revoked"
        elif isinstance(e, auth.InvalidIdTokenError):
            error_reason = "Invalid token"
        print(f"Token verification error: {str(e)}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=error_reason)
        return
    conversation_history = []
    
    try:
        # Try to acquire the semaphore
        if not await session_semaphore.acquire():
            await websocket.send_text("Server is at maximum capacity. Please try again later.")
            await websocket.close()
            return

        while True:
            try:
                async with client.aio.live.connect(model=model, config=config) as session:
                    # Send conversation history to establish context
                    if conversation_history:
                        await session.send_client_content(turns=conversation_history, turn_complete=False)

                    while True:
                        message = await websocket.receive()
                        data = message["text"]

                        try:
                            # Create user turn
                            user_turn = {
                                "role": "user",
                                "parts": [{"text": data}]
                            }
                            
                            # Add to history and send with turn_complete=True
                            conversation_history.append(user_turn)
                            await session.send_client_content(turns=[user_turn], turn_complete=True)

                            # Process model response
                            async for idx, response in async_enumerate(session.receive()):
                                # Send audio response if available
                                if response.server_content.model_turn:
                                    print("Model turn:", response.server_content.model_turn)
                                if response.server_content.output_transcription:
                                    print("Transcript:", response.server_content.output_transcription.text)

                                if response.data is not None:
                                    wav_header = create_wav_header(response.data)
                                    wav_data = wav_header + response.data
                                    await websocket.send_bytes(wav_data)
                                
                                # Store model response in history if text is available
                                if response.text is not None:
                                    model_turn = {
                                        "role": "model",
                                        "parts": [{"text": response.text}]
                                    }
                                    conversation_history.append(model_turn)

                            # Keep conversation history manageable
                            if len(conversation_history) > 10:
                                conversation_history = conversation_history[-10:]
                                    
                        except Exception as e:
                            print(f"Error in conversation: {e}")
                            if websocket.client_state == WebSocketState.CONNECTED:
                                await websocket.send_text(f"Error: {str(e)}")
                            break
                            
            except Exception as e:
                print(f"Session error: {e}")
                if "User location is not supported for the API use" in str(e):
                    client_ip = websocket.client.host
                    print(f"Service not available for IP: {client_ip} due to API limitations.")
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text("Service not available in your region due to API limitations.")
                        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Location not supported")
                elif websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(f"Session error: {str(e)}")
                await asyncio.sleep(1)  # Wait before reconnecting
    finally:
        # Release the semaphore when the connection is closed
        session_semaphore.release()
