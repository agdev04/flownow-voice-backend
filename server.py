#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RTVI Bot Server Implementation.

This FastAPI server manages RTVI bot instances and provides endpoints for both
direct browser access and RTVI client connections. It handles:
- Creating Daily rooms
- Managing bot processes
- Providing connection credentials
- Monitoring bot status

Requirements:
- Daily API key (set in .env file)
- Python 3.10+
- FastAPI
- Running bot implementation
"""

import argparse
import os
import subprocess
import base64
import json
import os
import subprocess

from contextlib import asynccontextmanager
from typing import Any, Dict

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from fastapi.responses import JSONResponse, RedirectResponse
import firebase_admin
from firebase_admin import auth, credentials
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams

# Load environment variables from .env file
load_dotenv(override=True)

# Maximum number of bot instances allowed per room
MAX_BOTS_PER_ROOM = 1

# Dictionary to track bot processes: {pid: (process, room_url)}
bot_procs = {}

# Store Daily API helpers
daily_helpers = {}

# Get the base64-encoded Firebase Admin key from environment
firebase_base64 = os.getenv('FIREBASE_ADMIN_KEY_B64')

# Decode the base64 string to JSON string
decoded_json = base64.b64decode(firebase_base64).decode('utf-8')

# Parse the JSON string
service_account_info = json.loads(decoded_json)

# Before initializing Firebase, check if it's already initialized
if not firebase_admin._apps:
    # Initialize Firebase Admin with the credentials
    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)

# Security bearer token
security = HTTPBearer()

def cleanup():
    """Cleanup function to terminate all bot processes.

    Called during server shutdown.
    """
    for entry in bot_procs.values():
        proc = entry[0]
        proc.terminate()
        proc.wait()


def get_bot_file():
    bot_implementation = os.getenv("BOT_IMPLEMENTATION", "openai").lower().strip()
    # If blank or None, default to openai
    if not bot_implementation:
        bot_implementation = "openai"
    if bot_implementation not in ["openai", "gemini"]:
        raise ValueError(
            f"Invalid BOT_IMPLEMENTATION: {bot_implementation}. Must be 'openai' or 'gemini'"
        )
    return f"bot-{bot_implementation}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager that handles startup and shutdown tasks.

    - Creates aiohttp session
    - Initializes Daily API helper
    - Cleans up resources on shutdown
    """
    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    yield
    await aiohttp_session.close()
    cleanup()


# Initialize FastAPI app with lifespan manager
app = FastAPI(lifespan=lifespan)

# Configure CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def create_room_and_token() -> tuple[str, str]:
    """Helper function to create a Daily room and generate an access token.

    Returns:
        tuple[str, str]: A tuple containing (room_url, token)

    Raises:
        HTTPException: If room creation or token generation fails
    """
    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL", None)
    token = os.getenv("DAILY_SAMPLE_ROOM_TOKEN", None)
    if not room_url:
        room = await daily_helpers["rest"].create_room(DailyRoomParams())
        if not room.url:
            raise HTTPException(status_code=500, detail="Failed to create room")
        room_url = room.url

        token = await daily_helpers["rest"].get_token(room_url)
        if not token:
            raise HTTPException(status_code=500, detail=f"Failed to get token for room: {room_url}")

    return room_url, token

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

@app.get("/")
async def start_agent(request: Request):
    """Endpoint for direct browser access to the bot.

    Creates a room, starts a bot instance, and redirects to the Daily room URL.

    Returns:
        RedirectResponse: Redirects to the Daily room URL

    Raises:
        HTTPException: If room creation, token generation, or bot startup fails
    """
    print("Creating room")
    room_url, token = await create_room_and_token()
    print(f"Room URL: {room_url}")

    # Check if there is already an existing process running in this room
    num_bots_in_room = sum(
        1 for proc in bot_procs.values() if proc[1] == room_url and proc[0].poll() is None
    )
    if num_bots_in_room >= MAX_BOTS_PER_ROOM:
        raise HTTPException(status_code=500, detail=f"Max bot limit reached for room: {room_url}")

    # Spawn a new bot process
    try:
        bot_file = get_bot_file()
        proc = subprocess.Popen(
            [f"python3 -m {bot_file} -u {room_url} -t {token}"],
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        bot_procs[proc.pid] = (proc, room_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    return RedirectResponse(room_url)


@app.post("/create-room")
async def create_new_room() -> Dict[str, str]:
    """Create a new Daily room and generate an access token.

    Returns:
        Dict[str, str]: Room URL and token for the new room
    """
    room_url, token = await create_room_and_token()
    return {"room_url": room_url, "token": token}

@app.post("/connect")
async def rtvi_connect(request: Request) -> Dict[Any, Any]:
    """RTVI connect endpoint that connects to an existing room.

    This endpoint is called by RTVI clients to establish a connection.
    The client must provide the room_url and token in the request body.

    Returns:
        Dict[Any, Any]: Authentication bundle containing room_url and token

    Raises:
        HTTPException: If bot startup fails or if room credentials are missing
    """
    # Get room credentials from request body
    body = await request.json()
    room_url = body.get("room_url")
    token = body.get("token")

    if not room_url or not token:
        raise HTTPException(
            status_code=400,
            detail="room_url and token are required in request body"
        )

    # Start the bot process
    try:
        bot_file = get_bot_file()
        proc = subprocess.Popen(
            [f"python3 -m {bot_file} -u {room_url} -t {token}"],
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        bot_procs[proc.pid] = (proc, room_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    # Return the authentication bundle in format expected by DailyTransport
    return {"room_url": room_url, "token": token}


@app.get("/status/{pid}")
def get_status(pid: int):
    """Get the status of a specific bot process.

    Args:
        pid (int): Process ID of the bot

    Returns:
        JSONResponse: Status information for the bot

    Raises:
        HTTPException: If the specified bot process is not found
    """
    # Look up the subprocess
    proc = bot_procs.get(pid)

    # If the subprocess doesn't exist, return an error
    if not proc:
        raise HTTPException(status_code=404, detail=f"Bot with process id: {pid} not found")

    # Check the status of the subprocess
    status = "running" if proc[0].poll() is None else "finished"
    return JSONResponse({"bot_id": pid, "status": status})


if __name__ == "__main__":
    import uvicorn

    # Parse command line arguments for server configuration
    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("FAST_API_PORT", "7860"))

    parser = argparse.ArgumentParser(description="Daily Storyteller FastAPI server")
    parser.add_argument("--host", type=str, default=default_host, help="Host address")
    parser.add_argument("--port", type=int, default=default_port, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Reload code on change")

    config = parser.parse_args()

    # Start the FastAPI server
    uvicorn.run(
        "server:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
    )
