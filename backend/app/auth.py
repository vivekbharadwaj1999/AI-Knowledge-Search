import os
import json
import bcrypt
import jwt
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
from pathlib import Path

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Users database file
USERS_DB_PATH = Path("data/users.json")

# Guest sessions - in-memory for simplicity (could use Redis in production)
GUEST_SESSIONS = {}


def ensure_users_db():
    """Ensure the users database file exists"""
    USERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not USERS_DB_PATH.exists():
        USERS_DB_PATH.write_text("{}")


def load_users() -> Dict:
    """Load users from JSON file"""
    ensure_users_db()
    try:
        with open(USERS_DB_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_users(users: Dict):
    """Save users to JSON file"""
    ensure_users_db()
    with open(USERS_DB_PATH, "w") as f:
        json.dump(users, f, indent=2)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )


def create_access_token(username: str, is_guest: bool = False) -> str:
    """Create a JWT access token"""
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {
        "sub": username,
        "is_guest": is_guest,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[Dict]:
    """Decode and verify a JWT token, return payload if valid"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        is_guest: bool = payload.get("is_guest", False)
        if username is None:
            return None
        return {"username": username, "is_guest": is_guest}
    except jwt.ExpiredSignatureError:
        return None
    except jwt.JWTError:
        return None


def create_user(username: str, password: str) -> bool:
    """Create a new user. Returns True if successful, False if username exists"""
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    
    users = load_users()
    
    if username in users:
        return False
    
    users[username] = {
        "password": hash_password(password),
        "created_at": datetime.utcnow().isoformat()
    }
    
    save_users(users)
    
    # Create user-specific directories
    user_upload_dir = Path(f"data/users/{username}/raw")
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    
    user_logs_dir = Path(f"data/users/{username}")
    user_logs_dir.mkdir(parents=True, exist_ok=True)
    
    return True


def authenticate_user(username: str, password: str) -> Optional[str]:
    """Authenticate a user. Returns JWT token if successful, None otherwise"""
    users = load_users()
    
    if username not in users:
        return None
    
    user_data = users[username]
    
    if not verify_password(password, user_data["password"]):
        return None
    
    return create_access_token(username, is_guest=False)


def create_guest_session() -> str:
    """Create a guest session and return token"""
    guest_id = f"guest_{secrets.token_hex(8)}"
    
    # Create guest directories
    guest_upload_dir = Path(f"data/guests/{guest_id}/raw")
    guest_upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Track guest session
    GUEST_SESSIONS[guest_id] = {
        "created_at": datetime.utcnow().isoformat(),
        "path": f"data/guests/{guest_id}"
    }
    
    return create_access_token(guest_id, is_guest=True)


def cleanup_guest_data(guest_id: str):
    """Delete all guest data"""
    import shutil
    guest_path = Path(f"data/guests/{guest_id}")
    if guest_path.exists():
        shutil.rmtree(guest_path)
    
    # Remove from sessions
    if guest_id in GUEST_SESSIONS:
        del GUEST_SESSIONS[guest_id]


def delete_user_account(username: str) -> bool:
    """
    Delete a user account completely:
    - Remove from users.json (credentials)
    - Delete all user files (uploads, vector store, logs)
    
    Returns True if successful, False if user not found
    """
    import shutil
    
    # Load users database
    users = load_users()
    
    # Check if user exists
    if username not in users:
        return False
    
    # Delete user from database
    del users[username]
    save_users(users)
    
    # Delete all user files
    user_path = Path(f"data/users/{username}")
    if user_path.exists():
        shutil.rmtree(user_path)
    
    return True


def get_user_upload_dir(username: str, is_guest: bool = False) -> str:
    """Get the upload directory for a specific user or guest"""
    if is_guest:
        return f"data/guests/{username}/raw"
    return f"data/users/{username}/raw"


def get_user_critique_log_path(username: str, is_guest: bool = False) -> str:
    """Get the critique log path for a specific user or guest"""
    if is_guest:
        return f"data/guests/{username}/critique_log.jsonl"
    return f"data/users/{username}/critique_log.jsonl"
