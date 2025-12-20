import os
import json
import bcrypt
import jwt
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
from pathlib import Path

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 

USERS_DB_PATH = Path("data/users.json")

GUEST_SESSIONS = {}

def ensure_users_db():
    USERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not USERS_DB_PATH.exists():
        USERS_DB_PATH.write_text("{}")

def load_users() -> Dict:
    ensure_users_db()
    try:
        with open(USERS_DB_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def save_users(users: Dict):
    ensure_users_db()
    with open(USERS_DB_PATH, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )

def create_access_token(username: str, is_guest: bool = False) -> str:
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
    
    user_upload_dir = Path(f"data/users/{username}/raw")
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    
    user_logs_dir = Path(f"data/users/{username}")
    user_logs_dir.mkdir(parents=True, exist_ok=True)
    
    return True

def authenticate_user(username: str, password: str) -> Optional[str]:
    users = load_users()
    
    if username not in users:
        return None
    
    user_data = users[username]
    
    if not verify_password(password, user_data["password"]):
        return None
    
    return create_access_token(username, is_guest=False)

def create_guest_session() -> str:
    guest_id = f"guest_{secrets.token_hex(8)}"
    
    guest_upload_dir = Path(f"data/guests/{guest_id}/raw")
    guest_upload_dir.mkdir(parents=True, exist_ok=True)
    
    GUEST_SESSIONS[guest_id] = {
        "created_at": datetime.utcnow().isoformat(),
        "path": f"data/guests/{guest_id}"
    }
    
    return create_access_token(guest_id, is_guest=True)

def cleanup_guest_data(guest_id: str):
    import shutil
    guest_path = Path(f"data/guests/{guest_id}")
    if guest_path.exists():
        shutil.rmtree(guest_path)
    
    if guest_id in GUEST_SESSIONS:
        del GUEST_SESSIONS[guest_id]

def delete_user_account(username: str) -> bool:
    import shutil
    
    users = load_users()
    
    if username not in users:
        return False
    
    del users[username]
    save_users(users)
    
    user_path = Path(f"data/users/{username}")
    if user_path.exists():
        shutil.rmtree(user_path)
    
    return True

def get_user_upload_dir(username: str, is_guest: bool = False) -> str:
    if is_guest:
        return f"data/guests/{username}/raw"
    return f"data/users/{username}/raw"

def get_user_critique_log_path(username: str, is_guest: bool = False) -> str:
    if is_guest:
        return f"data/guests/{username}/critique_log.jsonl"
    return f"data/users/{username}/critique_log.jsonl"
