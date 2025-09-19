"""
Authentication system for AI Conspects Server
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import uuid
from typing import Optional

from database import get_db, User
from schemas import DeviceRegistration, AuthResponse
from config import settings

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(user_id: str) -> str:
    """Create JWT access token"""
    expire = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    to_encode = {"sub": user_id, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[str]:
    """Verify JWT token and return user_id"""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        return user_id
    except JWTError:
        return None

async def get_user_by_device_id(db: Session, device_id: str) -> Optional[User]:
    """Get user by device ID"""
    return db.query(User).filter(User.device_id == device_id).first()

async def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    """Get user by ID"""
    return db.query(User).filter(User.id == uuid.UUID(user_id)).first()

async def create_user(db: Session, device_id: str) -> User:
    """Create new user"""
    user = User(device_id=device_id)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

async def get_or_create_user(db: Session, device_id: str) -> User:
    """Get existing user or create new one"""
    user = await get_user_by_device_id(db, device_id)
    if not user:
        user = await create_user(db, device_id)
    return user

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    user_id = verify_token(token)
    
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = await get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

@router.post("/register", response_model=AuthResponse)
async def register_device(
    request: DeviceRegistration,
    db: Session = Depends(get_db)
):
    """Register new device or get existing user"""
    user = await get_or_create_user(db, request.device_id)
    token = create_access_token(str(user.id))
    
    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user_id=str(user.id)
    )

@router.post("/login", response_model=AuthResponse)
async def login_device(
    request: DeviceRegistration,
    db: Session = Depends(get_db)
):
    """Login by device_id"""
    user = await get_user_by_device_id(db, request.device_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device not found"
        )
    
    token = create_access_token(str(user.id))
    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user_id=str(user.id)
    )

@router.get("/me")
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information"""
    return {
        "user_id": str(current_user.id),
        "device_id": current_user.device_id,
        "created_at": current_user.created_at.isoformat(),
        "is_active": current_user.is_active,
        "settings": current_user.settings
    }
