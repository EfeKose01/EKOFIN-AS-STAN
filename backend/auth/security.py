"""Kimlik doğrulama iş mantığı — framework'ten bağımsız (Streamlit import ETMEZ).

İki akış destekler:
  1) Manuel e-posta/şifre kaydı  -> create_user() / authenticate_user()
  2) Google OAuth 2.0            -> get_or_create_google_user()
     (Google akışının kendisi Streamlit'in yerleşik st.login() ile yürütülür;
     bu modül sadece "Google'dan gelen kullanıcıyı DB'de bul/oluştur" işini yapar.)
"""

import bcrypt
from sqlalchemy.orm import Session

from backend.models import AuthProvider, User
from backend.schemas import UserCreate

# Not: `passlib` yerine `bcrypt` paketi doğrudan kullanılıyor. passlib 1.7.x,
# bcrypt>=4.1 ile sürüm tespiti konusunda uyumsuz (bilinen bir sorun) ve kısa
# şifrelerde bile "password cannot be longer than 72 bytes" hatası fırlatıyor.
# bcrypt'in kendi 72 byte sınırını burada açıkça ele alıyoruz.
_BCRYPT_MAX_BYTES = 72


def hash_password(plain_password: str) -> str:
    pw_bytes = plain_password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    pw_bytes = plain_password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    try:
        return bcrypt.checkpw(pw_bytes, hashed_password.encode("utf-8"))
    except ValueError:
        return False


class AuthError(Exception):
    """Kullanıcıya doğrudan gösterilebilecek auth hataları için."""


def create_user(db: Session, data: UserCreate) -> User:
    existing = db.query(User).filter(User.email == data.email.lower()).first()
    if existing:
        raise AuthError("Bu e-posta adresiyle zaten bir hesap var. Giriş yapmayı deneyin.")

    user = User(
        email=data.email.lower(),
        full_name=data.full_name,
        hashed_password=hash_password(data.password),
        auth_provider=AuthProvider.EMAIL,
    )
    db.add(user)
    db.flush()  # id'yi almak için (commit, get_db context manager çıkışında olur)
    return user


def authenticate_user(db: Session, email: str, password: str) -> User:
    user = db.query(User).filter(User.email == email.lower()).first()
    if user is None or user.hashed_password is None:
        raise AuthError("E-posta veya şifre hatalı.")
    if not verify_password(password, user.hashed_password):
        raise AuthError("E-posta veya şifre hatalı.")
    if not user.is_active:
        raise AuthError("Bu hesap devre dışı bırakılmış.")
    return user


def get_or_create_google_user(db: Session, *, email: str, full_name: str | None, google_sub: str) -> User:
    """Google OAuth'tan dönen bilgilerle kullanıcıyı bulur, yoksa oluşturur.

    Aynı e-posta manuel kayıtla zaten varsa, onu Google hesabıyla eşleştirir
    (tek kullanıcının iki farklı girişle aynı hesaba ulaşabilmesi için).
    """
    user = db.query(User).filter(User.google_sub == google_sub).first()
    if user:
        return user

    user = db.query(User).filter(User.email == email.lower()).first()
    if user:
        user.google_sub = google_sub
        if user.auth_provider == AuthProvider.EMAIL and user.hashed_password is None:
            user.auth_provider = AuthProvider.GOOGLE
        db.flush()
        return user

    user = User(
        email=email.lower(),
        full_name=full_name,
        auth_provider=AuthProvider.GOOGLE,
        google_sub=google_sub,
        hashed_password=None,
    )
    db.add(user)
    db.flush()
    return user
