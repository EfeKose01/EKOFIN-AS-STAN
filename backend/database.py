"""SQLAlchemy engine ve session kurulumu.

Varsayılan: yerel dosya tabanlı SQLite (ekofin.db) — sıfır kurulumla çalışır.
Üretimde Postgres'e geçmek için sadece DATABASE_URL ortam değişkenini ayarlamak yeterli, örn:
    DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/ekofin
"""

import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ekofin.db")

# SQLite dosya tabanlı bağlantılar aynı thread dışından da kullanılabilsin diye
# (Streamlit her rerun'da farklı bir thread kullanabilir).
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Tabloları (yoksa) oluşturur. Uygulama açılışında bir kez çağrılır."""
    from backend import models  # noqa: F401  (modellerin Base.metadata'ya kaydolması için import şart)

    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db() -> Iterator[Session]:
    """`with get_db() as db:` şeklinde kullanılacak session context manager'ı."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
