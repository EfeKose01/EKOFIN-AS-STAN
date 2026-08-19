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
    """`with get_db() as db:` şeklinde kullanılacak session context manager'ı.

    DİKKAT — bloğun içinden akış kontrolü exception'ı fırlatan bir şey çağırırsanız
    (Streamlit'te `st.rerun()` / `st.stop()`, ya da KeyboardInterrupt) aşağıdaki
    `db.commit()` satırına HİÇ ULAŞILMAZ ve `db.close()` commit edilmemiş her şeyi
    geri alır. Bu exception'lar `Exception` değil `BaseException` alt sınıfı olduğu
    için eskiden sessizce yutuluyordu; artık `BaseException` yakalayıp açıkça
    rollback ediyoruz. Yani: st.rerun()'dan ÖNCE mutlaka `db.commit()` çağırın
    (bkz. pages_ui/auth.py ve pages_ui/portfolio_page.py).
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()
