"""ORM modelleri: Users ve Portfolios."""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AuthProvider(str, enum.Enum):
    EMAIL = "email"      # manuel e-posta/şifre ile kayıt
    GOOGLE = "google"    # Google OAuth 2.0 ile kayıt


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Manuel kayıtlarda dolu, Google ile girenlerde None (şifre saklanmaz).
    hashed_password: Mapped[str | None] = mapped_column(String(255), nullable=True)

    auth_provider: Mapped[AuthProvider] = mapped_column(
        Enum(AuthProvider), nullable=False, default=AuthProvider.EMAIL
    )
    # Google'ın kullanıcıya verdiği benzersiz "sub" kimliği (OAuth eşleştirmesi için).
    google_sub: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)

    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    portfolio_items: Mapped[list["PortfolioItem"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User {self.email} ({self.auth_provider.value})>"


class PortfolioItem(Base):
    """Kullanıcının portföyündeki tek bir hisse/varlık satırı."""

    __tablename__ = "portfolio_items"
    __table_args__ = (
        # Aynı kullanıcı aynı sembolü iki ayrı satırda tutmasın; miktar güncellensin.
        UniqueConstraint("user_id", "symbol", name="uq_user_symbol"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)

    symbol: Mapped[str] = mapped_column(String(20), nullable=False)  # örn. GARAN, THYAO, BTC-USD
    quantity: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    cost_basis: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)  # birim maliyet (TL/adet)

    # OCR'dan mı, manuel mi geldiğini iz sürmek için (destek/hata ayıklama amaçlı).
    source: Mapped[str] = mapped_column(String(20), default="manual")  # "manual" | "ocr"

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    owner: Mapped["User"] = relationship(back_populates="portfolio_items")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PortfolioItem {self.symbol} x{self.quantity} @ {self.cost_basis}>"
