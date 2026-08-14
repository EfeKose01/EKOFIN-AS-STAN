"""Pydantic şemaları — giriş doğrulama ve katmanlar arası veri sözleşmesi."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# ---- Kullanıcı ----

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = None

    @field_validator("password")
    @classmethod
    def password_not_trivial(cls, v: str) -> str:
        if v.strip().lower() in {"12345678", "password"}:
            raise ValueError("Lütfen daha güçlü bir şifre seçin.")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: EmailStr
    full_name: str | None
    auth_provider: str
    created_at: datetime


# ---- Portföy ----

class PortfolioItemCreate(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    quantity: float = Field(gt=0)
    cost_basis: float = Field(gt=0, description="Birim maliyet (TL/adet)")
    source: Literal["manual", "ocr"] = "manual"

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.strip().upper()


class PortfolioItemUpdate(BaseModel):
    quantity: float | None = Field(default=None, gt=0)
    cost_basis: float | None = Field(default=None, gt=0)


class PortfolioItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    symbol: str
    quantity: float
    cost_basis: float
    source: str
    updated_at: datetime


class OcrParsedRow(BaseModel):
    """OCR pipeline'ının ham çıktısı — DB'ye yazılmadan önce kullanıcı onayından geçer."""

    symbol: str
    quantity: float | None = None
    cost_basis: float | None = None
    confidence: float = Field(ge=0, le=1, description="0-1 arası ayrıştırma güven skoru")
    raw_line: str = ""
