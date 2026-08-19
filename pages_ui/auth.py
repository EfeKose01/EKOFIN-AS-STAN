"""Kimlik doğrulama arayüz bileşenleri (frontend katmanı — Streamlit'e bağımlı).

Backend'deki saf iş mantığını (backend/auth/security.py) Streamlit oturumuna
bağlayan "köprü" burada yaşıyor; böylece backend katmanı Streamlit'ten habersiz
ve bağımsız test edilebilir kalıyor.
"""

from __future__ import annotations

import streamlit as st
from sqlalchemy.orm import Session

from backend.auth.security import AuthError, authenticate_user, create_user, get_or_create_google_user
from backend.models import User
from backend.schemas import UserCreate

_SESSION_KEY = "ekofin_manual_user_id"


def _google_logged_in() -> bool:
    """st.user.is_logged_in güvenli kontrolü.

    secrets.toml'da [auth] bölümü (Google OAuth) yapılandırılmamışsa Streamlit
    st.user nesnesini oluşturur ama HERHANGİ BİR özelliğe erişim (is_logged_in
    dahil) AttributeError fırlatır — False DÖNMEZ. Bu yüzden düz `st.user.is_logged_in`
    kontrolü, Google girişi hiç kurulmamış ortamlarda (ör. yerel geliştirme, secrets.toml
    henüz doldurulmamışken) myPortfolio sayfasını tamamen çökertiyordu.
    """
    try:
        return bool(st.user.is_logged_in)
    except Exception:
        return False


def get_current_user(db: Session) -> User | None:
    """Google (st.login) veya manuel (session_state) girişten hangisi aktifse o kullanıcıyı döner."""

    # 1) Google OAuth (Streamlit'in yerleşik auth'u)
    if _google_logged_in():
        google_sub = st.user.get("sub") or st.user.get("email")
        user = get_or_create_google_user(
            db,
            email=st.user.email,
            full_name=st.user.get("name"),
            google_sub=google_sub,
        )
        # İlk Google girişinde kullanıcı burada yaratılıyor; sayfanın ilerleyen
        # kısmındaki bir st.rerun() kaydı geri almasın diye hemen kalıcı yapıyoruz.
        db.commit()
        return user

    # 2) Manuel e-posta/şifre girişi
    user_id = st.session_state.get(_SESSION_KEY)
    if user_id:
        return db.get(User, user_id)

    return None


def is_authenticated(db: Session) -> bool:
    return get_current_user(db) is not None


def logout() -> None:
    st.session_state.pop(_SESSION_KEY, None)
    if _google_logged_in():
        st.logout()
    else:
        st.rerun()


def render_login_register_widget(db: Session) -> None:
    """myPortfolio sayfasına girişi olmayan kullanıcıya gösterilecek giriş/kayıt bileşeni."""

    st.markdown(
        """
        <div class="ekofin-hero" style="padding-top:.6rem;">
            <div class="eyebrow"><span class="dot"></span> myPortfolio</div>
            <h1 style="font-size:2.1rem;">Devam etmek için giriş yapın</h1>
            <p>Portföyünüzü kaydetmek, kâr/zarar takibi yapmak ve size özel haber akışı almak için hesabınıza giriş yapın.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("#### 🔐 Google ile devam et")
        st.caption("Şifre yönetmenize gerek kalmadan, tek tıkla.")
        if st.button("Google ile Giriş Yap", type="primary", use_container_width=True):
            try:
                st.login("google")
            except Exception:
                st.error(
                    "Google ile giriş şu an yapılandırılmamış. `.streamlit/secrets.toml` içindeki "
                    "`[auth]` ve `[auth.google]` bölümlerini doldurmanız gerekiyor "
                    "(bkz. `.streamlit/secrets.toml.example`). Bu arada e-posta ile devam edebilirsiniz."
                )

    with col_r:
        st.markdown("#### ✉️ E-posta ile devam et")
        tab_login, tab_register = st.tabs(["Giriş Yap", "Kayıt Ol"])

        with tab_login:
            with st.form("login_form", border=False):
                email = st.text_input("E-posta", key="login_email")
                password = st.text_input("Şifre", type="password", key="login_password")
                submitted = st.form_submit_button("Giriş Yap", use_container_width=True)
            if submitted:
                try:
                    user = authenticate_user(db, email, password)
                    st.session_state[_SESSION_KEY] = user.id
                    st.rerun()
                except AuthError as e:
                    st.error(str(e))

        with tab_register:
            with st.form("register_form", border=False):
                full_name = st.text_input("Ad Soyad", key="reg_name")
                email = st.text_input("E-posta", key="reg_email")
                password = st.text_input("Şifre (en az 8 karakter)", type="password", key="reg_password")
                submitted = st.form_submit_button("Hesap Oluştur", use_container_width=True)
            if submitted:
                try:
                    data = UserCreate(email=email, password=password, full_name=full_name or None)
                    user = create_user(db, data)
                    # st.rerun() BaseException fırlattığı için get_db()'nin çıkıştaki
                    # commit'ine ulaşılmaz; yeni kullanıcı kaydını burada kalıcı yapmalıyız.
                    db.commit()
                    st.session_state[_SESSION_KEY] = user.id
                    st.success("Hesabınız oluşturuldu, giriş yapılıyor…")
                    st.rerun()
                except Exception as e:  # Pydantic ValidationError veya AuthError
                    st.error(str(e))


def render_user_badge(user: User) -> None:
    """Sidebar/üst bar için küçük 'giriş yapıldı' rozeti + çıkış butonu."""
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:.5rem;padding:.4rem 0 .8rem;">
            <div style="width:30px;height:30px;border-radius:50%;background:linear-gradient(135deg,#0a84ff,#5e5ce6);
                        display:flex;align-items:center;justify-content:center;font-size:.85rem;color:#fff;font-weight:700;">
                {(user.full_name or user.email)[0].upper()}
            </div>
            <div style="font-size:.85rem;line-height:1.2;">
                <div style="font-weight:600;">{user.full_name or user.email.split('@')[0]}</div>
                <div style="opacity:.6;font-size:.75rem;">{user.email}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Çıkış Yap", use_container_width=True):
        logout()
