"""EkoFin Asistan — ana giriş noktası.

Routing:
  - "Sohbet" sayfası  → HERKESE AÇIK, giriş gerektirmez (app_finetune_rag.run_streamlit_app)
  - "myPortfolio"     → SADECE GİRİŞ YAPMIŞ KULLANICILAR (Google OAuth veya e-posta/şifre)

Çalıştırma:
    streamlit run streamlit_app.py
"""

import streamlit as st

from backend.database import get_db, init_db
from pages_ui.auth import get_current_user
from pages_ui.theme import CUSTOM_CSS

APP_NAME = "EkoFin Asistan"

st.set_page_config(page_title=APP_NAME, page_icon="🤖", layout="wide")
init_db()


def _render_chat_page() -> None:
    from app_finetune_rag import run_streamlit_app

    run_streamlit_app()


def _render_portfolio_page() -> None:
    from pages_ui.portfolio_page import render_portfolio_page

    render_portfolio_page()


chat_page = st.Page(_render_chat_page, title="Sohbet", icon="💬", url_path="chat", default=True)
portfolio_page = st.Page(_render_portfolio_page, title="myPortfolio", icon="📊", url_path="portfolio")

# myPortfolio linki menüde her zaman görünür; sayfanın içine girildiğinde
# giriş yapılmamışsa render_portfolio_page zaten giriş/kayıt ekranını gösterir.
# (Linki tamamen gizlemek yerine göstermek, kullanıcıyı özelliğin varlığından
# haberdar edip giriş yapmaya davet ettiği için daha iyi bir UX.)
nav = st.navigation({"EkoFin Asistan": [chat_page, portfolio_page]})

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

nav.run()
