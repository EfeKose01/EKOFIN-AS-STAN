"""myPortfolio — giriş gerektiren kişisel dashboard sayfası."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.database import get_db
from backend.schemas import OcrParsedRow, PortfolioItemCreate
from backend.services.ocr_service import run_ocr_pipeline
from backend.services.portfolio_service import (
    delete_portfolio_item,
    fetch_current_prices,
    list_portfolio,
    portfolio_summary,
    portfolio_to_dataframe,
    sector_allocation,
    upsert_portfolio_item,
    value_portfolio,
)
from pages_ui.auth import get_current_user, render_login_register_widget, render_user_badge

_OCR_REVIEW_KEY = "ekofin_ocr_review_rows"
_OCR_CONFIDENCE_THRESHOLD = 0.7


def render_portfolio_page() -> None:
    with get_db() as db:
        user = get_current_user(db)

        if user is None:
            render_login_register_widget(db)
            return

        with st.sidebar:
            render_user_badge(user)

        st.markdown(
            """
            <div class="ekofin-hero" style="padding-top:.6rem;">
                <div class="eyebrow"><span class="dot"></span> myPortfolio</div>
                <h1 style="font-size:2.3rem;">Portföyünüz</h1>
                <p>Güncel değerleme, kâr/zarar ve sektörel dağılım — tek bakışta.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_overview, tab_manage = st.tabs(["📊 Genel Bakış", "✏️ Portföyümü Yönet"])

        items = list_portfolio(db, user.id)
        df = portfolio_to_dataframe(items)

        with tab_overview:
            _render_overview(df)

        with tab_manage:
            _render_manage(db, user.id, items)


def _render_overview(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Henüz portföyünüze varlık eklemediniz. **✏️ Portföyümü Yönet** sekmesinden ekleyebilirsiniz.")
        return

    with st.spinner("Güncel fiyatlar çekiliyor…"):
        prices = fetch_current_prices(df["symbol"].tolist())

    valued = value_portfolio(df, prices)
    summary = portfolio_summary(valued)

    if valued["price_missing"].any():
        missing_syms = ", ".join(valued.loc[valued["price_missing"], "symbol"].tolist())
        st.warning(f"Şu semboller için güncel fiyat alınamadı, maliyet fiyatı kullanıldı: **{missing_syms}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Değer", f"{summary['total_market_value']:,.2f} TL")
    c2.metric("Toplam Maliyet", f"{summary['total_cost']:,.2f} TL")
    c3.metric(
        "Kâr / Zarar",
        f"{summary['total_pnl']:,.2f} TL",
        delta=f"{summary['total_pnl_pct']:.2f}%",
    )
    c4.metric("Varlık Sayısı", summary["holdings_count"])

    col_table, col_pie = st.columns([3, 2], gap="large")

    with col_table:
        st.markdown("#### Holding Detayı")
        display_df = valued[["symbol", "quantity", "cost_basis", "current_price", "market_value", "pnl_pct"]].copy()
        display_df.columns = ["Sembol", "Adet", "Maliyet", "Güncel Fiyat", "Piyasa Değeri", "PnL %"]
        st.dataframe(
            display_df.style.format(
                {"Adet": "{:,.2f}", "Maliyet": "{:,.2f}", "Güncel Fiyat": "{:,.2f}",
                 "Piyasa Değeri": "{:,.2f}", "PnL %": "{:+.2f}%"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    with col_pie:
        st.markdown("#### Sektörel Dağılım")
        alloc = sector_allocation(valued)
        if alloc:
            fig = px.pie(
                names=list(alloc.keys()),
                values=list(alloc.values()),
                hole=0.55,
                color_discrete_sequence=["#0071e3", "#5e5ce6", "#34c759", "#ff9500", "#ff3b30", "#af52de", "#34c9ff"],
            )
            fig.update_traces(textinfo="percent+label", textfont_size=12)
            fig.update_layout(
                showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif", color="#1d1d1f"),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    _render_personalized_news(df["symbol"].tolist())


def _render_personalized_news(symbols: list[str]) -> None:
    from backend.services.news_service import filter_news_for_portfolio, load_recent_news

    st.markdown("#### 📰 Size Özel Haber Akışı")

    news = load_recent_news()
    if not news:
        st.caption(
            "Henüz taranmış bir haber yok. Sohbet sayfasının sol panelindeki "
            "**🌐 Güncel Veri Taraması → Şimdi Tara** ile haberleri güncelleyebilirsiniz "
            "(üretimde bu günlük otomatik bir görevle çalıştırılmalı)."
        )
        return

    matched = filter_news_for_portfolio(news, symbols)
    if not matched:
        st.caption("Portföyünüzdeki sembollerle ilgili güncel bir haber bulunamadı.")
        return

    for item in matched[:6]:
        tag = item.get("matched_symbol") or item.get("matched_keyword", "").title()
        st.markdown(
            f"""
            <div style="padding:.75rem 1rem; border-radius:14px; background:var(--ekofin-glass-bg);
                        backdrop-filter:var(--ekofin-glass-blur); border:1px solid var(--ekofin-glass-border);
                        margin-bottom:.6rem;">
                <div style="font-size:.72rem; font-weight:700; color:var(--ekofin-accent); text-transform:uppercase;">
                    {tag} · {item.get('source','')}
                </div>
                <div style="font-weight:600; margin-top:.2rem;">{item.get('title','')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_manage(db, user_id: str, items) -> None:
    st.markdown("#### Manuel Ekle / Güncelle")
    st.caption("Bir sembolü tekrar eklerseniz, mevcut satırla ağırlıklı ortalama alınarak birleştirilir.")

    with st.form("manual_add_form", clear_on_submit=True, border=False):
        c1, c2, c3 = st.columns(3)
        symbol = c1.text_input("Sembol (örn. GARAN)")
        quantity = c2.number_input("Adet", min_value=0.0, step=1.0, format="%.4f")
        cost_basis = c3.number_input("Birim Maliyet (TL)", min_value=0.0, step=0.01, format="%.2f")
        submitted = st.form_submit_button("➕ Ekle / Güncelle", type="primary")

    if submitted:
        if not symbol.strip() or quantity <= 0 or cost_basis <= 0:
            st.error("Lütfen sembol, adet ve maliyeti eksiksiz ve pozitif girin.")
        else:
            try:
                data = PortfolioItemCreate(symbol=symbol, quantity=quantity, cost_basis=cost_basis, source="manual")
                upsert_portfolio_item(db, user_id, data)
                st.success(f"{data.symbol} portföyünüze eklendi.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    _render_ocr_upload(db, user_id)

    st.markdown("---")
    st.markdown("#### Mevcut Satırlar")
    if not items:
        st.caption("Henüz kayıtlı bir varlık yok.")
        return

    for it in items:
        row = st.columns([2, 2, 2, 1])
        row[0].markdown(f"**{it.symbol}**")
        row[1].markdown(f"{float(it.quantity):,.4f} adet")
        row[2].markdown(f"{float(it.cost_basis):,.2f} TL maliyet")
        if row[3].button("🗑️", key=f"del_{it.id}", help="Bu satırı sil"):
            delete_portfolio_item(db, user_id, it.id)
            st.rerun()


def _render_ocr_upload(db, user_id: str) -> None:
    st.markdown("#### 📷 Ekran Görüntüsünden Otomatik Yükleme (OCR)")
    st.caption(
        "Aracı kurum uygulamanızdaki portföy ekranının görüntüsünü yükleyin; sembol/adet/maliyeti "
        "otomatik okumaya çalışırız. Bu bir **taslak** özelliktir — kaydetmeden önce her satırı "
        "kontrol edin, özellikle düşük güven skorlu (%70 altı) satırları."
    )
    uploaded = st.file_uploader("Portföy ekran görüntüsü (PNG/JPG)", type=["png", "jpg", "jpeg"], key="ocr_uploader")

    if uploaded is not None:
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("ekofin_ocr_last_file") != file_id:
            with st.spinner("Görüntü okunuyor (OCR)…"):
                try:
                    rows: list[OcrParsedRow] = run_ocr_pipeline(uploaded.getvalue())
                except Exception as e:
                    st.error(
                        f"OCR işlenemedi: {e}. Sunucuda `tesseract-ocr` kurulu olduğundan emin olun "
                        "veya bu görüntü için manuel giriş formunu kullanın."
                    )
                    rows = []
            st.session_state[_OCR_REVIEW_KEY] = [r.model_dump() for r in rows]
            st.session_state["ekofin_ocr_last_file"] = file_id

    review_rows = st.session_state.get(_OCR_REVIEW_KEY)
    if not review_rows:
        return

    if not review_rows:
        st.info("Görüntüde tanınabilir bir portföy satırı bulunamadı.")
        return

    if not any(r["confidence"] >= _OCR_CONFIDENCE_THRESHOLD for r in review_rows):
        st.warning(
            "Hiçbir satır yeterince güvenilir okunamadı. Aşağıdaki tablodan elle düzeltip "
            "yine de ekleyebilir, ya da üstteki manuel formu kullanabilirsiniz."
        )

    df = pd.DataFrame(review_rows)
    df["confidence_pct"] = (df["confidence"] * 100).round(0)
    df.insert(0, "dahil_et", df["confidence"] >= _OCR_CONFIDENCE_THRESHOLD)
    df = df[["dahil_et", "symbol", "quantity", "cost_basis", "confidence_pct", "raw_line"]]
    df.columns = ["Ekle", "Sembol", "Adet", "Maliyet", "Güven %", "Okunan Satır (referans)"]

    edited = st.data_editor(
        df,
        column_config={
            "Ekle": st.column_config.CheckboxColumn("Ekle"),
            "Güven %": st.column_config.ProgressColumn("Güven", min_value=0, max_value=100, format="%d%%"),
            "Okunan Satır (referans)": st.column_config.TextColumn("Okunan Satır (referans)", disabled=True),
        },
        hide_index=True,
        use_container_width=True,
        key="ocr_editor",
    )

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("✅ İşaretlileri Portföye Ekle", type="primary"):
            added = 0
            for _, row in edited.iterrows():
                if not row["Ekle"]:
                    continue
                symbol, qty, cost = row["Sembol"], row["Adet"], row["Maliyet"]
                if not symbol or pd.isna(qty) or pd.isna(cost) or qty <= 0 or cost <= 0:
                    continue
                try:
                    data = PortfolioItemCreate(symbol=symbol, quantity=float(qty), cost_basis=float(cost), source="ocr")
                    upsert_portfolio_item(db, user_id, data)
                    added += 1
                except Exception:
                    continue
            if added:
                st.success(f"{added} satır portföyünüze eklendi.")
                st.session_state.pop(_OCR_REVIEW_KEY, None)
                st.session_state.pop("ekofin_ocr_last_file", None)
                st.rerun()
            else:
                st.warning("Eklenecek geçerli (sembol + pozitif adet + pozitif maliyet) satır bulunamadı.")
    with c2:
        if st.button("Temizle", key="ocr_clear"):
            st.session_state.pop(_OCR_REVIEW_KEY, None)
            st.session_state.pop("ekofin_ocr_last_file", None)
            st.rerun()
