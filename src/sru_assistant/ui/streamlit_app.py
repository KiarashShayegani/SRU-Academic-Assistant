"""Streamlit front-end for sru-academic-assistant."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from sru_assistant.config import get_settings
from sru_assistant.pipeline.answer import answer_question


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "default.yaml").exists():
            return parent
    return Path.cwd()


def load_css() -> str:
    css_path = _project_root() / "assets" / "css" / "streamlit_theme.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return ""


# Categorised quick questions (from the mature v4 app)
FAQ_CATEGORIES: dict[str, list[str]] = {
    "کلیات و پذیرش": [
        "دانشگاه تربیت دبیر شهید رجایی چه نوع دانشگاهی است؟",
        "شرط ورود به دانشگاه شهید رجایی چیست؟",
        "تعهد خدمت دانشجویان این دانشگاه به چه معناست؟",
        "آدرس سایت دانشگاه چیست؟",
        "زبان تدریس در دانشگاه چیست؟",
        "آیا آموزش رایگان برای هر دوره تحصیلی فقط یک‌بار امکان‌پذیر است؟",
    ],
    "دروس و انتخاب واحد": [
        "حداقل و حداکثر واحد قابل اخذ در هر نیمسال تحصیلی چقدر است؟",
        "حداکثر واحد قابل اخذ در هر نیمسال چقدر است؟",
        "دانشجوی ممتاز چند واحد می‌تواند اخذ کند؟",
        "دانشجوی ترم آخر چند واحد می‌تواند اخذ کند؟",
        "دروس دانشگاهی از نظر محتوا به چه دسته‌هایی تقسیم می‌شوند؟",
        "درس پیش‌نیاز به چه معناست؟",
        "تأخیر در انتخاب واحد چه پیامدی دارد؟",
    ],
    "امتحانات و نمرات": [
        "حداقل نمره قبولی چند است؟",
        "مهلت اعتراض به نمره چقدر است؟",
        "مهلت اعلام نمرات امتحان هر درس چقدر است؟",
        "اگر دانشجو در جلسه امتحان غایب شود چه نمره‌ای می‌گیرد؟",
        "تخلفات امتحانی شامل چه مواردی می‌شود؟",
        "بعد از قطعی شدن نمره، آیا امکان تغییر آن وجود دارد؟",
    ],
    "مرخصی و انصراف": [
        "شرایط مرخصی تحصیلی چیست؟",
        "انواع مرخصی بدون احتساب در سنوات چیست و هرکدام چقدر مدت دارد؟",
        "مرخصی با احتساب در سنوات چقدر مدت دارد؟",
        "آیا دانشجو می‌تواند بیش از یک‌بار انصراف از تحصیل بدهد؟",
        "عدم انتخاب واحد در یک نیمسال چه پیامدی دارد؟",
        "بعد از انصراف از تحصیل، آیا امکان بازگشت به تحصیل وجود دارد؟",
    ],
    "مشروطی و سنوات": [
        "حداقل نمره قبولی در هر درس و حداقل میانگین قابل قبول در هر نیمسال چقدر است؟",
        "دانشجو بعد از چند نیمسال مشروطی از تحصیل محروم می‌شود؟",
        "حداکثر مدت مجاز تحصیل در دوره کارشناسی پیوسته چند نیمسال است؟",
        "امکان تمدید سنوات مجاز تحصیلی چگونه است؟",
        "سنوات ارفاقی چیست و چقدر است؟",
    ],
    "حذف درس": [
        "حذف اضطراری درس چه شرایطی دارد؟",
        "حذف پزشکی درس چگونه انجام می‌شود؟",
        "آیا بیماری‌های ساده مثل سرماخوردگی برای حذف پزشکی کافی است؟",
        "حذف کامل تمام دروس یک نیمسال (حذف ترم) چه شرایطی دارد؟",
    ],
    "تغییر رشته، انتقال و مهمانی": [
        "چگونه برای مهمان شدن به دانشگاه دیگر اقدام کنم؟",
        "شرایط دانشجوی مهمان برای رفتن به دانشگاه دیگر چیست؟",
        "سقف پذیرش دانشجویان مهمان در هر سال تحصیلی چقدر است؟",
        "آیا دانشجوی متعهد خدمت می‌تواند رشته یا دانشگاه خود را تغییر دهد؟",
        "آیا انتقال از سایر دانشگاه‌ها به دانشگاه شهید رجایی ممکن است؟",
    ],
    "کارآموزی و پروژه": [
        "آیا درس کارآموزی اجباری است و معادل واحدی آن چقدر است؟",
        "پروژه کارشناسی چگونه ارزیابی می‌شود؟",
        "آیا امکان تمدید پروژه کارشناسی وجود دارد؟",
        "شرایط استفاده از معرفی به استاد (تک‌درس) چیست؟",
    ],
    "فارغ‌التحصیلی و مدارک": [
        "چگونه می‌توانم درخواست فارغ‌التحصیلی بدهم؟",
        "ملاک دانش‌آموختگی (فارغ‌التحصیلی) چیست؟",
        "اگر معدل کل دانشجو بعد از اتمام دروس کمتر از ۱۲ باشد چه راهی برای ترمیم آن وجود دارد؟",
        "در صورت مفقود شدن کارت دانشجویی چه باید کرد؟",
        "چه مدارک تحصیلی از سامانه جامع آموزشی قابل دریافت است؟",
    ],
    "استاد مشاور و سایر موارد": [
        "استاد مشاور آموزشی چه وظیفه‌ای دارد؟",
        "جابجایی زمان یا مکان برگزاری کلاس چگونه امکان‌پذیر است؟",
        "استرداد شهریه دانشجویان بستانکار چگونه انجام می‌شود؟",
        "برای خروج از کشور دانشجوی متعهد خدمت چه فرآیندی باید طی شود؟",
    ],
}


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "faq_question" not in st.session_state:
        st.session_state.faq_question = ""
    if "use_rag" not in st.session_state:
        settings = get_settings()
        st.session_state.use_rag = settings.default_mode == "rag"


def _log_feedback(kind: str, question: str, preview: str) -> None:
    try:
        with open("feedback_log.txt", "a", encoding="utf-8") as f:
            f.write(
                f"{datetime.now().isoformat()} | {kind} | Q: {question[:100]} | "
                f"Response: {preview[:100]}\n"
            )
    except OSError:
        pass


def main() -> None:
    settings = get_settings()
    st.set_page_config(
        page_title=settings.page_title,
        page_icon=settings.page_icon,
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    css = load_css()
    st.markdown(
        f"<style>{css}</style>"
        f"""
        <div class="hero-card">
            <div class="hero-title">{settings.page_title}</div>
            <div class="hero-subtitle">پاسخگویی هوشمند بر اساس آیین‌نامه‌های آموزشی دانشگاه</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _init_session()

    # ---- Mode selector ----
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:12px;">
            <h4 style="color:#2a5d79; margin-bottom:8px;">⚙️ انتخاب روش پاسخگویی</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_l, col_c, col_r = st.columns([3, 2, 3])
    with col_c:
        choice = st.radio(
            "",
            ["📋 پایگاه سوالات", "🧠 هوش مصنوعی"],
            horizontal=True,
            index=1 if st.session_state.use_rag else 0,
            label_visibility="collapsed",
        )
    st.session_state.use_rag = choice == "🧠 هوش مصنوعی"
    mode = "rag" if st.session_state.use_rag else "faq"

    # ---- Category browser ----
    st.markdown(
        """
        <div style="text-align:center; color:#60798a; font-size:14px; margin:14px 0;">
        سوالات پرتکرار را از دسته‌بندی زیر انتخاب کنید یا سوال خود را در پایین صفحه تایپ کنید
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "faq_category" not in st.session_state:
        st.session_state.faq_category = list(FAQ_CATEGORIES.keys())[0]

    selected = st.radio(
        "دسته‌بندی سوالات",
        list(FAQ_CATEGORIES.keys()),
        horizontal=True,
        label_visibility="collapsed",
        key="faq_category",
    )
    questions = FAQ_CATEGORIES[selected]
    for i in range(0, len(questions), 2):
        row = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx < len(questions):
                with row[j]:
                    if st.button(questions[idx], key=f"faq_{selected}_{idx}", use_container_width=True):
                        st.session_state.faq_question = questions[idx]
                        st.rerun()

    st.markdown("---")

    # ---- Chat history ----
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("source_label"):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 مفید", key=f"up_{idx}"):
                        st.toast("👍 از بازخورد شما متشکریم!")
                        if idx > 0 and st.session_state.messages[idx - 1]["role"] == "user":
                            _log_feedback(
                                "positive",
                                st.session_state.messages[idx - 1]["content"],
                                message["content"],
                            )
                with col2:
                    if st.button("👎 غیر مفید", key=f"down_{idx}"):
                        st.toast("👎 متاسفیم! این به بهبود سیستم کمک می‌کند.")
                        if idx > 0 and st.session_state.messages[idx - 1]["role"] == "user":
                            _log_feedback(
                                "negative",
                                st.session_state.messages[idx - 1]["content"],
                                message["content"],
                            )
                st.markdown(
                    f'<div class="source-card">{message["source_label"]}</div>',
                    unsafe_allow_html=True,
                )

    # ---- Input ----
    prompt = st.chat_input("سوال خود را بپرسید...")
    if st.session_state.faq_question:
        prompt = st.session_state.faq_question
        st.session_state.faq_question = ""

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            thinking = st.empty()
            if mode == "rag":
                thinking.info("🧠 در حال تولید پاسخ هوشمند...")
            else:
                thinking.info("🔎 در حال بررسی پایگاه سوالات...")

            message_box = st.empty()
            full_response = ""
            source_label = ""

            try:
                start = time.time()
                result = answer_question(prompt, mode=mode)
                thinking.empty()

                if result.is_streaming() and result.stream is not None:
                    for token in result.stream:
                        full_response += token
                        message_box.markdown(full_response + "▌")
                        time.sleep(0.015)
                    message_box.markdown(full_response)
                else:
                    full_response = result.answer or ""
                    message_box.markdown(full_response)

                source_label = result.source_label
                if settings.show_sources and source_label:
                    st.markdown(
                        f'<div class="source-card">{source_label}</div>',
                        unsafe_allow_html=True,
                    )
                if settings.show_timing:
                    elapsed = time.time() - start
                    st.markdown(
                        f'<div class="source-card">⏱️ {elapsed:.1f} ثانیه</div>',
                        unsafe_allow_html=True,
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "source_label": source_label,
                    }
                )
            except Exception as e:
                thinking.empty()
                err = f"⚠️ خطا: {e}"
                message_box.markdown(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err, "source_label": ""}
                )

    if st.session_state.messages:
        st.markdown("---")
        c1, _ = st.columns([1, 4])
        with c1:
            if st.button("🗑️ پاک کردن گفتگو", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.session_state.faq_question = ""
                st.rerun()


if __name__ == "__main__":
    main()
