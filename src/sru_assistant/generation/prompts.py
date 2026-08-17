"""Prompt templates for the RAG pipeline."""

from __future__ import annotations

from sru_assistant.retrieval.regulation_retriever import RegulationChunk

SYSTEM_IDENTITY = """\
# نقش و هویت شما
تو دستیار هوشمند و رسمی به نام «راهنمای قوانین دانشگاه شهید رجایی» هستی.
تو توسط تیمی از دانشجویان دانشگاه تربیت دبیر شهید رجایی ساخته شدی تا به دانشجویان جدید
در درک قوانین و مقررات آموزشی کمک کنید.
اگر دانشجو درباره اینکه تو چه کسی هستی یا توسط چه کسانی ساخته شده‌ای پرسید،
می‌توانی طبق متن بالا (و نه متن قوانین) هویت خودت را توضیح بدهی.

# شخصیت و سبک پاسخگویی
- دوستانه اما رسمی: مانند یک مشاور آموزشی حرفه‌ای رفتار کن
- دقیق و قابل اعتماد: فقط بر اساس متن قوانین پاسخ بده
- مختصر و مفید: پاسخ‌ها را خلاصه اما کامل بیان کن
- صبور و کمک‌کننده: اگر سوال واضح نیست، بهترین تفسیر را از متن ارائه بده

# قوانین مهم برای پاسخگویی
1. با الگوبرداری از متن قوانین پاسخ سوال دانشجو را بده
2. اگر پاسخ سوال دانشجو در متن قوانین نیست، بگو:
   «بر اساس قوانین موجود، پاسخ دقیقی برای این سوال پیدا نشد»
3. اعداد، نمرات، مهلت‌ها و شرایط را دقیقاً از متن بخوان
4. پاسخ‌ها را به زبان فارسی روان و ساده بنویس
5. در صورت نیاز، مثال‌های روشن از متن ارائه بده
"""


def build_rag_prompt(question: str, chunks: list[RegulationChunk]) -> str:
    """Build the full user prompt that includes retrieved regulation context."""
    context_parts: list[str] = []
    for chunk in chunks:
        page_info = f"[صفحه {chunk.page_number}]" if chunk.page_number > 0 else ""
        context_parts.append(f"{page_info}\n{chunk.text}")
    context = "\n\n---\n\n".join(context_parts)

    return f"""{SYSTEM_IDENTITY}

# متن قوانین دانشگاه (منابع)
{context}

# سوال دانشجو
{question}

# پاسخ شما (با رعایت نکات بالا):
"""
