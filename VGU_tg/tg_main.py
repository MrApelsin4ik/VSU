import os
import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any
from functools import partial
import re
import ast

import aiosqlite
import requests
from rapidfuzz import fuzz, process

from aiogram import Bot, Dispatcher, types, F
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove
from aiogram.filters import Command
from aiogram.types.error_event import ErrorEvent

from dotenv import load_dotenv

load_dotenv(".env")

import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
API_URL = os.getenv("API_URL", "http://127.0.0.1:9000").rstrip("/")
SITE_URL = os.getenv("SITE_URL", "http://vsu.apeellsin4ik.ru").rstrip("/")
API_KEY = os.getenv("API_KEY", None)
LM_API_URL = os.getenv("LM_API_URL", '')
LM_API_KEY = os.getenv("LM_API_KEY", None)
DB_PATH = os.getenv("BOT_DB", "bot_data.sqlite3")
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "3000"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "400"))
EXECUTOR = ThreadPoolExecutor(max_workers=8)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not BOT_TOKEN:
    raise RuntimeError("TG_BOT_TOKEN not set")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# ------------------ База данных ------------------
async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                chat_id INTEGER PRIMARY KEY,
                notify_prefs TEXT DEFAULT '{}',
                classroom_info TEXT DEFAULT NULL,
                search_scope TEXT DEFAULT 'sections'
            )
        """)
        await db.commit()

async def get_user_settings(chat_id: int) -> Dict[str, Any]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT notify_prefs, classroom_info, search_scope FROM users WHERE chat_id = ?", (chat_id,))
        row = await cur.fetchone()
        if row:
            notify_prefs = json.loads(row[0]) if row[0] else {}
            classroom = json.loads(row[1]) if row[1] else None
            return {"chat_id": chat_id, "notify_prefs": notify_prefs, "classroom_info": classroom, "search_scope": row[2] or "sections"}
        else:
            await db.execute("INSERT INTO users (chat_id) VALUES (?)", (chat_id,))
            await db.commit()
            return {"chat_id": chat_id, "notify_prefs": {}, "classroom_info": None, "search_scope": "sections"}

async def update_user_settings(chat_id: int, **kwargs):
    async with aiosqlite.connect(DB_PATH) as db:
        if "notify_prefs" in kwargs:
            await db.execute("UPDATE users SET notify_prefs = ? WHERE chat_id = ?", (json.dumps(kwargs["notify_prefs"], ensure_ascii=False), chat_id))
        if "classroom_info" in kwargs:
            val = json.dumps(kwargs["classroom_info"], ensure_ascii=False) if kwargs["classroom_info"] is not None else None
            await db.execute("UPDATE users SET classroom_info = ? WHERE chat_id = ?", (val, chat_id))
        if "search_scope" in kwargs:
            await db.execute("UPDATE users SET search_scope = ? WHERE chat_id = ?", (kwargs["search_scope"], chat_id))
        await db.commit()

# ------------------ Text helpers & JSON extraction ------------------
_POSITIVE_WORDS = [
    "obvious", "obviously", "contains", "likely", "relevant", "yes",
    "явно", "очевид", "содерж", "вероятн", "релевант", "да", "есть", "подход"
]
_NEGATIVE_WORDS = [
    "not", "no", "irrelevant", "doesn't", "does not",
    "не", "не релевант", "не содержит", "нет", "не связано", "неважн"
]

def _clean_role_tokens(text: str) -> str:
    """
    Убирает маркеры каналов/ролевые токены вроде <|channel|>analysis<|message|>
    и похожие. Также обрезает лишние пробелы.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<\|[^|]+\|>", " ", text)

    text = re.sub(r"(analysis|assistant|user)\s*[:>-]*\s*<\|message\|>", " ", text, flags=re.IGNORECASE)

    text = re.sub(r"[\x00-\x1F\x7F\u200B-\u200F]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_first_json_object(text: str):
    """
    Попытка извлечь первый парсируемый JSON-объект из текста LLM.
    Особенности:
    - очищаем ролевые токены и невидимые символы,
    - если есть маркеры типа '<|channel|>final<|message|>' берём содержимое после
      последнего такого маркера (финальная роль),
    - затем пробуем json.loads по всему тексту, потом ищем корректный сбалансированный {...}
      начиная с конца (чтобы предпочесть финальный JSON), аккуратно нормализуем одинарные кавычки,
      в крайнем случае пробуем ast.literal_eval.
    """
    print(text)
    if not text or not isinstance(text, str):
        raise ValueError("empty input")


    txt = _clean_role_tokens(text)


    logger.debug("RAW_LLM_RESPONSE (truncated for parse): %s", (txt or "")[:2000])


    final_markers = [
        "<|channel|>final<|message|>",
        "<|assistant|>final<|message|>",
        "<|final|>",
        "<|channel|>assistant_final<|message|>",
        "<|assistant|>final",
    ]
    last_pos = -1
    last_marker_len = 0
    for marker in final_markers:
        pos = txt.rfind(marker)
        if pos > last_pos:
            last_pos = pos
            last_marker_len = len(marker)
    if last_pos != -1:
        txt = txt[last_pos + last_marker_len:].strip()


    try:
        return json.loads(txt)
    except Exception:
        pass


    start_positions = [m.start() for m in re.finditer(r'\{', txt)]
    if start_positions:

        for start in reversed(start_positions):
            depth = 0
            candidate = None
            for i in range(start, len(txt)):
                ch = txt[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = txt[start:i+1]
                        break
            if not candidate:
                continue

            try:
                return json.loads(candidate)
            except Exception:

                try:
                    cand = candidate
                    cand2 = re.sub(r"(?P<prefix>[:\s\{,])'(.*?)'(?P<suffix>[,\}\]\s])",
                                   lambda m: f"{m.group('prefix')}\"{m.group(2)}\"{m.group('suffix')}", cand)
                    return json.loads(cand2)
                except Exception:
                    try:
                        return ast.literal_eval(candidate)
                    except Exception:

                        continue


    m = re.search(r'\{.*?\}', txt, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            try:
                candidate2 = re.sub(r"'(\s*:\s*)'", lambda mm: '"' + mm.group(1) + '"', candidate)
                candidate2 = re.sub(r"'([^']*)'", lambda mo: '"' + mo.group(1).replace('"', '\\"') + '"', candidate2)
                return json.loads(candidate2)
            except Exception:
                try:
                    return ast.literal_eval(candidate)
                except Exception:
                    pass


    raise ValueError("No parsable JSON object found")




def _normalize_item(it):
    if isinstance(it, dict):
        return it
    if isinstance(it, str):
        try:
            parsed = json.loads(it)
            if isinstance(parsed, dict):
                return parsed
            return {"title": str(parsed), "raw": parsed}
        except Exception:
            return {"title": it, "raw": it}
    try:
        return {"title": str(it), "raw": it}
    except Exception:
        return {"title": "<unknown>", "raw": it}

# ------------------ Utilities ------------------
SENTENCE_END_RE = re.compile(r'[.!?…](?:\s+|$)')
WHITESPACE_RE = re.compile(r'\s+')
def strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    return WHITESPACE_RE.sub(" ", text).strip()


def chunk_text(text: str, size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    if overlap >= size:
        raise ValueError("Overlap must be smaller than chunk size")

    chunks = []
    start = 0
    n = len(text)

    while start < n:

        end = min(start + size, n)


        if end < n:

            chunk_segment = text[start:end]
            sentence_breaks = []

            for match in SENTENCE_END_RE.finditer(chunk_segment):
                sentence_breaks.append(match.end())

            if sentence_breaks:

                end = start + sentence_breaks[-1]
            else:

                last_space = chunk_segment.rfind(' ')
                if last_space != -1 and last_space > size * 0.5:
                    end = start + last_space


        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)


        if end == n:
            break


        next_start = end - overlap


        search_area = text[next_start:min(next_start + overlap, n)]
        sentence_match = SENTENCE_END_RE.search(search_area)

        if sentence_match:

            next_start = next_start + sentence_match.end()
        else:

            first_space = search_area.find(' ')
            if first_space != -1:
                next_start = next_start + first_space + 1


        start = max(next_start, start + 1)


        if start >= n:
            break

    return chunks




def _extract_final_content(text: str) -> str:
    """
    Извлекает только содержимое после финального маркера <|channel|>final<|message|>
    и обрезает все предыдущие рассуждения и анализ.
    """
    if not text or not isinstance(text, str):
        return ""


    final_markers = [
        "<|channel|>final<|message|>",
        "<|assistant|>final<|message|>",
        "<|final|>",
        "<|channel|>assistant_final<|message|>",
        "<|assistant|>final",
    ]

    last_pos = -1
    last_marker_len = 0
    for marker in final_markers:
        pos = text.rfind(marker)
        if pos > last_pos:
            last_pos = pos
            last_marker_len = len(marker)

    if last_pos != -1:

        final_content = text[last_pos + last_marker_len:].strip()
        # Очищаем от оставшихся технических токенов
        final_content = _clean_role_tokens(final_content)
        return final_content


    return _clean_role_tokens(text)

# ------------------ LLM ------------------
SYSTEM_PROMPT = "Ты — ассистент, который кратко извлекает полезную информацию из описаний и файлов. Отвечай по-русски."

def call_llm_sync(prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
    if not LM_API_URL:
        return '{"debug": "no-lm"}'
    payload = {"model": os.getenv("LM_MODEL", "openai/gpt-oss-20b"), "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}], "temperature": temperature}
    print(f'---------PAYLOAD---------\n{payload}')
    headers = {"Content-Type": "application/json"}
    if LM_API_KEY: headers["Authorization"] = f"Bearer {LM_API_KEY}"
    try:
        r = requests.post(LM_API_URL, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "choices" in data and data["choices"]:
            first = data["choices"][0]
            return first.get("message", {}).get("content") or first.get("text") or ""
        print(f'---------RESPONSE---------\n{str(data)}')
        return str(data)
    except Exception as e:
        logger.exception("LLM request failed")
        return f"[LLM_ERROR] {e}"

# ------------------ API Client ------------------
class SimpleAPIClientSync:
    def __init__(self, base_url: str = API_URL, api_key: Optional[str] = API_KEY, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-API-KEY"] = self.api_key
        return h

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_items(self, model: Optional[str] = None, rng: Optional[str] = None, all_items: bool = False, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            url = self._url("/api/get/")
            params = {}
            if model:
                params["model"] = model
            if rng:
                params["range"] = rng
            if all_items:
                params["all"] = "1"
            r = requests.get(url, headers=self._headers(), params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json() if r.status_code == 200 else []
            if limit and isinstance(data, list):
                return data[:limit]
            return data
        except Exception as e:
            logger.exception("get_items failed: %s", e)
            return []

    def search(self, query: Optional[str] = None, sections: Optional[List[str]] = None, limit: int = 200) -> List[Dict[str, Any]]:
        try:
            items = self.get_items(limit=limit)
            if not isinstance(items, list):
                items = []
        except Exception:
            items = []

        if sections:
            try:
                items = [it for it in items if (it.get("section") in sections) or (it.get("type") in sections) or (it.get("model") in sections)]
            except Exception:
                pass

        return items

    def get_full(self, model: str, pk: int) -> Dict[str, Any]:
        url = self._url(f"/api/ai/full/{model}/{pk}/")
        r = requests.get(url, headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

# ------------------ Typing indicator ------------------
async def periodic_typing(chat_id: int, stop_event: asyncio.Event, interval: float = 7.0):
    try:
        while not stop_event.is_set():
            try:
                await bot.send_chat_action(chat_id, action=types.ChatActions.TYPING)
            except Exception:
                pass
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass

# ------------------ Workflow with fuzzy fallback ------------------
async def workflow_search_and_answer(chat_id: int, query: str, user_settings: Dict[str, Any]):
    api = SimpleAPIClientSync()
    try:
        await bot.send_chat_action(chat_id, action=types.ChatActions.TYPING)
    except Exception:
        pass

    sections = ["news", "announcement", "courses"] if user_settings.get("search_scope") == "sections" else None
    loop = asyncio.get_event_loop()
    try:
        items = await loop.run_in_executor(EXECUTOR, partial(api.get_items, None, None, False, 1000))
        if items is None:
            items = []
        if isinstance(items, dict) and "items" in items and isinstance(items["items"], list):
            items = items["items"]
        normalized = []
        for it in items:
            nit = _normalize_item(it)
            normalized.append(nit)
        items = normalized

        if sections is not None:
            items = [it for it in items if (it.get("section") in sections) or (it.get("type") in sections) or (it.get("model") in sections)]
    except Exception:
        items = []

    async def do_fuzzy():
        def compute():
            choices = []
            mapping = {}
            for it in items:
                title = (it.get("title") or it.get("name") or "").strip()
                desc = (it.get("description_ai") or it.get("description") or "").strip()
                text = f"{title} — {desc}" if desc else title
                if not text:
                    continue
                choices.append(text)
                mapping[text] = it
            if not choices:
                return []
            try:
                raw = process.extract(query, choices, scorer=fuzz.token_sort_ratio, limit=6)
            except Exception:
                raw = []
            out = []
            for row in raw:
                if not row:
                    continue
                match = row[0]
                score = int(row[1]) if len(row) > 1 else 0
                if score < 40:
                    continue
                it = mapping.get(match)
                if it:
                    out.append({"score": score, "item": it})
            return out

        res = await loop.run_in_executor(EXECUTOR, compute)
        if not res:
            return
        lines = ["*Быстрый поиск*\nПохожие элементы (подробный ответ на ваш вопрос все еще готовится):"]
        for s in res:
            it = s["item"]
            model = it.get("model") or it.get("type") or "news"
            pk = it.get("id") or it.get("pk")
            title = it.get("title") or it.get("name") or "(без названия)"
            score = s.get("score")
            link = f"{SITE_URL}/{model}/{pk}/" if pk else API_URL
            #lines.append(f"{title} — {model}:{pk} (score={score}) \n{link}")
            lines.append(f"{title}\n{link}")
        try:
            await bot.send_message(chat_id, "\n".join(lines))
        except Exception:
            pass

    async def do_llm():
        L1_context_limit = 3000
        ctx = []
        ctx_size = 0
        candidates = []
        for it in items:
            desc = it.get("description_ai") or it.get("title") or ""
            model = it.get("model") or it.get("type") or "news"
            pk = it.get("id") or it.get("pk")
            title = it.get("title") or it.get("name") or ""
            if not pk:
                continue
            part = f"{model}:{pk} {title}\n{desc}\n"
            if ctx_size + len(part) > L1_context_limit and ctx_size > 0:
                break
            ctx.append(part)
            ctx_size += len(part)
            candidates.append({"model": model, "id": int(pk), "title": title, "description_ai": desc, "raw": it})

        prompt_l1 = (
            "У нас есть вопрос пользователя:\n\n"
            f"Q: {query}\n\n"
            "Даны следующие краткие описания (L1). Для каждого элемента ответь JSON-объектом, где ключ — идентификатор в формате model:id, а значение — 'y' если в описании вероятно есть полезная информация для ответа, иначе 'n'.\n\n"
            "L1:\n"
            + "\n---\n".join(ctx)
            + "\n\nОтвечай только JSON, например: {\"news:123\": \"y\", \"course:1\": \"n\"}"
        )

        stop_event = asyncio.Event()
        typing_task = asyncio.create_task(periodic_typing(chat_id, stop_event, interval=7.0))

        try:
            print('pl1',prompt_l1)
            raw_resp = await loop.run_in_executor(EXECUTOR, partial(call_llm_sync, prompt_l1, SYSTEM_PROMPT, 0.0))
            print('rr',raw_resp)
            if raw_resp is None:
                raw_resp = ""
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            raw_resp = ""

        useful_map = {}
        try:
            parsed = _extract_first_json_object(raw_resp)
            if isinstance(parsed, dict):
                useful_map = parsed
            else:
                logger.warning("L1: parsed JSON is not dict (type=%s). Using heuristic.", type(parsed))
        except Exception:
            logger.exception("Failed parse L1 usefulness JSON, using heuristic; raw LLM response (truncated): %s", (raw_resp or '')[:800])
            qlow = query.lower()
            for c in candidates:
                useful_map[f"{c['model']}:{c['id']}"] = "y" if (
                    qlow in (c['description_ai'] or "").lower() or qlow in c['title'].lower()) else "n"

        results_texts = []
        processed_l2_keys = set()
        for c in candidates:

            key = f"{c['model']}:{c['id']}"
            if useful_map.get(key) != "y":
                continue
            if key in processed_l2_keys:
                continue
            processed_l2_keys.add(key)
            try:
                full = await loop.run_in_executor(EXECUTOR, partial(api.get_full, c['model'], c['id']))
            except Exception as e:
                logger.exception("Failed get_full for %s: %s", key, e)
                continue

            descr_l2 = full.get("description_ai_l2") or full.get("description_ai_l2_text") or ""
            if descr_l2 and len(descr_l2) > 50:
                chunks = chunk_text(descr_l2, size=2000, overlap=300)
                print(chunks)

                for ch in chunks:
                    prompt_chunk = (
                        f"Вопрос: {query}\n\n"
                        f"Фрагмент:\n'''{ch}'''\n\n"
                        "Найди внутри этого фрагмента все предложения или короткие фрагменты, которые прямо отвечают на вопрос или содержат факты/данные полезные для ответа. "
                        "Если ничего полезного — верни пустую строку (или 'none'). Ограничь ответ 500 символами."
                    )

                    resp_chunk = await loop.run_in_executor(EXECUTOR, partial(call_llm_sync, prompt_chunk, SYSTEM_PROMPT, 0.0))
                    resp_chunk = _extract_final_content(resp_chunk)
                    if 'none' in resp_chunk:
                        resp_chunk = ''
                    resp_chunk = resp_chunk.strip() if resp_chunk else ""
                    if resp_chunk:
                        #results_texts.append(f"Из {key} ({c['title']}) найдено:\n{resp_chunk}")
                        results_texts.append(f"Найдено:\n{resp_chunk}")
            else:
                text = ""
                for k in ("text", "content", "body", "description", "full_text", "article", "html"):
                    v = full.get(k)
                    if v and isinstance(v, str) and v.strip():
                        text = strip_html(v) if k == "html" else v.strip()
                        break

                attachments = full.get("attachments") or full.get("files") or full.get("resources") or []
                if not text and attachments:
                    try:
                        await bot.send_message(chat_id, f"Для более точного ответа хочу прочитать файлы из {key} — начинаю загрузку и извлечение.")
                    except Exception:
                        pass
                    extracted_texts = []
                    tmp_dir = os.path.join("tmp_bot_files", f"{c['model']}_{c['id']}")
                    os.makedirs(tmp_dir, exist_ok=True)
                    for att in attachments:
                        url = None
                        if isinstance(att, dict):
                            url = att.get("url") or att.get("download_url") or att.get("file")
                        elif isinstance(att, str):
                            url = att
                        if not url:
                            continue
                        filename = os.path.basename(url.split("?")[0]) or f"{c['id']}"
                        dest_path = os.path.join(tmp_dir, filename)
                        try:

                            if hasattr(api, 'download_file'):
                                await loop.run_in_executor(EXECUTOR, partial(api.download_file, url, dest_path))
                            lower = dest_path.lower()
                            extracted = ""
                            if lower.endswith(".txt") and os.path.exists(dest_path):
                                with open(dest_path, "r", encoding="utf-8", errors="ignore") as f:
                                    extracted = f.read()
                            elif lower.endswith(".pdf") and os.path.exists(dest_path):
                                try:
                                    import fitz
                                    doc = fitz.open(dest_path)
                                    pages = []
                                    for p in doc:
                                        pages.append(p.get_text("text"))
                                    extracted = "\n".join(pages)
                                except Exception:
                                    logger.exception("pdf extraction failed for %s", dest_path)
                            elif lower.endswith(".docx") and os.path.exists(dest_path):
                                try:
                                    from docx import Document
                                    doc = Document(dest_path)
                                    parts = [p.text for p in doc.paragraphs if p.text.strip()]
                                    extracted = "\n".join(parts)
                                except Exception:
                                    logger.exception("docx extract failed")
                            if extracted:
                                extracted_texts.append(extracted)
                        except Exception:
                            logger.exception("Download/extract failed for %s", url)
                    text = "\n\n".join(extracted_texts)

                if text:
                    chunks = chunk_text(text, size=2000, overlap=300)
                    for ch in chunks:
                        prompt_chunk = (
                            f"Вопрос: {query}\n\n"
                            f"Фрагмент:\n'''{ch}'''\n\n"
                            "Если есть в этом фрагменте информация, которая помогает ответить на вопрос — выдай её кратко (1-3 предложения). Иначе — возвращай пустую строку."
                        )
                        resp_chunk = await loop.run_in_executor(EXECUTOR, partial(call_llm_sync, prompt_chunk, SYSTEM_PROMPT, 0.0))
                        resp_chunk = resp_chunk.strip() if resp_chunk else ""
                        if resp_chunk:
                            results_texts.append(f"Из {key} ({c['title']}) найдено:\n{resp_chunk}")

        stop_event.set()
        try:
            await typing_task
        except Exception:
            pass

        if not results_texts:
            await bot.send_message(chat_id, "Не удалось найти прямых фактов по запросу в доступных описаниях/файлах.")
        else:
            try:
                await bot.send_chat_action(chat_id, action=types.ChatActions.TYPING)
            except Exception:
                pass
            final_answer = "\n\n".join(results_texts[:10])
            if len(final_answer) > 3500:
                fname = f"answer_{chat_id}.txt"
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(final_answer)
                try:
                    await bot.send_document(chat_id, types.InputFile(fname))
                    os.remove(fname)
                except Exception as e:
                    logger.exception("Failed send document: %s", e)
                    await bot.send_message(chat_id, final_answer[:3500] + "\n\n(и ещё...)")
            else:
                await bot.send_message(chat_id, final_answer)

    await asyncio.gather(asyncio.create_task(do_fuzzy()), asyncio.create_task(do_llm()))

# ------------------ Keyboards ------------------
def main_menu_kb() -> types.InlineKeyboardMarkup:
    kb = types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="выбор оповещений", callback_data="menu_notify")],
        [types.InlineKeyboardButton(text="подключить classroom", callback_data="menu_classroom")],
        [types.InlineKeyboardButton(text="настройка поиска", callback_data="menu_search")],
    ])
    return kb


def search_scope_kb(current_scope: str) -> types.InlineKeyboardMarkup:
    kb = types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(
            text="Искать только в соответствующих разделах (быстрее)" + (" ✅" if current_scope == "sections" else ""),
            callback_data="scope_sections"
        )],
        [types.InlineKeyboardButton(
            text="Искать везде (медленнее)" + (" ✅" if current_scope == "all" else ""),
            callback_data="scope_all"
        )],
        [types.InlineKeyboardButton(text="Назад в меню", callback_data="menu_back")],
    ])
    return kb


def notify_kb(current: Dict[str, Any]) -> types.InlineKeyboardMarkup:
    kb = types.InlineKeyboardMarkup(inline_keyboard=[])
    rows = []
    for key in ("news", "announcement", "courses"):
        on = bool(current.get(key, False))
        label = f"{'✅' if on else '◻️'} {key}"
        rows.append([types.InlineKeyboardButton(text=label, callback_data=f"toggle_notify:{key}")])
    rows.append([types.InlineKeyboardButton(text="Назад в меню", callback_data="menu_back")])
    kb = types.InlineKeyboardMarkup(inline_keyboard=rows)
    return kb

# ------------------ Handlers ------------------
@dp.message(Command(commands=["start", "help"]))
async def cmd_start(message: types.Message):
    await init_db()
    await get_user_settings(message.chat.id)
    text = (
        "Привет! Я — поисковый ассистент. Выберите действие или просто напишите вопрос — я попробую найти ответ по базе.\n\n"
    )
    await message.answer(text, reply_markup=main_menu_kb())


@dp.message(Command("search_settings"))
async def open_search_settings(msg: types.Message):
    settings = await get_user_settings(msg.chat.id)
    kb = search_scope_kb(settings["search_scope"])
    await msg.answer("Настройка поиска:", reply_markup=kb)


@dp.callback_query(F.data.startswith("menu_"))
async def callbacks_menu(cb: types.CallbackQuery):
    data = cb.data
    chat_id = cb.message.chat.id if cb.message else cb.from_user.id
    if data == "menu_notify":
        settings = await get_user_settings(chat_id)
        kb = notify_kb(settings.get("notify_prefs", {}))
        try:
            await cb.message.edit_text("Настройка оповещений:", reply_markup=kb)
        except Exception:
            await bot.send_message(chat_id, "Настройка оповещений:", reply_markup=kb)
    elif data == "menu_classroom":
        try:
            await cb.message.edit_text("Подключение classroom: отправьте токен или URL авторизации.", reply_markup=None)
        except Exception:
            await bot.send_message(chat_id, "Подключение classroom: отправьте токен или URL авторизации.")
    elif data == "menu_search":
        settings = await get_user_settings(chat_id)
        try:
            await cb.message.edit_text("Настройка поиска:", reply_markup=search_scope_kb(settings["search_scope"]))
        except Exception:
            await bot.send_message(chat_id, "Настройка поиска:", reply_markup=search_scope_kb(settings["search_scope"]))
    elif data == "menu_back":
        try:
            await cb.message.edit_text("Главное меню:", reply_markup=main_menu_kb())
        except Exception:
            await bot.send_message(chat_id, "Главное меню:", reply_markup=main_menu_kb())
    await cb.answer()


@dp.callback_query(F.data.startswith("scope_"))
async def callbacks_scope(cb: types.CallbackQuery):
    chat_id = cb.message.chat.id if cb.message else cb.from_user.id
    data = cb.data
    if data == "scope_sections":
        await update_user_settings(chat_id, search_scope="sections")
        try:
            await cb.message.edit_text("Теперь поиск будет выполняться только в соответствующих разделах (быстрее).", reply_markup=main_menu_kb())
        except Exception:
            await bot.send_message(chat_id, "Теперь поиск будет выполняться только в соответствующих разделах (быстрее).", reply_markup=main_menu_kb())
    elif data == "scope_all":
        await update_user_settings(chat_id, search_scope="all")
        try:
            await cb.message.edit_text("Теперь поиск будет выполняться во всех разделах (медленнее).", reply_markup=main_menu_kb())
        except Exception:
            await bot.send_message(chat_id, "Теперь поиск будет выполняться во всех разделах (медленнее).", reply_markup=main_menu_kb())
    await cb.answer()


@dp.callback_query(F.data.startswith("toggle_notify:"))
async def callbacks_toggle_notify(cb: types.CallbackQuery):
    key = cb.data.split(":", 1)[1]
    chat_id = cb.message.chat.id if cb.message else cb.from_user.id
    settings = await get_user_settings(chat_id)
    prefs = settings.get("notify_prefs", {})
    prefs[key] = not prefs.get(key, False)
    await update_user_settings(chat_id, notify_prefs=prefs)
    await cb.answer(f"{key}: {'вкл' if prefs[key] else 'выкл'}")
    try:
        await cb.message.edit_reply_markup(reply_markup=notify_kb(prefs))
    except Exception:
        pass


@dp.message(F.text.startswith("http"), F.reply_to_message.is_(None))
async def handle_classroom_url(message: types.Message):
    chat_id = message.chat.id
    info = {"url": message.text}
    await update_user_settings(chat_id, classroom_info=info)
    await message.answer("Classroom подключен (сохранён URL).", reply_markup=main_menu_kb())


@dp.message(F.text)
async def handle_text(message: types.Message):
    if message.text.startswith("/"):
        return
    chat_id = message.chat.id
    text = message.text.strip()
    settings = await get_user_settings(chat_id)
    await message.answer("Ищу информацию...", reply_markup=ReplyKeyboardRemove())

    asyncio.create_task(workflow_search_and_answer(chat_id, text, settings))


async def global_error_handler(event: ErrorEvent):
    exc = event.exception
    upd = event.update

    logger.exception("Unhandled exception: %s", exc)

    try:
        if upd and getattr(upd, "message", None):
            await upd.message.answer("❗ Произошла ошибка, попробуйте снова.")
    except Exception as e:
        logger.exception("Failed to notify user about error: %s", e)

    return True


dp.errors.register(global_error_handler)


async def main():
    await init_db()
    logger.info("Starting bot (aiogram 3.x)...")
    try:
        await dp.start_polling(bot, allowed_updates=[
            "message",
            "callback_query",
            "inline_query",
        ])
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
