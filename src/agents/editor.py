import aiosqlite
import aiohttp
import logging
import os
import re
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from src.categories import CANONICAL_CATEGORIES
from src.database.client import get_curated_posts, insert_digest, insert_digest_with_items, get_topics_for_posts
from src.llm.claude_headless import generate_with_claude, trace_run

logger = logging.getLogger(__name__)

CATEGORY_LABELS = {
    "ai":         "🤖 Искусственный интеллект",
    "crypto":     "🔗 Крипто / Web3",
    "startup":    "🚀 Стартапы / Продукт",
    "psychology": "🧠 Психология / Саморазвитие",
    "science":    "🔬 Наука / Нейронаука",
    "research":   "📄 Статья дня",
    "fitness":    "💪 Спорт / Здоровье",
    "fintech":    "💰 Финансы / Инвестиции",
    "other":      "📌 Разное",
    "newsletter": "📧 Newsletters / Email",
}

# Phase 12 / TOPIC-04: human-readable display labels for snake_case topic keys.
# Falls back to title-cased version if the topic is missing (defense against
# future ALLOWED_TOPICS additions outpacing this dict).
TOPIC_DISPLAY_LABELS: dict[str, str] = {
    # ai/ml
    "ml": "ML / Machine Learning",
    "llm": "LLM",
    "agents": "AI Agents",
    "ai_tools": "AI Tools",
    "ml_research": "ML Research",
    "llm_eval": "LLM Evaluation",
    "prompt_eng": "Prompt Engineering",
    # crypto
    "defi": "DeFi",
    "btc_eth": "BTC / ETH",
    "blockchain_infra": "Blockchain Infrastructure",
    # startup
    "startup_funding": "Funding",
    "indie_dev": "Indie Dev",
    "founders": "Founders",
    "hiring": "Hiring",
    "content_creation": "Content Creation",
    # fitness
    "fitness_strength": "Strength Training",
    "jiu_jitsu": "Jiu-Jitsu",
    "longevity": "Longevity",
    # fintech
    "fintech": "Fintech",
    "banking": "Banking",
    # tech
    "devops": "DevOps",
    "infra": "Infrastructure",
    "observability": "Observability",
    "security": "Security",
    # knowledge
    "philosophy": "Philosophy",
    "psychology_self": "Psychology",
    "learning": "Learning",
    "books": "Books",
    "productivity": "Productivity",
    # science
    "science_news": "Science",
}


def _topic_label(snake: str) -> str:
    """Phase 12 / TOPIC-04: human-readable display label for a topic snake_case key.
    Falls back to title-cased version if the topic is missing from the map
    (defense against future ALLOWED_TOPICS additions outpacing this dict).
    """
    return TOPIC_DISPLAY_LABELS.get(snake, snake.replace("_", " ").title())


def _group_posts_by_topic(
    posts: list[dict],
    topics_by_post: dict[int, list[dict]],
) -> tuple[list[tuple[str, list[dict]]], list[dict]]:
    """Phase 12 / TOPIC-04 + D-18, D-19: split a category's posts into
    (topic_groups, untopiced_posts).

    topic_groups: list of (topic_key, [posts...]) tuples, sorted by topic_key
                  ASC for deterministic digest output. Each post appears in
                  exactly ONE group — its FIRST topic (D-18 primary).
    untopiced_posts: posts with no topics, rendered under the category
                     heading WITHOUT a sub-header (D-19).
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    untopiced: list[dict] = []
    for post in posts:
        post_topics = topics_by_post.get(post["id"], [])
        if not post_topics:
            untopiced.append(post)
            continue
        # D-18: primary topic = first entry (Plan 02 ORDER BY confidence DESC)
        primary = post_topics[0]["name"]
        groups[primary].append(post)
    # Sort groups by topic key for deterministic output
    sorted_groups = sorted(groups.items())
    return sorted_groups, untopiced

STRICT_CONSTRAINTS = """
- ЗАПРЕЩЕНО: Любые вступления типа "Вот ваш дайджест" или "Этот пост обсуждает".
- ЗАПРЕЩЕНО: Притягивать связь за уши. Если связи с проектами нет — пиши [Общий контекст].
- ЗАПРЕЩЕНО: Давать советы по созданию контента или "следующие шаги".
- ФОРМАТ: Строго по шаблону. Никаких лишних слов.
"""

_STATIC_PROJECTS_FALLBACK = """Your projects (configure via PROJECT_GRAPH_PATH or EDITOR_PROJECTS_FALLBACK env):
- Project A: short description
- Project B: short description"""

# Default backend per category. Claude CLI (free headless) for fresh-tech categories;
# Ollama for voice-sensitive categories where local model is tuned.
_DEFAULT_CATEGORY_BACKENDS: dict[str, str] = {
    "ai": "claude",
    "crypto": "claude",
}


def _category_backend(cat: str, global_backend: str) -> str:
    """Return the editor backend to use for this category.

    Resolution order:
      1. EDITOR_BACKEND_{CAT} env var (e.g. EDITOR_BACKEND_AI=ollama overrides default)
      2. _DEFAULT_CATEGORY_BACKENDS dict
      3. global_backend (the pipeline-level default)
    """
    env_val = os.environ.get(f"EDITOR_BACKEND_{cat.upper()}")
    if env_val:
        return env_val
    return _DEFAULT_CATEGORY_BACKENDS.get(cat, global_backend)


def _load_projects_block() -> str:
    """Read project-graph.md from second_brain output. Strip frontmatter, prepend
    instruction header. Fall back to static list on read error.

    DO NOT mention autorss-feed itself — it's the project the user is reading
    the digest IN. Self-references are tautological. Editor should connect to
    OTHER projects only.

    ME-02 fix: env-var read moved INSIDE the function (was module-level, which
    binds once at import time and breaks monkeypatch.setenv in tests AND fires
    a Windows-only file read on import that always fails on CI/Linux).
    ME-03 fix: malformed frontmatter (opening `---\\n` with no closing fence)
    falls through to the static fallback instead of leaking the unparsed
    `---\\n` prefix into the LLM prompt.
    """
    path = os.environ.get(
        "PROJECT_GRAPH_PATH",
        "project-graph.md",
    )
    # Read env-overridable fallback INSIDE function (not module-level) per CLAUDE.md gotcha
    static_fallback = os.environ.get("EDITOR_PROJECTS_FALLBACK", _STATIC_PROJECTS_FALLBACK)
    try:
        from pathlib import Path
        text = Path(path).read_text(encoding="utf-8")
        if text.startswith("---\n"):
            end = text.find("\n---\n", 4)
            if end > 0:
                text = text[end + 5:]
            else:
                # ME-03: malformed frontmatter — fall back to static list
                # rather than feeding `---\n...` into the prompt as content.
                return static_fallback + "\n\nВАЖНО: НЕ ссылайся на autorss-feed — это сам дайджест."
        return (
            "Проекты пользователя (используй для привязки Связь:). "
            "ВАЖНО: НЕ ссылайся на сам autorss-feed — пользователь уже читает дайджест ИЗ него, связи к нему тавтологичны. "
            "Используй другие проекты ниже:\n\n" + text.strip()
        )
    except Exception:
        return static_fallback + "\n\nВАЖНО: НЕ ссылайся на autorss-feed — это сам дайджест."

FEW_SHOT = """ПРИМЕР ПРАВИЛЬНОЙ ЗАПИСИ (не копируй содержание, копируй структуру):
**[AI of the Day]** ([→ Оригинал](https://t.me/aioftheday/100)): Meta внедрила систему Andromeda — таргетинг переведён с «ручных настроек интересов» на «анализ атрибутов креатива», нейросеть сама решает показ на основе содержания поста. CTR вырос на 22% в пилоте. Связь: Ковчег — вместо сложных анкет строить систему анализа сырого запроса инди-фаундера; это переход от ручного к агентному, который ты уже реализуешь через Claude Code."""


def _make_tool_dispatcher(db, projects_block: str):
    async def dispatch(name: str, args: dict) -> str:
        if name == "fetch_full_article":
            url = args.get("url", "")
            try:
                from src.collectors.web_browse import fetch_article_body
                body = await fetch_article_body(url)
            except ImportError:
                import trafilatura
                body = trafilatura.fetch_url(url)
            return body or "[could not fetch article]"

        elif name == "find_related_posts":
            cat = args.get("category", "")
            days = int(args.get("days", 7))
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            async with db.execute(
                "SELECT rp.raw_text, rp.url FROM raw_posts rp "
                "JOIN sources s ON s.id = rp.source_id "
                "WHERE rp.status='curated' AND s.category=? AND rp.published_at>=? LIMIT 5",
                (cat, cutoff),
            ) as cur:
                rows = await cur.fetchall()
            if not rows:
                return "[no related posts found]"
            return "\n---\n".join(
                f"{r[1] or 'no-url'}: {(r[0] or '')[:300]}" for r in rows
            )

        elif name == "lookup_project_context":
            slug = args.get("slug", "").lower()
            for line in projects_block.split("\n"):
                if slug in line.lower():
                    return line.strip()
            return f"[project '{slug}' not found]"

        return "[unknown tool]"

    return dispatch

async def generate_research_section(
    host: str,
    model: str,
    posts: list[dict],
    backend: str = "ollama",
) -> str:
    """Idea 6 — Paper of the day: 3-paragraph research format per paper.

    Paragraph 1: Methods (what was done and how).
    Paragraph 2: Results and conclusions (what was found).
    Paragraph 3: Relevance to the user's projects (1-2 sentences).
    """
    projects_block = _load_projects_block()
    entries = "\n\n---\n\n".join(
        f"Статья [{i+1}]\n"
        f"Источник: {p.get('source_display_name', p.get('source_target', ''))}"
        + (f"\nURL: {p['url']}" if p.get('url') else "")
        + f"\n\n{p['raw_text'][:2000]}"
        for i, p in enumerate(posts)
    )
    prompt = (
        f"Ты — научный аналитик. Для каждой статьи ниже напиши СТРОГО 3 абзаца на русском языке.\n\n"
        f"{projects_block}\n\n"
        f"ФОРМАТ КАЖДОЙ ЗАПИСИ:\n"
        f"**[Название статьи или источника]** ([→ Оригинал](URL)):\n\n"
        f"Методы: <2-3 предложения — что сделали и как, конкретные методы/архитектура/дата>\n\n"
        f"Выводы: <2-3 предложения — что нашли, цифры, главный результат>\n\n"
        f"Связь: <название проекта> — <конкретная применимость в 1-2 предложениях>\n\n"
        f"ЗАПРЕЩЕНО: вступления, маркетинговый язык, «данная работа исследует», буллеты внутри записи.\n"
        f"Если связи с проектами нет — пиши: Связь: [Общий контекст].\n\n"
        f"Статьи:\n\n{entries}"
    )
    if backend == "claude":
        return await generate_with_claude(prompt)
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{host}/api/generate", json=payload) as response:
            data = await response.json()
            return data["response"].strip()


async def _resolve_mixer_block(db, posts: list[dict], category: str) -> str:
    """Phase 81: build mixture-of-prompts block from per-source adapter_weights.

    Resolves weights for ALL source_ids in the group (batch query), then merges
    by summing weights and renormalising. Falls back to category default per
    source whose adapter_weights is NULL. On any DB error / missing db, returns
    "" (no-op — fail-soft so editor stays byte-identical for legacy paths when
    the feature is not wired).
    """
    from src.editor.prompt_mixer import (
        build_mixer_block,
        resolve_weights_for_source,
    )

    if db is None:
        return ""
    # WR-03: coerce source_id to int — string ids bind as TEXT against an INTEGER
    # column and SQLite would silently match 0 rows, falling back to category default.
    source_ids: list[int] = sorted({
        int(p["source_id"]) for p in posts
        if p.get("source_id") is not None and str(p["source_id"]).lstrip("-").isdigit()
    })
    if not source_ids:
        return ""
    try:
        placeholders = ",".join("?" * len(source_ids))
        async with db.execute(
            f"SELECT id, adapter_weights FROM sources WHERE id IN ({placeholders})",
            source_ids,
        ) as cur:
            rows = await cur.fetchall()
    except Exception as e:  # noqa: BLE001 — DB shape varies in tests
        logger.warning("_resolve_mixer_block DB error: %s", e)
        return ""
    if not rows:
        return ""
    merged: dict[str, float] = {}
    for _sid, aw_json in rows:
        weights = resolve_weights_for_source(aw_json, category)
        for style, w in weights.items():
            merged[style] = merged.get(style, 0.0) + w
    # Renormalise so the block reads as proportions across the group
    total = sum(merged.values())
    if total > 0:
        merged = {k: v / total for k, v in merged.items()}
    return build_mixer_block(merged)


async def generate_section(
    host: str,
    model: str,
    category: str,
    posts: list[dict],
    backend: str = "ollama",
    hint: list[dict] | None = None,
    db=None,
) -> str:
    if category == "research":
        return await generate_research_section(host, model, posts, backend=backend)

    # Phase 81 / MIX-PROMPTS-01: per-source mixture-of-prompts block.
    # Empty string for category=="research" (handled above) or missing db.
    _mixer_block = await _resolve_mixer_block(db, posts, category)

    if (
        os.environ.get("EDITOR_AGENTIC", "true").lower() == "true"
        and backend == "ollama"
        and db is not None
    ):
        from src.llm.ollama_tools import chat_with_tools, EDITOR_TOOLS
        projects_block = _load_projects_block()
        dispatch = _make_tool_dispatcher(db, projects_block)
        numbered = "\n\n".join(
            f"[{i+1}] Канал: {p['source_display_name']} ({p['source_target']})"
            + (f"\nСсылка: {p['url']}" if p.get("url") else "")
            + f"\n{p['raw_text'][:600]}"
            for i, p in enumerate(posts)
        )
        hint_block = ""
        if hint:
            detected = ", ".join(f'"{h["label"]}"' for h in hint)
            hint_block = (
                f"\n\nПРЕДЫДУЩАЯ ПОПЫТКА содержала запрещённые элементы: {detected}. "
                f"Перепиши БЕЗ этих фраз/конструкций.\n"
            )
        system = (
            f"Ты — аналитик-фильтр для персонального дайджеста. {STRICT_CONSTRAINTS}\n\n"
            f"{hint_block}"
            f"{projects_block}\n\n"
            + (f"{_mixer_block}\n\n" if _mixer_block else "")
            + f"{FEW_SHOT}\n\n"
            f"Ты можешь использовать инструменты для получения полного текста статьи, "
            f"поиска связанных постов или уточнения контекста проекта.\n"
            f"ЗАПРЕЩЕНО НАВСЕГДА:\n"
            f"- Фразы: «если у вас есть вопросы», «надеюсь это поможет», «рад помочь»\n"
            f"- Нумерованные списки ВНУТРИ записи\n"
            f"- Вступления и подведения итогов к разделу\n\n"
            f"ФОРМАТ КАЖДОЙ ЗАПИСИ (строго):\n"
            f"**[Название канала]** ([→ Оригинал](URL)): <2-3 предложения>. "
            f"Связь: <проект> — <польза>.\n\n"
            f"Категория: {CATEGORY_LABELS.get(category, category)}"
        )
        return await chat_with_tools(
            host=host,
            model=model,
            system=system,
            user_msg=f"Напиши раздел дайджеста:\n\n{numbered}",
            tools=EDITOR_TOOLS,
            tool_dispatcher=dispatch,
            max_rounds=3,
        )

    numbered = "\n\n".join(
        f"[{i+1}] Канал: {p['source_display_name']} ({p['source_target']})"
        + (f"\nСсылка: {p['url']}" if p.get('url') else "")
        + "\n" + (
            " | ".join(p['claims']) if p.get('claims') else p['raw_text'][:600]
        )
        for i, p in enumerate(posts)
    )
    hint_block = ""
    if hint:
        detected = ", ".join(f'"{h["label"]}"' for h in hint)
        hint_block = (
            f"\n\nПРЕДЫДУЩАЯ ПОПЫТКА содержала запрещённые элементы: {detected}. "
            f"Перепиши БЕЗ этих фраз/конструкций.\n"
        )
    prompt = (
        f"Ты — аналитик-фильтр для персонального дайджеста. {STRICT_CONSTRAINTS}\n\n"
        f"{hint_block}"
        f"{_load_projects_block()}\n\n"
        + (f"{_mixer_block}\n\n" if _mixer_block else "")
        + f"{FEW_SHOT}\n\n"
        f"ЗАПРЕЩЕНО НАВСЕГДА:\n"
        f"- Фразы: «если у вас есть вопросы», «надеюсь это поможет», «рад помочь», «в заключение», «таким образом»\n"
        f"- Нумерованные списки или буллеты ВНУТРИ записи\n"
        f"- Вступления и подведения итогов к разделу\n"
        f"- Краткие перечисления без механизма (что именно произошло технически)\n\n"
        f"ФОРМАТ КАЖДОЙ ЗАПИСИ (строго):\n"
        f"**[Название канала]** ([→ Оригинал](URL)): <2-3 предложения — конкретный механизм/факт/цифра>. "
        f"Связь: <название проекта из списка выше> — <конкретная техническая или когнитивная польза>.\n\n"
        f"Если связи с проектами нет — пиши: Связь: [общий контекст].\n"
        f"Если пост без реального содержания — пропускай молча.\n\n"
        f"Категория постов: {CATEGORY_LABELS.get(category, category)}\n\n"
        f"Посты:\n{numbered}"
    )
    if backend == "claude":
        return await generate_with_claude(prompt)

    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{host}/api/generate", json=payload) as response:
            data = await response.json()
            return data["response"].strip()

# Phase 73 / STRUCT-GUARD-01: ollama (qwen 7b) failure modes — sometimes
# breaks out of the strict-format prompt and emits chat-style meta-narrative
# ("Updated Project Notes", "Do you want me to ...") OR a Russian summary
# ("Статья [1] * Название: ...") instead of the canonical
# `**[Channel](url)** ([→ Оригинал](url)): ...` entry shape. Both kill react
# buttons (no marker injection) and pollute the digest. Detect + fallback.
_CHAT_MODE_TRIGGERS: tuple[str, ...] = (
    "Updated Project Notes",
    "Do you want me to",
    "Надеюсь, это поможет",
    "I hope this helps",
    "Вот ваш дайджест",
    "Here is the digest",
    "Here are the structured notes",
    "Вот структурированные заметки",
    "Okay, here's a breakdown",
    "Here's a breakdown",
    "formatted as requested",
    "The repeated entries are included",
    "ensure all information",
    "(→ Original)",     # English template artefact — editor mandates Russian "Оригинал"
    "[Package Description]",
)
# Accept channel-with-URL `**[X](url)** ...` and channel-without-URL `**[X]** ...`
# shapes; only the `([→ Оригинал](http...))` link is mandatory.
_CANONICAL_ENTRY_RE = re.compile(
    r"\*\*\[[^\]]+\](?:\([^)]+\))?\*\*\s*\(\[→ Оригинал\]\(https?://[^)]+\)\)",
)


def _validate_structure(section_text: str, posts: list[dict]) -> str | None:
    """Return None if OK, else short reason string. Phase 73 STRUCT-GUARD-01."""
    if not section_text:
        return None
    for trig in _CHAT_MODE_TRIGGERS:
        if trig in section_text:
            return f"chat_mode:{trig[:40]}"
    # Canonical-shape gate only fires on egregious LLM failure: 3+ posts in the
    # group AND zero canonical entries rendered. Lower-bar checks would catch
    # legitimate mock outputs in tests and minor LLM stylistic drift.
    if len(posts) >= 3:
        canonical = len(_CANONICAL_ENTRY_RE.findall(section_text))
        if canonical == 0:
            return f"zero_canonical:{len(posts)}_posts"
    # Hallucinated-duplication gate: LLM rendered N>posts fake entries (e.g.
    # `Entry 1:`, `Entry 2:` numbered list with identical bodies). Catches the
    # 1-post → 6-fake-copies failure mode seen in digest 69 ML/research section.
    fake_entries = len(re.findall(r"^\s*Entry\s+\d+\s*[:\.\)]", section_text, re.MULTILINE))
    if fake_entries >= 3 and fake_entries > len(posts):
        return f"hallucinated_entries:{fake_entries}_vs_{len(posts)}_posts"
    return None


def _fallback_section(posts: list[dict]) -> str:
    """Emit canonical entries from raw post data — last-resort fallback when
    the LLM repeatedly breaks the structural contract. Markers can match URLs
    so buttons still render. Phase 73 STRUCT-GUARD-01."""
    lines = []
    for p in posts:
        ch = p.get("source_display_name") or p.get("source_target") or "Source"
        url = p.get("url") or ""
        snippet = ((p.get("raw_text") or "")[:280]).replace("\n", " ").strip()
        lines.append(
            f"**[{ch}]({url})** ([→ Оригинал]({url})): {snippet} *Связь: [Общий контекст]*"
        )
    return "\n\n".join(lines)


def _strip_empty_subheaders(markdown: str) -> str:
    """Drop `### Subheader\\n\\n` blocks that have no entries below them
    (next non-empty line is another header or end-of-section).
    Phase 73 STRUCT-GUARD-01."""
    return re.sub(
        r"### [^\n]+\n+(?=(?:### |## |\Z))",
        "",
        markdown,
    )


def _inject_markers(
    section_text: str,
    posts: list[dict],
    marker_name: str,
    fallback_posts: list[dict] | None = None,
) -> str:
    """LO-03 DRY factory: append `<!-- {marker_name}:{id} -->` HTML comment
    markers per entry, matched by source URL (falls back to positional).

    `fallback_posts` (optional) is searched by URL when the local `posts` list
    misses — covers the case where the LLM relocated an entry to a different
    category/topic section. Without it, ollama digests lose ~30% of markers
    (cross-category bug, 2026-05-20).
    """
    if not posts and not fallback_posts:
        return section_text

    url_to_post: dict[str, dict] = {p["url"]: p for p in posts if p.get("url")}
    if fallback_posts:
        for p in fallback_posts:
            url = p.get("url")
            if url and url not in url_to_post:
                url_to_post[url] = p

    parts = section_text.split("**[")
    preamble = parts[0]
    raw_entries = parts[1:]

    augmented: list[str] = []
    matched_ids: set[int] = set()
    pos_idx = 0

    for entry in raw_entries:
        entry = entry.rstrip()
        matched_post: dict | None = None

        url_m = re.search(r'\((https?://[^)]+)\)', entry)
        if url_m:
            matched_post = url_to_post.get(url_m.group(1))

        if matched_post is None:
            while pos_idx < len(posts) and posts[pos_idx]["id"] in matched_ids:
                pos_idx += 1
            if pos_idx < len(posts):
                matched_post = posts[pos_idx]
                pos_idx += 1

        if matched_post:
            matched_ids.add(matched_post["id"])
            # Insert marker after the first paragraph (channel line) so the
            # button row renders directly below the entry, not after any
            # trailing [пропуск ...] lines that belong to the same text block.
            first_para, sep, rest = entry.partition("\n\n")
            if sep:
                marked = f"**[{first_para} <!-- {marker_name}:{matched_post['id']} -->{sep}{rest}"
            else:
                marked = f"**[{entry} <!-- {marker_name}:{matched_post['id']} -->"
            augmented.append(marked)
        else:
            augmented.append(f"**[{entry}")

    # NOTE (2026-05-14): unpaired markers (posts the LLM dropped from output)
    # used to be appended as trailing `<!-- marker:id -->` blocks. They became
    # orphan button rows in the rendered HTML (markdown doesn't wrap consecutive
    # HTML comments in <p>). Since the user can't read a post the LLM dropped,
    # react buttons for it are useless. We now drop unpaired markers entirely;
    # digest_items DB rows still get created for ALL posts (see _post_to_item
    # in create_daily_digest), so accounting is preserved.
    return preamble + "\n\n".join(augmented)


def _inject_post_id_markers(
    section_text: str, posts: list[dict], fallback_posts: list[dict] | None = None
) -> str:
    """Append `<!-- post-id:{id} -->` markers per entry (Phase 5 D-13/D-15).

    Thin wrapper over _inject_markers (LO-03 DRY).
    """
    return _inject_markers(section_text, posts, "post-id", fallback_posts=fallback_posts)


def _inject_item_id_markers(
    section_text: str, posts: list[dict], fallback_posts: list[dict] | None = None
) -> str:
    """Append `<!-- item-id:{id} -->` markers per entry (Phase 15 / REACT-01).

    Per G-1, item_id == raw_posts.id (so N is the same value as post-id), but
    the marker is a separate namespace so the worker reads `<!-- item-id:N -->`
    even if dedup logic ever decouples item_id from post_id in v1.3+.

    Runs BEFORE _inject_post_id_markers in create_daily_digest so both markers
    coexist on the same line: `**[Channel <!-- item-id:N --> <!-- post-id:N --> ...`.
    Mirrors the Phase 10 two-stage injector chain `_inject_extraction → _inject_buttons`.

    Thin wrapper over _inject_markers (LO-03 DRY).
    """
    return _inject_markers(section_text, posts, "item-id", fallback_posts=fallback_posts)


async def _generate_with_voice_check(
    host: str,
    model: str,
    category: str,
    posts: list[dict],
    backend: str,
    label: str,
    max_retries: int = 2,
    db=None,
) -> tuple[str, bool]:
    """D-07/D-08 retry orchestrator. Returns (section_text, was_stripped).

    Loop up to max_retries+1 times. On each iteration:
      1. Call generate_section (hint = last detected violations, or None on first try).
      2. Run check_section over the result.
      3. If no violations, return (text, False) — happy path.
      4. If violations and attempt < max_retries, save violations as next-iter hint and retry.
      5. If max_retries exhausted: strip banned phrases via strip_residual; structures persist
         as-is (D-11) with WARNING log; return (stripped_or_unchanged_text, True).

    NEVER raises — caller's outer try/except in create_daily_digest must NOT see exceptions
    from VOICE machinery. Treats unrecoverable errors as fail-open (return last attempt unchanged).
    """
    from src.voice.check import check_section, strip_residual

    last_section = ""
    last_violations: list[dict] = []
    for attempt in range(max_retries + 1):
        try:
            last_section = await generate_section(
                host, model, category, posts, backend=backend,
                hint=last_violations or None,
                db=db,
            )
        except Exception:
            logger.exception("generate_section raised on category=%s attempt=%d", category, attempt)
            return (last_section, False)

        # Phase 73 STRUCT-GUARD-01: detect ollama chat-mode / non-canonical drift.
        # Fallback to raw-post canonical render so buttons + markers still work.
        struct_violation = _validate_structure(last_section, posts)
        if struct_violation:
            logger.warning(
                "STRUCTURE violation cat=%s attempt=%d: %s — emitting fallback section",
                category, attempt, struct_violation,
            )
            last_section = _fallback_section(posts)
            return (last_section, True)

        result = check_section(last_section, label=label)
        if not result.violations:
            return (last_section, False)

        last_violations = result.violations
        logger.info(
            "VOICE retry %d/%d for category=%s — violations: %s",
            attempt + 1, max_retries, category,
            [v["label"] for v in result.violations],
        )

    has_phrase = any(v["kind"] == "phrase" for v in last_violations)
    has_structure = any(v["kind"] == "structure" for v in last_violations)
    if has_phrase:
        try:
            stripped = strip_residual(last_section)
            logger.warning(
                "VOICE retry exhausted for category=%s; auto-stripped phrases (D-08). Violations: %s",
                category, [v["label"] for v in last_violations],
            )
            if has_structure:
                logger.warning(
                    "VOICE structure violations persist as-is per D-11 for category=%s: %s",
                    category, [v["label"] for v in last_violations if v["kind"] == "structure"],
                )
            return (stripped, True)
        except Exception:
            logger.exception("strip_residual raised; returning original text")
            return (last_section, False)

    logger.warning(
        "VOICE retry exhausted for category=%s; structures persist as-is per D-11: %s",
        category, [v["label"] for v in last_violations],
    )
    return (last_section, False)


# Phase 72 / EVENT-CLUSTER-01: module-level counter exposed for run_pipeline
# OK-log accretion (event_clusters=N field). Fail-soft default 0. Reset at the
# START of every create_daily_digest call so the value is per-run.
_LAST_EVENT_CLUSTERS_COUNT: int = 0


def get_event_clusters_count() -> int:
    """Phase 72: number of event clusters that fired during the LAST
    create_daily_digest call. 0 when EVENT_CLUSTER_MODE!=on or no cluster
    qualified. Used by run_pipeline to emit `event_clusters=N` on the OK log
    line (fail-soft 6-file lockstep accretion)."""
    return _LAST_EVENT_CLUSTERS_COUNT


def _set_event_clusters_count(n: int) -> None:
    global _LAST_EVENT_CLUSTERS_COUNT
    try:
        _LAST_EVENT_CLUSTERS_COUNT = int(n)
    except Exception:
        _LAST_EVENT_CLUSTERS_COUNT = 0


# Phase 75 / REACT-PRED-01: env-gated within-category ranking by predicted
# P(non-skip). Default OFF — digest output byte-identical to baseline. When
# REACT_PRED_RANK=on, lazy-load weights once per process; fall back to
# baseline order with a single WARNING if weights file missing. score()
# failures degrade to 0.5 so the post still appears.
_PREDICTOR_STATE: dict = {"warned": False, "weights": "__unset__"}
_PREDICTOR_LOCK = threading.Lock()


def _reset_predictor_cache() -> None:
    """Test helper — clears the per-process predictor cache + warn flag."""
    with _PREDICTOR_LOCK:
        _PREDICTOR_STATE["warned"] = False
        _PREDICTOR_STATE["weights"] = "__unset__"


def _maybe_rank_by_predictor(posts: list[dict], category: str) -> list[dict]:
    """Re-order `posts` within a single category by predicted P(non-skip) desc.

    No-op unless `REACT_PRED_RANK=on` (case-insensitive). Any other value =
    off. Returns the original list (or a new sorted list) — never mutates.

    WR-04: lazy-load guarded by `_PREDICTOR_LOCK` so concurrent
    `create_daily_digest` calls cannot duplicate `load_latest()` or spam the
    missing-weights warning.
    """
    if os.environ.get("REACT_PRED_RANK", "off").lower() != "on":
        return posts
    with _PREDICTOR_LOCK:
        if _PREDICTOR_STATE["weights"] == "__unset__":
            try:
                from src.agents.reaction_predictor import load_latest
                _PREDICTOR_STATE["weights"] = load_latest()
            except Exception:
                logger.exception("REACT_PRED_RANK: load_latest crashed; disabling")
                _PREDICTOR_STATE["weights"] = None
        weights = _PREDICTOR_STATE["weights"]
    if weights is None:
        with _PREDICTOR_LOCK:
            if not _PREDICTOR_STATE["warned"]:
                logger.warning(
                    "REACT_PRED_RANK=on but no weights file found in data/; "
                    "falling back to baseline order"
                )
                _PREDICTOR_STATE["warned"] = True
        return posts
    from src.agents.reaction_predictor import score as _score

    # CR-01 (Phase 84): open ONE sqlite conn per rerank call (was per-post,
    # which churned the page cache + raced uvicorn under default rollback
    # journal). timeout=5.0 survives a concurrent writer.
    import sqlite3 as _sqlite3
    _db_path = os.environ.get("DB_PATH", "curator.db")
    try:
        _conn = _sqlite3.connect(_db_path, timeout=5.0)
    except _sqlite3.Error:
        logger.exception("REACT_PRED_RANK: sqlite connect failed; baseline order")
        return posts

    def _safe(p: dict) -> float:
        try:
            return float(_score(p["id"], category, weights, db_conn=_conn))
        except Exception:
            logger.debug(
                "reaction_predictor.score failed for post %s", p.get("id"),
                exc_info=True,
            )
            return 0.5

    try:
        return sorted(posts, key=_safe, reverse=True)
    finally:
        try:
            _conn.close()
        except Exception:  # pragma: no cover - defensive
            pass


def _sanitize_cluster_title(title: str) -> str:
    """T-72-01 mitigation: strip newlines + backticks + markdown-breakable
    chars from cluster strings (canonical title AND source names) so a
    malicious post/channel title cannot inject a fake section header, fake
    link, or break the cluster line / bold span.

    WR-03 fix (Phase 72 review): also strip ``[``, ``]``, ``(``, ``)``, ``*``
    so a channel display_name like ``Evil](http://x)`` or a title containing
    ``**`` cannot corrupt the ``**{title}** — N каналов: [{names}]`` line.
    """
    if not title:
        return ""
    cleaned = title
    for ch in ("`", "\r", "\n", "[", "]", "(", ")", "*"):
        cleaned = cleaned.replace(ch, " " if ch in ("\r", "\n") else "")
    return " ".join(cleaned.split())


async def create_daily_digest(db: aiosqlite.Connection, ollama_host: str, model: str, backend: str = "ollama"):
    # Phase 72: reset per-run cluster counter unconditionally — even early
    # returns (no curated posts) must clear the previous run's value.
    _set_event_clusters_count(0)
    posts = await get_curated_posts(db)
    if not posts:
        return

    # Phase 40: bind one run_id for the entire digest so every claude_cli trace
    # row shares a queryable scope.
    with trace_run(agent="editor"):
        return await _create_daily_digest_impl(db, ollama_host, model, backend, posts)


async def _create_daily_digest_impl(db, ollama_host, model, backend, posts):

    # Idea 4: pre-extract claims for denser editor input (idempotent — skips already extracted).
    try:
        from src.analysis.claim_extractor import extract_claims_for_posts
        _claims_map = await extract_claims_for_posts(db, posts, ollama_host, model)
        for _p in posts:
            _p['claims'] = _claims_map.get(_p['id'], [])
    except Exception:
        logger.warning("Claim extraction failed — falling back to raw_text", exc_info=True)
        for _p in posts:
            _p['claims'] = []

    # Phase 12 / TOPIC-04: batch-fetch topics for all curated posts (one DB call).
    topics_by_post = await get_topics_for_posts(db, [p["id"] for p in posts])

    # Idea 1 (IIT submodular): compute intra-digest saliency for all posts first,
    # before per-category submodular selection (needs all posts for novelty calc).
    from src.llm.saliency import compute_saliency_batch
    saliency_by_post = compute_saliency_batch(posts)

    # Phase 72 / EVENT-CLUSTER-01: env-gated cluster compute.
    # Read env INSIDE function (reference_module_level_env_binding.md). When
    # EVENT_CLUSTER_MODE=on AND ≥1 cluster of ≥3 posts fires, render a
    # "🔥 Событие дня" section above category sections and EXCLUDE clustered
    # post ids from per-category rendering (no duplication). Fail-soft: any
    # error → clusters=[], excluded_ids=set(), counter=0; pipeline continues.
    event_cluster_section_md: str = ""
    excluded_cluster_ids: set[int] = set()
    _event_cluster_mode = os.environ.get("EVENT_CLUSTER_MODE", "off").lower()
    if _event_cluster_mode == "on":
        try:
            from src.editor.event_cluster import cluster_recent_events

            # Pull simhash/published_at for the current curated set in one query.
            post_ids = [p["id"] for p in posts]
            placeholders = ",".join("?" * len(post_ids))
            sim_map: dict[int, int | None] = {}
            pub_map: dict[int, str | None] = {}
            if post_ids:
                async with db.execute(
                    f"SELECT id, simhash, published_at FROM raw_posts WHERE id IN ({placeholders})",
                    post_ids,
                ) as _scur:
                    for _r in await _scur.fetchall():
                        sim_map[_r[0]] = _r[1]
                        pub_map[_r[0]] = _r[2]

            cluster_input = []
            for p in posts:
                sh = sim_map.get(p["id"])
                pub = pub_map.get(p["id"])
                if sh is None or pub is None:
                    continue
                cluster_input.append({
                    "id": p["id"],
                    "simhash": sh,
                    "published_at": pub,
                    "relevance_score": p.get("relevance_score") or 0.0,
                    "source_display_name": (
                        p.get("source_display_name") or p.get("source_target") or ""
                    ),
                    "title": (p.get("raw_text") or "").split("\n", 1)[0][:200],
                })
            clusters = cluster_recent_events(cluster_input)
            if clusters:
                lines: list[str] = ["## 🔥 Событие дня", ""]
                for c in clusters:
                    title = _sanitize_cluster_title(c.get("canonical_title", ""))
                    size = int(c.get("size", 0))
                    # WR-03 fix: sanitize each source_name — a channel
                    # display_name containing ``]`` or ``\n## ...`` would
                    # otherwise break the cluster line or inject a fake heading.
                    sanitized_names = [
                        _sanitize_cluster_title(n)
                        for n in c.get("source_names", [])
                    ]
                    sanitized_names = [n for n in sanitized_names if n]
                    src_list = ", ".join(sanitized_names)
                    lines.append(
                        f"**{title}** — {size} каналов покрывают: [{src_list}]"
                    )
                    for mid in c.get("member_ids", []):
                        try:
                            excluded_cluster_ids.add(int(mid))
                        except Exception:
                            pass
                event_cluster_section_md = "\n".join(lines)
            _set_event_clusters_count(len(clusters))
        except Exception:
            logger.exception("event_cluster compute failed (non-fatal)")
            event_cluster_section_md = ""
            excluded_cluster_ids = set()
            _set_event_clusters_count(0)
    else:
        _set_event_clusters_count(0)

    by_category = defaultdict(list)
    for p in posts:
        if p["id"] in excluded_cluster_ids:
            continue
        by_category[p["source_category"]].append(p)

    category_order = list(CANONICAL_CATEGORIES)
    sections = []
    # Phase 72: prepend cluster section above categories.
    if event_cluster_section_md:
        sections.append(event_cluster_section_md)
    # ME-01 fix: collect digest_items incrementally inside per-category try so
    # categories swallowed by `except Exception` do not pollute digest_items
    # with rows whose text never made it into the digest markdown. REACT-06:
    # cache holds only what the user actually sees in the digest.
    items: list[dict] = []

    def _post_to_item(p: dict) -> dict:
        sal = saliency_by_post.get(p["id"], {})
        return {
            "item_id": p["id"],   # G-1: item_id == raw_posts.id
            "post_id": p["id"],
            "channel": p.get("source_display_name") or p.get("source_target"),
            "url": p.get("url"),
            "snippet": (p.get("raw_text") or "")[:600],
            "linked_project": None,
            "saliency_novelty": sal.get("novelty"),
            "saliency_relevance": sal.get("relevance"),
            "saliency_urgency": sal.get("urgency"),
        }

    from src.llm.iit_scorer import submodular_select

    for cat in category_order:
        cat_posts = by_category.get(cat, [])
        if not cat_posts:
            continue
        # Idea 1 (IIT submodular): select max-Phi diverse subset per category.
        cat_posts = submodular_select(cat_posts)
        # Phase 75 / REACT-PRED-01: env-gated reorder by predicted P(non-skip).
        # Default off — pre-Phase-75 baseline order preserved byte-identical.
        cat_posts = _maybe_rank_by_predictor(cat_posts, cat)
        label = CATEGORY_LABELS.get(cat, cat)
        cat_backend = _category_backend(cat, backend)

        # Phase 12 / TOPIC-04: split this category's posts by primary topic;
        # render hybrid layout with `### {topic}` sub-headers under the
        # `## {category}` heading. Untopiced posts render under the category
        # heading WITHOUT a sub-header (D-19).
        topic_groups, untopiced = _group_posts_by_topic(cat_posts, topics_by_post)

        category_chunks: list[str] = []
        category_items: list[dict] = []
        try:
            for topic_key, group_posts in topic_groups:
                section_text, was_stripped = await _generate_with_voice_check(
                    ollama_host, model, cat, group_posts, backend=cat_backend, label=label, db=db,
                )
                section_text = _inject_item_id_markers(section_text, group_posts, fallback_posts=posts)
                section_text = _inject_post_id_markers(section_text, group_posts, fallback_posts=posts)
                category_chunks.append(f"### {_topic_label(topic_key)}\n\n{section_text}")
                category_items.extend(_post_to_item(p) for p in group_posts)
                if was_stripped:
                    logger.info(
                        "Generated section '%s/%s' (%d posts) [VOICE auto-strip applied]",
                        cat, topic_key, len(group_posts),
                    )
                else:
                    logger.info(
                        "Generated section '%s/%s' (%d posts)",
                        cat, topic_key, len(group_posts),
                    )
            if untopiced:
                section_text, was_stripped = await _generate_with_voice_check(
                    ollama_host, model, cat, untopiced, backend=cat_backend, label=label, db=db,
                )
                section_text = _inject_item_id_markers(section_text, untopiced, fallback_posts=posts)
                section_text = _inject_post_id_markers(section_text, untopiced, fallback_posts=posts)
                # D-19: untopiced posts render WITHOUT a ### sub-header — they
                # go directly under the ## category heading.
                category_chunks.append(section_text)
                category_items.extend(_post_to_item(p) for p in untopiced)
                if was_stripped:
                    logger.info(
                        "Generated section '%s' default (no-topic) (%d posts) [VOICE auto-strip applied]",
                        cat, len(untopiced),
                    )
                else:
                    logger.info(
                        "Generated section '%s' default (no-topic) (%d posts)",
                        cat, len(untopiced),
                    )
            if category_chunks:
                sections.append(f"## {label}\n\n" + "\n\n".join(category_chunks))
                # Promote category_items into the global items list ONLY if the
                # category fully completed (no exception swallowed mid-way).
                items.extend(category_items)
        except Exception:
            logger.exception("Failed to generate section for category '%s'", cat)

    if not sections:
        return

    digest_md = "\n\n---\n\n".join(sections)
    # Phase 73 STRUCT-GUARD-01: drop empty `### Topic` sub-headers that survive
    # when the LLM dropped all entries under a topic group.
    digest_md = _strip_empty_subheaders(digest_md)

    # Phase 64 / DIGEST-CONFORM-01: env-gated spec-conformance gate.
    # Hook placement: AFTER section assembly + _inject_markers (already applied
    # per-section above), BEFORE insert_digest_with_items. Read env INSIDE the
    # function (reference_module_level_env_binding.md). Lazy-import the
    # validator so a broken module can't break the editor (fail-soft layer 2:
    # outer try/except below). Counter increments per VIOLATION inside
    # validate_digest itself; block mode flips the module-level sentinel so
    # run_pipeline can emit ``digest=conformance_fail``.
    _conformance_mode = os.environ.get("DIGEST_CONFORMANCE_MODE", "warn").lower()
    # WR-02 fix (Phase 64 review): module-level ``_LAST_BLOCK_FAIL`` sentinel
    # is reset in a ``finally`` block so a mid-validate exception cannot leave
    # the flag stuck ``True`` and bleed into the next pipeline run (overnight
    # loop / cron overlap). Reset happens at the START of the hook to clear any
    # stale value from a previous run, and is also guaranteed by the finally.
    _block_fail_flag = False
    try:
        from src.editor.spec_conformance import (
            reset_last_conformance_block_fail,
            set_last_conformance_block_fail,
            validate_digest,
        )
        reset_last_conformance_block_fail()
        try:
            _conf_result = validate_digest(digest_md, CANONICAL_CATEGORIES)
            if _conf_result is not None and not _conf_result.ok:
                if _conformance_mode == "block":
                    logger.error(
                        "digest_conformance_block_fail violations=%s",
                        _conf_result.violations,
                    )
                    _block_fail_flag = True
                    set_last_conformance_block_fail(True)
                    return
                else:
                    logger.warning(
                        "digest_conformance_violations=%s",
                        _conf_result.violations,
                    )
        finally:
            # If we didn't intentionally flag block-fail, ensure sentinel is
            # clean — guards against exceptions inside validate_digest leaving
            # a partial set_last_conformance_block_fail(True) state.
            if not _block_fail_flag:
                reset_last_conformance_block_fail()
    except Exception:
        # Fail-soft layer 2 — validator crash MUST NOT break the editor.
        logger.exception("validate_digest hook crashed (non-fatal)")
        # Defensive: ensure sentinel reset even if the import itself failed
        # (so a follow-up run isn't tainted by a previous block-fail flag).
        try:
            from src.editor.spec_conformance import (
                reset_last_conformance_block_fail as _reset_cf,
            )
            _reset_cf()
        except Exception:
            pass

    try:
        digest_id = await insert_digest_with_items(db, digest_md, items, backend=backend)
    except Exception:
        logger.exception("Failed to save digest")
        return

    # Index digest chunks for semantic search (non-fatal if sqlite-vec unavailable)
    try:
        from src.search.embeddings import index_digest
        n_chunks = await index_digest(db, digest_id=digest_id, host=ollama_host)
        if n_chunks:
            logger.info("Indexed %d chunks for digest %d", n_chunks, digest_id)
    except Exception:
        logger.exception("Digest indexing failed (non-fatal)")
