"""Rate-limited async Gemini calls for a free-tier key shared by several projects.

Free tier is 15 requests/min per model per project, shared with sibling projects, so we
pace ourselves well below it, honour the server's retryDelay on 429, and stop cleanly if
the per-day quota is exhausted (DailyQuotaExceeded). Everything is cached by the callers.
"""
import asyncio
import os
import random
import re
import time

from google import genai

MAX_CONCURRENT = 4  # RULES.md


class DailyQuotaExceeded(Exception):
    pass


class Limiter:
    def __init__(self, rpm):
        self.gap, self.next, self.lock = 60.0 / rpm, 0.0, asyncio.Lock()

    async def wait(self):
        async with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        await asyncio.sleep(max(0.0, t - now))


class Gemini:
    def __init__(self, rpm=6):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.sem = asyncio.Semaphore(MAX_CONCURRENT)
        self.limiters, self.rpm = {}, rpm
        self.n429 = 0

    async def generate(self, model, contents, config, tries=12):
        lim = self.limiters.setdefault(model, Limiter(self.rpm))
        delay = 10
        for attempt in range(tries):
            await lim.wait()
            async with self.sem:
                try:
                    return await self.client.aio.models.generate_content(model=model, contents=contents, config=config)
                except Exception as e:
                    err = str(e)
            if "PerDay" in err:
                raise DailyQuotaExceeded(err[:400])
            transient = any(c in err for c in ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE",
                                               "DEADLINE", "imeout", "isconnect"))
            if not transient or attempt == tries - 1:
                raise RuntimeError(err[:500])
            if "429" in err:
                self.n429 += 1
            m = re.search(r"retry in ([\d.]+)s", err)
            wait = float(m.group(1)) + 1 if m else delay + random.random() * delay
            await asyncio.sleep(wait)
            delay = min(delay * 2, 120)
