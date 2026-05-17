**Efficiency Summary and Action Items**

This file lists focused, actionable updates to make the project leaner, faster, and fully local-only.

**High-Level Goal:**
- **Reduce runtime overhead** and remove leftover persona/online-model bloat so the repository runs 100% locally with one prompt file feeding the model.

**Top Tasks**
- **Logger / Encoding:** [core/logger.py](core/logger.py) (or the logging init) — Ensure console and file logging use UTF-8 (set `PYTHONUTF8=1` on startup or explicitly encode/replace characters). This will prevent `UnicodeEncodeError` seen in `run.log`.
- **Remove UI Bloat:** [webconfig/](webconfig/) — Prune or archive remaining web UI templates/static JS/CSS that are not required for local runtime.
- **Remove Online Provider Artifacts:** [core/config.py](core/config.py), [requirements.txt](requirements.txt) — Remove unused environment keys and dependencies for cloud providers, and update `requirements.txt` to only include required local libs.
- **Consolidate Prompt Pipeline:** `core/mean_workout_prompt.txt` and [core/session.py](core/session.py) — Confirm `INSTRUCTIONS` is the single source of truth and remove any code paths that modify/override it at runtime.
- **Runtime Smoke Test:** `main.py` — Perform one end-to-end run, reproduce recent errors, and fix remaining runtime exceptions (logger encoding, provider init, etc.).
- **Static Cleanup & Formatting:** run a linter/formatter and remove dead imports across `core/` (e.g. leftover persona imports). Files to check: [core/session.py](core/session.py), [core/profile_manager.py](core/profile_manager.py), [core/wakeup.py](core/wakeup.py).
- **Motor & Movement Sanity Check:** [core/movements.py](core/movements.py) — Verify PWM sign/directions, timing, and constants after recent edits.
- **Docs Update:** `README.md`, `CHANGELOG.md`, `.env.example` — Remove persona/online references and document the single prompt workflow and local-only runtime steps.

**Suggested Order (minimal friction):**
1. Fix logger/encoding (blocks runtime logs).  
2. Run smoke test and capture errors.  
3. Consolidate and lock prompt pipeline.  
4. Remove UI / webconfig files (archive instead of immediate delete).  
5. Trim requirements and remove online-provider code.  
6. Static lint + formatting.  
7. Update docs.

**Quick Commands**
- Run a smoke test and capture output:
```
set PYTHONUTF8=1
python -u main.py 2>&1 | Tee-Object -FilePath run.log
```
- Run lint & format (example):
```
pip install -r requirements.txt
.\venv\Scripts\python -m pip install black flake8
.\venv\Scripts\python -m black .
.\venv\Scripts\python -m flake8 core
```

**Notes / Rationale**
- The most immediate blocker in recent runs is the logger/encoding failure (see `run.log`) — this prevents useful startup diagnostics.  
- Many persona-related files were removed; remaining references in docs, examples, and web UI should be removed to avoid confusion.  
- Consolidating the prompt into `core/mean_workout_prompt.txt` reduces runtime branching and simplifies testing.

If you want, I can start with the highest-priority items now: fix logger encoding and run the smoke test to capture remaining errors.
