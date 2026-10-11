"""Set conservative throughput defaults without altering experiment logic."""
from datetime import datetime
from pathlib import Path
from shutil import copy2

root = Path(__file__).resolve().parents[1]
env = root / ".env"
source = env.read_text(encoding="utf-8") if env.exists() else (root / ".env.example").read_text(encoding="utf-8")
if env.exists():
    backup = env.with_name(".env.backup-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    copy2(env, backup)
    print("Backed up:", backup)
settings = {"ODR_RESEARCH_CONCURRENCY": "2", "MRDRE_EVAL_WORKERS": "2", "MRDRE_SUMMARY_WORKERS": "2"}
lines = []
seen = set()
for line in source.splitlines():
    key = line.split("=", 1)[0].strip()
    if key in settings and not line.lstrip().startswith("#"):
        if key not in seen:
            lines.append(key + "=" + settings[key])
            seen.add(key)
    else:
        lines.append(line)
for key, value in settings.items():
    if key not in seen:
        lines.append(key + "=" + value)
env.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Applied starting profile: two research, judge and summary workers.")
print("This does not configure the separate Ollama server.")
