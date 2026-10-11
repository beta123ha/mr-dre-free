"""Configure an existing clone of *official* LangChain Open Deep Research.

This edits only local server configuration (authentication / telemetry),
NOT the research agent's logic or prompt/scaffold.
"""
import argparse
import json
from pathlib import Path


def configure(path: Path):
    path = path.resolve()
    config_file = path / "langgraph.json"
    if not config_file.exists():
        raise SystemExit(f"Not an Open Deep Research clone: {config_file} missing")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if "Deep Researcher" not in config.get("graphs", {}):
        raise SystemExit("Expected official LangChain graph 'Deep Researcher'")
    # Official ODR ships optional Supabase authentication for deployment. A
    # local benchmark client cannot provide Supabase bearer tokens. Disable it
    # ONLY in the *local* clone; bind the dev server to localhost.
    if "auth" in config:
        backup = config_file.with_suffix(".json.upstream-backup")
        if not backup.exists():
            backup.write_text(config_file.read_text(encoding="utf-8"), encoding="utf-8")
        del config["auth"]
        config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Disabled cloud authentication in local development config; backup: {backup}")
    env = path / ".env"
    if not env.exists():
        example = path / ".env.example"
        env.write_text(example.read_text(encoding="utf-8") if example.exists() else "", encoding="utf-8")
    old = env.read_text(encoding="utf-8")
    import re
    for key, value in {
        "GET_API_KEYS_FROM_CONFIG": "false",
        "LANGSMITH_TRACING": "false",
    }.items():
        pattern = rf"(?m)^{re.escape(key)}=.*$"
        if re.search(pattern, old):
            old = re.sub(pattern, f"{key}={value}", old)
        else:
            old = old.rstrip() + f"\n{key}={value}\n"
    env.write_text(old, encoding="utf-8")
    print("Local ODR configuration prepared. No original research code was modified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("odr_dir", help="Path to official langchain-ai/open_deep_research clone")
    args = parser.parse_args()
    configure(Path(args.odr_dir))
