"""Interactive local CLI for manually testing POST /chat against a running
server. Tracks conversation history for you across turns - just type each
message, no need to hand-build the JSON body yourself.

Run (with the server already running, e.g. via uvicorn app.main:app):
    python scripts/chat_cli.py
    python scripts/chat_cli.py --url http://127.0.0.1:8000

Commands while chatting:
    /reset   start a new conversation (clears history)
    /history print the full message history sent so far
    /quit    exit
"""
import argparse
import sys

import httpx


def print_response(data: dict) -> None:
    print(f"\nAgent: {data['reply']}")
    recs = data.get("recommendations") or []
    if recs:
        print(f"\nRecommendations ({len(recs)}):")
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r['name']}  [{r['test_type']}]")
            print(f"     {r['url']}")
    if data.get("end_of_conversation"):
        print("\n(end_of_conversation: true)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of the running server")
    args = parser.parse_args()

    chat_url = f"{args.url.rstrip('/')}/chat"
    health_url = f"{args.url.rstrip('/')}/health"

    try:
        httpx.get(health_url, timeout=5.0).raise_for_status()
    except Exception as e:
        print(f"Could not reach {health_url} ({e}). Is the server running?")
        sys.exit(1)

    history: list[dict] = []
    print(f"Connected to {args.url}. Type a message, or /quit to exit, /reset to start over.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/reset":
            history = []
            print("(history cleared)\n")
            continue
        if user_input == "/history":
            for m in history:
                print(f"  [{m['role']}] {m['content']}")
            print()
            continue

        history.append({"role": "user", "content": user_input})

        try:
            resp = httpx.post(chat_url, json={"messages": history}, timeout=35.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"\nRequest failed: {e}\n")
            history.pop()  # don't poison history with a turn that never got a reply
            continue

        print_response(data)
        history.append({"role": "assistant", "content": data["reply"]})


if __name__ == "__main__":
    main()
