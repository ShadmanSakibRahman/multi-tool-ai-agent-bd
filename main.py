"""Tiny REPL so you can chat with the agent from a terminal."""

import sys

from agent import ask, build_agent


def main():
    print("Building agent (this loads the LLM and the 4 tools)...")
    executor = build_agent(verbose=False)
    print("Ready. Type your question. Type 'exit' to quit.\n")

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(f"> {q}")
        print(ask(executor, q))
        return

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", ":q"):
            break
        try:
            print(ask(executor, q))
        except Exception as e:
            print(f"[error] {e}")
        print()


if __name__ == "__main__":
    main()
