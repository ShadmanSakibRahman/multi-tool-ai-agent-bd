"""Run the 5 assignment example queries and verify the agent picks the right tool."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import build_agent
from langchain_classic.agents import AgentExecutor
from langchain_core.callbacks import BaseCallbackHandler


class ToolTracker(BaseCallbackHandler):
    def __init__(self):
        self.tools_used = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "?")
        self.tools_used.append(name)
        print(f"   [tool called] {name}({input_str[:120]}{'...' if len(input_str) > 120 else ''})")


QUERIES = [
    ("List top 10 hospitals in Dhaka with bed capacity.", "HospitalsDBTool"),
    ("Which universities in Bangladesh offer medical degrees?", "InstitutionsDBTool"),
    ("Find restaurants in Chattogram serving biryani.", "RestaurantsDBTool"),
    ("What is the healthcare policy of Bangladesh?", "WebSearchTool"),
    ("How many government institutions are in Rajshahi?", "InstitutionsDBTool"),
]


def main():
    print("Building agent ...")
    executor: AgentExecutor = build_agent(verbose=False)
    print("Ready.\n")

    pass_count = 0
    for i, (q, expected_tool) in enumerate(QUERIES, 1):
        print(f"=== Q{i}: {q}")
        print(f"   expected tool: {expected_tool}")
        tracker = ToolTracker()
        try:
            out = executor.invoke({"input": q}, config={"callbacks": [tracker]})
            answer = out.get("output", "")
        except Exception as e:
            answer = f"[error] {e}"
        print(f"   tools used:  {tracker.tools_used}")
        print(f"   answer:      {answer[:400]}{'...' if len(answer) > 400 else ''}")
        ok = expected_tool in tracker.tools_used
        print(f"   routed correctly: {'YES' if ok else 'NO'}\n")
        if ok:
            pass_count += 1

    print(f"\nRouting score: {pass_count}/{len(QUERIES)}")


if __name__ == "__main__":
    main()
