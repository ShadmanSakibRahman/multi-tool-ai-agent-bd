"""
Builds the main routing agent.

The agent has 4 tools:
  InstitutionsDBTool, HospitalsDBTool, RestaurantsDBTool, WebSearchTool

A data / statistics question goes to the matching DB tool.
A general-knowledge question goes to the web search tool.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from tools import build_tools

load_dotenv()

SYSTEM_PROMPT = """You are an AI agent for Bangladesh data and general knowledge.

You have four tools:
  - InstitutionsDBTool: educational institutions in Bangladesh
  - HospitalsDBTool: hospitals and health facilities in Bangladesh
  - RestaurantsDBTool: restaurants in Bangladesh
  - WebSearchTool: general knowledge questions

Routing rules:
- If the question is about counts, lists, locations, statistics, or any concrete
  fact that lives in one of the three databases, call the matching DB tool.
- If the question is conceptual ("what is", "explain", "policy", "role of",
  "history of", recent news, definitions), call WebSearchTool.
- Pick exactly one tool per question. Do not call multiple tools unless the
  first one truly fails to answer.

How to write the final answer (very important - read carefully):
- For DB tools, your final answer MUST be based only on the rows that appear in
  the tool output. Do NOT add facts from your own knowledge. Do NOT invent
  numbers like bed counts, ratings, or addresses that are not in the rows.
- If the user asked for a column that is not in the result (for example "bed
  capacity" when the SQL only returned name and code), state clearly that this
  specific information is not available in the dataset and show what IS
  available instead.
- If the tool returns "(no rows)" or an SQL error, say the database does not
  contain that information.
- For "how many" questions, just state the number from the COUNT result.
- For WebSearchTool, summarise the search results in your own words and keep it
  concise (3 to 5 sentences).
"""


def build_agent(model_name: str = "llama-3.3-70b-versatile", verbose: bool = True) -> AgentExecutor:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    llm = ChatGroq(model=model_name, temperature=0, api_key=api_key)
    tools = build_tools(llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=5,
    )
    return executor


def ask(executor: AgentExecutor, question: str) -> str:
    out = executor.invoke({"input": question})
    return out.get("output", "")
