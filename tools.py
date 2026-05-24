"""
The 4 tools the agent can call.

  InstitutionsDBTool  -> SQLite: data/institutions.db   (schools / colleges / madrasahs)
  HospitalsDBTool     -> SQLite: data/hospitals.db      (hospitals / clinics)
  RestaurantsDBTool   -> SQLite: data/restaurants.db    (restaurants)
  WebSearchTool       -> DuckDuckGo (general knowledge)

The DB tools take a natural-language question, ask the LLM to write an SQL query
against the relevant DB, run the SQL, and return the rows (plus the SQL itself,
so the main agent can format a final answer).
"""

import re
import sqlite3
from pathlib import Path
from typing import Optional

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool, Tool
from pydantic import BaseModel, Field


class QuestionInput(BaseModel):
    question: str = Field(description="A single natural-language question to answer from the database.")


class SearchInput(BaseModel):
    query: str = Field(description="A web search query string.")

DATA_DIR = Path(__file__).parent / "data"


SQL_PROMPT = PromptTemplate.from_template(
    """You are a SQLite expert. Given the user question, write ONE valid SQLite query
that answers it using the schema below. Return ONLY the SQL query, nothing else.
Do not wrap it in markdown. Do not add commentary. Do not include the word "sqlite".

Rules:
- Use only the table and columns shown in the schema.
- For text matching use LOWER(col) LIKE '%term%' to be case-insensitive.
- Limit the result to {top_k} rows unless the question asks for a count or aggregate.
- If the question asks "how many" use COUNT(*).
- Prefer exact equality only when the user names something you can see in the sample rows.
- Bangladeshi cities have old and new spellings - match BOTH using OR. Examples:
  Chittagong / Chattogram, Dacca / Dhaka, Comilla / Cumilla, Barisal / Barishal,
  Jessore / Jashore. So for "Chattogram" use:
    (LOWER(address) LIKE '%chattogram%' OR LOWER(address) LIKE '%chittagong%')

Schema:
{schema}

Question: {question}
SQLQuery:"""
)


def _clean_sql(raw: str) -> str:
    """Strip code fences and prefixes the LLM sometimes adds."""
    s = raw.strip()
    s = re.sub(r"^```(?:sql|sqlite)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    s = re.sub(r"^(SQLQuery|SQL|Query)\s*[:\-]\s*", "", s, flags=re.IGNORECASE)
    return s.strip().rstrip(";")


def _run_sql(db_path: Path, sql: str, max_rows: int = 30) -> str:
    """Execute SQL and return a compact text representation of the result."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows)
        if not rows:
            return "(no rows)"
        cols = rows[0].keys()
        header = " | ".join(cols)
        lines = [header, "-" * len(header)]
        for r in rows:
            lines.append(" | ".join("" if r[c] is None else str(r[c]) for c in cols))
        more = cur.fetchone()
        if more is not None:
            lines.append(f"... (result truncated at {max_rows} rows)")
        return "\n".join(lines)
    finally:
        conn.close()


def _make_db_tool(
    name: str,
    description: str,
    db_filename: str,
    table_name: str,
    llm: BaseChatModel,
    top_k: int = 15,
) -> Tool:
    db_path = DATA_DIR / db_filename
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} not found. Run 'python build_databases.py' first."
        )

    db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)
    schema = db.get_table_info([table_name])

    def _query(question: str) -> str:
        prompt = SQL_PROMPT.format(schema=schema, question=question, top_k=top_k)
        raw = llm.invoke(prompt).content
        sql = _clean_sql(raw if isinstance(raw, str) else str(raw))
        try:
            result = _run_sql(db_path, sql)
        except sqlite3.Error as e:
            return f"SQL error from {name}: {e}\nSQL tried:\n{sql}"
        return f"SQL: {sql}\n\n{result}"

    return StructuredTool.from_function(
        func=_query,
        name=name,
        description=description,
        args_schema=QuestionInput,
    )


def build_tools(llm: BaseChatModel) -> list[Tool]:
    institutions_tool = _make_db_tool(
        name="InstitutionsDBTool",
        description=(
            "Use this tool for any question about educational institutions in "
            "Bangladesh: schools, colleges, madrasahs, government / non-government / "
            "autonomous institutions, EIIN codes, division / district / thana / union "
            "level counts, management type, education level, MPO status, affiliation, "
            "or anything else that is a fact about a registered institution. "
            "Input: a single natural-language question."
        ),
        db_filename="institutions.db",
        table_name="institutions",
        llm=llm,
    )
    hospitals_tool = _make_db_tool(
        name="HospitalsDBTool",
        description=(
            "Use this tool for any question about hospitals, clinics, community "
            "clinics, diagnostic centers, blood banks, and other health facilities "
            "in Bangladesh. You can filter by division, district, upazila, city "
            "corporation, type, agency (e.g. DGHS), and whether the facility is "
            "private. Input: a single natural-language question."
        ),
        db_filename="hospitals.db",
        table_name="hospitals",
        llm=llm,
    )
    restaurants_tool = _make_db_tool(
        name="RestaurantsDBTool",
        description=(
            "Use this tool for any question about restaurants in Bangladesh. You can "
            "search by name (e.g. 'biryani' in the name), filter by address keyword "
            "(e.g. address contains 'Chattogram' or 'Dhaka'), sort by rating or "
            "number_of_reviews, or use latitude/longitude. Note the dataset has no "
            "explicit cuisine column, so cuisine-style searches happen by matching "
            "the restaurant name. Input: a single natural-language question."
        ),
        db_filename="restaurants.db",
        table_name="restaurants",
        llm=llm,
    )

    web_search = DuckDuckGoSearchRun()

    def _search(query: str) -> str:
        try:
            return web_search.run(query)
        except Exception as e:
            return f"Web search failed: {e}"

    web_search_tool = StructuredTool.from_function(
        func=_search,
        name="WebSearchTool",
        description=(
            "Use this tool for general knowledge questions that cannot be answered "
            "from the institution / hospital / restaurant databases. Examples: "
            "'What is the healthcare policy of Bangladesh?', 'What is the role of "
            "DGHS?', definitions, recent news, history, government policies. "
            "Input: a search query string."
        ),
        args_schema=SearchInput,
    )

    return [institutions_tool, hospitals_tool, restaurants_tool, web_search_tool]
