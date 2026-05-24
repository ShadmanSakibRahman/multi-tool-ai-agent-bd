# Multi-Tool AI Agent for Bangladesh

A LangChain agent that can answer questions about Bangladesh using three real datasets and falls back to web search for general knowledge questions.

**Live demo:** https://multi-tool-ai-agent-bd-bfaqzk7glurhyfkxkkhwdb.streamlit.app/

## What it does

You ask a question in plain English. The agent figures out where the answer should come from and uses the right tool.

- "How many government institutions are in Rajshahi?" → checks the institutions database → "110"
- "List 10 hospitals in Dhaka district." → checks the hospitals database → returns the rows
- "Find restaurants in Chattogram serving biryani." → checks the restaurants database → returns the matches
- "What is the role of DGHS in Bangladesh?" → searches the web → gives a summary

## The four tools

1. **InstitutionsDBTool** — schools, colleges, madrasahs, technical and vocational institutes. 34,901 rows.
2. **HospitalsDBTool** — hospitals, clinics, community clinics, diagnostic centers, blood banks. 38,886 rows.
3. **RestaurantsDBTool** — restaurants with names, locations, ratings. 12,703 rows.
4. **WebSearchTool** — DuckDuckGo, for general knowledge that is not in the three databases.

Each DB tool takes a natural-language question, asks the LLM to write a SQLite query against that database, runs the query, and returns the rows to the agent. The main agent then writes a final answer based only on the rows it got back (so it does not make things up).

## Datasets

All three are public on HuggingFace, uploaded by `Mahadih534`.

- https://huggingface.co/datasets/Mahadih534/Institutional-Information-of-Bangladesh
- https://huggingface.co/datasets/Mahadih534/all-bangladeshi-hospitals
- https://huggingface.co/datasets/Mahadih534/Bangladeshi-Restaurant-Data

`build_databases.py` downloads them and writes one SQLite file per dataset:

```
data/institutions.db   → table: institutions
data/hospitals.db      → table: hospitals
data/restaurants.db    → table: restaurants
```

Column names are normalised to snake_case and the column types are set (TEXT, INTEGER, REAL) so SQL queries work the way you expect.

## How the agent decides which tool to use

It is a LangChain tool-calling agent built on a Groq-hosted LLM (default: `llama-3.3-70b-versatile`). The system prompt explains the four tools and tells the agent two simple rules:

- Data, counts, names, lists, anything that lives in the three databases → call the matching DB tool.
- Concepts, definitions, policies, news, history → call the web search tool.

The agent also has anti-hallucination rules in the prompt. If the tool result does not contain the information the user asked for (for example, the hospitals dataset has no "bed capacity" column), the agent has to say so instead of making up numbers.

## How to run it locally

You need Python 3.11 or newer.

1. Clone the repo:
   ```
   git clone https://github.com/ShadmanSakibRahman/multi-tool-ai-agent-bd.git
   cd multi-tool-ai-agent-bd
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate            # Windows
   source venv/bin/activate          # macOS / Linux
   ```

3. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

4. Add your Groq API key. Copy `.env.example` to `.env` and paste your key:
   ```
   GROQ_API_KEY=your_groq_key_here
   ```
   You can get a free key at https://console.groq.com/keys.

5. Build the SQLite databases (downloads the HF datasets, takes about 30 seconds):
   ```
   python build_databases.py
   ```

6. Run the Streamlit UI:
   ```
   streamlit run app.py
   ```
   Open http://localhost:8501 in the browser.

   Or use the command line REPL:
   ```
   python main.py
   ```

## Project layout

```
.
├── app.py                  Streamlit UI
├── main.py                 command line REPL
├── agent.py                builds the LangChain AgentExecutor
├── tools.py                the 4 tools (3 DB + 1 web search)
├── build_databases.py      downloads HF datasets and writes the SQLite DBs
├── demo.ipynb              Jupyter / Colab notebook with example queries
├── requirements.txt
├── .env.example
├── .gitignore
├── data/                   gets filled after running build_databases.py
└── scripts/                a few helper scripts (schema inspection, smoke tests)
```

## Example queries to try

Routed to **InstitutionsDBTool**:
- "How many government institutions are in Rajshahi?"
- "How many madrasahs are in Sylhet division?"
- "List 5 colleges in Khulna district."

Routed to **HospitalsDBTool**:
- "How many hospitals are in Dhaka district?"
- "Which private clinics are in Chittagong?"
- "List community clinics in Barisal division."

Routed to **RestaurantsDBTool**:
- "Top 10 restaurants in Dhaka by rating."
- "Find restaurants whose name contains biryani."
- "Average rating of all restaurants in the dataset."

Routed to **WebSearchTool**:
- "What is the role of DGHS in Bangladesh?"
- "What is the healthcare policy of Bangladesh?"
- "Explain the EIIN code system."

## Deploy to Streamlit Community Cloud (for the live link)

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and connect your GitHub.
3. New app → pick this repo → main branch → `app.py`.
4. In the app's *Secrets* section, add:
   ```
   GROQ_API_KEY = "your_groq_key_here"
   ```
5. Click Deploy. The first run downloads the HF datasets and builds the three SQLite DBs (takes about a minute), after that startup is fast because Streamlit caches the agent.

## Notes I figured out the hard way

- The hospitals dataset does not have a bed-capacity column. The agent will now say so explicitly instead of inventing numbers.
- Bangladeshi cities have old and new spellings (Chittagong / Chattogram, Dacca / Dhaka, Comilla / Cumilla, Barisal / Barishal, Jessore / Jashore). The SQL prompt tells the LLM to match both. Without that hint, asking for "Chattogram" misses every row that has "Chittagong" in the address.
- The institutions dataset uses uppercase district names (e.g. `'RAJSHAHI'`) while the hospitals dataset uses title case (`'Dhaka'`). The SQL prompt uses case-insensitive `LIKE` for text matching so this does not matter at query time.
- LangChain 1.x moved `AgentExecutor` and `create_tool_calling_agent` out of `langchain.agents`. They now live in `langchain_classic.agents` (which is a separate pip package).
- The DuckDuckGo Python package was renamed from `duckduckgo-search` to `ddgs`. If you see "No module named 'ddgs'", you have the old package installed.

## Author

Md. Shadman Sakib Rahman
