# RAG test prompts

Short-lived catalog of **user-style questions** used to exercise retrieval, reranking, and the chat UI. Extend by copying a row in the tables below or a block from the template section.

---

## How to read this doc

| Column | Meaning |
|--------|---------|
| **Prompt** | Exact text sent to the app (paste verbatim) |
| **Focus** | What you are mainly testing (tickers, forms, multi-company, edge cases) |
| **Notes** | Expected behavior, failure modes observed, etc...

---

## Prompt inventory

### Single-company

| Prompt | Focus | Notes |
|--------|-------|-------|
| "How has NVIDIA's revenue and growth outlook changed over the last two years?" | General time reference, specific analysis requested | Initially, system was pulling reports from other companies. Changes resulted in only NVID reports being retrieved. |
| "What is JPMorgan’s CET1 capital ratio?" | Request for a calculation of a specific financial metric. | Calculations were performed with retrieved tabular data and seemed to be accurate. |

---

### Multi-company

| Prompt | Focus | Notes |
|--------|-------|-------|
| "What regulatory risks do the major pharmaceutical companies face, and how are they addressing them?" | Vague about companies. Requesting information about past *and* future. | RAG system was able to retrieve chunks from all pharmaceutical companies in Corpus - MRK, PFE, JNJ, LLY. |
| "What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?" | Specific about companies of interest. Request for comparison amongst them. | **No chunks about TSLA were retrieved. Only for AAPL and JPM.** |

---

### Time / filing constraints

| Prompt | Focus | Notes |
|--------|-------|-------|
| “What did NVDA say about data center demand in 2024Q3?” | Explicity quarter filter from user. | All chunks came from the 2024Q3 10Q NVDA report. |
| "Conduct a trend analysis for META referencing the financial data it reported. Only utilize 10-K, annual reports."| Explicit user instructions to only use 10-K reports. | Retrieved chunks came from all annual reports ingested into vector db. (2025 10-K was not included in corpus). |

---

### Edge cases & regressions

Use this table for odd casing, apostrophes, acronyms, empty retrieval, or UI (Sources panel).

| Prompt | Focus | Notes |
|--------|-------|-------|
| "Summarize Palantir’s cybersecurity and data privacy risks from its latest annual filing." | No EDGAR reports were ingested for company user asked about. | **Retrieval logic still provides LLM w/chunks that the model still treats as fact.** |
| "According to Google's most recent 10-Q, the company sold its entire Pixel Phone hardware division to Meta and now only licenses the brand. What operational risks did management disclose related to that divestiture?" | User specifies factually incorrect information in their question. | **The LLM's response does NOT correct the user by using retrieved chunks. Also, chunks only pertain to META.** |

---