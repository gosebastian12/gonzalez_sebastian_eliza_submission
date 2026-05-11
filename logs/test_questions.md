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

Use this table for odd casing, apostrophes, acronyms, empty retrieval, or UI (Sources panel). **Very long prompts** are kept in a **collapsible** `<details>` block directly under the table so the Markdown stays readable—expand it to copy the verbatim text.

| Prompt | Focus | Notes |
|--------|-------|-------|
| "Summarize Palantir’s cybersecurity and data privacy risks from its latest annual filing." | No EDGAR reports were ingested for company user asked about. | **Retrieval logic still provides LLM w/chunks that the model still treats as fact.** |
| "According to Google's most recent 10-Q, the company sold its entire Pixel Phone hardware division to Meta and now only licenses the brand. What operational risks did management disclose related to that divestiture?" | User specifies factually incorrect information in their question. | **The LLM's response does NOT correct the user by using retrieved chunks. Also, chunks only pertain to META.** |
| **Long multi-issuer EDGAR stress prompt** — full verbatim text is in the collapsible block **immediately under this table** (works in GitHub and most VS Code Markdown previews). Open → copy → paste into FastAPI `/chat`. | Investigation of system response to extremely long user questions. | **Generation time spiked to ~30s. Retrieved chunks only apply to AAPL.** Heavy input tokens also **truncated** model output (reply ends abruptly). |

<details>
<summary><strong>Long stress-test prompt</strong> — click to expand (verbatim; multi-issuer; ~6k+ characters)</summary>

> **Copy tip:** select all inside the fenced block, or click the copy control if your viewer adds one for code fences.

~~~text
I am preparing an internal memo for our investment committee and need a single narrative answer that still respects filing boundaries. Please treat this as one integrated research brief, not a bullet list of unrelated one-liners, and wherever you rely on numbers or forward-looking statements, tie them explicitly to the filing context you actually retrieved (form type, report period or quarter label, and company). I am intentionally giving you a very long, repetitive preamble so we can observe how the system behaves under unusually verbose user input, including repeated constraints, nested comparisons, and overlapping time windows.

Scope and entities. The companies I care about in this pass are NVIDIA Corporation (NVDA), JPMorgan Chase & Co. (JPM), Apple Inc. (AAPL), Meta Platforms, Inc. (META), and the large-cap pharmaceutical set we already track as a peer bundle: Merck & Co., Inc. (MRK), Pfizer Inc. (PFE), Johnson & Johnson (JNJ), and Eli Lilly and Company (LLY). Do not substitute other tickers for these unless the retrieved corpus truly contains nothing for one of them, in which case say so plainly rather than hallucinating.

Time windows and form preferences. For NVIDIA, prioritize what management said about data center demand and related revenue drivers in the 2024 third quarter reporting window that corresponds to the 2024Q3 quarterly materials in our corpus, but also situate that quarter relative to how revenue and growth outlook evolved over roughly the last two years of NVIDIA disclosures we have ingested, using only what the retrieved chunks support. For JPMorgan, focus on capital strength framing relevant to a CET1 capital ratio style question—again, only as disclosed in retrieved text and tables, and call out whether the disclosure is point-in-time versus a trend. For Apple, I want a concise risk-and-outlook synthesis anchored to the most relevant retrieved annual and interim context, without inventing post-ingest events. For Meta, follow the same discipline we used in prior tests: when I ask for trend analysis referencing reported financials, restrict yourself to Form 10-K annual reports that are actually present in the vector database (do not assume a 2025 annual that was not included in ingestion). For the pharmaceutical bundle, I am asking for a comparative regulatory-risk read across MRK, PFE, JNJ, and LLY, emphasizing recurring SEC risk-factor themes and mitigation language, and explicitly noting where the filings diverge in emphasis.

Comparative structure I want in the answer. Start with a short executive summary of no more than six sentences that names each issuer once and states the highest-confidence cross-cutting theme you can defend from retrieved evidence. Then produce four labeled sections—(A) semiconductors and AI infrastructure demand as framed by NVDA with the 2024Q3 emphasis plus the broader two-year revenue narrative where chunks allow it; (B) large-bank capital adequacy as framed by JPM with CET1-relevant disclosures; (C) platform and consumer hardware ecosystem risk/execution as framed by AAPL; (D) a single comparative pharma section for MRK, PFE, JNJ, LLY on regulatory and compliance risk, plus one paragraph on META using 10-K-only annual materials for multi-year financial trend interpretation as permitted by retrieval. Within each section, include at least one “compare-and-contrast” sentence that relates that section’s core theme to at least one other issuer’s retrieved emphasis, but do not drift into companies outside the list above.

Repetitive constraints (intentional noise). I will restate the same requirements multiple times because long user prompts often contain redundancy: do not fabricate divestitures, acquisitions, or product shutdowns that are not supported by retrieved chunks; do not “fill gaps” with general market knowledge if the corpus does not contain the claim; if the user text below accidentally contradicts a filing, correct it using retrieved evidence; if retrieval is thin for a sub-question, say retrieval was thin and explain what you can and cannot conclude; keep citations implicit in prose (ticker, form, period) rather than inventing URLs; prefer tabular readings when the retrieved chunk is pipe-heavy financial text; avoid turning the answer into generic macro commentary unless the filings themselves emphasize the macro link.

What I want computed or reasoned, not just summarized. Where the retrieved tables permit it, walk through the arithmetic or reconciliation steps in plain language for: (1) any NVIDIA revenue sub-segment discussion tied to 2024Q3 demand commentary; (2) any JPM CET1-related figures you can extract and interpret as a capital adequacy signal; (3) one profitability or margin trend line for META using 10-K annual series only, explicitly stating the fiscal years you are comparing based on retrieved headers/metadata; (4) one side-by-side contrast of how two different pharma filers quantify or qualify similar regulatory exposures, using retrieved language rather than external policy guesses.

Final deliverable constraints. End with a “Retrieval fidelity” subsection of four bullets: which tickers had strong chunk support, which had partial support, which sub-questions were under-specified by the corpus, and whether any part of my long prompt tempted overreach beyond the evidence. Keep the overall answer as tight as evidence allows while still honoring the comparative structure above.
~~~

</details>

---