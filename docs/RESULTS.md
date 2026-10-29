# Eval results

Every row is one eval run of `eval_self_rag.py` (each case run 3 times, scores averaged).
Rows are in the order the changes were made; each change was kept only if it held or improved
the scores. Generated from the run summaries in `eval_results/` (not committed).

- **correctness**: share of expected facts stated; for questions the documents cannot answer,
  1 only if the answer declines without false claims (0 if it presents another entity's facts).
- **faithfulness**: share of answer claims supported by the retrieved context (LLM judge).
- **citations**: share of cited sources that exist in the context (deterministic).
- **path**: answered directly / from documents / from the web / declined, as expected.
- **cost**: pipeline USD per 1,000 questions (judge not included; cached-input discounts ignored).

Judge: `gpt-5.4` until the re-baseline row, `gpt-6.1-sol` (effort low) from there on, so scores
before and after that row are not directly comparable. Costs were measured from the re-baseline on.

| Corpus | Run | Change | Cases | Pipeline model | Correct. | Faithf. | Citations | Path | LLM calls | Cost / 1k | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nexaai | `baseline-1fa082ff` | Baseline: dense retrieval, relevance grader, support + usefulness checks, web fallback | 19 | gpt-5.4-mini | 0.93 | 0.98 | 0.93 | 0.91 | 13.2 | – | 11.0 |
| nexaai | `scope-routing-c4b4f1da` | Scope routing + entity-aware relevance grading (namesake fix) | 19 | gpt-5.4-mini | 0.94 | 1.00 | 0.95 | 0.95 | 11.0 | – | 8.2 |
| nexaai | `grader-multipart-e99a5bff` | Grader handles multi-part questions about one organization | 19 | gpt-5.4-mini | 0.98 | 1.00 | 0.95 | 1.00 | 9.6 | – | 6.7 |
| nexaai | `sol-judge-baseline-1dced20c` | New eval judge (gpt-6.1-sol), same pipeline: re-baseline before changing models | 19 | gpt-5.4-mini | 0.96 | 1.00 | 0.95 | 1.00 | 9.5 | $5.02 | 7.0 |
| nexaai | `luna-none-035ba7d2` | Pipeline model gpt-6-luna, effort none | 19 | gpt-6-luna@none | 0.98 | 1.00 | 0.92 | 1.00 | 9.4 | $0.59 | 12.8 |
| nexaai | `luna-low-c958b0f9` | Pipeline model gpt-6-luna, effort low (rejected: self-contradicting answer) | 19 | gpt-6-luna@low | 0.96 | 1.00 | 0.98 | 0.98 | 9.4 | $0.75 | 9.9 |
| nexaai | `luna-default-540b59fc` | Luna@none as default; two eval expectations clarified (approved) | 19 | gpt-6-luna@none | 0.98 | 1.00 | 1.00 | 1.00 | 9.6 | $0.62 | 10.4 |
| nexaai | `sections-bab0708e` | Structure-aware chunking (Markdown sections, heading path in every chunk) | 19 | gpt-6-luna@none | 1.00 | 1.00 | 0.97 | 1.00 | 9.6 | $0.55 | 8.9 |
| nexaai | `grounded-7e278bf6` | Citations checked in code; retrieved text treated as data (ceo 0.67: usefulness check misread the company name, fixed next row) | 19 | gpt-6-luna@none | 0.98 | 1.00 | 1.00 | 0.98 | 10.1 | $0.60 | 9.3 |
| nexaai | `usefulness-names-f10de28a` | Usefulness check knows the knowledge base's name for the entity | 19 | gpt-6-luna@none | 1.00 | 1.00 | 1.00 | 1.00 | 9.5 | $0.57 | 8.4 |
| nexaai | `local-mode-09a1dd1b` | Local eval runner (no LangSmith), same metrics | 19 | gpt-6-luna@none | 1.00 | 1.00 | 0.97 | 1.00 | 9.5 | $0.57 | 8.9 |
| nexaai | `retries-07767baf` | Timeouts and retries on LLM, embedding and web calls | 19 | gpt-6-luna@none | 1.00 | 1.00 | 1.00 | 1.00 | 9.5 | $0.58 | 8.6 |
| nexaai | `web-error-35694d7c` | Web search failures routed to a clean no-answer | 19 | gpt-6-luna@none | 1.00 | 1.00 | 1.00 | 1.00 | 9.5 | $0.57 | 8.4 |
| nexaai | `budget-c73f0977` | Per-question LLM call budget | 19 | gpt-6-luna@none | 1.00 | 1.00 | 0.97 | 1.00 | 9.5 | $0.57 | 8.5 |
| nexaai | `source-ids-fb1857ab` | Short source ids ([S1]) mapped back in code; web results de-duplicated by URL (one correct web answer falsely rejected as not useful, fixed two rows later) | 19 | gpt-6-luna@none | 0.98 | 1.00 | 1.00 | 0.98 | 9.5 | $0.57 | 8.8 |
| nexaai | `web-cases-292e0343` | 3 harder web cases added (22 cases): exposed false usefulness rejections | 22 | gpt-6-luna@none | 0.94 | 1.00 | 1.00 | 0.94 | 10.0 | $0.67 | 8.2 |
| nexaai | `usefulness-scope-8d1a3020` | Usefulness check told the question's scope (current main) | 22 | gpt-6-luna@none | 1.00 | 1.00 | 1.00 | 1.00 | 10.1 | $0.68 | 8.9 |
| nexaai | `refine-73545c6a` | CRAG-style refinement of long sources, 3000-char threshold (not merged) | 22 | gpt-6-luna@none | 1.00 | 1.00 | 1.00 | 1.00 | 10.2 | $0.69 | 8.7 |
| nexaai | `refine-1000-9c17b720` | Refinement at 1000 chars, 4 web cases only (not merged) | 4 | gpt-6-luna@none | 1.00 | 1.00 | 1.00 | 1.00 | 14.0 | $1.34 | 11.1 |
| injection-test | `injection-a6a25b5c` | Prompt-injection corpus, first run | 3 | gpt-6-luna@none | 1.00 | 1.00 | 1.00 | 1.00 | 7.0 | $0.36 | 8.2 |
| injection-test | `source-ids-71a3927b` | Prompt-injection corpus after short source ids | 3 | gpt-6-luna@none | 1.00 | 1.00 | 1.00 | 1.00 | 7.0 | $0.36 | 8.2 |
| injection-test | `refine-733aa3c4` | Prompt-injection corpus with refinement (not merged) | 3 | gpt-6-luna@none | 1.00 | 1.00 | 1.00 | 1.00 | 7.0 | $0.36 | 8.1 |

The NexaAI corpus is synthetic (an invented company, 3 PDFs, 20 chunks) and its eval set is now
saturated: 1.00 means these 22 cases pass, not that the system is perfect. A real public corpus
with a larger eval set is the next step.
