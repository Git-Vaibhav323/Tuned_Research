# ResearchPilot RAG Evaluation Report

**Generated:** 2026-10-15 11:03

| Metric | Count |
|--------|-------|
| PASS   | 39 |
| FAIL   | 0 |
| SKIP   | 0 |
| HUMAN-REVIEW | 1 |
| **Total** | **40** |

## Results by Category

### Setup

| ID | Description | Status | Detail | Latency (ms) |
|----|-------------|--------|--------|--------------|
| IMPORT | All RAG modules import successfully | ✅ PASS |  | 0 |

### Index

| ID | Description | Status | Detail | Latency (ms) |
|----|-------------|--------|--------|--------------|
| IDX-01 | Index not empty (5796 chunks, 2000 docs) | ✅ PASS | chunks=5796 docs=2000 | 0 |
| IDX-02 | Index has at least 50 documents | ✅ PASS | unique_docs=2000 | 0 |

### Retrieval

| ID | Description | Status | Detail | Latency (ms) |
|----|-------------|--------|--------|--------------|
| RET-01 | retrieve('transformer attention mechanism') | ✅ PASS | top_score=0.63 hits=5 kw=✗ | 37624 |
| RET-02 | retrieve('object detection deep learning YOLO') | ✅ PASS | top_score=0.75 hits=5 kw=✓ | 35 |
| RET-03 | retrieve('convolutional neural network image') | ✅ PASS | top_score=0.53 hits=5 kw=✗ | 35 |
| RET-04 | retrieve('natural language processing BERT GPT') | ✅ PASS | top_score=0.59 hits=5 kw=✓ | 26 |
| RET-05 | retrieve('random forest classification') | ✅ PASS | top_score=0.65 hits=5 kw=✓ | 29 |
| RET-NEG | Nonsense query → low relevance scores | ✅ PASS | max_score=0.110 | 35 |
| RET-EMPTY | Empty query returns no results | ✅ PASS | empty string | 0 |

### Metadata

| ID | Description | Status | Detail | Latency (ms) |
|----|-------------|--------|--------|--------------|
| META-TITLE | Hit has non-empty 'title' | ✅ PASS | value='Understanding of Machine Learn' | 0 |
| META-YEAR | Hit has non-empty 'year' | ✅ PASS | value='2023' | 0 |
| META-SOURCE | Hit has non-empty 'source' | ✅ PASS | value='openalex_csv' | 0 |
| META-CHUNK_ID | Hit has non-empty 'chunk_id' | ✅ PASS | value='97524acf5b3d329927ee' | 0 |
| META-DOC_ID | Hit has non-empty 'doc_id' | ✅ PASS | value='da792c1d1f2f318f' | 0 |

### Citations

| ID | Description | Status | Detail | Latency (ms) |
|----|-------------|--------|--------|--------------|
| CIT-01 | format_citations returns non-empty list | ✅ PASS | count=3 | 0 |
| CIT-02 | Citation has label field '[N]' | ✅ PASS | label=[1] | 0 |
| CIT-03 | Citation title is non-empty | ✅ PASS | title=Gradient-based learning applied to | 0 |
| CIT-04 | Citation number starts at 1 | ✅ PASS | number=1 | 0 |
| CIT-05 | DOI URL format correct when present | ✅ PASS | doi_url=https://doi.org/10.57702/yobif9b | 0 |
| CIT-06 | Evidence type is 'Full text' or 'Abstract only' | ✅ PASS | evidence_type=Abstract only | 0 |
| CIT-MD | format_citations_markdown returns non-empty string | ✅ PASS | len=580 | 0 |

### Loading

| ID | Description | Status | Detail | Latency (ms) |
|----|-------------|--------|--------|--------------|
| LOAD-CSV | load_openalex_csv loads 2000 docs | ✅ PASS | loaded=2000 | 256 |
| LOAD-SCHEMA | Loaded doc has required fields | ✅ PASS | keys=['doc_id', 'title', 'abstract', 'te | 0 |
| PDF-EMPTY | load_pdf_bytes(b'') returns None | ✅ PASS | empty bytes | 0 |
| PDF-GARBAGE | load_pdf_bytes(garbage) returns None | ✅ PASS | random bytes | 0 |
| CSV-BAD-COLS | CSV with no title/abstract returns error | ✅ PASS | docs=0 warnings=1 | 0 |
| CSV-GOOD | Valid CSV with title+abstract loads correctly | ✅ PASS | docs=1 | 0 |
| CSV-SKIP-EMPTY | CSV skips empty rows, loads valid ones | ✅ PASS | docs=2 | 0 |

### Chunking

| ID | Description | Status | Detail | Latency (ms) |
|----|-------------|--------|--------|--------------|
| CHUNK-01 | chunk_documents produces chunks for 5 docs | ✅ PASS | chunks=15 | 0 |
| CHUNK-FIELDS | Chunk has required provenance fields | ✅ PASS | keys=['chunk_id', 'doc_id', 'text', 'tit | 0 |
| CHUNK-MINLEN | All chunks >= MIN_CHUNK_CHARS | ✅ PASS | min_len=177 | 0 |

### Indexing

| ID | Description | Status | Detail | Latency (ms) |
|----|-------------|--------|--------|--------------|
| IDEM-01 | Re-indexing same docs adds 0 new chunks | ✅ PASS | added=0 | 0 |
| IDEM-02 | Total chunk count unchanged after re-index | ✅ PASS | before=5796 after=5796 | 0 |

### Persistence

| ID | Description | Status | Detail | Latency (ms) |
|----|-------------|--------|--------|--------------|
| PERSIST-01 | Chunk count matches after collection re-open | ✅ PASS | before=5796 after=5796 | 0 |

### Generation

| ID | Description | Status | Detail | Latency (ms) |
|----|-------------|--------|--------|--------------|
| OLLAMA-01 | Ollama is reachable and model loaded | ✅ PASS | model_ready=True | 0 |
| OLLAMA-02 | rag_chat() returns tokens and citations | ✅ PASS | tokens=201 cits=5 | 6619 |
| OLLAMA-03 | rag_chat() first-token latency < 30s | ✅ PASS | latency=6619ms | 6619 |
| OLLAMA-QUALITY | Answer quality — does it use evidence and cite sources? | 👁️ HUMAN-REVIEW | Requires human review of answer content | 0 |

### Latency

| ID | Description | Status | Detail | Latency (ms) |
|----|-------------|--------|--------|--------------|
| LAT-RET | Average retrieval latency < 2000ms | ✅ PASS | avg=41ms q1=33ms q2=50ms | 41 |


## Notes

- AUTO tests verify structural correctness only.
- HUMAN-REVIEW tests require reading actual model outputs.
- Answer faithfulness, relevance, and citation accuracy require human evaluation.
- SKIP means the test condition (non-empty index, Ollama running) was not met.
