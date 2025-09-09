# Current Query Flow with LangGraph

## 🔄 Complete Query Processing Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUERY ENDPOINT                               │
│  POST /query                                                    │
│  { "query": "What are safety requirements?", "pdf_name": "..." }│
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              LangGraphQueryProcessor.process_query()            │
│  - Initialize state with query, PDF name, chunk size = 3       │
│  - Execute LangGraph workflow                                   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH WORKFLOW                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ 1. Initialize   │───▶│ 2. Retrieve     │───▶│ 3. Generate     │
│    Query        │    │    Chunks       │    │    LLM Response │
│                 │    │                 │    │                 │
│ - Check PDF     │    │ - Query vector  │    │ - Process with  │
│   exists        │    │   DB with 3     │    │   Azure AI      │
│ - Validate      │    │   chunks        │    │ - Generate      │
│   collection    │    │ - Get embedded  │    │   response      │
│                 │    │   images        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Validate Response Quality                                    │
│                                                                 │
│ - Check response length (>50 chars)                            │
│ - Check if mentions query terms                                │
│ - Check if uses chunks                                         │
│ - Calculate confidence score (0.0-1.0)                        │
│                                                                 │
│ Confidence = has_content(0.3) + mentions_query(0.3) +          │
│              uses_chunks(0.2) + response_length(0.2)           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Decision Logic                                               │
│                                                                 │
│ IF confidence >= 0.6 AND response is adequate:                 │
│   └─▶ Go to Finalize Response                                  │
│                                                                 │
│ IF confidence < 0.7 OR needs human review:                     │
│   └─▶ Go to Human Decision (simulated)                         │
│                                                                 │
│ IF iteration < 3:                                               │
│   └─▶ Go to Increase Chunk Size                                │
│                                                                 │
│ IF iteration >= 3:                                              │
│   └─▶ Go to Finalize Response                                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6A. Human Decision (Simulated)                                 │
│                                                                 │
│ IF confidence >= 0.8:                                          │
│   └─▶ Decision = "process"                                     │
│                                                                 │
│ IF confidence >= 0.6:                                          │
│   └─▶ Decision = "requery"                                     │
│                                                                 │
│ ELSE:                                                           │
│   └─▶ Decision = "requery"                                     │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6B. Increase Chunk Size                                         │
│                                                                 │
│ - Current size: 3 → 5 → 7                                      │
│ - Increment: +2 each time                                       │
│ - Increment iteration counter                                   │
│                                                                 │
│ IF iteration < 3:                                               │
│   └─▶ Go back to Retrieve Chunks (with new size)               │
│                                                                 │
│ IF iteration >= 3:                                              │
│   └─▶ Go to Finalize Response                                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. Finalize Response                                            │
│                                                                 │
│ - Collect embedded images from used chunks                     │
│ - Collect tables from used chunks                              │
│ - Match chunks used by LLM                                     │
│ - Create QueryResponse object                                  │
│ - Calculate processing time                                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RETURN RESPONSE                              │
│                                                                 │
│ QueryResponse {                                                 │
│   success: true,                                               │
│   message: "Query processed successfully...",                  │
│   response: "LLM generated response...",                       │
│   chunks_used: ["Section 1", "Section 2"],                    │
│   images: [ImageData objects with base64 data],                │
│   tables: ["<table>...</table>"],                              │
│   processing_time: "2.34s"                                     │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Retry Loop Example

```
Iteration 1: 3 chunks → Confidence: 0.4 → Retry with 5 chunks
Iteration 2: 5 chunks → Confidence: 0.7 → Process (good enough)
Iteration 3: 7 chunks → (if still needed)
```

## 📊 Key Features

1. **Automatic Quality Assessment**: Each response gets a confidence score
2. **Progressive Chunk Increase**: 3 → 5 → 7 chunks for better context
3. **Maximum 3 Iterations**: Prevents infinite loops
4. **Embedded Images**: Images stored directly in chunks (base64)
5. **Smart Decision Making**: Automatically decides retry vs. finalize
6. **Same API**: Maintains existing endpoint structure

## 🎯 Quality Metrics

- **has_content**: Response has meaningful content (0.3 weight)
- **mentions_query**: Response mentions query terms (0.3 weight)  
- **uses_chunks**: Response references retrieved chunks (0.2 weight)
- **response_length**: Response is sufficiently long (0.2 weight)

## 🚀 Benefits

- **Better Responses**: More context with increased chunks
- **Quality Control**: Automatic validation of response quality
- **Efficient**: Only retries when necessary
- **Reliable**: Maximum iteration limit prevents hanging
- **Fast**: Embedded images don't require additional queries
