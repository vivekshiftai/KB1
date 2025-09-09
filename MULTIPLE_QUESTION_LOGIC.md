# Multiple Question Logic - Improved Implementation

## Overview
Enhanced the LangGraph workflow to properly handle multiple questions by combining chunks from all individual questions and generating one comprehensive response.

## How It Works Now

### 1. **Query Analysis** 
- **Input**: User query (e.g., "What are the safety requirements and how do I perform maintenance?")
- **Analysis**: LLM breaks down into individual questions:
  ```json
  {
    "is_single_question": false,
    "question_count": 2,
    "individual_questions": [
      "What are the safety requirements?",
      "How do I perform maintenance?"
    ],
    "complexity": "moderate",
    "reasoning": "Contains two distinct questions about safety and maintenance"
  }
  ```

### 2. **Chunk Retrieval for ALL Questions**
- **Process**: Get chunks for each individual question separately
- **Combination**: Merge all chunks together (avoiding duplicates)
- **Result**: Comprehensive chunk collection covering all aspects

```python
# Get chunks for each individual query
for i, query in enumerate(individual_queries):
    chunks = await self.vector_db.query_chunks(
        collection_name=state["collection_name"],
        query=query,
        top_k=chunk_size
    )
    # Add to combined list (avoid duplicates)
    for chunk in chunks:
        if chunk not in all_combined_chunks:
            all_combined_chunks.append(chunk)
```

### 3. **Comprehensive Response Generation**
- **Input**: Combined chunks from all questions + Original user query
- **Enhanced Prompt**: Mentions individual questions for context
- **Output**: One comprehensive response addressing the original query

```python
enhanced_query = f"""Original Query: {original_query}

This query contains {len(individual_queries)} individual questions:
1. What are the safety requirements?
2. How do I perform maintenance?

Please provide a comprehensive answer that addresses all aspects of the original query."""
```

### 4. **Validation Against Original Query**
- **Target**: Always validates against the original user query
- **Metrics**: 
  - Mentions original query terms
  - Addresses individual questions (bonus points)
  - Uses chunks effectively
  - Has sufficient content

```python
quality_metrics = {
    "mentions_original_query": any(word.lower() in response.lower() for word in original_query.split()),
    "mentions_individual_queries": sum(1 for q in individual_queries if any(word.lower() in response.lower() for word in q.split())),
    "uses_chunks": len(state["llm_response"].get("chunks_used", [])) > 0,
    # ... other metrics
}

# Bonus for addressing multiple questions
if len(individual_queries) > 1:
    questions_addressed_ratio = quality_metrics["mentions_individual_queries"] / len(individual_queries)
    confidence += questions_addressed_ratio * 0.2
```

### 5. **Retry Logic with Chunk Size Increase**
- **Increment**: +1 chunk for each retry (for all questions)
- **Scope**: Applies to all individual questions in the retry
- **Validation**: Always against original user query

## Workflow Flow

```
User Query: "What are safety requirements and maintenance procedures?"

1. Analyze Query
   ↓
2. Individual Questions: ["safety requirements", "maintenance procedures"]
   ↓
3. Retrieve Chunks (for each question)
   - Safety chunks: [chunk1, chunk2, chunk3]
   - Maintenance chunks: [chunk4, chunk5, chunk6]
   ↓
4. Combine Chunks: [chunk1, chunk2, chunk3, chunk4, chunk5, chunk6]
   ↓
5. Generate Response (using original query + combined chunks)
   ↓
6. Validate Response (against original query)
   ↓
7. If inadequate → Increase chunk size (+1) → Retry from step 3
   ↓
8. Finalize Response
```

## Key Improvements

### ✅ **Comprehensive Coverage**
- Gets chunks for ALL individual questions
- Combines them for maximum context
- Generates one unified response

### ✅ **Original Query Focus**
- Always validates against original user query
- Response addresses the user's actual intent
- Maintains context of the complete question

### ✅ **Efficient Retry Logic**
- Increases chunk size by +1 for all questions
- Single retry loop (not per question)
- Faster processing and better results

### ✅ **Better Validation**
- Checks if response addresses original query
- Bonus points for covering individual questions
- More accurate quality assessment

## Example Scenarios

### Scenario 1: Single Question
```
Input: "What are the safety requirements?"
Analysis: Single question
Chunks: Safety-related chunks only
Response: Direct answer about safety
Validation: Against "safety requirements"
```

### Scenario 2: Multiple Questions
```
Input: "What are safety requirements and how do I perform maintenance?"
Analysis: 2 questions
Chunks: Safety chunks + Maintenance chunks (combined)
Response: Comprehensive answer covering both aspects
Validation: Against original query + bonus for addressing both questions
```

### Scenario 3: Complex Query
```
Input: "What are safety requirements, maintenance procedures, and troubleshooting steps?"
Analysis: 3 questions
Chunks: Safety + Maintenance + Troubleshooting chunks (combined)
Response: Complete guide covering all three areas
Validation: Against original query + bonus for addressing all three questions
```

## Benefits

1. **Better Context**: More comprehensive chunk collection
2. **Unified Response**: One coherent answer instead of fragmented responses
3. **Original Intent**: Always addresses the user's actual question
4. **Efficient Processing**: Single retry loop instead of multiple
5. **Better Validation**: More accurate quality assessment
6. **Improved User Experience**: Complete, comprehensive answers

## Technical Implementation

### Chunk Combination Logic
```python
all_combined_chunks = []
for query in individual_queries:
    chunks = await self.vector_db.query_chunks(query, top_k=chunk_size)
    for chunk in chunks:
        if chunk not in all_combined_chunks:
            all_combined_chunks.append(chunk)
```

### Enhanced Prompt Generation
```python
if len(individual_queries) > 1:
    enhanced_query = f"""Original Query: {original_query}

This query contains {len(individual_queries)} individual questions:
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(individual_queries)])}

Please provide a comprehensive answer that addresses all aspects of the original query."""
```

### Validation with Bonus Scoring
```python
# Bonus for addressing multiple questions
if len(individual_queries) > 1:
    questions_addressed_ratio = quality_metrics["mentions_individual_queries"] / len(individual_queries)
    confidence += questions_addressed_ratio * 0.2
```

## Result
The improved logic now provides:
- **Comprehensive chunk collection** from all individual questions
- **Unified response generation** addressing the original query
- **Better validation** against user intent
- **Efficient retry mechanism** with chunk size increase
- **Enhanced user experience** with complete, coherent answers
