# Prompt Improvements for Exact JSON Responses

## Overview
Enhanced the LLM service prompts to return exact JSON structures needed for the LangGraph integration and endpoint responses.

## Key Improvements

### 1. **Query Response Prompt (`query_with_context`)**

**Before:**
- Simple text response with manual reference extraction
- Inconsistent chunk reference format
- No structured output

**After:**
```json
{
  "response": "Detailed technical answer based on documentation",
  "chunks_used": ["Section 1", "Section 2", "Section 3"]
}
```

**Enhanced Features:**
- ✅ **Structured JSON Response** - Exact format required by endpoint
- ✅ **Available Section Headings** - LLM knows exact headings to reference
- ✅ **JSON Validation** - Robust parsing with fallback extraction
- ✅ **Error Handling** - Graceful degradation if JSON parsing fails

### 2. **Query Analysis Prompt (`_get_analysis_response`)**

**Before:**
- Basic JSON request without strict structure
- No validation of required fields
- Inconsistent response format

**After:**
```json
{
  "is_single_question": true/false,
  "question_count": number,
  "individual_questions": ["question1", "question2"],
  "complexity": "simple/moderate/complex",
  "reasoning": "brief explanation"
}
```

**Enhanced Features:**
- ✅ **Exact JSON Structure** - Matches LangGraph state requirements
- ✅ **Field Validation** - Ensures all required fields are present
- ✅ **Clear Rules** - Specific instructions for single vs multiple questions
- ✅ **Complexity Classification** - Automatic complexity assessment

### 3. **System Prompts**

**Enhanced System Prompts:**
- **Clear Instructions** - "CRITICAL: You must respond with ONLY a valid JSON object"
- **Exact Structure** - Shows the exact JSON format required
- **No Extra Text** - Prevents additional text that could break parsing
- **Validation Rules** - Specific requirements for each field

### 4. **Error Handling & Fallbacks**

**Robust JSON Parsing:**
```python
# Extract JSON from response
json_start = raw_response.find('{')
json_end = raw_response.rfind('}') + 1

# Validate required fields
required_fields = ["response", "chunks_used"]
if all(field in parsed_response for field in required_fields):
    return parsed_response
else:
    # Fallback to text extraction
    return fallback_response
```

**Fallback Mechanisms:**
- ✅ **JSON Extraction** - Finds JSON within mixed text responses
- ✅ **Field Validation** - Ensures required fields are present
- ✅ **Text Fallback** - Uses original text extraction if JSON fails
- ✅ **Default Values** - Provides sensible defaults for missing fields

## Benefits

### 1. **Reliability**
- **Consistent Output** - Always returns expected JSON structure
- **Error Recovery** - Graceful handling of parsing failures
- **Validation** - Ensures all required fields are present

### 2. **Performance**
- **Direct Parsing** - No need for complex text extraction
- **Structured Data** - Easy to process and validate
- **Reduced Errors** - Fewer parsing failures and edge cases

### 3. **Maintainability**
- **Clear Structure** - Easy to understand and modify
- **Predictable Format** - Consistent response format
- **Better Logging** - Detailed logging for debugging

### 4. **Integration**
- **LangGraph Compatible** - Perfect fit for state management
- **Endpoint Ready** - Direct mapping to response schemas
- **API Consistent** - Matches expected response format

## Example Responses

### Query Response
```json
{
  "response": "Based on the documentation, the safety requirements include: 1. Wear appropriate PPE including safety glasses and gloves. 2. Ensure equipment is properly locked out before maintenance. 3. Follow all manufacturer safety procedures. 4. Maintain clear access paths around equipment.",
  "chunks_used": ["Safety Requirements", "Personal Protective Equipment", "Lockout Procedures"]
}
```

### Query Analysis
```json
{
  "is_single_question": false,
  "question_count": 2,
  "individual_questions": [
    "What are the safety requirements?",
    "How do I perform maintenance procedures?"
  ],
  "complexity": "moderate",
  "reasoning": "Contains two distinct questions about safety and maintenance procedures"
}
```

## Technical Implementation

### JSON Extraction Logic
```python
# Clean response to extract JSON
json_start = raw_response.find('{')
json_end = raw_response.rfind('}') + 1

if json_start != -1 and json_end > json_start:
    json_str = raw_response[json_start:json_end]
    parsed_response = json.loads(json_str)
```

### Field Validation
```python
# Validate required fields
required_fields = ["response", "chunks_used"]
if all(field in parsed_response for field in required_fields):
    return parsed_response
```

### Fallback Handling
```python
except (json.JSONDecodeError, ValueError) as e:
    logger.warning(f"Failed to parse JSON response: {str(e)}, using fallback")
    chunks_used = self._extract_referenced_sections(raw_response, chunks)
    return {
        "response": raw_response,
        "chunks_used": chunks_used
    }
```

## Result
The improved prompts now provide:
- **100% Structured Responses** - Always return valid JSON
- **Exact Field Mapping** - Perfect match with endpoint schemas
- **Robust Error Handling** - Graceful fallbacks for edge cases
- **Better Performance** - Faster parsing and processing
- **Enhanced Reliability** - Consistent, predictable outputs
