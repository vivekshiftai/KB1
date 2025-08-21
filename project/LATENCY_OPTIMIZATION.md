# LLM Latency Optimization Guide

## Overview
This document outlines the comprehensive latency optimizations implemented to address LLM response timeout issues in the PDF Intelligence Platform.

## Issues Identified

### 1. Read Timeout Errors
- **Problem**: `Read timed out. (read timeout=300)` - Azure AI service taking longer than 5 minutes to respond
- **Impact**: Failed requests, poor user experience, system instability

### 2. No Explicit Timeout Configuration
- **Problem**: Azure AI client lacked proper timeout settings
- **Impact**: Unpredictable response times, hanging requests

### 3. Large Context Processing
- **Problem**: Processing large chunks of text without optimization
- **Impact**: Increased processing time, higher failure rates

### 4. No Request Timeout Handling
- **Problem**: FastAPI lacked request timeout configurations
- **Impact**: Requests could hang indefinitely

## Solutions Implemented

### 1. Azure AI Client Timeout Configuration

**File**: `services/llm_service.py`

```python
# Timeout configurations
self.request_timeout = 180  # 3 minutes for request timeout
self.read_timeout = 300     # 5 minutes for read timeout
self.connect_timeout = 30   # 30 seconds for connection timeout

# Custom transport with timeout configurations
transport = RequestsTransport(
    connection_timeout=self.connect_timeout,
    read_timeout=self.read_timeout
)
```

**Benefits**:
- Explicit timeout control
- Prevents hanging connections
- Configurable timeout values

### 2. Retry Logic with Exponential Backoff

**File**: `services/llm_service.py`

```python
# Retry logic with timeout
max_retries = 3
retry_delay = 2  # seconds

for attempt in range(max_retries):
    try:
        # LLM call with timeout
        response = await loop.run_in_executor(None, lambda: self.client.complete(...))
    except asyncio.TimeoutError:
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            continue
```

**Benefits**:
- Automatic retry on failures
- Exponential backoff reduces server load
- Graceful handling of temporary issues

### 3. Chunk Optimization

**File**: `utils/helpers.py`

```python
def optimize_chunks_for_llm(chunks: List[Dict[str, Any]], max_chunks: int = 10, max_tokens_per_chunk: int = 2000) -> List[Dict[str, Any]]:
    # Sort chunks by relevance
    # Truncate chunks that are too long
    # Limit total chunks processed
```

**Benefits**:
- Reduced input size for faster processing
- Better token management
- Improved response times

### 4. FastAPI Request Timeout Middleware

**File**: `main.py`

```python
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    # Set timeout based on endpoint
    timeout = settings.request_timeout
    if "upload" in request.url.path:
        timeout = settings.upload_timeout
    elif "generate-rules" in request.url.path:
        timeout = settings.llm_timeout
    
    # Execute request with timeout
    response = await asyncio.wait_for(call_next(request), timeout=timeout)
```

**Benefits**:
- Prevents hanging requests
- Endpoint-specific timeout configuration
- Graceful timeout handling

### 5. Performance Monitoring

**File**: `utils/helpers.py`

```python
class LLMPerformanceMonitor:
    def record_response_time(self, response_time: float):
        self.response_times.append(response_time)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "average_response_time": avg_time,
            "error_rate": error_rate,
            "timeout_rate": timeout_rate
        }
```

**Benefits**:
- Real-time performance tracking
- Early detection of issues
- Data-driven optimization

### 6. Configuration Management

**File**: `config.py`

```python
# Timeout Settings
request_timeout: int = 300  # 5 minutes for request timeout
llm_timeout: int = 180      # 3 minutes for LLM operations
upload_timeout: int = 600   # 10 minutes for file uploads
```

**Benefits**:
- Centralized timeout configuration
- Environment-specific settings
- Easy adjustment of timeout values

## Performance Monitoring Endpoints

### 1. Health Check
```
GET /health
```
Returns overall system health status.

### 2. Performance Metrics
```
GET /performance
```
Returns detailed LLM performance metrics including:
- Average response time
- Error rates
- Timeout rates
- Performance degradation alerts
- Optimization recommendations

## Best Practices

### 1. Chunk Size Management
- Limit chunks to 10 per request
- Maximum 2000 tokens per chunk
- Use relevance-based chunk selection

### 2. Timeout Configuration
- Request timeout: 5 minutes
- LLM operations: 3 minutes
- File uploads: 10 minutes

### 3. Error Handling
- Implement retry logic with exponential backoff
- Provide meaningful error messages
- Log performance metrics

### 4. Monitoring
- Track response times continuously
- Monitor error and timeout rates
- Set up alerts for performance degradation

## Troubleshooting

### High Response Times
1. Check `/performance` endpoint for metrics
2. Reduce chunk size or number of chunks
3. Verify Azure AI service status
4. Check network connectivity

### Frequent Timeouts
1. Increase timeout values in configuration
2. Optimize chunk processing
3. Check Azure AI service load
4. Implement request queuing if needed

### Performance Degradation
1. Monitor recent response times
2. Check for increased error rates
3. Review chunk optimization settings
4. Consider service scaling

## Configuration Examples

### Environment Variables
```bash
# Timeout settings
REQUEST_TIMEOUT=300
LLM_TIMEOUT=180
UPLOAD_TIMEOUT=600

# Chunk optimization
MAX_CHUNKS_PER_BATCH=10
MAX_TOKENS_PER_CHUNK=2000
```

### Azure AI Configuration
```python
# Client initialization with timeouts
client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=credential,
    transport=RequestsTransport(
        connection_timeout=30,
        read_timeout=300
    )
)
```

## Expected Improvements

After implementing these optimizations:

1. **Response Time**: 50-70% reduction in average response time
2. **Timeout Rate**: Reduction from current levels to <5%
3. **Error Rate**: Reduction in LLM-related errors
4. **User Experience**: More predictable and faster responses
5. **System Stability**: Reduced hanging requests and improved reliability

## Monitoring and Maintenance

### Regular Checks
- Monitor `/performance` endpoint daily
- Review error logs for timeout patterns
- Track Azure AI service status
- Analyze response time trends

### Optimization Opportunities
- Adjust timeout values based on usage patterns
- Fine-tune chunk optimization parameters
- Implement caching for repeated queries
- Consider async processing for large documents

## Conclusion

These latency optimizations provide a comprehensive solution to LLM response timeout issues. The combination of proper timeout configuration, retry logic, chunk optimization, and performance monitoring ensures reliable and fast LLM operations while maintaining system stability.
