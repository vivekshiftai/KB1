"""
LLM Service - Clean Version
Handles interactions with Azure AI for intelligent responses

Version: 0.2 - Cleaned up for LangGraph integration
"""

import json
import logging
import re
import tiktoken
import threading
import asyncio
from typing import List, Dict, Any, Optional
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # Azure AI Configuration
        self.endpoint = "https://chgai.services.ai.azure.com/models"
        self.api_version = "2024-05-01-preview"
        
        # Model configurations for different use cases
        self.models = {
            "maintenance": {
                "name": settings.maintenance_model_name,
                "endpoint": settings.maintenance_model_endpoint or self.endpoint
            },
            "rules": {
                "name": settings.rules_model_name,
                "endpoint": settings.rules_model_endpoint or self.endpoint
            },
            "safety": {
                "name": settings.safety_model_name,
                "endpoint": settings.safety_model_endpoint or self.endpoint
            },
            "query": {
                "name": settings.query_model_name,
                "endpoint": settings.query_model_endpoint or self.endpoint
            },
            "analysis": {
                "name": settings.analysis_model_name,
                "endpoint": settings.analysis_model_endpoint or self.endpoint
            }
        }
        
        # Validate Azure AI key
        if not settings.azure_openai_key:
            logger.error("Azure AI key not configured. Please set AZURE_OPENAI_KEY in your environment.")
            raise ValueError("Azure AI key is required but not configured")
        
        try:
            # Initialize Azure AI client
            self.client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(settings.azure_openai_key),
                api_version=self.api_version
            )
            logger.info("Azure AI client initialized successfully")
            logger.info(f"Available models: {list(self.models.keys())}")
        except Exception as e:
            logger.error(f"Failed to initialize Azure AI client: {str(e)}")
            raise e
        
        # Token limits
        self.max_tokens = 8192
        self.max_completion_tokens = 1500
        self.max_context_tokens = self.max_tokens - self.max_completion_tokens
        
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception as e:
            logger.warning(f"Failed to load GPT-4 encoding, using default: {str(e)}")
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"LLM Service initialized - max_tokens: {self.max_tokens}")
        
        # Thread-safe locks for concurrent operations
        self._request_locks = {}  # Per-model request locks
        self._global_lock = threading.Lock()
        # Semaphore to limit concurrent API requests (prevent rate limiting)
        self._api_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
    
    def _get_request_lock(self, model_name: str) -> threading.Lock:
        """Get or create a lock for a specific model to prevent rate limiting"""
        with self._global_lock:
            if model_name not in self._request_locks:
                self._request_locks[model_name] = threading.Lock()
            return self._request_locks[model_name]
    
    def _get_model_config(self, use_case: str) -> dict:
        """Get model configuration for specific use case"""
        if use_case not in self.models:
            logger.warning(f"Unknown use case '{use_case}', using default query model")
            use_case = "query"
        
        model_config = self.models[use_case]
        logger.info(f"Using {use_case} model: {model_config['name']} at {model_config['endpoint']}")
        return model_config
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def _needs_table_data(self, query: str) -> bool:
        """Determine if the query requires table data - Always return True to include all data"""
        # Always include tables and full data for comprehensive responses
        return True

    def _filter_chunk_content(self, chunk: Dict[str, Any], needs_tables: bool) -> str:
        """Extract full chunk content including all data and tables"""
        try:
            if "metadata" in chunk and "document" in chunk:
                heading = chunk.get("metadata", {}).get("heading", "")
                content = chunk.get("document", "")
            else:
                heading = chunk.get("heading", "")
                content = chunk.get("text", chunk.get("content", ""))
            
            if not content:
                return ""
            
            # Always return full content including tables, specifications, and all data
            # No filtering - use complete chunk data for comprehensive responses
            return content
        except Exception as e:
            logger.warning(f"Error extracting chunk content: {str(e)}")
            return ""

    async def assess_information_sufficiency(self, chunks: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Assess if we have sufficient information to answer the query completely"""
        logger.info(f"Assessing information sufficiency for query: {query}")
        
        if not chunks:
            return {
                "has_sufficient_info": False,
                "missing_information": ["No information available"],
                "additional_queries_needed": [query],
                "confidence_score": 0.0,
                "reasoning": "No chunks available for analysis"
            }
        
        # Prepare context from chunks
        context_parts = []
        for chunk in chunks:
            try:
                if "metadata" in chunk and "document" in chunk:
                    heading = chunk.get("metadata", {}).get("heading", "")
                    content = chunk.get("document", "")
                else:
                    heading = chunk.get("heading", "")
                    content = chunk.get("text", chunk.get("content", ""))
                
                if content:
                    context_parts.append(f"**{heading}**\n{content[:500]}...")  # Truncate for assessment
            except Exception as e:
                logger.warning(f"Error processing chunk for assessment: {str(e)}")
                continue
        
        context = "\n\n".join(context_parts)
        
        assessment_prompt = f"""Analyze if you have sufficient information to answer the user's question completely.

Available Information:
{context}

User Question: {query}

CRITICAL: Respond with ONLY a valid JSON object with this exact structure:
{{
    "has_sufficient_info": true/false,
    "missing_information": ["list of specific missing information"],
    "additional_queries_needed": ["specific search terms for missing info"],
    "confidence_score": 0.0-1.0,
    "reasoning": "brief explanation of assessment"
}}

ASSESSMENT CRITERIA:
- Can you provide a complete, actionable answer?
- Do you have all necessary steps, procedures, or details?
- Are there any gaps that would require "refer to section X" responses?
- Can you include specific values, measurements, and technical details?

If missing information, provide specific search terms that would help find the missing details.

Return ONLY the JSON object, no additional text."""
        
        try:
            # Get analysis model configuration
            model_config = self._get_model_config("analysis")
            
            # Use semaphore to limit concurrent requests
            async with self._api_semaphore:
                response = self.client.complete(
                    messages=[
                        SystemMessage(content=assessment_prompt),
                        UserMessage(content="Assess the information sufficiency.")
                    ],
                    max_tokens=1000,
                    temperature=0.1,
                    top_p=0.1,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    model=model_config["name"]
                )
            
            raw_response = response.choices[0].message.content.strip()
            logger.info(f"Assessment response: {raw_response[:200]}...")
            
            # Parse JSON response
            try:
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = raw_response[json_start:json_end]
                    assessment = json.loads(json_str)
                    
                    # Validate required fields
                    required_fields = ["has_sufficient_info", "missing_information", "additional_queries_needed", "confidence_score", "reasoning"]
                    for field in required_fields:
                        if field not in assessment:
                            logger.warning(f"Missing field {field} in assessment response")
                    
                    logger.info(f"Information assessment: sufficient={assessment.get('has_sufficient_info')}, confidence={assessment.get('confidence_score')}")
                    return assessment
                else:
                    raise ValueError("No valid JSON found in response")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse assessment JSON: {str(e)}")
                logger.error(f"Raw response: {raw_response}")
                # Fallback assessment
                return {
                    "has_sufficient_info": False,
                    "missing_information": ["Unable to assess information"],
                    "additional_queries_needed": [query],
                    "confidence_score": 0.0,
                    "reasoning": f"JSON parsing error: {str(e)}"
                }
            
        except Exception as e:
            logger.error(f"Error in information assessment: {str(e)}")
            return {
                "has_sufficient_info": False,
                "missing_information": ["Assessment failed"],
                "additional_queries_needed": [query],
                "confidence_score": 0.0,
                "reasoning": f"Assessment error: {str(e)}"
            }

    async def query_with_context(self, chunks: List[Dict[str, Any]], query: str, query_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query with context chunks using structured JSON response"""
        logger.info(f"Processing query with {len(chunks)} context chunks")
        
        # If no chunks available, provide a helpful response
        if not chunks:
            logger.warning("No chunks available, providing general response")
            return {
                "response": f"I don't have access to specific documentation about '{query}'. Please ensure the document is properly uploaded and processed.",
                "chunks_used": []
            }
        
        # Determine if query needs table data
        needs_tables = self._needs_table_data(query)
        logger.info(f"Query analysis - Tables needed: {needs_tables}")
        
        # Prepare context from chunks
        context_parts = []
        chunk_headings = []
        for chunk in chunks:
            try:
                if "metadata" in chunk and "document" in chunk:
                    heading = chunk.get("metadata", {}).get("heading", "")
                else:
                    heading = chunk.get("heading", "")
                
                # Filter content based on table needs
                content = self._filter_chunk_content(chunk, needs_tables)
                
                if content:
                    context_parts.append(f"**{heading}**\n{content}")
                    chunk_headings.append(heading)
            except Exception as e:
                logger.warning(f"Error processing chunk: {str(e)}")
                continue
        
        if not context_parts:
            logger.warning("No valid context could be extracted from chunks")
            return {
                "response": f"I couldn't extract meaningful content from the provided chunks for '{query}'.",
                "chunks_used": []
            }
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt for structured JSON response
        table_info = "Full documentation data including tables, specifications, and all technical details has been included in the context for comprehensive analysis."
        
        # Add query analysis information if available
        analysis_info = ""
        if query_analysis:
            complexity = query_analysis.get("complexity", "unknown")
            question_count = query_analysis.get("question_count", 1)
            is_single = query_analysis.get("is_single_question", True)
            reasoning = query_analysis.get("reasoning", "")
            
            analysis_info = f"""
Query Analysis:
- Complexity: {complexity}
- Question Count: {question_count}
- Single Question: {is_single}
- Reasoning: {reasoning}
"""
        
        system_prompt = f"""You are a technical documentation assistant. You must respond with a valid JSON object containing the answer and referenced sections.

Context Information: {table_info}{analysis_info}

CRITICAL: Your response must be ONLY a valid JSON object with this exact structure:
{{
    "response": "Your detailed answer to the user's question",
    "chunks_used": ["List of section headings you referenced"]
}}

ABSOLUTE REQUIREMENT: NEVER mention chapter numbers, section numbers, or give generic references like "as described in chapter X". ALWAYS provide the actual content from the documentation chunks.

CRITICAL CONTENT REQUIREMENTS - NO EXCEPTIONS:
- ONLY use information from the provided documentation chunks above
- NEVER mention chapter numbers, section numbers, or page references
- NEVER say "as described in", "refer to", "see chapter", "check section", or "please refer to"
- NEVER give generic responses like "as described in chapter 4.1" or "refer to section X"
- ALWAYS provide the actual content from the chunks instead of references
- If you have the information in the chunks, provide the complete details
- If you don't have the information, say "This information is not available in the provided documentation"
- Extract and present the actual steps, procedures, and details from the documentation
- Quote specific values, measurements, and technical details directly from the chunks

RESPONSE FORMAT REQUIREMENTS:
- Structure your response in clear, numbered steps (1., 2., 3., etc.)
- Use bullet points or numbered lists for procedures and instructions
- Break down complex processes into sequential steps
- Make each step clear and actionable
- Use proper formatting with line breaks between steps
- Include specific values, measurements, and technical details from the documentation

Do not include any text before or after the JSON object."""
        
        # Add query analysis guidance if available
        analysis_guidance = ""
        if query_analysis and not query_analysis.get("is_single_question", True):
            individual_questions = query_analysis.get("individual_questions", [])
            analysis_guidance = f"""

IMPORTANT: This query has been analyzed as containing multiple questions: {individual_questions}
Please ensure your response addresses all aspects of the original query comprehensively."""
        
        user_prompt = f"""Based on the following documentation, answer the user's question and return a JSON response.

Documentation Context:
{context}

Available Section Headings: {chunk_headings}

User Question: {query}{analysis_guidance}

INFORMATION ASSESSMENT:
Before responding, evaluate:
1. Do you have complete information to answer this question?
2. What specific details are available in the provided chunks?
3. What additional information might be needed for a complete answer?
4. Can you provide actionable steps based on available information?

MANDATORY RESPONSE RULES - FOLLOW EXACTLY:
- READ the provided documentation chunks carefully
- EXTRACT the actual information from the chunks
- PROVIDE the complete details from the chunks, not references to them
- NEVER mention "chapter 4.1", "section X", or any chapter/section numbers
- NEVER say "as described in", "refer to", "see chapter", or "check section"
- NEVER give generic responses like "as described in chapter 4.1 Preparing for operational readiness"
- ALWAYS provide the actual steps, procedures, and details from the documentation
- If the information is in the chunks, provide it completely
- If the information is not in the chunks, say "This information is not available in the provided documentation"

IMPORTANT FORMATTING INSTRUCTIONS:
- Structure your response as clear, numbered steps (1., 2., 3., etc.)
- Break down procedures into sequential, actionable steps
- Use bullet points for lists and sub-items
- Make each step specific and easy to follow
- Include proper line breaks between steps for readability
- Include specific values, measurements, and technical details from the documentation

EXAMPLE OF CORRECT RESPONSE:
❌ WRONG: "1. Ensure the machine is ready for operation as described in chapter 4.1"
✅ CORRECT: "1. Check all safety systems are operational. 2. Verify power supply connections. 3. Inspect machine components for damage."

Provide a comprehensive answer based ONLY on the documentation provided above. 

CRITICAL: In the "chunks_used" array, list the EXACT section headings from the available headings that you referenced in your answer. This is essential for collecting the correct images and tables from those chunks.

Available Section Headings: {chunk_headings}

Return ONLY the JSON object, no additional text."""
        
        try:
            # Get query model configuration
            model_config = self._get_model_config("query")
            
            # Use semaphore to limit concurrent requests
            async with self._api_semaphore:
                response = self.client.complete(
                    messages=[
                        SystemMessage(content=system_prompt),
                        UserMessage(content=user_prompt)
                    ],
                    max_tokens=self.max_completion_tokens,
                    temperature=0.1,
                    top_p=0.1,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    model=model_config["name"]
                )
            
            raw_response = response.choices[0].message.content.strip()
            logger.info(f"Raw LLM response using {model_config['name']}: {raw_response[:200]}...")
            
            # Try to parse JSON response
            try:
                # Clean the response to extract JSON
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = raw_response[json_start:json_end]
                    parsed_response = json.loads(json_str)
                    
                    # Validate required fields
                    if "response" in parsed_response and "chunks_used" in parsed_response:
                        # Post-process to remove any remaining generic references
                        response_text = parsed_response["response"]
                        
                        # Ensure re module is available
                        import re
                        
                        # Check for and fix generic references
                        generic_patterns = [
                            r"as described in chapter [\d\.]+[^.]*",
                            r"refer to chapter [\d\.]+[^.]*",
                            r"see chapter [\d\.]+[^.]*",
                            r"check chapter [\d\.]+[^.]*",
                            r"as described in section [\d\.]+[^.]*",
                            r"refer to section [\d\.]+[^.]*",
                            r"see section [\d\.]+[^.]*",
                            r"check section [\d\.]+[^.]*"
                        ]
                        
                        for pattern in generic_patterns:
                            if re.search(pattern, response_text, re.IGNORECASE):
                                logger.warning(f"Found generic reference in response: {pattern}")
                                # Replace with a more appropriate message
                                response_text = re.sub(pattern, "Please refer to the specific procedures in the documentation", response_text, flags=re.IGNORECASE)
                        
                        parsed_response["response"] = response_text
                        
                        logger.info(f"Successfully parsed JSON response with {len(parsed_response.get('chunks_used', []))} referenced chunks")
                        return parsed_response
                    else:
                        logger.warning("JSON response missing required fields, using fallback")
                        raise ValueError("Missing required fields")
                else:
                    logger.warning("No valid JSON found in response, using fallback")
                    raise ValueError("No JSON found")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON response: {str(e)}, using fallback extraction")
                # Fallback: extract response and chunks using old method
                chunks_used = self._extract_referenced_sections(raw_response, chunks)
                return {
                    "response": raw_response,
                    "chunks_used": chunks_used
                }
            
        except Exception as e:
            logger.error(f"Error in LLM query: {str(e)}")
            raise e
    
    async def _get_analysis_response(self, prompt: str) -> str:
        """Get analysis response from LLM for query analysis"""
        try:
            system_prompt = """You are an expert at analyzing user queries and breaking them down into individual questions. 

CRITICAL: You must respond with ONLY a valid JSON object with this exact structure:
{
  "is_single_question": true/false,
  "question_count": number,
  "individual_questions": ["question1", "question2"],
  "complexity": "simple/moderate/complex",
  "reasoning": "brief explanation"
}

Do not include any text before or after the JSON object."""
            
            # Get analysis model configuration
            model_config = self._get_model_config("analysis")
            
            # Use semaphore to limit concurrent requests
            async with self._api_semaphore:
                response = self.client.complete(
                    messages=[
                        SystemMessage(content=system_prompt),
                        UserMessage(content=prompt)
                    ],
                    max_tokens=1000,
                    temperature=0.1,
                    top_p=0.1,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    model=model_config["name"]
                )
            
            raw_response = response.choices[0].message.content.strip()
            logger.info(f"Raw analysis response using {model_config['name']}: {raw_response[:200]}...")
            
            # Try to parse JSON response
            try:
                # Clean the response to extract JSON
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = raw_response[json_start:json_end]
                    parsed_response = json.loads(json_str)
                    
                    # Validate required fields
                    required_fields = ["is_single_question", "question_count", "individual_questions", "complexity", "reasoning"]
                    if all(field in parsed_response for field in required_fields):
                        logger.info("Successfully parsed analysis JSON response")
                        return json_str
                    else:
                        logger.warning("Analysis JSON response missing required fields, using fallback")
                        raise ValueError("Missing required fields")
                else:
                    logger.warning("No valid JSON found in analysis response, using fallback")
                    raise ValueError("No JSON found")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse analysis JSON response: {str(e)}, using fallback")
                # Return fallback JSON
                return '{"is_single_question": true, "question_count": 1, "individual_questions": ["Original query"], "complexity": "simple", "reasoning": "Error in analysis"}'
            
        except Exception as e:
            logger.error(f"Error getting analysis response: {str(e)}")
            # Return fallback JSON
            return '{"is_single_question": true, "question_count": 1, "individual_questions": ["Original query"], "complexity": "simple", "reasoning": "Error in analysis"}'
    
    def _extract_referenced_sections(self, answer: str, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract referenced sections from LLM response"""
        referenced_sections = []
        
        try:
            logger.info(f"Extracting references from answer: {answer[:200]}...")
            
            # Look for REFERENCES section in the answer
            if "REFERENCES:" in answer:
                # Extract the REFERENCES section
                ref_start = answer.find("REFERENCES:")
                ref_section = answer[ref_start:]
                logger.info(f"Found REFERENCES section: {ref_section}")
                
                # Extract section headings from the references
                lines = ref_section.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('-') or line.startswith('•'):
                        # Remove bullet points and extract heading
                        heading = line[1:].strip()
                        if heading:
                            referenced_sections.append(heading)
                            logger.info(f"Extracted bullet reference: {heading}")
                    elif ':' in line and not line.startswith('REFERENCES:'):
                        # Handle format like "Section 1: Introduction"
                        heading = line.strip()
                        if heading:
                            referenced_sections.append(heading)
                            logger.info(f"Extracted colon reference: {heading}")
            
            # If no explicit REFERENCES section, try to match headings in the answer
            if not referenced_sections:
                logger.warning("No explicit REFERENCES section found, trying text matching")
                for chunk in chunks:
                    try:
                        # Handle both possible chunk structures
                        if "metadata" in chunk and "document" in chunk:
                            # Vector DB format
                            heading = chunk.get("metadata", {}).get("heading", "")
                        else:
                            # Fallback format
                            heading = chunk.get("heading", "")
                        
                        if heading and heading.lower() in answer.lower():
                            referenced_sections.append(heading)
                            logger.info(f"Matched heading in text: {heading}")
                    except Exception as e:
                        logger.warning(f"Error extracting heading from chunk: {str(e)}")
                        continue
            
            # If still no references found, use all chunks (fallback)
            if not referenced_sections:
                logger.warning("No references found, using all chunks as fallback")
                for chunk in chunks:
                    try:
                        if "metadata" in chunk and "document" in chunk:
                            heading = chunk.get("metadata", {}).get("heading", "")
                        else:
                            heading = chunk.get("heading", "")
                        
                        if heading:
                            referenced_sections.append(heading)
                    except Exception as e:
                        logger.warning(f"Error getting heading from chunk: {str(e)}")
                        continue
            
            # If still no references, use chunk indices as fallback
            if not referenced_sections:
                logger.warning("No headings found, using chunk indices as fallback")
                for i, chunk in enumerate(chunks):
                    referenced_sections.append(f"Chunk {i+1}")
            
            logger.info(f"Final extracted referenced sections: {referenced_sections}")
            return referenced_sections
            
        except Exception as e:
            logger.error(f"Error extracting referenced sections: {str(e)}")
            # Fallback: return chunk indices
            fallback_sections = []
            for i, chunk in enumerate(chunks):
                fallback_sections.append(f"Chunk {i+1}")
            
            logger.info(f"Using fallback sections: {fallback_sections}")
            return fallback_sections

    async def generate_rules(self, chunks: List[Dict[str, Any]]) -> str:
        """Generate IoT monitoring rules from chunks"""
        logger.info(f"Generating rules from {len(chunks)} chunks")
        
        if not chunks:
            return "No content available to generate rules from."
        
        # Prepare context from chunks (include full data with tables)
        context_parts = []
        for chunk in chunks:
            try:
                if "metadata" in chunk and "document" in chunk:
                    heading = chunk.get("metadata", {}).get("heading", "")
                    content = chunk.get("document", "")
                    # Extract tables from metadata if available
                    tables = chunk.get("metadata", {}).get("tables", [])
                else:
                    heading = chunk.get("heading", "")
                    content = chunk.get("text", chunk.get("content", ""))
                    tables = chunk.get("tables", [])
                
                if content:
                    # Include full content including tables, specifications, and all data
                    context_text = f"**{heading}**\n{content}"
                    
                    # Add tables to context if they exist
                    if tables:
                        context_text += f"\n\n**Tables in this section:**\n"
                        for i, table in enumerate(tables, 1):
                            context_text += f"\nTable {i}:\n{table}\n"
                    
                    context_parts.append(context_text)
            except Exception as e:
                logger.warning(f"Error processing chunk for rules: {str(e)}")
                continue
        
        if not context_parts:
            return "No valid content could be extracted to generate rules."
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are an expert in IoT monitoring and industrial automation. Your task is to analyze technical documentation and generate comprehensive IoT monitoring rules.

IMPORTANT: The provided documentation includes complete technical data, tables, specifications, and detailed information. Use all available data to create precise monitoring rules.

CRITICAL: You must respond with ONLY a valid JSON object with this exact structure:
{
  "rules": [
    {
      "name": "Rule Name",
      "description": "Detailed description of the rule",
      "metric": "sensor_metric_name",
      "metric_value": "numerical_value_with_unit",
      "threshold": "numerical_threshold_condition",
      "consequence": "What happens when threshold is exceeded",
      "condition": "IF condition statement",
      "action": "Actions to take (SEND_ALERT, LOG_EVENT, etc.)",
      "priority": "HIGH/MEDIUM/LOW"
    }
  ]
}

CRITICAL IoT MONITORING REQUIREMENTS:
- metric must be a SENSOR-MEASURABLE parameter (temperature, pressure, voltage, current, speed, vibration, etc.)
- metric_value must be a specific numerical value with unit (e.g., "75°C", "1500 N", "85%", "220 V", "50 Hz")
- threshold must be numerical conditions (e.g., "> 75°C", "< 20%", "> 1500 N", "= 220 V")
- DO NOT use descriptive states like "hinged down", "pushed in", "equal distance" - these are not IoT metrics
- Focus on measurable physical parameters that sensors can detect
- Use actual numerical values from the documentation specifications

VALID IoT METRICS EXAMPLES:
- temperature, pressure, voltage, current, speed, vibration, frequency, power, flow_rate, level, humidity, etc.

INVALID METRICS (DO NOT USE):
- bolt_status, alignment_status, installation_status, position_status, etc.

Do not include any text before or after the JSON object."""
        
        user_prompt = f"""Based on the following technical documentation, generate comprehensive IoT monitoring rules:

Documentation:
{context}

Generate a comprehensive set of IoT monitoring rules that cover:
1. Equipment monitoring parameters (temperature, pressure, voltage, current, speed, vibration)
2. Safety thresholds and alerts (temperature limits, pressure limits, electrical limits)
3. Performance metrics (power consumption, efficiency, output levels)
4. Maintenance indicators (vibration levels, wear indicators, performance degradation)
5. Operational conditions (operating ranges, normal vs abnormal conditions)

CRITICAL: The documentation includes tables with specific IoT monitoring data. Use the table information extensively for:
- Exact threshold values from specification tables (temperature, pressure, voltage, current limits)
- Precise monitoring parameters from technical data tables
- Specific sensor ranges and operating conditions from equipment tables
- Performance metrics and efficiency data from measurement tables
- Safety limits and alert thresholds from safety specification tables

Pay special attention to any tables containing technical specifications, operating parameters, or monitoring data as these contain the most relevant IoT monitoring information.

CRITICAL IoT MONITORING REQUIREMENTS:
- ONLY create rules for SENSOR-MEASURABLE parameters that IoT devices can monitor
- Extract actual numerical values from tables, specifications, and technical data
- Use specific numbers with units (e.g., "75°C", "1500 N", "85%", "220 V", "50 Hz")
- Create precise threshold conditions with real numbers from the documentation
- Focus on physical parameters that can be measured by sensors
- DO NOT create rules for mechanical states, positions, or installation status

VALID SENSOR METRICS TO USE:
- Temperature, pressure, voltage, current, speed, vibration, frequency, power, flow rate, level, humidity, torque, force, etc.

Return ONLY the JSON object with the rules array."""
        
        try:
            # Get rules model configuration
            model_config = self._get_model_config("rules")
            
            # Use semaphore to limit concurrent requests
            async with self._api_semaphore:
                response = self.client.complete(
                    messages=[
                        SystemMessage(content=system_prompt),
                        UserMessage(content=user_prompt)
                    ],
                    max_tokens=self.max_completion_tokens,
                    temperature=0.2,
                    top_p=0.1,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    model=model_config["name"]
                )
            
            raw_response = response.choices[0].message.content.strip()
            logger.info(f"Raw LLM response for rules using {model_config['name']}: {raw_response[:200]}...")
            
            # Parse JSON response
            try:
                # Ensure re module is available
                import re
                
                # Remove markdown code blocks if present
                cleaned_response = raw_response
                if "```json" in cleaned_response:
                    cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
                if "```" in cleaned_response:
                    cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
                
                parsed_response = json.loads(cleaned_response.strip())
                return parsed_response.get("rules", [])
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse rules JSON: {str(e)}")
                logger.error(f"Raw response: {raw_response}")
                return []
            
        except Exception as e:
            logger.error(f"Error generating rules: {str(e)}")
            return []

    async def generate_safety_information(self, chunks: List[Dict[str, Any]]) -> str:
        """Generate safety information from chunks"""
        logger.info(f"Generating safety information from {len(chunks)} chunks")
        
        if not chunks:
            return "No content available to generate safety information from."
        
        # Prepare context from chunks (include full data with tables)
        context_parts = []
        for chunk in chunks:
            try:
                if "metadata" in chunk and "document" in chunk:
                    heading = chunk.get("metadata", {}).get("heading", "")
                    content = chunk.get("document", "")
                    # Extract tables from metadata if available
                    tables = chunk.get("metadata", {}).get("tables", [])
                else:
                    heading = chunk.get("heading", "")
                    content = chunk.get("text", chunk.get("content", ""))
                    tables = chunk.get("tables", [])
                
                if content:
                    # Include full content including tables, specifications, and all data
                    context_text = f"**{heading}**\n{content}"
                    
                    # Add tables to context if they exist
                    if tables:
                        context_text += f"\n\n**Tables in this section:**\n"
                        for i, table in enumerate(tables, 1):
                            context_text += f"\nTable {i}:\n{table}\n"
                    
                    context_parts.append(context_text)
            except Exception as e:
                logger.warning(f"Error processing chunk for safety: {str(e)}")
                continue
        
        if not context_parts:
            return "No valid content could be extracted to generate safety information."
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are a safety expert specializing in industrial equipment and machinery safety. Your task is to analyze technical documentation and generate comprehensive safety information.

IMPORTANT: The provided documentation includes complete technical data, tables, specifications, and detailed information. Use all available data to create comprehensive safety guidelines.

CRITICAL: You must respond with ONLY a valid JSON object with this exact structure:
{
  "safety_precautions": [
    {
      "title": "Safety Title",
      "description": "Description of the safety concern",
      "category": "category_type",
      "severity": "HIGH/MEDIUM/LOW",
      "mitigation": "How to mitigate the risk",
      "about_reaction": "What happens if not addressed",
      "causes": "What causes this safety issue",
      "how_to_avoid": "How to avoid the issue",
      "safety_info": "Additional safety information",
      "type": "warning/caution/procedure",
      "recommended_action": "Recommended action to take"
    }
  ],
  "safety_information": [
    {
      "title": "Information Title",
      "description": "Description of safety information",
      "category": "category_type",
      "severity": "HIGH/MEDIUM/LOW",
      "mitigation": "How to mitigate the risk",
      "about_reaction": "What happens if not addressed",
      "causes": "What causes this safety issue",
      "how_to_avoid": "How to avoid the issue",
      "safety_info": "Additional safety information",
      "type": "warning/caution/procedure",
      "recommended_action": "Recommended action to take"
    }
  ]
}

Do not include any text before or after the JSON object."""
        
        user_prompt = f"""Based on the following technical documentation, generate comprehensive safety information:

Documentation:
{context}

Generate detailed safety information covering:
1. Safety procedures and protocols
2. Hazard identification and mitigation
3. Personal protective equipment (PPE) requirements
4. Emergency procedures
5. Safety warnings and precautions
6. Risk assessments
7. Error codes and their meanings
8. Alert conditions and responses

CRITICAL: The documentation includes tables with specific safety data. Use the table information extensively for:
- Exact error codes and their descriptions from error code tables
- Precise safety limits and thresholds (temperature, pressure, voltage, current)
- Specific precautionary measures from safety tables
- Alert conditions and their severity levels from alert tables
- PPE requirements and specifications from safety tables
- Emergency response procedures and contact information

Pay special attention to any tables labeled as "error codes", "precaution", "alert", or "safety" as these contain the most relevant safety data.

IMPORTANT: Use specific numerical values from the documentation for:
- Temperature limits (e.g., "60°C", "100°F")
- Pressure limits (e.g., "150 psi", "10 bar")
- Voltage/current limits (e.g., "220 V", "5 A")
- Distance requirements (e.g., "2 meters", "6 feet")
- Time limits and durations from safety procedures

Return ONLY the JSON object with safety_precautions and safety_information arrays."""
        
        try:
            # Get safety model configuration
            model_config = self._get_model_config("safety")
            
            # Use semaphore to limit concurrent requests
            async with self._api_semaphore:
                response = self.client.complete(
                    messages=[
                        SystemMessage(content=system_prompt),
                        UserMessage(content=user_prompt)
                    ],
                    max_tokens=self.max_completion_tokens,
                    temperature=0.2,
                    top_p=0.1,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    model=model_config["name"]
                )
            
            raw_response = response.choices[0].message.content.strip()
            logger.info(f"Raw LLM response for safety using {model_config['name']}: {raw_response[:200]}...")
            
            # Parse JSON response
            try:
                # Ensure re module is available
                import re
                
                # Remove markdown code blocks if present
                cleaned_response = raw_response
                if "```json" in cleaned_response:
                    cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
                if "```" in cleaned_response:
                    cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
                
                parsed_response = json.loads(cleaned_response.strip())
                return {
                    "safety_precautions": parsed_response.get("safety_precautions", []),
                    "safety_information": parsed_response.get("safety_information", [])
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse safety JSON: {str(e)}")
                logger.error(f"Raw response: {raw_response}")
                return {"safety_precautions": [], "safety_information": []}
            
        except Exception as e:
            logger.error(f"Error generating safety information: {str(e)}")
            return {"safety_precautions": [], "safety_information": []}

    async def dynamic_query_processing(self, vector_db, collection_name: str, query: str, query_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Dynamic information gathering flow with multi-stage processing"""
        logger.info(f"Starting dynamic query processing for: {query}")
        logger.info(f"Using collection: {collection_name}")
        
        try:
            # Stage 1: Initial query to get base chunks
            logger.info(f"Stage 1: Getting initial chunks from collection: {collection_name}")
            initial_chunks = await vector_db.query_chunks(
                collection_name=collection_name,
                query=query,
                top_k=5
            )
            
            if not initial_chunks:
                logger.warning("No initial chunks found")
                return {
                    "response": f"I don't have access to specific documentation about '{query}'. Please ensure the document is properly uploaded and processed.",
                    "chunks_used": [],
                    "processing_stages": ["initial_query"],
                    "confidence_score": 0.0
                }
            
            # Stage 2: Assess information sufficiency
            logger.info("Stage 2: Assessing information sufficiency")
            assessment = await self.assess_information_sufficiency(initial_chunks, query)
            
            all_chunks = initial_chunks.copy()
            processing_stages = ["initial_query", "assessment"]
            
            # Stage 3: Get additional information if needed
            if not assessment.get("has_sufficient_info", False):
                logger.info("Stage 3: Gathering additional information")
                logger.info(f"Using same collection for additional queries: {collection_name}")
                additional_queries = assessment.get("additional_queries_needed", [])
                
                for additional_query in additional_queries[:3]:  # Limit to 3 additional queries
                    try:
                        logger.info(f"Querying for additional info: {additional_query} in collection: {collection_name}")
                        additional_chunks = await vector_db.query_chunks(
                            collection_name=collection_name,
                            query=additional_query,
                            top_k=3
                        )
                        
                        # Add new chunks (avoid duplicates)
                        for chunk in additional_chunks:
                            chunk_id = chunk.get("metadata", {}).get("chunk_index", "")
                            if not any(existing.get("metadata", {}).get("chunk_index", "") == chunk_id for existing in all_chunks):
                                all_chunks.append(chunk)
                        
                        logger.info(f"Added {len(additional_chunks)} additional chunks for: {additional_query}")
                        
                    except Exception as e:
                        logger.warning(f"Error getting additional chunks for '{additional_query}': {str(e)}")
                        continue
                
                processing_stages.append("additional_queries")
            
            # Stage 4: Generate comprehensive response
            logger.info("Stage 4: Generating comprehensive response")
            final_response = await self.query_with_context(all_chunks, query, query_analysis)
            
            # Stage 4.5: Check if response indicates need for more information
            logger.info("Stage 4.5: Checking response completeness")
            response_text = final_response.get("response", "")
            
            # Simple check for incomplete responses that might need more information
            needs_more_info = any(phrase in response_text.lower() for phrase in [
                "need more specific information",
                "additional information is needed", 
                "more details are required",
                "insufficient information"
            ])
            
            if needs_more_info:
                logger.info("Response indicates need for more information, attempting additional queries")
                
                # Try to get more relevant chunks based on the original query
                additional_queries = [
                    f"{query} detailed steps procedures",
                    f"{query} complete instructions",
                    f"{query} specific requirements"
                ]
                
                for additional_query in additional_queries[:2]:  # Limit to 2 additional queries
                    try:
                        logger.info(f"Querying for additional info: {additional_query}")
                        additional_chunks = await vector_db.query_chunks(
                            collection_name=collection_name,
                            query=additional_query,
                            top_k=3
                        )
                        
                        # Add new chunks (avoid duplicates)
                        for chunk in additional_chunks:
                            chunk_id = chunk.get("metadata", {}).get("chunk_index", "")
                            if not any(existing.get("metadata", {}).get("chunk_index", "") == chunk_id for existing in all_chunks):
                                all_chunks.append(chunk)
                        
                        logger.info(f"Added {len(additional_chunks)} additional chunks for: {additional_query}")
                        
                    except Exception as e:
                        logger.warning(f"Error getting additional chunks for '{additional_query}': {str(e)}")
                        continue
                
                # Regenerate response with additional chunks
                logger.info("Regenerating response with additional information")
                final_response = await self.query_with_context(all_chunks, query, query_analysis)
                processing_stages.append("additional_information_gathering")
            
            # Stage 5: Evaluate response quality
            logger.info("Stage 5: Evaluating response quality")
            evaluation = await self.evaluate_response_quality(final_response, query, assessment)
            
            # Combine all information
            result = {
                **final_response,
                "processing_stages": processing_stages,
                "initial_chunks_count": len(initial_chunks),
                "total_chunks_count": len(all_chunks),
                "assessment": assessment,
                "evaluation": evaluation,
                "confidence_score": evaluation.get("confidence_score", 0.0),
                "collection_used": collection_name  # Track which collection was used throughout
            }
            
            logger.info(f"Dynamic processing completed: {len(processing_stages)} stages, {len(all_chunks)} total chunks, confidence: {result['confidence_score']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in dynamic query processing: {str(e)}")
            return {
                "response": f"Error processing query: {str(e)}",
                "chunks_used": [],
                "processing_stages": ["error"],
                "confidence_score": 0.0,
                "error": str(e)
            }

    async def evaluate_response_quality(self, response: Dict[str, Any], query: str, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the quality and completeness of the response"""
        logger.info("Evaluating response quality")
        
        try:
            response_text = response.get("response", "")
            chunks_used = response.get("chunks_used", [])
            
            # Quality metrics
            quality_metrics = {
                "response_length": len(response_text),
                "has_content": len(response_text.strip()) > 50,
                "uses_chunks": len(chunks_used) > 0,
                "addresses_query": any(word.lower() in response_text.lower() for word in query.split()),
                "has_specific_details": any(char.isdigit() for char in response_text),  # Has numbers/measurements
                "has_procedural_steps": "1." in response_text or "step" in response_text.lower(),
                "no_generic_references": not any(phrase in response_text.lower() for phrase in [
                    "refer to", "see section", "check the", "please refer", "see documentation"
                ])
            }
            
            # Calculate confidence score
            confidence_score = 0.0
            if quality_metrics["has_content"]:
                confidence_score += 0.2
            if quality_metrics["uses_chunks"]:
                confidence_score += 0.2
            if quality_metrics["addresses_query"]:
                confidence_score += 0.2
            if quality_metrics["has_specific_details"]:
                confidence_score += 0.15
            if quality_metrics["has_procedural_steps"]:
                confidence_score += 0.15
            if quality_metrics["no_generic_references"]:
                confidence_score += 0.1
            
            # Bonus for addressing assessment concerns
            if assessment.get("has_sufficient_info", False):
                confidence_score += 0.1
            
            # Cap at 1.0
            confidence_score = min(confidence_score, 1.0)
            
            evaluation = {
                "confidence_score": confidence_score,
                "quality_metrics": quality_metrics,
                "is_comprehensive": confidence_score >= 0.7,
                "needs_improvement": confidence_score < 0.5,
                "assessment_met": assessment.get("has_sufficient_info", False)
            }
            
            logger.info(f"Response evaluation: confidence={confidence_score:.2f}, comprehensive={evaluation['is_comprehensive']}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating response quality: {str(e)}")
            return {
                "confidence_score": 0.0,
                "quality_metrics": {},
                "is_comprehensive": False,
                "needs_improvement": True,
                "assessment_met": False,
                "error": str(e)
            }

    async def generate_maintenance_schedule(self, chunks: List[Dict[str, Any]]) -> str:
        """Generate maintenance schedule from chunks"""
        logger.info(f"Generating maintenance schedule from {len(chunks)} chunks")
        
        if not chunks:
            return "No content available to generate maintenance schedule from."
        
        # Prepare context from chunks (always include tables for maintenance)
        context_parts = []
        for chunk in chunks:
            try:
                if "metadata" in chunk and "document" in chunk:
                    heading = chunk.get("metadata", {}).get("heading", "")
                    content = chunk.get("document", "")
                    # Extract tables from metadata if available
                    tables = chunk.get("metadata", {}).get("tables", [])
                else:
                    heading = chunk.get("heading", "")
                    content = chunk.get("text", chunk.get("content", ""))
                    tables = chunk.get("tables", [])
                
                if content:
                    # For maintenance, always include all content including tables
                    context_text = f"**{heading}**\n{content}"
                    
                    # Add tables to context if they exist
                    if tables:
                        context_text += f"\n\n**Tables in this section:**\n"
                        for i, table in enumerate(tables, 1):
                            context_text += f"\nTable {i}:\n{table}\n"
                    
                    context_parts.append(context_text)
            except Exception as e:
                logger.warning(f"Error processing chunk for maintenance: {str(e)}")
                continue
        
        if not context_parts:
            return "No valid content could be extracted to generate maintenance schedule."
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are a maintenance expert specializing in industrial equipment and machinery maintenance. Your task is to analyze technical documentation and generate comprehensive maintenance schedules.

IMPORTANT: The provided documentation includes tables, specifications, and detailed technical data. Use this information to create precise maintenance schedules.

CRITICAL: You must respond with ONLY a valid JSON object with this exact structure:
{
  "maintenance_tasks": [
    {
      "task": "Task description",
      "task_name": "Specific task name",
      "description": "Detailed description of the task",
      "frequency": "frequency_interval",
      "priority": "HIGH/MEDIUM/LOW",
      "estimated_duration": "duration_in_time",
      "required_tools": "List of required tools",
      "category": "PREVENTIVE/CORRECTIVE/PREDICTIVE",
      "safety_notes": "Safety considerations and notes"
    }
  ]
}

Do not include any text before or after the JSON object."""
        
        user_prompt = f"""Based on the following technical documentation, generate a comprehensive maintenance schedule:

Documentation:
{context}

Generate a detailed maintenance schedule covering:
1. Preventive maintenance tasks
2. Maintenance intervals and frequencies
3. Required tools and materials
4. Step-by-step maintenance procedures
5. Inspection checklists
6. Maintenance priorities

CRITICAL: The documentation includes tables with specific maintenance data. Use the table information extensively for:
- Exact frequencies from maintenance schedules (e.g., "every 30 days", "monthly", "every 1000 hours")
- Precise durations from maintenance procedures (e.g., "2 hours", "30 minutes", "4 hours")
- Specific quantities, measurements, and specifications from tables
- Tool requirements and material lists from maintenance tables
- Priority levels and categories from maintenance schedules

Pay special attention to any tables labeled as "maintenance list", "maintenance tasks", "maintenance schedules", or "maintenance procedure" as these contain the most relevant data.

Return ONLY the JSON object with the maintenance_tasks array."""
        
        try:
            # Get maintenance model configuration
            model_config = self._get_model_config("maintenance")
            
            # Use semaphore to limit concurrent requests
            async with self._api_semaphore:
                response = self.client.complete(
                    messages=[
                        SystemMessage(content=system_prompt),
                        UserMessage(content=user_prompt)
                    ],
                    max_tokens=self.max_completion_tokens,
                    temperature=0.2,
                    top_p=0.1,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    model=model_config["name"]
                )
            
            raw_response = response.choices[0].message.content.strip()
            logger.info(f"Raw LLM response for maintenance using {model_config['name']}: {raw_response[:200]}...")
            
            # Parse JSON response
            try:
                # Ensure re module is available
                import re
                
                # Remove markdown code blocks if present
                cleaned_response = raw_response
                if "```json" in cleaned_response:
                    cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
                if "```" in cleaned_response:
                    cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
                
                parsed_response = json.loads(cleaned_response.strip())
                return parsed_response.get("maintenance_tasks", [])
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse maintenance JSON: {str(e)}")
                logger.error(f"Raw response: {raw_response}")
                return []
            
        except Exception as e:
            logger.error(f"Error generating maintenance schedule: {str(e)}")
            return []