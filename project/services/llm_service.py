"""
LLM Service
Handles interactions with Azure AI for intelligent responses

Version: 0.1
"""

import json
import logging
import tiktoken
from typing import List, Dict, Any, Optional
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential
from config import settings
from models.schemas import Rule, MaintenanceTask, SafetyInfo

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # Azure AI Configuration
        self.endpoint = "https://chgai.services.ai.azure.com/models"
        self.model_name = "Llama-4-Maverick-17B-128E-Instruct-FP8"
        self.api_version = "2024-05-01-preview"
        
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
        except Exception as e:
            logger.error(f"Failed to initialize Azure AI client: {str(e)}")
            raise e
        
        # Increase token limits for better handling of large documents
        self.max_tokens = 8192
        self.max_completion_tokens = 1500  # Reduced to allow more context
        self.max_context_tokens = self.max_tokens - self.max_completion_tokens
        
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")  # Keep using GPT-4 encoding for token counting
        except Exception as e:
            logger.warning(f"Failed to load GPT-4 encoding, using default: {str(e)}")
            # Fallback to a basic encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"LLM Service initialized with Azure AI - max_tokens: {self.max_tokens}, max_context_tokens: {self.max_context_tokens}")
        logger.info(f"Using Azure AI endpoint: {self.endpoint}")
        logger.info(f"Using model: {self.model_name}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def truncate_chunks_to_fit_context(self, chunks: List[Dict[str, Any]], system_prompt: str = "", user_prompt_template: str = "") -> List[Dict[str, Any]]:
        """Truncate chunks to fit within context limit"""
        # Create a simple version of the template without placeholders for token counting
        simple_template = user_prompt_template.replace("{context}", "CONTEXT_PLACEHOLDER").replace("{query}", "QUERY_PLACEHOLDER")
        
        # Calculate available tokens for chunks
        system_tokens = self.count_tokens(system_prompt)
        user_template_tokens = self.count_tokens(simple_template)
        reserved_tokens = system_tokens + user_template_tokens + 200  # Increased buffer
        
        available_tokens = self.max_context_tokens - reserved_tokens
        
        logger.info(f"Available tokens for chunks: {available_tokens}")
        logger.info(f"System prompt tokens: {system_tokens}")
        logger.info(f"User template tokens: {user_template_tokens}")
        logger.info(f"Total chunks to process: {len(chunks)}")
        
        # Log each chunk's token count
        for i, chunk in enumerate(chunks):
            try:
                if "metadata" in chunk and "document" in chunk:
                    heading = chunk.get("metadata", {}).get("heading", "")
                    content = chunk.get("document", "")
                else:
                    heading = chunk.get("heading", "")
                    content = chunk.get("text", chunk.get("content", ""))
                
                chunk_text = f"**{heading}**\n{content}"
                chunk_tokens = self.count_tokens(chunk_text)
                logger.info(f"Chunk {i+1}: '{heading}' ({chunk_tokens} tokens, {len(content)} chars)")
            except Exception as e:
                logger.warning(f"Error counting tokens for chunk {i}: {str(e)}")
        
        # If we have very few tokens available, try to use at least one chunk
        if available_tokens < 1000:
            logger.warning(f"Very few tokens available ({available_tokens}), will try to use at least one chunk")
            available_tokens = min(available_tokens, 2000)  # Ensure we have some minimum tokens
        
        selected_chunks = []
        current_tokens = 0
        
        for i, chunk in enumerate(chunks):
            try:
                # Handle both possible chunk structures
                if "metadata" in chunk and "document" in chunk:
                    # Vector DB format
                    heading = chunk.get("metadata", {}).get("heading", "")
                    content = chunk.get("document", "")
                else:
                    # Fallback format
                    heading = chunk.get("heading", "")
                    content = chunk.get("text", chunk.get("content", ""))
                
                chunk_text = f"**{heading}**\n{content}"
                chunk_tokens = self.count_tokens(chunk_text)
                
                logger.info(f"Chunk {i+1}: '{heading}' ({chunk_tokens} tokens)")
                
                # If this is the first chunk and it's too large, truncate it
                if i == 0 and chunk_tokens > available_tokens:
                    logger.warning(f"First chunk is too large ({chunk_tokens} tokens), truncating content")
                    # Truncate the content to fit
                    max_content_tokens = available_tokens - 100  # Leave some space for heading
                    truncated_content = self._truncate_text_to_tokens(content, max_content_tokens)
                    chunk["document"] = truncated_content  # Update the chunk with truncated content
                    chunk_tokens = self.count_tokens(f"**{heading}**\n{truncated_content}")
                    logger.info(f"Truncated chunk to {chunk_tokens} tokens")
                
                if current_tokens + chunk_tokens <= available_tokens:
                    selected_chunks.append(chunk)
                    current_tokens += chunk_tokens
                    logger.info(f"Added chunk '{heading}' ({chunk_tokens} tokens), total: {current_tokens}")
                else:
                    logger.info(f"Skipping chunk '{heading}' ({chunk_tokens} tokens) - would exceed limit")
                    break
            except Exception as e:
                logger.warning(f"Error processing chunk {i} in truncation: {str(e)}")
                continue
        
        logger.info(f"Selected {len(selected_chunks)} chunks with {current_tokens} tokens")
        
        # If no chunks were selected, try to use at least the first chunk with heavy truncation
        if not selected_chunks and chunks:
            logger.warning("No chunks could fit, trying to use first chunk with heavy truncation")
            first_chunk = chunks[0]
            if "metadata" in first_chunk and "document" in first_chunk:
                heading = first_chunk.get("metadata", {}).get("heading", "")
                content = first_chunk.get("document", "")
            else:
                heading = first_chunk.get("heading", "")
                content = first_chunk.get("text", first_chunk.get("content", ""))
            
            # Truncate to a very small size
            max_tokens = min(available_tokens - 200, 1000)
            truncated_content = self._truncate_text_to_tokens(content, max_tokens)
            first_chunk["document"] = truncated_content
            selected_chunks = [first_chunk]
            logger.info(f"Using heavily truncated first chunk: {len(truncated_content)} characters")
        
        return selected_chunks
    
    def _truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        try:
            # Encode the text to get tokens
            tokens = self.encoding.encode(text)
            
            # If text is already within limit, return as is
            if len(tokens) <= max_tokens:
                return text
            
            # Truncate tokens and decode back to text
            truncated_tokens = tokens[:max_tokens]
            truncated_text = self.encoding.decode(truncated_tokens)
            
            # Try to end at a sentence boundary
            last_period = truncated_text.rfind('.')
            last_exclamation = truncated_text.rfind('!')
            last_question = truncated_text.rfind('?')
            
            end_pos = max(last_period, last_exclamation, last_question)
            if end_pos > len(truncated_text) * 0.8:  # Only if we're not cutting too much
                truncated_text = truncated_text[:end_pos + 1]
            
            return truncated_text + " [Content truncated due to length]"
            
        except Exception as e:
            logger.error(f"Error truncating text: {str(e)}")
            # Fallback: simple character-based truncation
            return text[:max_tokens * 4] + " [Content truncated]"
    
    async def query_with_context(self, chunks: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Query with context chunks using Azure AI"""
        logger.info(f"Processing query with {len(chunks)} context chunks")
        
        # Debug: Log chunk structure
        if chunks:
            logger.info(f"First chunk keys: {list(chunks[0].keys())}")
            logger.info(f"First chunk metadata keys: {list(chunks[0].get('metadata', {}).keys())}")
            logger.info(f"First chunk document preview: {chunks[0].get('document', '')[:200]}...")
            logger.info(f"First chunk heading: {chunks[0].get('metadata', {}).get('heading', 'No heading')}")
            
            # Log all chunks for debugging
            for i, chunk in enumerate(chunks):
                heading = chunk.get('metadata', {}).get('heading', 'No heading')
                doc_length = len(chunk.get('document', ''))
                logger.info(f"Chunk {i}: heading='{heading}', document_length={doc_length}")
        else:
            logger.warning("No chunks provided for context")
        
        # System prompt
        system_prompt = "You are a technical documentation assistant. Provide accurate, detailed answers based on the provided manual sections."
        
        # User prompt template (simplified to use fewer tokens)
        user_prompt_template = """Answer the user's query based on the provided context.

Context:
{context}

Query: {query}

Provide a clear answer. At the end, add "REFERENCES:" followed by the exact section headings you used."""
        
        # If no chunks available, provide a helpful response
        if not chunks:
            logger.warning("No chunks available, providing general response")
            system_prompt = "You are a helpful technical assistant. Provide a general response when specific documentation is not available."
            user_prompt = f"""The user asked: {query}

Unfortunately, I don't have access to specific documentation content at the moment. Please provide a helpful general response about this topic.

Provide a clear answer. At the end, add "REFERENCES: General knowledge"."""
        else:
            # Truncate chunks to fit within token limit
            selected_chunks = self.truncate_chunks_to_fit_context(chunks, system_prompt, user_prompt_template)
            
            if not selected_chunks:
                logger.warning("No chunks could fit within token limit, using first chunk with heavy truncation")
                if chunks:
                    # Use the first chunk with heavy truncation
                    first_chunk = chunks[0]
                    if "metadata" in first_chunk and "document" in first_chunk:
                        heading = first_chunk.get("metadata", {}).get("heading", "General Information")
                        content = first_chunk.get("document", "")[:1000] + " [Content truncated]"
                    else:
                        heading = first_chunk.get("heading", "General Information")
                        content = first_chunk.get("text", first_chunk.get("content", ""))[:1000] + " [Content truncated]"
                    
                    context = f"**{heading}**\n{content}"
                else:
                    context = "No specific content available."
                    logger.warning("No chunks available for context")
            else:
                # Prepare context from selected chunks
                context_parts = []
                for i, chunk in enumerate(selected_chunks):
                    try:
                        # Handle both possible chunk structures
                        if "metadata" in chunk and "document" in chunk:
                            # Vector DB format
                            heading = chunk.get("metadata", {}).get("heading", f"Section {i+1}")
                            content = chunk.get("document", "")
                            logger.info(f"Processing chunk {i}: Vector DB format - heading: '{heading}', content length: {len(content)}")
                        else:
                            # Fallback format
                            heading = chunk.get("heading", f"Section {i+1}")
                            content = chunk.get("text", chunk.get("content", ""))
                            logger.info(f"Processing chunk {i}: Fallback format - heading: '{heading}', content length: {len(content)}")
                        
                        if content:
                            context_parts.append(f"**{heading}**\n{content}")
                            logger.info(f"Added chunk {i} to context: '{heading}' with {len(content)} characters")
                        else:
                            logger.warning(f"Chunk {i} has no content: heading='{heading}'")
                    except Exception as e:
                        logger.warning(f"Error processing chunk {i}: {str(e)}")
                        continue
                
                if not context_parts:
                    logger.warning("No valid context could be extracted from chunks")
                    context = "No specific content available."
                else:
                    context = "\n\n".join(context_parts)
                    logger.info(f"Final context length: {len(context)} characters")
                    logger.info(f"Context preview: {context[:500]}...")
            
            # Format the user prompt with context
            try:
                user_prompt = user_prompt_template.format(context=context, query=query)
                logger.info(f"User prompt length: {len(user_prompt)} characters")
                logger.info(f"User prompt preview: {user_prompt[:500]}...")
            except KeyError as e:
                logger.error(f"Error formatting user prompt - missing placeholder: {str(e)}")
                # Fallback to simple prompt
                user_prompt = f"Please answer this query based on the following context:\n\n{context}\n\nQuery: {query}"
                logger.info(f"Using fallback user prompt, length: {len(user_prompt)} characters")
            except Exception as e:
                logger.error(f"Error formatting user prompt: {str(e)}")
                # Fallback to simple prompt
                user_prompt = f"Please answer this query based on the following context:\n\n{context}\n\nQuery: {query}"
                logger.info(f"Using fallback user prompt, length: {len(user_prompt)} characters")
        
        try:
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
                model=self.model_name
            )
            
            answer = response.choices[0].message.content
            logger.info(f"LLM response length: {len(answer)} characters")
            logger.info(f"LLM response preview: {answer[:500]}...")
            
            # Parse referenced sections from LLM response
            chunks_used = self._extract_referenced_sections(answer, chunks)
            logger.info(f"LLM referenced {len(chunks_used)} sections: {chunks_used}")
            
            return {
                "response": answer,
                "chunks_used": chunks_used
            }
            
        except Exception as e:
            logger.error(f"Error in LLM query: {str(e)}")
            raise e
    
    async def generate_rules(self, chunks: List[Dict[str, Any]]) -> List[Rule]:
        """Generate IoT monitoring rules from chunks"""
        logger.info(f"Generating rules from {len(chunks)} chunks")
        
        # System prompt
        system_prompt = """You are a senior IoT systems engineer with 15+ years of experience in industrial automation and monitoring systems.
You specialize in creating precise, actionable IoT monitoring rules from technical documentation.
Your expertise includes sensor networks, threshold monitoring, predictive analytics, and automated response systems.
You understand the critical importance of accurate thresholds and clear consequences for equipment reliability and safety."""
        
        # User prompt template with detailed requirements
        user_prompt_template = """Analyze the provided technical documentation sections and extract specific IoT monitoring rules.

Context:
{context}

Please generate detailed IoT monitoring rules in JSON format with the following structure:
[
  {{
    "rule_name": "High Temperature Alert",
    "threshold": "Temperature > 85°C",
    "metric": "temperature",
    "metric_value": "85°C",
    "description": "Monitor equipment temperature to prevent overheating and thermal damage",
    "consequence": "Equipment shutdown, thermal damage, reduced lifespan, safety hazard"
  }}
]

IMPORTANT REQUIREMENTS:
1. **Rule Name**: Be specific and descriptive (e.g., "High Temperature Alert" not just "Temperature Rule")
2. **Threshold**: Use exact numerical values with units (e.g., "Temperature > 85°C", "Pressure < 50 PSI")
3. **Metric**: The specific parameter being monitored (temperature, pressure, vibration, speed, flow, level, voltage, current, etc.)
4. **Metric Value**: The exact numerical threshold value with units
5. **Description**: Clear explanation of what this rule monitors and why it's important
6. **Consequence**: What happens if the rule is exceeded (equipment damage, safety issues, performance degradation, etc.)

Focus on extracting:
- Temperature monitoring thresholds
- Pressure limits and ranges
- Vibration and speed parameters
- Flow and level measurements
- Electrical parameters (voltage, current)
- Performance indicators
- Operational status conditions
- Equipment health metrics

Ensure all rules are:
- Based on actual numerical values from the documentation
- Practical and implementable in IoT systems
- Clear about consequences when exceeded
- Specific to the equipment or system being monitored"""
        
        # Truncate chunks to fit within token limit
        selected_chunks = self.truncate_chunks_to_fit_context(chunks, system_prompt, user_prompt_template)
        
        if not selected_chunks:
            logger.warning("No chunks could fit within token limit")
            return []
        
        # Prepare context from selected chunks
        context_parts = []
        for chunk in selected_chunks:
            try:
                # Handle both possible chunk structures
                if "metadata" in chunk and "document" in chunk:
                    # Vector DB format
                    heading = chunk.get("metadata", {}).get("heading", "")
                    content = chunk.get("document", "")
                else:
                    # Fallback format
                    heading = chunk.get("heading", "")
                    content = chunk.get("text", chunk.get("content", ""))
                
                if content:
                    context_parts.append(f"**{heading}**\n{content}")
            except Exception as e:
                logger.warning(f"Error processing chunk in rules generation: {str(e)}")
                continue
        
        if not context_parts:
            logger.warning("No valid context could be extracted for rules generation")
            return []
        
        context = "\n\n".join(context_parts)
        
        # Format the user prompt with context
        try:
            user_prompt = user_prompt_template.format(context=context)
        except Exception as e:
            logger.error(f"Error formatting user prompt in rules generation: {str(e)}")
            # Fallback to simple prompt
            user_prompt = f"Please generate IoT monitoring rules based on this context:\n\n{context}"
        
        try:
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
                model=self.model_name
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                # Find JSON in response
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                json_str = content[start_idx:end_idx]
                
                rules_data = json.loads(json_str)
                rules = [Rule(**rule) for rule in rules_data]
                
                logger.info(f"Generated {len(rules)} rules")
                return rules
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON response, creating fallback rules: {e}")
                # Create fallback rules from text
                return self._parse_rules_from_text(content)
                
        except Exception as e:
            logger.error(f"Error generating rules: {str(e)}")
            raise e
    
    async def generate_maintenance_schedule(self, chunks: List[Dict[str, Any]]) -> List[MaintenanceTask]:
        """Generate maintenance schedule from chunks"""
        logger.info(f"Generating maintenance schedule from {len(chunks)} chunks")
        
        # System prompt
        system_prompt = """You are a senior maintenance engineer with 20+ years of experience in industrial equipment maintenance. 
You specialize in creating comprehensive, detailed maintenance schedules from technical documentation.
Your expertise includes preventive maintenance, predictive maintenance, and equipment reliability optimization.
You understand the critical importance of proper maintenance procedures for equipment safety and performance."""
        
        # User prompt template with detailed requirements
        user_prompt_template = """Analyze the provided technical documentation sections and extract comprehensive maintenance information.

Context:
{context}

Please generate detailed maintenance tasks in JSON format with the following structure:
[
  {{
    "task": "Check hydraulic oil level in main reservoir",
    "frequency": "daily",
    "category": "lubrication",
    "description": "Verify hydraulic oil level is between MIN and MAX marks on sight glass. Top up if necessary using approved hydraulic oil grade.",
    "priority": "high",
    "estimated_duration": "5 minutes",
    "required_tools": "oil level gauge, approved hydraulic oil",
    "safety_notes": "Ensure equipment is stopped and depressurized before checking oil level"
  }}
]

IMPORTANT REQUIREMENTS:
1. **Task Name**: Be specific and descriptive (e.g., "Check hydraulic oil level" not just "Check oil")
2. **Frequency**: Use specific intervals (daily, weekly, monthly, quarterly, semi-annually, annually, as-needed)
3. **Category**: Choose from: lubrication, inspection, cleaning, calibration, safety, performance, electrical, mechanical, preventive, predictive
4. **Description**: Provide detailed step-by-step instructions including:
   - What to check/inspect
   - How to perform the task
   - What to look for (acceptable ranges, signs of wear, etc.)
   - What actions to take if issues are found
5. **Priority**: high, medium, or low based on:
   - Safety implications
   - Equipment criticality
   - Impact on production
6. **Estimated Duration**: Realistic time estimate for the task
7. **Required Tools**: List specific tools, equipment, or materials needed
8. **Safety Notes**: Any safety precautions, PPE requirements, or warnings

Focus on extracting:
- Preventive maintenance procedures
- Inspection and monitoring tasks
- Cleaning and housekeeping activities
- Lubrication requirements
- Calibration and adjustment procedures
- Safety checks and procedures
- Performance monitoring tasks
- Equipment-specific maintenance requirements

Ensure all tasks are practical, actionable, and based on the actual content in the provided documentation."""
        
        # Truncate chunks to fit within token limit
        selected_chunks = self.truncate_chunks_to_fit_context(chunks, system_prompt, user_prompt_template)
        
        if not selected_chunks:
            logger.warning("No chunks could fit within token limit")
            return []
        
        # Prepare context from selected chunks
        context_parts = []
        for chunk in selected_chunks:
            try:
                # Handle both possible chunk structures
                if "metadata" in chunk and "document" in chunk:
                    # Vector DB format
                    heading = chunk.get("metadata", {}).get("heading", "")
                    content = chunk.get("document", "")
                else:
                    # Fallback format
                    heading = chunk.get("heading", "")
                    content = chunk.get("text", chunk.get("content", ""))
                
                if content:
                    context_parts.append(f"**{heading}**\n{content}")
            except Exception as e:
                logger.warning(f"Error processing chunk in maintenance generation: {str(e)}")
                continue
        
        if not context_parts:
            logger.warning("No valid context could be extracted for maintenance generation")
            return []
        
        context = "\n\n".join(context_parts)
        
        # Format the user prompt with context
        try:
            user_prompt = user_prompt_template.format(context=context)
        except Exception as e:
            logger.error(f"Error formatting user prompt in maintenance generation: {str(e)}")
            # Fallback to simple prompt
            user_prompt = f"Please generate maintenance tasks based on this context:\n\n{context}"
        
        try:
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
                model=self.model_name
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                json_str = content[start_idx:end_idx]
                
                tasks_data = json.loads(json_str)
                
                # Convert frequency text to numeric values
                for task in tasks_data:
                    frequency_text = task.get('frequency', '').lower()
                    if 'daily' in frequency_text or 'day' in frequency_text:
                        task['frequency'] = 1
                    elif 'weekly' in frequency_text or 'week' in frequency_text:
                        task['frequency'] = 7
                    elif 'monthly' in frequency_text or 'month' in frequency_text:
                        task['frequency'] = 30
                    elif 'quarterly' in frequency_text or 'quarter' in frequency_text:
                        task['frequency'] = 90
                    elif 'semi-annually' in frequency_text or 'semi-annual' in frequency_text or '6 month' in frequency_text:
                        task['frequency'] = 180
                    elif 'annually' in frequency_text or 'yearly' in frequency_text or 'annual' in frequency_text:
                        task['frequency'] = 365
                    else:
                        task['frequency'] = 1  # Default to daily (1) if unclear or not found
                
                tasks = [MaintenanceTask(**task) for task in tasks_data]
                
                logger.info(f"Generated {len(tasks)} maintenance tasks")
                return tasks
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                return self._parse_maintenance_from_text(content)
                
        except Exception as e:
            logger.error(f"Error generating maintenance schedule: {str(e)}")
            raise e
    
    async def generate_safety_information(self, chunks: List[Dict[str, Any]]) -> List[SafetyInfo]:
        """Generate safety information from chunks"""
        logger.info(f"Generating safety information from {len(chunks)} chunks")
        
        # System prompt
        system_prompt = """You are a senior safety engineer with 20+ years of experience in industrial safety and risk management.
You specialize in identifying, analyzing, and providing comprehensive safety solutions for industrial equipment and machinery.
Your expertise includes hazard identification, risk assessment, safety procedures, emergency response, and compliance with safety standards.
You understand the critical importance of preventing accidents and protecting workers from machine-related hazards."""
        
        # User prompt template with detailed requirements
        user_prompt_template = """Analyze the provided technical documentation sections and extract comprehensive safety information.

Context:
{context}

Please generate detailed safety information in JSON format with the following structure:
[
  {{
    "name": "High Temperature Hazard",
    "about_reaction": "Equipment surfaces can reach dangerously high temperatures during operation",
    "causes": "Continuous operation, lack of cooling, mechanical friction, electrical resistance",
    "how_to_avoid": "Monitor temperature sensors, ensure proper ventilation, follow operating procedures, use thermal protection",
    "safety_info": "Wear heat-resistant PPE, maintain safe distance, implement temperature monitoring, establish emergency shutdown procedures"
  }}
]

IMPORTANT REQUIREMENTS:
1. **Name**: Specific hazard or safety concern (e.g., "High Temperature Hazard" not just "Temperature Warning")
2. **About Reaction**: What happens when this hazard occurs or what the danger is
3. **Causes**: What leads to this safety issue (mechanical, electrical, operational, environmental factors)
4. **How to Avoid**: Specific preventive measures and safety procedures to prevent the hazard
5. **Safety Info**: Additional safety information including PPE requirements, emergency procedures, and safety protocols

Focus on extracting:
- Machine operation hazards
- Equipment failure risks
- Environmental dangers
- Human factor risks
- Emergency situations
- Maintenance safety concerns
- Installation and startup hazards
- Shutdown and isolation procedures

Ensure all safety information is:
- Based on actual content from the documentation
- Practical and actionable
- Specific to the equipment or system
- Clear about consequences and prevention methods
- Comprehensive in covering all safety aspects"""
        
        # Process chunks in batches of 10
        batch_size = 10
        all_safety_info = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: chunks {i+1}-{min(i+batch_size, len(chunks))}")
            
            # Truncate chunks to fit within token limit
            selected_chunks = self.truncate_chunks_to_fit_context(batch_chunks, system_prompt, user_prompt_template)
            
            if not selected_chunks:
                logger.warning(f"No chunks could fit within token limit for batch {i//batch_size + 1}")
                continue
            
            # Prepare context from selected chunks
            context_parts = []
            for chunk in selected_chunks:
                try:
                    # Handle both possible chunk structures
                    if "metadata" in chunk and "document" in chunk:
                        # Vector DB format
                        heading = chunk.get("metadata", {}).get("heading", "")
                        content = chunk.get("document", "")
                    else:
                        # Fallback format
                        heading = chunk.get("heading", "")
                        content = chunk.get("text", chunk.get("content", ""))
                    
                    if content:
                        context_parts.append(f"**{heading}**\n{content}")
                except Exception as e:
                    logger.warning(f"Error processing chunk in safety generation: {str(e)}")
                    continue
            
            if not context_parts:
                logger.warning(f"No valid context could be extracted for batch {i//batch_size + 1}")
                continue
            
            context = "\n\n".join(context_parts)
            
            # Format the user prompt with context
            try:
                user_prompt = user_prompt_template.format(context=context)
            except Exception as e:
                logger.error(f"Error formatting user prompt in safety generation: {str(e)}")
                # Fallback to simple prompt
                user_prompt = f"Please generate safety information based on this context:\n\n{context}"
            
            try:
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
                    model=self.model_name
                )
                
                content = response.choices[0].message.content
                
                # Extract JSON from response
                try:
                    start_idx = content.find('[')
                    end_idx = content.rfind(']') + 1
                    json_str = content[start_idx:end_idx]
                    
                    safety_data = json.loads(json_str)
                    batch_safety_info = [SafetyInfo(**info) for info in safety_data]
                    
                    all_safety_info.extend(batch_safety_info)
                    logger.info(f"Generated {len(batch_safety_info)} safety items from batch {i//batch_size + 1}")
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse JSON response for batch {i//batch_size + 1}: {e}")
                    batch_fallback = self._parse_safety_from_text(content)
                    all_safety_info.extend(batch_fallback)
                    
            except Exception as e:
                logger.error(f"Error generating safety information for batch {i//batch_size + 1}: {str(e)}")
                continue
        
        logger.info(f"Generated total {len(all_safety_info)} safety items from all batches")
        return all_safety_info
    
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

    def _parse_rules_from_text(self, text: str) -> List[Rule]:
        """Parse rules from plain text response"""
        rules = []
        lines = text.split('\n')
        
        metrics = ['temperature', 'pressure', 'vibration', 'speed', 'flow', 'level', 'voltage', 'current', 'power', 'efficiency']
        
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ['temperature', 'pressure', 'vibration', 'speed', 'flow', 'level', 'voltage', 'current', 'monitor', 'threshold', 'limit']):
                # Determine metric
                metric = "general"
                for m in metrics:
                    if m in line_lower:
                        metric = m
                        break
                
                # Extract threshold if present
                threshold = line.strip()
                metric_value = "N/A"
                
                # Try to extract numerical values
                import re
                numbers = re.findall(r'\d+(?:\.\d+)?', line)
                if numbers:
                    metric_value = f"{numbers[0]} units"
                
                # Determine consequence based on metric type
                consequence = "Equipment performance degradation"
                if 'temperature' in line_lower:
                    consequence = "Thermal damage, equipment shutdown, safety hazard"
                elif 'pressure' in line_lower:
                    consequence = "System failure, pressure vessel damage, safety risk"
                elif 'vibration' in line_lower:
                    consequence = "Mechanical damage, bearing failure, equipment wear"
                elif 'voltage' in line_lower or 'current' in line_lower:
                    consequence = "Electrical damage, component failure, safety hazard"
                
                rules.append(Rule(
                    rule_name=f"{metric.title()} Monitoring Rule",
                    threshold=threshold,
                    metric=metric,
                    metric_value=metric_value,
                    description=f"Monitor {metric} to ensure optimal equipment performance and prevent damage",
                    consequence=consequence
                ))
        
        return rules
    
    def _parse_maintenance_from_text(self, text: str) -> List[MaintenanceTask]:
        """Parse maintenance tasks from plain text response"""
        tasks = []
        lines = text.split('\n')
        
        frequencies = ['daily', 'weekly', 'monthly', 'quarterly', 'semi-annually', 'annually', 'as-needed']
        categories = ['lubrication', 'inspection', 'cleaning', 'calibration', 'safety', 'performance', 'electrical', 'mechanical', 'preventive', 'predictive']
        
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ['maintenance', 'check', 'inspect', 'clean', 'lubricate', 'calibrate', 'test', 'monitor']):
                # Determine frequency and convert to numeric values
                frequency = 1  # default to daily
                frequency_text = "daily"
                for freq in frequencies:
                    if freq in line_lower:
                        frequency_text = freq
                        break
                
                # Convert frequency text to numeric values
                if 'daily' in frequency_text or 'day' in frequency_text:
                    frequency = 1
                elif 'weekly' in frequency_text or 'week' in frequency_text:
                    frequency = 7
                elif 'monthly' in frequency_text or 'month' in frequency_text:
                    frequency = 30
                elif 'quarterly' in frequency_text or 'quarter' in frequency_text:
                    frequency = 90
                elif 'semi-annually' in frequency_text or 'semi-annual' in frequency_text or '6 month' in frequency_text:
                    frequency = 180
                elif 'annually' in frequency_text or 'yearly' in frequency_text or 'annual' in frequency_text:
                    frequency = 365
                
                # Determine category
                category = "general"  # default
                for cat in categories:
                    if cat in line_lower:
                        category = cat
                        break
                
                # Determine priority based on keywords
                priority = "medium"  # default
                if any(word in line_lower for word in ['critical', 'safety', 'emergency', 'urgent']):
                    priority = "high"
                elif any(word in line_lower for word in ['routine', 'basic', 'simple']):
                    priority = "low"
                
                tasks.append(MaintenanceTask(
                    task=line.strip(),
                    frequency=str(frequency),
                    category=category,
                    description=line.strip(),
                    priority=priority,
                    estimated_duration="10 minutes",
                    required_tools="Standard maintenance tools",
                    safety_notes="Follow standard safety procedures"
                ))
        
        return tasks
    
    def _parse_safety_from_text(self, text: str) -> List[SafetyInfo]:
        """Parse safety information from plain text response"""
        safety_items = []
        lines = text.split('\n')
        
        safety_keywords = ['warning', 'caution', 'danger', 'safety', 'hazard', 'risk', 'emergency', 'protective', 'injury', 'damage']
        
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in safety_keywords):
                # Determine hazard type based on keywords
                hazard_type = "General Safety Hazard"
                if 'temperature' in line_lower or 'heat' in line_lower:
                    hazard_type = "High Temperature Hazard"
                elif 'pressure' in line_lower:
                    hazard_type = "Pressure Hazard"
                elif 'electrical' in line_lower or 'voltage' in line_lower or 'current' in line_lower:
                    hazard_type = "Electrical Hazard"
                elif 'mechanical' in line_lower or 'moving' in line_lower:
                    hazard_type = "Mechanical Hazard"
                elif 'chemical' in line_lower or 'toxic' in line_lower:
                    hazard_type = "Chemical Hazard"
                
                # Determine causes based on hazard type
                causes = "Equipment malfunction, operational error, environmental factors"
                if 'temperature' in line_lower:
                    causes = "Continuous operation, lack of cooling, mechanical friction, electrical resistance"
                elif 'pressure' in line_lower:
                    causes = "System overpressure, valve failure, pump malfunction, pressure vessel damage"
                elif 'electrical' in line_lower:
                    causes = "Electrical fault, insulation failure, overcurrent, short circuit"
                elif 'mechanical' in line_lower:
                    causes = "Mechanical failure, wear and tear, improper maintenance, operational stress"
                
                # Determine prevention methods
                how_to_avoid = "Follow safety procedures, use proper PPE, maintain equipment, monitor conditions"
                if 'temperature' in line_lower:
                    how_to_avoid = "Monitor temperature sensors, ensure proper ventilation, follow operating procedures, use thermal protection"
                elif 'pressure' in line_lower:
                    how_to_avoid = "Monitor pressure gauges, check relief valves, follow pressure limits, maintain pressure systems"
                elif 'electrical' in line_lower:
                    how_to_avoid = "Use proper electrical PPE, check insulation, follow lockout procedures, maintain electrical systems"
                elif 'mechanical' in line_lower:
                    how_to_avoid = "Use mechanical guards, follow lockout procedures, maintain equipment, monitor for unusual sounds/vibration"
                
                # Determine safety information
                safety_info = "Wear appropriate PPE, follow safety procedures, maintain safe distance, implement monitoring"
                if 'temperature' in line_lower:
                    safety_info = "Wear heat-resistant PPE, maintain safe distance, implement temperature monitoring, establish emergency shutdown procedures"
                elif 'pressure' in line_lower:
                    safety_info = "Wear pressure-resistant PPE, maintain safe distance, implement pressure monitoring, establish emergency relief procedures"
                elif 'electrical' in line_lower:
                    safety_info = "Wear electrical PPE, maintain safe distance, implement electrical monitoring, establish emergency shutdown procedures"
                elif 'mechanical' in line_lower:
                    safety_info = "Wear mechanical PPE, maintain safe distance, implement vibration monitoring, establish emergency stop procedures"
                
                safety_items.append(SafetyInfo(
                    name=hazard_type,
                    about_reaction=line.strip(),
                    causes=causes,
                    how_to_avoid=how_to_avoid,
                    safety_info=safety_info
                ))
        
        return safety_items