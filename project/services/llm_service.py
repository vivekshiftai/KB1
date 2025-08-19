"""
LLM Service
Handles interactions with OpenAI GPT-4 for intelligent responses

Version: 0.1
"""

import json
import logging
import tiktoken
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from config import settings
from models.schemas import Rule, MaintenanceTask, SafetyInfo

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4"
        # Increase token limits for better handling of large documents
        self.max_tokens = 8192
        self.max_completion_tokens = 1500  # Reduced to allow more context
        self.max_context_tokens = self.max_tokens - self.max_completion_tokens
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        logger.info(f"LLM Service initialized with max_tokens: {self.max_tokens}, max_context_tokens: {self.max_context_tokens}")
    
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
        """Query with context chunks using GPT-4"""
        logger.info(f"Processing query with {len(chunks)} context chunks")
        
        # Debug: Log chunk structure
        if chunks:
            logger.info(f"First chunk keys: {list(chunks[0].keys())}")
            logger.info(f"First chunk metadata keys: {list(chunks[0].get('metadata', {}).keys())}")
        
        # System prompt
        system_prompt = "You are a technical documentation assistant. Provide accurate, detailed answers based on the provided manual sections."
        
        # User prompt template (simplified to use fewer tokens)
        user_prompt_template = """Answer the user's query based on the provided context.

Context:
{context}

Query: {query}

Provide a clear answer. At the end, add "REFERENCES:" followed by the exact section headings you used."""
        
        # Truncate chunks to fit within token limit
        selected_chunks = self.truncate_chunks_to_fit_context(chunks, system_prompt, user_prompt_template)
        
        if not selected_chunks:
            logger.error("No chunks could fit within token limit")
            logger.error(f"Available tokens: {available_tokens}")
            logger.error(f"Total chunks received: {len(chunks)}")
            if chunks:
                first_chunk = chunks[0]
                if "metadata" in first_chunk and "document" in first_chunk:
                    first_chunk_tokens = self.count_tokens(first_chunk.get("document", ""))
                else:
                    first_chunk_tokens = self.count_tokens(first_chunk.get("text", ""))
                logger.error(f"First chunk tokens: {first_chunk_tokens}")
            
            return {
                "response": "I apologize, but the content is too large to process. Please try a more specific query or upload a smaller document.",
                "chunks_used": []
            }
        
        # Prepare context from selected chunks
        context_parts = []
        for i, chunk in enumerate(selected_chunks):
            try:
                # Handle both possible chunk structures
                if "metadata" in chunk and "document" in chunk:
                    # Vector DB format
                    heading = chunk.get("metadata", {}).get("heading", f"Section {i+1}")
                    content = chunk.get("document", "")
                else:
                    # Fallback format
                    heading = chunk.get("heading", f"Section {i+1}")
                    content = chunk.get("text", chunk.get("content", ""))
                
                if content:
                    context_parts.append(f"**{heading}**\n{content}")
            except Exception as e:
                logger.warning(f"Error processing chunk {i}: {str(e)}")
                continue
        
        if not context_parts:
            logger.warning("No valid context could be extracted from chunks")
            return {
                "response": "I apologize, but I couldn't extract meaningful content from the document. Please try a different query or upload a different document.",
                "chunks_used": []
            }
        
        context = "\n\n".join(context_parts)
        
        # Format the user prompt with context
        try:
            user_prompt = user_prompt_template.format(context=context, query=query)
        except KeyError as e:
            logger.error(f"Error formatting user prompt - missing placeholder: {str(e)}")
            # Fallback to simple prompt
            user_prompt = f"Please answer this query based on the following context:\n\n{context}\n\nQuery: {query}"
        except Exception as e:
            logger.error(f"Error formatting user prompt: {str(e)}")
            # Fallback to simple prompt
            user_prompt = f"Please answer this query based on the following context:\n\n{context}\n\nQuery: {query}"
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_completion_tokens,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            
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
        system_prompt = "You are an IoT systems engineer specializing in industrial monitoring rules. Generate practical, actionable monitoring rules based on technical documentation."
        
        # User prompt template
        user_prompt_template = """Analyze these technical manual sections and generate IoT monitoring rules. 
Focus on operational parameters, thresholds, and automated responses. 
Format as structured rules with conditions and actions.
Avoid safety procedures - focus on operational monitoring.

Context:
{context}

Please generate rules in JSON format with the following structure:
[
  {{
    "condition": "Temperature > 30°C",
    "action": "Send notification to operator",
    "category": "temperature_monitoring",
    "priority": "medium"
  }}
]

Focus on:
- Temperature thresholds
- Pressure limits
- Operational parameters
- Performance indicators
- Maintenance triggers
- System status monitoring"""
        
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
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_completion_tokens,
                temperature=0.2
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
        system_prompt = "You are a maintenance engineer specializing in industrial equipment maintenance schedules. Extract and structure maintenance tasks from technical documentation."
        
        # User prompt template
        user_prompt_template = """Extract maintenance schedules from these manual sections. 
Identify daily, weekly, monthly, and periodic maintenance tasks.
Return structured data with task descriptions and frequencies.

Context:
{context}

Please generate maintenance tasks in JSON format with the following structure:
[
  {{
    "task": "Check oil level",
    "frequency": "daily",
    "category": "lubrication",
    "description": "Verify oil level is within acceptable range"
  }}
]

Focus on:
- Preventive maintenance tasks
- Inspection procedures
- Cleaning and lubrication
- Calibration requirements
- Safety checks
- Performance monitoring"""
        
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
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_completion_tokens,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                json_str = content[start_idx:end_idx]
                
                tasks_data = json.loads(json_str)
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
        system_prompt = "You are a safety engineer specializing in industrial equipment safety. Extract comprehensive safety information from technical documentation."
        
        # User prompt template
        user_prompt_template = """Extract safety procedures and warnings from these manual sections.
Generate comprehensive safety guidelines categorized by type.

Context:
{context}

Please generate safety information in JSON format with the following structure:
[
  {{
    "type": "warning",
    "title": "High Temperature Warning",
    "description": "Equipment surfaces may reach temperatures exceeding 80°C during operation",
    "category": "thermal_hazard"
  }}
]

Focus on:
- Warnings and cautions
- Safety procedures
- Emergency procedures
- Personal protective equipment
- Hazard identification
- Risk mitigation"""
        
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
                logger.warning(f"Error processing chunk in safety generation: {str(e)}")
                continue
        
        if not context_parts:
            logger.warning("No valid context could be extracted for safety generation")
            return []
        
        context = "\n\n".join(context_parts)
        
        # Format the user prompt with context
        try:
            user_prompt = user_prompt_template.format(context=context)
        except Exception as e:
            logger.error(f"Error formatting user prompt in safety generation: {str(e)}")
            # Fallback to simple prompt
            user_prompt = f"Please generate safety information based on this context:\n\n{context}"
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_completion_tokens,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                json_str = content[start_idx:end_idx]
                
                safety_data = json.loads(json_str)
                safety_info = [SafetyInfo(**info) for info in safety_data]
                
                logger.info(f"Generated {len(safety_info)} safety items")
                return safety_info
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                return self._parse_safety_from_text(content)
                
        except Exception as e:
            logger.error(f"Error generating safety information: {str(e)}")
            raise e
    
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
                logger.warning("No explicit REFERENCES section found, falling back to text matching")
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
            
            logger.info(f"Final extracted referenced sections: {referenced_sections}")
            return referenced_sections
            
        except Exception as e:
            logger.error(f"Error extracting referenced sections: {str(e)}")
            # Fallback: return all chunk headings
            fallback_sections = []
            for chunk in chunks:
                try:
                    if "metadata" in chunk and "document" in chunk:
                        heading = chunk.get("metadata", {}).get("heading", "")
                    else:
                        heading = chunk.get("heading", "")
                    
                    if heading:
                        fallback_sections.append(heading)
                except Exception:
                    continue
            
            logger.info(f"Using fallback sections: {fallback_sections}")
            return fallback_sections

    def _parse_rules_from_text(self, text: str) -> List[Rule]:
        """Parse rules from plain text response"""
        rules = []
        lines = text.split('\n')
        
        for line in lines:
            if 'if' in line.lower() and ('then' in line.lower() or '>' in line or '<' in line):
                rules.append(Rule(
                    condition=line.strip(),
                    action="Monitor and alert",
                    category="general",
                    priority="medium"
                ))
        
        return rules
    
    def _parse_maintenance_from_text(self, text: str) -> List[MaintenanceTask]:
        """Parse maintenance tasks from plain text response"""
        tasks = []
        lines = text.split('\n')
        
        frequencies = ['daily', 'weekly', 'monthly', 'annually']
        
        for line in lines:
            line_lower = line.lower()
            for freq in frequencies:
                if freq in line_lower:
                    tasks.append(MaintenanceTask(
                        task=line.strip(),
                        frequency=freq,
                        category="general",
                        description=line.strip()
                    ))
                    break
        
        return tasks
    
    def _parse_safety_from_text(self, text: str) -> List[SafetyInfo]:
        """Parse safety information from plain text response"""
        safety_items = []
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ['warning', 'caution', 'danger', 'safety', 'hazard']):
                if 'warning' in line_lower:
                    safety_type = 'warning'
                elif 'procedure' in line_lower:
                    safety_type = 'procedure'
                else:
                    safety_type = 'warning'
                
                safety_items.append(SafetyInfo(
                    type=safety_type,
                    title=line.strip()[:100],
                    description=line.strip(),
                    category="general"
                ))
        
        return safety_items