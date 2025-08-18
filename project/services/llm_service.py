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
        self.max_tokens = 8192
        self.max_completion_tokens = 2000
        self.max_context_tokens = self.max_tokens - self.max_completion_tokens
        self.encoding = tiktoken.encoding_for_model("gpt-4")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def truncate_chunks_to_fit_context(self, chunks: List[Dict[str, Any]], system_prompt: str = "", user_prompt_template: str = "") -> List[Dict[str, Any]]:
        """Truncate chunks to fit within context limit"""
        # Calculate available tokens for chunks
        system_tokens = self.count_tokens(system_prompt)
        user_template_tokens = self.count_tokens(user_prompt_template)
        reserved_tokens = system_tokens + user_template_tokens + 100  # Buffer
        
        available_tokens = self.max_context_tokens - reserved_tokens
        
        logger.info(f"Available tokens for chunks: {available_tokens}")
        logger.info(f"System prompt tokens: {system_tokens}")
        logger.info(f"User template tokens: {user_template_tokens}")
        
        selected_chunks = []
        current_tokens = 0
        
        for chunk in chunks:
            heading = chunk["metadata"].get("heading", "")
            content = chunk["document"]
            chunk_text = f"**{heading}**\n{content}"
            chunk_tokens = self.count_tokens(chunk_text)
            
            if current_tokens + chunk_tokens <= available_tokens:
                selected_chunks.append(chunk)
                current_tokens += chunk_tokens
                logger.info(f"Added chunk '{heading}' ({chunk_tokens} tokens), total: {current_tokens}")
            else:
                logger.info(f"Skipping chunk '{heading}' ({chunk_tokens} tokens) - would exceed limit")
                break
        
        logger.info(f"Selected {len(selected_chunks)} chunks with {current_tokens} tokens")
        return selected_chunks
    
    async def query_with_context(self, chunks: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Query with context chunks using GPT-4"""
        logger.info(f"Processing query with {len(chunks)} context chunks")
        
        # System prompt
        system_prompt = "You are a technical documentation assistant. Provide accurate, detailed answers based on the provided manual sections."
        
        # User prompt template
        user_prompt_template = """Based on the following technical manual sections, please answer the user's query. 
Please identify which specific sections you used to formulate your answer.

Context:
{context}

User Query: {query}

Please provide a comprehensive answer and then list the specific sections/headings you referenced."""
        
        # Truncate chunks to fit within token limit
        selected_chunks = self.truncate_chunks_to_fit_context(chunks, system_prompt, user_prompt_template.format(query=query))
        
        if not selected_chunks:
            logger.warning("No chunks could fit within token limit")
            return {
                "response": "I apologize, but the content is too large to process. Please try a more specific query or upload a smaller document.",
                "chunks_used": []
            }
        
        # Prepare context from selected chunks
        context_parts = []
        for i, chunk in enumerate(selected_chunks):
            heading = chunk["metadata"].get("heading", f"Section {i+1}")
            content = chunk["document"]
            context_parts.append(f"**{heading}**\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Format the user prompt with context
        user_prompt = user_prompt_template.format(context=context, query=query)
        
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
            
            # Simple parsing to extract referenced sections
            chunks_used = []
            for chunk in chunks:
                heading = chunk["metadata"].get("heading", "")
                if heading.lower() in answer.lower():
                    chunks_used.append(heading)
            
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
            heading = chunk["metadata"].get("heading", "")
            content = chunk["document"]
            context_parts.append(f"**{heading}**\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Format the user prompt with context
        user_prompt = user_prompt_template.format(context=context)
        
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
            heading = chunk["metadata"].get("heading", "")
            content = chunk["document"]
            context_parts.append(f"**{heading}**\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Format the user prompt with context
        user_prompt = user_prompt_template.format(context=context)
        
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
            heading = chunk["metadata"].get("heading", "")
            content = chunk["document"]
            context_parts.append(f"**{heading}**\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Format the user prompt with context
        user_prompt = user_prompt_template.format(context=context)
        
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