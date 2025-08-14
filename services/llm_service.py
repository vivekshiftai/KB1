import json
import logging
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from config import settings
from models.schemas import Rule, MaintenanceTask, SafetyInfo

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4"
    
    async def query_with_context(self, chunks: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Query with context chunks using GPT-4"""
        logger.info(f"Processing query with {len(chunks)} context chunks")
        
        # Prepare context
        context_parts = []
        for i, chunk in enumerate(chunks):
            heading = chunk["metadata"].get("heading", f"Section {i+1}")
            content = chunk["document"]
            context_parts.append(f"**{heading}**\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following technical manual sections, please answer the user's query. 
Please identify which specific sections you used to formulate your answer.

Context:
{context}

User Query: {query}

Please provide a comprehensive answer and then list the specific sections/headings you referenced."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a technical documentation assistant. Provide accurate, detailed answers based on the provided manual sections."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
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
        
        # Prepare context
        context_parts = []
        for chunk in chunks:
            heading = chunk["metadata"].get("heading", "")
            content = chunk["document"]
            context_parts.append(f"**{heading}**\n{content}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Analyze these technical manual sections and generate IoT monitoring rules. 
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
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an IoT systems engineer specializing in industrial monitoring rules. Generate practical, actionable monitoring rules based on technical documentation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
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
        
        # Prepare context
        context_parts = []
        for chunk in chunks:
            heading = chunk["metadata"].get("heading", "")
            content = chunk["document"]
            context_parts.append(f"**{heading}**\n{content}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Extract maintenance schedules from these manual sections. 
Identify daily, weekly, monthly, and periodic maintenance tasks.
Return structured data with task descriptions and frequencies.

Context:
{context}

Please generate maintenance tasks in JSON format:
[
  {{
    "task": "Check oil levels",
    "frequency": "daily",
    "category": "lubrication",
    "description": "Visual inspection of oil levels in main reservoir"
  }}
]

Focus on:
- Daily inspections
- Weekly cleaning tasks
- Monthly servicing
- Periodic overhauls
- Calibration requirements
- Filter replacements"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a maintenance engineer specializing in industrial equipment. Extract comprehensive maintenance schedules from technical documentation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
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
        
        # Prepare context
        context_parts = []
        for chunk in chunks:
            heading = chunk["metadata"].get("heading", "")
            content = chunk["document"]
            context_parts.append(f"**{heading}**\n{content}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Extract safety procedures and warnings from these manual sections.
Generate comprehensive safety guidelines categorized by type.

Context:
{context}

Please generate safety information in JSON format:
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
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a safety engineer specializing in industrial equipment safety. Extract comprehensive safety information from technical documentation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
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