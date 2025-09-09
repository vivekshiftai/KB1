"""
LLM Service - Clean Version
Handles interactions with Azure AI for intelligent responses

Version: 0.2 - Cleaned up for LangGraph integration
"""

import json
import logging
import tiktoken
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
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    async def query_with_context(self, chunks: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Query with context chunks using structured JSON response"""
        logger.info(f"Processing query with {len(chunks)} context chunks")
        
        # If no chunks available, provide a helpful response
        if not chunks:
            logger.warning("No chunks available, providing general response")
            return {
                "response": f"I don't have access to specific documentation about '{query}'. Please ensure the document is properly uploaded and processed.",
                "chunks_used": []
            }
        
        # Prepare context from chunks
        context_parts = []
        chunk_headings = []
        for chunk in chunks:
            try:
                if "metadata" in chunk and "document" in chunk:
                    heading = chunk.get("metadata", {}).get("heading", "")
                    content = chunk.get("document", "")
                else:
                    heading = chunk.get("heading", "")
                    content = chunk.get("text", chunk.get("content", ""))
                
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
        system_prompt = """You are a technical documentation assistant. You must respond with a valid JSON object containing the answer and referenced sections.

CRITICAL: Your response must be ONLY a valid JSON object with this exact structure:
{
    "response": "Your detailed answer to the user's question",
    "chunks_used": ["List of section headings you referenced"]
}

Do not include any text before or after the JSON object."""
        
        user_prompt = f"""Based on the following documentation, answer the user's question and return a JSON response.

Documentation Context:
{context}

Available Section Headings: {chunk_headings}

User Question: {query}

Provide a comprehensive answer based on the documentation. In the "chunks_used" array, list the exact section headings from the available headings that you referenced in your answer.

Return ONLY the JSON object, no additional text."""
        
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
            
            raw_response = response.choices[0].message.content.strip()
            logger.info(f"Raw LLM response: {raw_response[:200]}...")
            
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
                model=self.model_name
            )
            
            raw_response = response.choices[0].message.content.strip()
            logger.info(f"Raw analysis response: {raw_response[:200]}...")
            
            # Try to parse JSON response
            try:
                import json
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