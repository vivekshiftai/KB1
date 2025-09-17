"""
LangGraph-based Query Processor with Response Validation and Retry Logic
Implements intelligent query processing with response validation and chunk size optimization

Version: 0.1
"""

import logging
import time
from typing import TypedDict, List, Dict, Any, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from services.vector_db import VectorDatabase
from services.llm_service import LLMService
from utils.helpers import sanitize_filename, calculate_processing_time
from models.schemas import QueryRequest, QueryResponse, ImageData
# Configuration constants
MAX_ITERATIONS = 3
BASE_CHUNK_SIZE = 2
CHUNK_SIZE_INCREMENT = 1
MIN_CONFIDENCE_SCORE = 0.6
MIN_RESPONSE_LENGTH = 50
HUMAN_REVIEW_THRESHOLD = 0.7
QUALITY_WEIGHTS = {
    "has_content": 0.3,
    "mentions_query": 0.3,
    "uses_chunks": 0.2,
    "response_length": 0.2
}

logger = logging.getLogger(__name__)

class QueryState(TypedDict):
    """State for the LangGraph query processing workflow"""
    # Input data
    original_query: str
    pdf_name: str
    collection_name: str
    
    # Query analysis
    query_analysis: Dict[str, Any]
    individual_queries: List[str]
    current_query_index: int
    
    # Processing state
    iteration: int
    max_iterations: int
    current_chunk_size: int
    base_chunk_size: int
    
    # Data
    all_chunks: List[Dict[str, Any]]
    current_chunks: List[Dict[str, Any]]
    llm_response: Dict[str, Any]
    individual_responses: List[Dict[str, Any]]
    images: List[ImageData]
    tables: List[str]
    
    # Validation results
    response_quality: Dict[str, Any]
    user_decision: Optional[str]  # "process" or "requery"
    
    # Final output
    final_response: Optional[QueryResponse]
    processing_time: str
    start_time: float

class LangGraphQueryProcessor:
    """Enhanced query processor using LangGraph for intelligent workflow management"""
    
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.llm_service = LLMService()
        
        # Build the LangGraph workflow
        self.graph = self._build_query_workflow()
        
        logger.info("LangGraph Query Processor initialized successfully")
    
    def _build_query_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for query processing"""
        
        builder = StateGraph(QueryState)
        
        # Add nodes
        builder.add_node("initialize_query", self._initialize_query)
        builder.add_node("analyze_query", self._analyze_query)
        builder.add_node("process_individual_query", self._process_individual_query)
        builder.add_node("retrieve_chunks", self._retrieve_chunks)
        builder.add_node("generate_llm_response", self._generate_llm_response)
        builder.add_node("validate_response", self._validate_response)
        builder.add_node("human_decision", self._human_decision)
        builder.add_node("increase_chunk_size", self._increase_chunk_size)
        builder.add_node("combine_responses", self._combine_responses)
        builder.add_node("finalize_response", self._finalize_response)
        
        # Add edges
        builder.add_edge(START, "initialize_query")
        builder.add_edge("initialize_query", "analyze_query")
        builder.add_conditional_edges(
            "analyze_query",
            self._decide_query_processing,
            {
                "single_query": "process_individual_query",
                "multiple_queries": "process_individual_query"
            }
        )
        builder.add_edge("process_individual_query", "retrieve_chunks")
        builder.add_edge("retrieve_chunks", "generate_llm_response")
        builder.add_edge("generate_llm_response", "validate_response")
        
        # Conditional edges
        builder.add_conditional_edges(
            "validate_response",
            self._decide_next_step,
            {
                "human_decision": "human_decision",
                "finalize": "combine_responses",
                "retry": "increase_chunk_size"
            }
        )
        
        builder.add_conditional_edges(
            "human_decision",
            self._decide_from_human_input,
            {
                "process": "combine_responses",
                "requery": "increase_chunk_size",
                "end": "combine_responses"
            }
        )
        
        builder.add_conditional_edges(
            "increase_chunk_size",
            self._decide_retry_or_end,
            {
                "retry": "retrieve_chunks",
                "end": "combine_responses"
            }
        )
        
        builder.add_edge("combine_responses", "finalize_response")
        builder.add_edge("finalize_response", END)
        
        # Compile with checkpointing
        checkpointer = InMemorySaver()
        return builder.compile(checkpointer=checkpointer)
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process query using LangGraph workflow"""
        start_time = time.time()
        
        logger.info(f"Starting LangGraph query processing for: {request.query}")
        
        # Initialize state
        initial_state = {
            "original_query": request.query,
            "pdf_name": request.pdf_name,
            "collection_name": sanitize_filename(request.pdf_name),
            "query_analysis": {},
            "individual_queries": [],
            "current_query_index": 0,
            "iteration": 0,
            "max_iterations": MAX_ITERATIONS,
            "current_chunk_size": BASE_CHUNK_SIZE,
            "base_chunk_size": BASE_CHUNK_SIZE,
            "all_chunks": [],
            "current_chunks": [],
            "llm_response": {},
            "individual_responses": [],
            "images": [],
            "tables": [],
            "response_quality": {},
            "user_decision": None,
            "final_response": None,
            "processing_time": "",
            "start_time": start_time
        }
        
        try:
            # Execute the workflow
            config = {"configurable": {"thread_id": f"query_{int(time.time())}"}}
            final_state = await self.graph.ainvoke(initial_state, config=config)
            
            logger.info(f"LangGraph query processing completed in {final_state['processing_time']}")
            return final_state["final_response"]
            
        except Exception as e:
            logger.error(f"Error in LangGraph query processing: {str(e)}")
            # Return fallback response
            return QueryResponse(
                success=False,
                message=f"Query processing failed: {str(e)}",
                response="I apologize, but I encountered an error while processing your query. Please try again.",
                chunks_used=[],
                images=[],
                tables=[],
                processing_time=calculate_processing_time(start_time),
                suggested_images=[],
                images_used_for_response=[]
            )
    
    def _initialize_query(self, state: QueryState) -> QueryState:
        """Initialize query processing"""
        logger.info(f"Initializing query processing - Iteration {state['iteration'] + 1}")
        
        # Check if collection exists
        if not self.vector_db.collection_exists(state["collection_name"]):
            raise ValueError(f"PDF '{state['pdf_name']}' not found. Please upload the PDF first.")
        
        # Check collection type
        collection_type = self.vector_db.get_collection_type(state["collection_name"])
        if collection_type == "image":
            # Try to find document collection
            base_collection_name = state["collection_name"].replace("_images", "")
            if self.vector_db.collection_exists(base_collection_name):
                state["collection_name"] = base_collection_name
            else:
                raise ValueError(f"Document collection not found for PDF '{state['pdf_name']}'")
        
        return state
    
    async def _analyze_query(self, state: QueryState) -> QueryState:
        """Analyze the user query to determine if it's single or multiple questions"""
        logger.info("Analyzing user query for complexity")
        
        try:
            # Use LLM to analyze the query
            analysis_prompt = f"""Analyze the following user query and determine if it contains a single question or multiple questions.

User Query: "{state['original_query']}"

Return ONLY a valid JSON object with this exact structure:
{{
  "is_single_question": true/false,
  "question_count": number,
  "individual_questions": ["question1", "question2"],
  "complexity": "simple/moderate/complex",
  "reasoning": "brief explanation"
}}

Rules:
- If single question: is_single_question=true, question_count=1, individual_questions=[original_query]
- If multiple questions: is_single_question=false, question_count=actual_count, individual_questions=[broken_down_questions]
- Complexity: "simple" (1 question), "moderate" (2-3 questions), "complex" (4+ questions)
- Break down complex queries into clear, specific individual questions
- Each individual question should be self-contained and answerable

Examples:
Single: {{"is_single_question": true, "question_count": 1, "individual_questions": ["What are the safety requirements?"], "complexity": "simple", "reasoning": "Single clear question about safety requirements"}}

Multiple: {{"is_single_question": false, "question_count": 2, "individual_questions": ["What are the safety requirements?", "How do I perform maintenance?"], "complexity": "moderate", "reasoning": "Contains two distinct questions about safety and maintenance"}}"""
            
            # Get analysis from LLM
            analysis_response = await self.llm_service._get_analysis_response(analysis_prompt)
            
            # Parse the analysis
            import json
            try:
                analysis = json.loads(analysis_response)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                analysis = {
                    "is_single_question": True,
                    "question_count": 1,
                    "individual_questions": [state['original_query']],
                    "complexity": "simple",
                    "reasoning": "Fallback: treating as single question"
                }
            
            state["query_analysis"] = analysis
            state["individual_queries"] = analysis.get("individual_questions", [state['original_query']])
            
            logger.info(f"Query analysis: {analysis['complexity']} complexity, {analysis['question_count']} questions")
            logger.info(f"Individual queries: {state['individual_queries']}")
            
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            # Fallback to single question
            state["query_analysis"] = {
                "is_single_question": True,
                "question_count": 1,
                "individual_questions": [state['original_query']],
                "complexity": "simple",
                "reasoning": f"Error in analysis: {str(e)}"
            }
            state["individual_queries"] = [state['original_query']]
        
        return state
    
    def _decide_query_processing(self, state: QueryState) -> str:
        """Decide how to process the query based on analysis"""
        analysis = state["query_analysis"]
        
        if analysis.get("is_single_question", True):
            logger.info("Processing as single question")
            return "single_query"
        else:
            logger.info("Processing as multiple questions")
            return "multiple_queries"
    
    def _process_individual_query(self, state: QueryState) -> QueryState:
        """Process all individual queries together to get comprehensive chunks"""
        individual_queries = state["individual_queries"]
        
        logger.info(f"Processing {len(individual_queries)} individual queries together: {individual_queries}")
        
        # Reset iteration and chunk size for combined processing
        state["iteration"] = 0
        state["current_chunk_size"] = BASE_CHUNK_SIZE
        state["all_chunks"] = []  # Clear previous chunks
        state["current_chunks"] = []
        state["llm_response"] = {}
        state["response_quality"] = {}
        
        return state
    
    async def _retrieve_chunks(self, state: QueryState) -> QueryState:
        """Retrieve chunks from vector database for ALL individual queries"""
        chunk_size = state["current_chunk_size"]
        individual_queries = state["individual_queries"]
        
        logger.info(f"Retrieving {chunk_size} chunks for {len(individual_queries)} individual queries")
        
        all_combined_chunks = []
        
        try:
            # Get chunks for each individual query
            for i, query in enumerate(individual_queries):
                logger.info(f"Retrieving chunks for query {i+1}: {query}")
                
                chunks = await self.vector_db.query_chunks(
                    collection_name=state["collection_name"],
                    query=query,
                    top_k=chunk_size
                )
                
                if not chunks:
                    logger.warning(f"No chunks retrieved for query {i+1}, trying with empty query")
                    chunks = await self.vector_db.query_chunks(
                        collection_name=state["collection_name"],
                        query="",
                        top_k=chunk_size
                    )
                
                # Add chunks to combined list (avoid duplicates)
                for chunk in chunks:
                    if chunk not in all_combined_chunks:
                        all_combined_chunks.append(chunk)
                
                logger.info(f"Retrieved {len(chunks)} chunks for query {i+1}")
            
            # Set combined chunks for processing
            state["current_chunks"] = all_combined_chunks
            state["all_chunks"] = all_combined_chunks
            
            logger.info(f"Total combined chunks for all queries: {len(all_combined_chunks)}")
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            state["current_chunks"] = []
            state["all_chunks"] = []
        
        return state
    
    async def _generate_llm_response(self, state: QueryState) -> QueryState:
        """Generate LLM response from combined chunks for original user query"""
        logger.info("Generating LLM response for original user query")
        
        try:
            if not state["current_chunks"]:
                raise ValueError("No chunks available for processing")
            
            # Use the original user query for response generation
            original_query = state["original_query"]
            individual_queries = state["individual_queries"]
            
            # Create enhanced prompt that mentions the individual questions
            if len(individual_queries) > 1:
                enhanced_query = f"""Original Query: {original_query}

This query contains {len(individual_queries)} individual questions:
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(individual_queries)])}

Please provide a comprehensive answer that addresses all aspects of the original query."""
            else:
                enhanced_query = original_query
            
            # Use dynamic query processing for comprehensive responses
            llm_result = await self.llm_service.dynamic_query_processing(
                self.vector_db,
                state["collection_name"],
                enhanced_query,
                state.get("query_analysis")
            )
            
            state["llm_response"] = llm_result
            logger.info(f"LLM response generated for original query: {len(llm_result.get('response', ''))} characters")
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            state["llm_response"] = {
                "response": f"Error generating response: {str(e)}",
                "chunks_used": []
            }
        
        return state
    
    def _validate_response(self, state: QueryState) -> QueryState:
        """Validate LLM response quality against original user query"""
        logger.info("Validating LLM response quality against original user query")
        
        response = state["llm_response"].get("response", "")
        original_query = state["original_query"]
        individual_queries = state["individual_queries"]
        chunks = state["current_chunks"]
        
        # Quality metrics - use dynamic processing evaluation if available
        if "evaluation" in state["llm_response"]:
            # Use evaluation from dynamic processing
            evaluation = state["llm_response"]["evaluation"]
            quality_metrics = {
                "response_length": len(response),
                "has_content": evaluation.get("quality_metrics", {}).get("has_content", False),
                "mentions_original_query": evaluation.get("quality_metrics", {}).get("addresses_query", False),
                "mentions_individual_queries": sum(1 for q in individual_queries if any(word.lower() in response.lower() for word in q.split())),
                "uses_chunks": evaluation.get("quality_metrics", {}).get("uses_chunks", False),
                "confidence_score": evaluation.get("confidence_score", 0.0),
                "is_comprehensive": evaluation.get("is_comprehensive", False),
                "has_specific_details": evaluation.get("quality_metrics", {}).get("has_specific_details", False),
                "no_generic_references": evaluation.get("quality_metrics", {}).get("no_generic_references", False)
            }
        else:
            # Fallback to original validation
            quality_metrics = {
                "response_length": len(response),
                "has_content": len(response.strip()) > 50,
                "mentions_original_query": any(word.lower() in response.lower() for word in original_query.split()),
                "mentions_individual_queries": sum(1 for q in individual_queries if any(word.lower() in response.lower() for word in q.split())),
                "uses_chunks": len(state["llm_response"].get("chunks_used", [])) > 0,
                "confidence_score": 0.0
            }
        
        # Calculate confidence score using quality weights
        confidence = 0.0
        
        if quality_metrics["has_content"]:
            confidence += QUALITY_WEIGHTS["has_content"]
        if quality_metrics["mentions_original_query"]:
            confidence += QUALITY_WEIGHTS["mentions_query"]
        if quality_metrics["uses_chunks"]:
            confidence += QUALITY_WEIGHTS["uses_chunks"]
        if quality_metrics["response_length"] > MIN_RESPONSE_LENGTH:
            confidence += QUALITY_WEIGHTS["response_length"]
        
        # Bonus for addressing multiple questions
        if len(individual_queries) > 1:
            questions_addressed_ratio = quality_metrics["mentions_individual_queries"] / len(individual_queries)
            confidence += questions_addressed_ratio * 0.2  # Bonus for addressing multiple questions
        
        quality_metrics["confidence_score"] = confidence
        
        # Check if response is adequate
        is_adequate = (
            quality_metrics["has_content"] and
            quality_metrics["uses_chunks"] and
            confidence >= MIN_CONFIDENCE_SCORE
        )
        
        state["response_quality"] = {
            **quality_metrics,
            "is_adequate": is_adequate,
            "needs_human_review": confidence < HUMAN_REVIEW_THRESHOLD or not is_adequate
        }
        
        logger.info(f"Response validation - Confidence: {confidence:.2f}, Adequate: {is_adequate}, Questions addressed: {quality_metrics['mentions_individual_queries']}/{len(individual_queries)}")
        
        return state
    
    def _decide_next_step(self, state: QueryState) -> str:
        """Decide next step based on validation results"""
        quality = state["response_quality"]
        iteration = state["iteration"]
        
        # If response is adequate and we're not in first iteration, finalize
        if quality["is_adequate"] and iteration > 0:
            logger.info("Response is adequate, proceeding to finalization")
            return "finalize"
        
        # If we've reached max iterations, finalize
        if iteration >= state["max_iterations"]:
            logger.info("Max iterations reached, finalizing response")
            return "finalize"
        
        # If response needs human review, ask for human input
        if quality["needs_human_review"]:
            logger.info("Response needs human review")
            return "human_decision"
        
        # Otherwise, retry with more chunks
        logger.info("Retrying with increased chunk size")
        return "retry"
    
    def _human_decision(self, state: QueryState) -> QueryState:
        """Get human decision on whether to process or requery"""
        response = state["llm_response"].get("response", "")
        confidence = state["response_quality"]["confidence_score"]
        iteration = state["iteration"]
        
        # Create human-readable summary
        summary = f"""
Query: {state['query']}
Iteration: {iteration + 1}/{state['max_iterations']}
Confidence Score: {confidence:.2f}
Chunks Used: {len(state['llm_response'].get('chunks_used', []))}

Response Preview:
{response[:300]}{'...' if len(response) > 300 else ''}

Please decide:
- Type 'process' to accept this response
- Type 'requery' to try again with more context
- Type 'end' to stop processing
"""
        
        # In a real implementation, this would be an interrupt for human input
        # For now, we'll simulate the decision based on confidence
        if confidence >= 0.8:
            user_decision = "process"
        elif confidence >= MIN_CONFIDENCE_SCORE:
            user_decision = "requery"
        else:
            user_decision = "requery"
        
        logger.info(f"Human decision: {user_decision}")
        state["user_decision"] = user_decision
        
        return state
    
    def _decide_from_human_input(self, state: QueryState) -> str:
        """Decide next step based on human input"""
        decision = state["user_decision"]
        
        if decision == "process":
            return "process"
        elif decision == "requery":
            return "requery"
        else:  # "end"
            return "end"
    
    def _increase_chunk_size(self, state: QueryState) -> QueryState:
        """Increase chunk size for next iteration"""
        current_size = state["current_chunk_size"]
        new_size = current_size + CHUNK_SIZE_INCREMENT
        
        logger.info(f"Increasing chunk size from {current_size} to {new_size}")
        
        state["current_chunk_size"] = new_size
        state["iteration"] += 1
        
        return state
    
    def _decide_retry_or_end(self, state: QueryState) -> str:
        """Decide whether to retry or end processing"""
        iteration = state["iteration"]
        max_iterations = state["max_iterations"]
        
        if iteration < max_iterations:
            logger.info(f"Retrying with iteration {iteration + 1}")
            return "retry"
        else:
            logger.info("Max iterations reached, ending processing")
            return "end"
    
    def _combine_responses(self, state: QueryState) -> QueryState:
        """Finalize the comprehensive response (already combined)"""
        logger.info("Finalizing comprehensive response")
        
        try:
            # The response is already comprehensive since we processed all questions together
            if state["llm_response"]:
                logger.info("Response already comprehensive, no combination needed")
            else:
                logger.warning("No response available, creating fallback")
                state["llm_response"] = {
                    "response": "No response generated",
                    "chunks_used": []
                }
            
            logger.info(f"Finalized response with {len(state['llm_response'].get('chunks_used', []))} referenced chunks")
            
        except Exception as e:
            logger.error(f"Error finalizing response: {str(e)}")
            # Fallback to current response
            if not state["llm_response"]:
                state["llm_response"] = {
                    "response": "Error finalizing response",
                    "chunks_used": []
                }
        
        return state
    
    def _finalize_response(self, state: QueryState) -> QueryState:
        """Finalize the response and collect images/tables"""
        logger.info("Finalizing response")
        
        try:
            # Collect images and tables from used chunks (keep original behavior)
            images, tables = self._collect_media_from_chunks(state)
            
            # Validate and log suggested images for tracking purposes (doesn't change response)
            self._validate_suggested_images(state)
            
            logger.info(f"Returning {len(images)} images from used chunks (original behavior maintained)")
            
            # Create final response with dynamic processing information
            final_response = QueryResponse(
                success=True,
                message="Query processed successfully with dynamic information gathering",
                response=state["llm_response"].get("response", ""),
                chunks_used=state["llm_response"].get("chunks_used", []),
                images=images,
                tables=tables,
                processing_time=calculate_processing_time(state["start_time"]),
                suggested_images=state["llm_response"].get("suggested_images", []),
                images_used_for_response=state["llm_response"].get("images_used_for_response", [])
            )
            
            # Add dynamic processing metadata if available
            if "processing_stages" in state["llm_response"]:
                final_response.processing_stages = state["llm_response"]["processing_stages"]
            if "confidence_score" in state["llm_response"]:
                final_response.confidence_score = state["llm_response"]["confidence_score"]
            if "initial_chunks_count" in state["llm_response"]:
                final_response.initial_chunks_count = state["llm_response"]["initial_chunks_count"]
            if "total_chunks_count" in state["llm_response"]:
                final_response.total_chunks_count = state["llm_response"]["total_chunks_count"]
            if "collection_used" in state["llm_response"]:
                final_response.collection_used = state["llm_response"]["collection_used"]
            
            state["final_response"] = final_response
            state["images"] = images
            state["tables"] = tables
            state["processing_time"] = final_response.processing_time
            
            logger.info(f"Response finalized - {len(images)} images, {len(tables)} tables")
            
        except Exception as e:
            logger.error(f"Error finalizing response: {str(e)}")
            # Create error response
            state["final_response"] = QueryResponse(
                success=False,
                message=f"Error finalizing response: {str(e)}",
                response="I apologize, but I encountered an error while finalizing your response.",
                chunks_used=[],
                images=[],
                tables=[],
                processing_time=calculate_processing_time(state["start_time"]),
                suggested_images=[],
                images_used_for_response=[]
            )
        
        return state
    
    def _collect_media_from_chunks(self, state: QueryState) -> tuple[List[ImageData], List[str]]:
        """Collect images from all chunks and tables from referenced chunks"""
        images = []
        tables = []
        seen_image_filenames = set()
        
        # Get current chunks and chunks referenced by LLM
        current_chunks = state.get("current_chunks", [])
        chunks_used = state["llm_response"].get("chunks_used", [])
        
        if not isinstance(current_chunks, list):
            logger.warning(f"current_chunks is not a list: {type(current_chunks)}")
            current_chunks = []
        
        logger.info(f"Collecting images from all {len(current_chunks)} chunks (LLM had access to all images)")
        logger.info(f"Collecting tables only from referenced chunks: {chunks_used}")
        
        # Collect images from ALL chunks (since LLM saw all images)
        for i, chunk in enumerate(current_chunks):
            if not isinstance(chunk, dict):
                logger.warning(f"Skipping non-dict chunk: {type(chunk)}")
                continue
                
            chunk_heading = chunk.get("metadata", {}).get("heading", f"Chunk {i+1}")
            
            # Always collect images from this chunk
            embedded_images = chunk.get("embedded_images", [])
            if isinstance(embedded_images, list) and embedded_images:
                logger.info(f"Found {len(embedded_images)} images in chunk '{chunk_heading}'")
                for img in embedded_images:
                    if hasattr(img, 'filename'):
                        filename = img.filename
                    else:
                        filename = str(img)
                    
                    if filename not in seen_image_filenames:
                        images.append(img)
                        seen_image_filenames.add(filename)
                        logger.info(f"Added image: {filename} from chunk '{chunk_heading}'")
                    else:
                        logger.info(f"Skipped duplicate image: {filename}")
            
            # Only collect tables from referenced chunks
            chunk_is_referenced = False
            for referenced_heading in chunks_used:
                if (chunk_heading.lower() == referenced_heading.lower() or
                    chunk_heading.lower() in referenced_heading.lower() or
                    referenced_heading.lower() in chunk_heading.lower() or
                    referenced_heading.lower() == f"chunk {i+1}"):
                    chunk_is_referenced = True
                    break
            
            if chunk_is_referenced:
                chunk_tables = chunk.get("tables", [])
                if isinstance(chunk_tables, list):
                    tables.extend(chunk_tables)
                    logger.info(f"Added {len(chunk_tables)} tables from referenced chunk '{chunk_heading}'")
        
        # Remove duplicate tables
        tables = list(set(tables))
        
        logger.info(f"Final collection: {len(images)} embedded images and {len(tables)} tables")
        
        return images, tables
    
    def _validate_suggested_images(self, state: QueryState) -> None:
        """Validate and log LLM suggested images for tracking purposes (doesn't change response)"""
        suggested_images = state["llm_response"].get("suggested_images", [])
        
        if not suggested_images:
            logger.info("No suggested images to validate")
            return
        
        logger.info(f"Validating {len(suggested_images)} suggested images: {suggested_images}")
        
        # Collect all available images from current chunks for validation
        # Use the same numbering logic as the LLM service
        all_available_images = set()
        current_chunks = state.get("current_chunks", [])
        
        # Get the image reference mapping from LLM response to know the correct numbering
        image_reference_mapping = state["llm_response"].get("image_reference_mapping", {})
        
        # Add all numbered image names from the mapping
        for numbered_name in image_reference_mapping.keys():
            all_available_images.add(numbered_name)
        
        logger.info(f"Available numbered images for validation: {sorted(all_available_images)}")
        logger.info(f"Image reference mapping: {image_reference_mapping}")
        
        # Validate suggested images (for logging purposes only)
        valid_suggestions = 0
        for suggested_image_name in suggested_images:
            if suggested_image_name in all_available_images:
                valid_suggestions += 1
                logger.info(f"✓ Valid suggestion: {suggested_image_name}")
            else:
                logger.warning(f"✗ Invalid suggestion: {suggested_image_name} (not found in available numbered images: {sorted(all_available_images)})")
        
        logger.info(f"Suggestion validation: {valid_suggestions}/{len(suggested_images)} valid suggestions")
