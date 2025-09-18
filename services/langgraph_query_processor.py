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
from utils.image_processor import ImageProcessor
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
        self.image_processor = ImageProcessor()
        
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
        
        logger.info(f"=== STARTING LANGGRAPH QUERY PROCESSING ===")
        logger.info(f"Query: {request.query}")
        logger.info(f"PDF: {request.pdf_name}")
        logger.info(f"Top-k: {request.top_k}")
        
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
            logger.info(f"Executing LangGraph workflow with config: {config}")
            
            final_state = await self.graph.ainvoke(initial_state, config=config)
            
            logger.info(f"=== LANGGRAPH QUERY PROCESSING COMPLETED ===")
            logger.info(f"Total processing time: {final_state['processing_time']}")
            logger.info(f"Final response success: {final_state['final_response'].success}")
            logger.info(f"Images in final response: {len(final_state['final_response'].images)}")
            logger.info(f"Suggested images: {final_state['final_response'].suggested_images}")
            
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
        logger.info(f"=== STAGE: INITIALIZE QUERY ===")
        logger.info(f"Iteration: {state['iteration'] + 1}")
        logger.info(f"Collection: {state['collection_name']}")
        
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
        logger.info(f"=== STAGE: GENERATE LLM RESPONSE ===")
        logger.info(f"Processing query: {state['original_query']}")
        logger.info(f"Available chunks: {len(state.get('current_chunks', []))}")
        
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
        logger.info(f"=== STAGE: FINALIZE RESPONSE ===")
        logger.info(f"LLM response keys: {list(state['llm_response'].keys())}")
        logger.info(f"Response text length: {len(state['llm_response'].get('response', ''))}")
        logger.info(f"Chunks used: {state['llm_response'].get('chunks_used', [])}")
        logger.info(f"Suggested images: {state['llm_response'].get('suggested_images', [])}")
        logger.info(f"Image reference mapping: {state['llm_response'].get('image_reference_mapping', {})}")
        
        try:
            # Collect all available images and tables from used chunks
            all_images, tables = self._collect_media_from_chunks(state)
            
            # Validate and log suggested images for tracking purposes
            self._validate_suggested_images(state)
            
            # Option: Return only LLM-suggested images or all images
            suggested_images = state["llm_response"].get("suggested_images", [])
            image_reference_mapping = state["llm_response"].get("image_reference_mapping", {})
            
            if suggested_images:
                # LLM service already did STRICT filtering - only images referenced in response text
                suggested_image_data = []
                logger.info(f"🎯 STRICT FILTERING: LLM service found {len(suggested_images)} images actually referenced in response text")
                
                for img in all_images:
                    img_filename = img.filename if hasattr(img, 'filename') else str(img)
                    # Check if this image corresponds to a referenced image
                    for suggested_ref in suggested_images:
                        expected_filename = f"{suggested_ref.replace(' ', '_')}.jpg"
                        if img_filename == expected_filename:
                            suggested_image_data.append(img)
                            logger.info(f"✅ Including referenced image: {img_filename}")
                            break
                
                logger.info(f"🎯 FINAL RESULT: {len(suggested_image_data)} images actually referenced in response")
                logger.info(f"Referenced images: {suggested_images}")
                logger.info(f"Returned image filenames: {[img.filename if hasattr(img, 'filename') else str(img) for img in suggested_image_data]}")
                
                # Images are already labeled from LLM service - just return the pre-labeled ones
                logger.info("📋 Using pre-labeled images that LLM already saw...")
                logger.info(f"✅ Returning {len(suggested_image_data)} pre-labeled images")
                
                images = suggested_image_data
            else:
                logger.info(f"❌ No images referenced in response text - returning empty list")
                images = []
            
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
            
            logger.info(f"=== RESPONSE FINALIZATION COMPLETE ===")
            logger.info(f"Final response contains:")
            logger.info(f"- Images: {len(images)}")
            logger.info(f"- Tables: {len(tables)}")
            logger.info(f"- Suggested images: {final_response.suggested_images}")
            logger.info(f"- Images used for response: {final_response.images_used_for_response}")
            logger.info(f"Final image filenames in response: {[img.filename if hasattr(img, 'filename') else str(img) for img in images]}")
            
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
        """Collect pre-labeled images and tables from processed chunks"""
        images = []
        tables = []
        seen_image_filenames = set()
        
        # Get the chunk data that was processed by LLM service (contains pre-labeled images)
        processed_chunks = state["llm_response"].get("processed_chunks", [])
        current_chunks = state.get("current_chunks", [])
        chunks_used = state["llm_response"].get("chunks_used", [])
        
        # Check if LLM service provided chunk_data_with_images in the response
        chunk_data_with_images = state["llm_response"].get("chunk_data_with_images", [])
        
        if chunk_data_with_images:
            logger.info(f"🎯 Using pre-processed chunk data with labeled images from LLM service")
            logger.info(f"Found {len(chunk_data_with_images)} chunks with pre-labeled images")
            chunks_to_search = chunk_data_with_images
        elif processed_chunks and isinstance(processed_chunks, list):
            chunks_to_search = processed_chunks
            logger.info(f"Using {len(processed_chunks)} processed chunks from LLM service")
        else:
            chunks_to_search = current_chunks
            logger.info(f"Using {len(current_chunks)} current chunks (fallback)")
        
        if not isinstance(chunks_to_search, list):
            logger.warning(f"chunks_to_search is not a list: {type(chunks_to_search)}")
            chunks_to_search = []
        
        # Check if we have all chunks that LLM processed
        image_reference_mapping = state["llm_response"].get("image_reference_mapping", {})
        expected_image_count = len(image_reference_mapping)
        
        logger.info(f"Collecting pre-labeled images from {len(chunks_to_search)} chunks")
        logger.info(f"Expected {expected_image_count} images based on LLM mapping: {list(image_reference_mapping.values())}")
        logger.info(f"Collecting tables only from referenced chunks: {chunks_used}")
        
        # Collect images from ALL chunks (since LLM saw all images)
        for i, chunk in enumerate(chunks_to_search):
            if not isinstance(chunk, dict):
                logger.warning(f"Skipping non-dict chunk: {type(chunk)}")
                continue
                
            # Handle different chunk formats
            if "heading" in chunk:
                # This is chunk_data_with_images format (pre-labeled)
                chunk_heading = chunk.get("heading", f"Chunk {i+1}")
                chunk_images = chunk.get("images", [])
            else:
                # This is regular chunk format
                chunk_heading = chunk.get("metadata", {}).get("heading", f"Chunk {i+1}")
                chunk_images = chunk.get("embedded_images", [])
            
            # Always collect images from this chunk
            if isinstance(chunk_images, list) and chunk_images:
                logger.info(f"Found {len(chunk_images)} images in chunk '{chunk_heading}'")
                for img in chunk_images:
                    # Handle pre-labeled images (dict format) or regular ImageData objects
                    if isinstance(img, dict):
                        original_filename = img.get('filename', str(img))
                        # Convert dict to ImageData for consistency
                        from models.schemas import ImageData
                        img_data = ImageData(
                            filename=img['filename'],
                            data=img['data'],
                            mime_type=img['mime_type'],
                            size=img['size']
                        )
                    elif hasattr(img, 'filename'):
                        original_filename = img.filename
                        img_data = img
                    else:
                        original_filename = str(img)
                        img_data = img
                    
                    if original_filename not in seen_image_filenames:
                        # Find the numbered name for this image from the mapping
                        numbered_name = None
                        for num_name, mapped_filename in image_reference_mapping.items():
                            if mapped_filename == original_filename:
                                numbered_name = num_name
                                break
                        
                        if numbered_name:
                            # For pre-labeled images, they should already have the correct filename
                            if original_filename.startswith("labeled_"):
                                # Image is already pre-labeled, use as-is
                                images.append(img_data)
                                logger.info(f"✓ PRE-LABELED: Using {original_filename} from chunk '{chunk_heading}'")
                            else:
                                # Create numbered filename (fallback case)
                                numbered_filename = f"{numbered_name.replace(' ', '_')}.jpg"
                                if hasattr(img_data, 'filename'):
                                    numbered_img = ImageData(
                                        filename=numbered_filename,
                                        data=img_data.data,
                                        mime_type=img_data.mime_type,
                                        size=img_data.size
                                    )
                                    images.append(numbered_img)
                                    logger.info(f"✓ CONVERTED: {original_filename} -> {numbered_filename} from chunk '{chunk_heading}'")
                                else:
                                    images.append(img_data)
                                    logger.info(f"✓ FALLBACK: Using original {original_filename}")
                        else:
                            # No mapping found, use original
                            images.append(img_data)
                            logger.warning(f"❌ NO MAPPING: {original_filename} not found in mapping {list(image_reference_mapping.keys())}, using original filename")
                        
                        seen_image_filenames.add(original_filename)
                    else:
                        logger.info(f"Skipped duplicate image: {original_filename}")
            
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
                # Handle different chunk formats for tables
                if "tables" in chunk:
                    chunk_tables = chunk.get("tables", [])
                else:
                    chunk_tables = chunk.get("tables", [])
                    
                if isinstance(chunk_tables, list):
                    tables.extend(chunk_tables)
                    logger.info(f"Added {len(chunk_tables)} tables from referenced chunk '{chunk_heading}'")
        
        # Remove duplicate tables
        tables = list(set(tables))
        
        logger.info(f"Final collection: {len(images)} embedded images and {len(tables)} tables")
        
        # Check if we found all images that LLM had access to
        image_reference_mapping = state["llm_response"].get("image_reference_mapping", {})
        expected_images = set(image_reference_mapping.values())  # All filenames LLM knew about
        found_images = {img.filename if hasattr(img, 'filename') else str(img) for img in images}
        
        missing_images = expected_images - found_images
        if missing_images:
            logger.warning(f"Missing {len(missing_images)} images that LLM had access to: {missing_images}")
            logger.info("Searching all chunks for missing images...")
            
            # Search all chunks more thoroughly for missing images
            for chunk in chunks_to_search:
                if not isinstance(chunk, dict):
                    continue
                    
                chunk_heading = chunk.get("metadata", {}).get("heading", "Unknown")
                embedded_images = chunk.get("embedded_images", [])
                
                if isinstance(embedded_images, list):
                    for img in embedded_images:
                        if hasattr(img, 'filename'):
                            filename = img.filename
                        elif isinstance(img, dict) and 'filename' in img:
                            filename = img['filename']
                        else:
                            filename = str(img)
                        
                        if filename in missing_images and filename not in seen_image_filenames:
                            # Find the numbered name for this missing image
                            numbered_name = None
                            for num_name, mapped_filename in image_reference_mapping.items():
                                if mapped_filename == filename:
                                    numbered_name = num_name
                                    break
                            
                            if numbered_name:
                                # Create numbered image
                                if hasattr(img, 'filename'):
                                    from models.schemas import ImageData
                                    numbered_img = ImageData(
                                        filename=f"{numbered_name}.jpg",
                                        data=img.data,
                                        mime_type=img.mime_type,
                                        size=img.size
                                    )
                                    images.append(numbered_img)
                                    logger.info(f"Found missing image: {filename} as '{numbered_name}.jpg' in chunk '{chunk_heading}'")
                                else:
                                    numbered_img = img.copy() if isinstance(img, dict) else img
                                    if isinstance(numbered_img, dict):
                                        numbered_img['filename'] = f"{numbered_name}.jpg"
                                    images.append(numbered_img)
                                    logger.info(f"Found missing image: {filename} as '{numbered_name}.jpg' in chunk '{chunk_heading}'")
                            else:
                                images.append(img)
                                logger.info(f"Found missing image: {filename} (no numbered mapping) in chunk '{chunk_heading}'")
                            
                            seen_image_filenames.add(filename)
                            missing_images.remove(filename)
            
            if missing_images:
                logger.error(f"Still missing images after thorough search: {missing_images}")
            else:
                logger.info("All missing images found and added")
        
        logger.info(f"Final collection after missing image search: {len(images)} embedded images and {len(tables)} tables")
        
        return images, tables
    
    def _filter_relevant_suggestions(self, suggested_images: List[str], state: QueryState) -> List[str]:
        """Filter suggested images to remove irrelevant ones like error codes or emergency symbols"""
        try:
            query = state.get("original_query", "").lower()
            image_reference_mapping = state["llm_response"].get("image_reference_mapping", {})
            
            # Keywords that indicate the query is about errors, safety, or warnings
            error_safety_keywords = [
                "error", "warning", "alarm", "emergency", "safety", "fault", "alert", 
                "code", "troubleshoot", "problem", "issue", "malfunction"
            ]
            
            # Check if the query is about errors or safety
            is_error_safety_query = any(keyword in query for keyword in error_safety_keywords)
            
            filtered_suggestions = []
            
            for suggested_image in suggested_images:
                # Get the original filename to analyze
                original_filename = image_reference_mapping.get(suggested_image, "")
                
                # Check if this appears to be an error/warning/emergency image
                is_error_warning_image = any(keyword in original_filename.lower() for keyword in [
                    "error", "warning", "alarm", "emergency", "danger", "caution", 
                    "symbol", "sign", "code", "alert", "fault", "prohibited", "stop",
                    "hazard", "risk", "attention", "notice", "indication"
                ])
                
                # Also check chunk headings for context
                chunk_context = ""
                current_chunks = state.get("current_chunks", [])
                for chunk in current_chunks:
                    if isinstance(chunk, dict):
                        embedded_images = chunk.get("embedded_images", [])
                        for img in embedded_images:
                            img_filename = img.filename if hasattr(img, 'filename') else str(img)
                            if img_filename == original_filename:
                                chunk_heading = chunk.get("metadata", {}).get("heading", "").lower()
                                chunk_context = chunk_heading
                                break
                
                is_error_context = any(keyword in chunk_context for keyword in [
                    "error", "warning", "alarm", "emergency", "safety", "sign", "symbol", "prohibited",
                    "indication", "alert", "fault", "code", "hazard", "caution", "danger", "stop"
                ])
                
                # Decision logic
                if is_error_warning_image or is_error_context:
                    if is_error_safety_query:
                        # Query is about errors/safety, so include error/warning images
                        filtered_suggestions.append(suggested_image)
                        logger.info(f"Including error/safety image '{suggested_image}' for error/safety query")
                    else:
                        # Query is not about errors/safety, exclude error/warning images
                        logger.info(f"🚫 FILTERED OUT: '{suggested_image}' - emergency/warning image for procedural query")
                        logger.info(f"   Reason: chunk='{chunk_context}', filename='{original_filename[:50]}...'")
                        logger.info(f"   Query type: procedural, Image type: emergency/warning")
                else:
                    # Regular procedural image, include it
                    filtered_suggestions.append(suggested_image)
                    logger.info(f"✅ INCLUDED: '{suggested_image}' - procedural image relevant to query")
                    logger.info(f"   Context: chunk='{chunk_context}', filename='{original_filename[:50]}...'")
            
            logger.info(f"Image filtering: {len(suggested_images)} suggested -> {len(filtered_suggestions)} filtered")
            logger.info(f"Query type: {'error/safety' if is_error_safety_query else 'procedural'}")
            
            return filtered_suggestions
            
        except Exception as e:
            logger.error(f"Error filtering image suggestions: {str(e)}")
            return suggested_images  # Return original suggestions if filtering fails
    
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
