# services/agents/tools.py
from __future__ import annotations
import json
import asyncio
from langchain_core.messages import BaseMessage
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import Query, ModelProvider
from .runtime import AgentRuntime

# Will be set at app startup (via DI)
runtime: Optional[AgentRuntime] = None

# services/agents/tools.py (add near imports)
AGENT_TOP_K = 5
UI_PAGE_SIZE = 100

class AgentSearchQuery(BaseModel):
    """
    Defines a single search operation.
    For composite searches, multiple fields can be populated.
    Each field corresponds to a different search modality.
    """
    text: Optional[str] = Field(None, description="Semantic search query for general visual/textual content.")
    asr: Optional[str] = Field(None, description="Query for spoken words in the video audio (speech-to-text).")
    ocr: Optional[str] = Field(None, description="Query for visible text in video frames (optical character recognition).")
    obj: Optional[List[str]] = Field(None, description="List of objects to detect in video frames.")
    image: Optional[str] = Field(None, description="Base64 encoded image string for visual similarity search.")
    
    @field_validator('obj', mode='before')
    @classmethod
    def ensure_obj_list(cls, v):
        """Ensure obj is always a list or None."""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        return v
    
    @field_validator('*', mode='after')
    @classmethod
    def strip_whitespace(cls, v):
        """Clean up string inputs."""
        if isinstance(v, str):
            return v.strip() if v.strip() else None
        return v

    def has_any_query(self) -> bool:
        """Check if at least one search field is populated."""
        return any([
            self.text,
            self.asr,
            self.ocr,
            self.obj and len(self.obj) > 0,
            self.image
        ])

    def get_populated_fields(self) -> List[str]:
        """Return list of populated field names for debugging."""
        fields = []
        if self.text: fields.append("text")
        if self.asr: fields.append("asr")
        if self.ocr: fields.append("ocr")
        if self.obj: fields.append("obj")
        if self.image: fields.append("image")
        return fields


def _to_domain_query(agent_query: AgentSearchQuery, default_lang: str = "eng") -> Query:
    """
    Convert AgentSearchQuery to domain Query format.
    Fills in required fields with sensible defaults.
    """
    query_dict = agent_query.model_dump(exclude_unset=True)
    
    # Base structure with all required fields
    base = {
        "text": query_dict.get("text", ""),
        "asr": query_dict.get("asr", ""),
        "ocr": query_dict.get("ocr", ""),
        "origin": "",  # Mark queries as coming from agent
        "obj": query_dict.get("obj", []),
        "lang": default_lang,
        "image": query_dict.get("image", ""),
    }
    
    return Query(**base)
def make_tool_output(
    tool_name: str,
    type_: str,
    items: list | dict,
    meta: Optional[dict] = None
) -> dict:
    """
    Standardizes all tool outputs into a common structure.
    """
    if not isinstance(items, list):
        items = [items]
    return {
        "tool": tool_name,
        "ok": True,
        "type": type_,
        "items": items,
        "meta": meta or {}
    }


# ----------- Pydantic arg schemas -----------

class VideoSearchArgs(BaseModel):
    """Arguments for the video_search tool."""
    queries: List[Dict[str, Any]] = Field(
        ...,
        description='List of query dicts. Each query should have at least one field populated: text, asr, ocr, obj, or image.'
    )
    provider: Optional[Dict[str, Any]] = Field(
        None,
        description='Optional: ModelProvider configuration, e.g. {"clip":true,"siglip2":true,"beit3":true}. Defaults to all available models if not provided.'
    )
    page: int = Field(1, ge=1, description="1-based page number.")
    page_size: int = Field(100, ge=1, le=1000, description="Results per page.")


class ChainSearchArgs(BaseModel):
    """Arguments for the chain_search tool."""
    queries: List[Dict[str, Any]] = Field(
        ...,
        description='Ordered query dicts for sequential stages. Each represents a temporal step.'
    )
    provider: Optional[Dict[str, Any]] = Field(
        None,
        description='Optional: ModelProvider configuration, e.g. {"clip":true,"siglip2":true,"beit3":true}. Defaults to all available models if not provided.'
    )
    page: int = Field(1, ge=1)
    page_size: int = Field(100, ge=1, le=1000)
# services/agents/tools.py

# (Keep all other Pydantic schemas as they are)

class QueryRefinementArgs(BaseModel):
    """Arguments for the query_refinement tool."""
    initial_queries: List[Dict[str, Any]] = Field(
        ...,
        description="The initial list of queries to be refined."
    )
    refinement_strategy: str = Field(
        "web_enriched",
        description="Strategy for refinement: 'llm_only' (rewrite for precision) or 'web_enriched' (rewrite and add context from web searches)."
    )
class DuckDuckGoImageArgs(BaseModel):
    """Arguments for DuckDuckGo image search."""
    query: str = Field(..., description="Image search query, e.g., 'red ao dai near bun bo sign'.")
    max_results: int = Field(10, ge=1, le=50, description="Max number of images to return (<=50).")
    region: str = Field("wt-wt", description="Region code, e.g., 'us-en', 'vn-vi', 'wt-wt'.")
    safesearch: str = Field("moderate", description="Safe search level: 'off' | 'moderate' | 'strict'")


class DuckDuckGoWebArgs(BaseModel):
    """Arguments for DuckDuckGo web search."""
    query: str = Field(..., description="Web search query.")
    max_results: int = Field(10, ge=1, le=50)


class FrameInfoArgs(BaseModel):
    """Arguments for the frame_information tool."""
    video_id: str = Field(..., description="Video ID, e.g. 'L01_V001'.")
    frame_id: int = Field(..., ge=0, description="Frame index.")


class HistoryGetArgs(BaseModel):
    """Arguments for the history_get tool."""
    user_id: str = Field("anonymous", description="User ID.")
    limit: int = Field(10, ge=1, le=50, description="Number of entries.")


# ---------------------- Tools ----------------------

@tool(args_schema=VideoSearchArgs)
async def video_search_tool(
    queries: List[Dict[str, Any]], 
    provider: Optional[Dict[str, Any]] = None, 
    page: int = 1, 
    page_size: int = 100
):
    """
    Search across videos using embeddings with temporal boosting.
    
    Supports multiple query modalities:
    - text: General semantic search
    - ocr: Search for visible text in frames
    - asr: Search for spoken words in audio
    - obj: Detect specific objects
    - image: Visual similarity search
    
    Multiple fields can be used together for composite searches.
    """
    if runtime is None:
        raise RuntimeError("Agent runtime is not initialized.")
    
    # Validate and convert each query dictionary
    agent_queries = []
    for i, q in enumerate(queries):
        try:
            aq = AgentSearchQuery(**q)
            if not aq.has_any_query():
                print(f"Warning: Query {i} has no populated fields, skipping")
                continue
            agent_queries.append(aq)
            print(f"Query {i} populated fields: {aq.get_populated_fields()}")
        except Exception as e:
            print(f"Error parsing query {i}: {e}")
            raise ValueError(f"Invalid query at index {i}: {e}")
    
    if not agent_queries:
        raise ValueError("No valid queries provided")
    
    domain_queries = [_to_domain_query(q) for q in agent_queries]
    p = ModelProvider(**provider) if provider else ModelProvider(clip=True, siglip2=True, beit3=True)

    # Always fetch enough for UI (e.g., 100). Ignore incoming page_size for the agent path.
    full_results = await runtime.search(domain_queries, p, page=page, page_size=UI_PAGE_SIZE)
    agent_view = full_results[:AGENT_TOP_K]

    # Return BOTH: small slice for LLM + full list for UI
    return {"agent_view": [r.model_dump() for r in agent_view],
            "full":       [r.model_dump() for r in full_results]}


@tool(args_schema=ChainSearchArgs)
async def chain_search_tool(
    queries: List[Dict[str, Any]], 
    provider: Optional[Dict[str, Any]] = None, 
    page: int = 1, 
    page_size: int = 100
):
    """
    Chain search with sequential query stages using dynamic programming alignment.
    
    Each query in the list represents a temporal stage.
    Useful for finding sequences like: "person walking" → "person sitting"
    """
    if runtime is None:
        raise RuntimeError("Agent runtime is not initialized.")
    
    agent_queries = []
    for i, q in enumerate(queries):
        try:
            aq = AgentSearchQuery(**q)
            if not aq.has_any_query():
                raise ValueError(f"Query {i} has no populated fields")
            agent_queries.append(aq)
        except Exception as e:
            raise ValueError(f"Invalid query at index {i}: {e}")
    
    domain_queries = [_to_domain_query(q) for q in agent_queries]
    p = ModelProvider(**provider) if provider else ModelProvider(clip=True, siglip2=True, beit3=True)

    full_results = await runtime.chain_search(domain_queries, p, page=page, page_size=UI_PAGE_SIZE)
    agent_view = full_results[:AGENT_TOP_K]

    return {"agent_view": [r.model_dump() for r in agent_view],
            "full":       [r.model_dump() for r in full_results]}

# services/agents/tools.py

# services/agents/tools.py

# services/agents/tools.py

# services/agents/tools.py
# NEW: A helper function to safely parse JSON from LLM output
def _parse_llm_json(message: BaseMessage, key: str) -> str:
    """
    Safely parses a JSON string from an LLM message and extracts a value.
    Returns the raw content as fallback.
    """
    try:
        content = message.content.strip()
        # Handle cases where the LLM might wrap the JSON in markdown code blocks
        if content.startswith("```json"):
            content = content.strip("```json").strip("```")
        
        data = json.loads(content)
        return data.get(key, content)
    except (json.JSONDecodeError, KeyError, AttributeError):
        # If anything goes wrong, return the raw content
        return message.content.strip()
async def _query_refinement_core(
    *,
    initial_queries: List[Dict[str, Any]],
    refinement_strategy: str,
    llm
) -> Dict[str, Any]:
    if llm is None:
        raise RuntimeError("LLM was not provided to the query refinement core.")

    refinement_log = [f"Starting query refinement with strategy: '{refinement_strategy}'."]
    refined_queries = []

    # --- LLM Chains (no changes here) ---
    from langchain_core.prompts import ChatPromptTemplate
    ner_prompt = ChatPromptTemplate.from_template(
    "From the following text, extract the key entities (like people, places, or specific items) "
    "and any surrounding contextual keywords (like actions, descriptions, or related objects). "
    "Combine them into a concise search query. For example, if the text is 'show me videos of that person walking near the Eiffel Tower', "
    "the output should be 'person walking near Eiffel Tower'. If no key entities are found, respond with an empty string. Text: '{input}'"
)
    refine_prompt = ChatPromptTemplate.from_template(
        "Rewrite the following user query to be a more effective and precise video search query. "
        "Focus on clear, descriptive language. Original Query: '{input}'\n\n"
        "Your response MUST be a single JSON object with a single key 'refined_query'. "
        "For example: {{\"refined_query\": \"your rewritten query\"}}. "
        "Do not include any other text, explanation, or markdown."
    )
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query synthesizer. Merge web search context into a video query to make it more specific."),
        ("human", "Refined Video Query: '{refined_query}'\n\nWeb Context: '{web_context}'\n\n"
                  "Synthesize these into a single, improved video query. "
                  "Your response MUST be a single JSON object with a single key 'refined_query'. "
                  "For example: {{\"refined_query\": \"your synthesized query\"}}. "
                  "Do not include any other text, explanation, or markdown.")
    ])

    ner_chain = ner_prompt | llm
    refine_chain = refine_prompt | llm
    synthesis_chain = synthesis_prompt | llm

    for query_dict in initial_queries:
        # CHANGED: Standardize on the 'text' key for consistency.
        original_text = query_dict.get("text")
        if not original_text:
            refined_queries.append(query_dict)
            continue

        final_query_text = ""

        # --- START: PARALLEL EXECUTION LOGIC ---
        if refinement_strategy == "web_enriched":
            # Run base refinement and NER extraction in parallel
            refinement_log.append("Executing base refinement and NER check in parallel.")
            base_refinement_task = refine_chain.ainvoke({"input": original_text})
            ner_task = ner_chain.ainvoke({"input": original_text})

            # Wait for both independent tasks to complete
            refined_text_msg, proper_nouns_msg = await asyncio.gather(base_refinement_task, ner_task)
            
            refined_text = _parse_llm_json(refined_text_msg, "refined_query")
            proper_nouns = proper_nouns_msg.content.strip()
            refinement_log.append(f"LLM refined '{original_text}' to '{refined_text}'")

            if proper_nouns:
                refinement_log.append(f"Found proper nouns: '{proper_nouns}'. Searching web for context.")
                try:
                    web_results_dict = await duckduckgo_web_search_tool.ainvoke({"query": proper_nouns, "max_results": 3})
                    snippets = [r.get('snippet', '') for r in web_results_dict.get('items', []) if r.get('snippet')]
                    web_context = " ".join(snippets)
                    
                    if web_context:
                        synthesis_msg = await synthesis_chain.ainvoke({
                            "refined_query": refined_text, "web_context": web_context
                        })
                        final_query_text = _parse_llm_json(synthesis_msg, "refined_query")
                        refinement_log.append("Successfully synthesized web context into the final query.")
                    else:
                        refinement_log.append("Web search yielded no useful context.")
                        final_query_text = refined_text # Fallback to the already refined text
                except Exception as e:
                    refinement_log.append(f"Web search for context failed: {e}")
                    final_query_text = refined_text # Fallback on error
            else:
                refinement_log.append("No proper nouns found. Using LLM-only refinement.")
                final_query_text = refined_text
        else: # llm_only strategy
            refinement_log.append("Executing LLM-only refinement.")
            refined_text_msg = await refine_chain.ainvoke({"input": original_text})
            final_query_text = _parse_llm_json(refined_text_msg, "refined_query")
            refinement_log.append(f"LLM refined '{original_text}' to '{final_query_text}'")
        
        # --- END: PARALLEL EXECUTION LOGIC ---

        new_query = query_dict.copy()
        new_query["text"] = final_query_text
        refined_queries.append(new_query)

    refinement_log.append("Refinement process complete.")
    return make_tool_output(
        tool_name="query_refinement_tool",
        type_="refined_query",
        items=refined_queries,
        meta={"refinement_log": refinement_log}
    )
# services/agents/tools.py
from langchain_core.tools import StructuredTool

def make_query_refinement_tool(llm):
    async def _call(
        initial_queries: List[Dict[str, Any]],
        refinement_strategy: str = "web_enriched",
    ):
        return await _query_refinement_core(
            initial_queries=initial_queries,
            refinement_strategy=refinement_strategy,
            llm=llm,
        )

    return StructuredTool.from_function(
        name="query_refinement_tool",
        description=("Refines and transforms a list of queries to improve search "
                     "effectiveness, optionally enriching with web context."),
        args_schema=QueryRefinementArgs,
        coroutine=_call,
    )

# ============ External Search Tools ============

@tool(args_schema=DuckDuckGoImageArgs)
async def duckduckgo_image_search_tool(
    query: str,
    max_results: int = 10,
    region: str = "wt-wt",
    safesearch: str = "moderate",
):
    """
    Search images with DuckDuckGo (via LangChain wrapper).
    
    Useful for finding reference images to compare with video frames,
    or to gather visual context about objects, locations, or concepts.
    
    Returns image URLs, thumbnails, and source page links.
    """
    if runtime is None:
        raise RuntimeError("Agent runtime is not initialized.")

    items = await runtime.ddg_images(
        query=query,
        max_results=max_results,
        region=region,
        safesearch=safesearch,
        user_id="agent"
    )

    return make_tool_output(
    tool_name="duckduckgo_image_search_tool",
    type_="image_results",
    items=items,
    meta={"source": "duckduckgo", "count": len(items)}
)



@tool(args_schema=DuckDuckGoWebArgs)
async def duckduckgo_web_search_tool(query: str, max_results: int = 10):
    """
    Search the web with DuckDuckGo for text results.
    
    Useful for gathering context about:
    - Events or locations mentioned in queries
    - Background information on detected objects
    - Verification of OCR/ASR content
    
    Returns web page titles, URLs, and snippets.
    """
    if runtime is None:
        raise RuntimeError("Agent runtime is not initialized.")
    
    rows = await runtime.ddg_web(query=query, max_results=max_results, user_id="agent")
    
    return  make_tool_output(
    tool_name="duckduckgo_web_search_tool",
    type_="web_results",
    items=rows,
    meta={"source": "duckduckgo", "count": len(rows)}
)


# ============ Utility Tools ============

@tool(args_schema=FrameInfoArgs)
async def frame_information_tool(video_id: str, frame_id: int):
    """
    Fetch OCR/ASR text and detected objects for a specific frame.
    
    Returns detailed information including:
    - OCR: Text visible in the frame
    - ASR: Words spoken at this timestamp
    - Objects: Detected objects and their positions
    """
    if runtime is None:
        raise RuntimeError("Agent runtime is not initialized.")
    
    info = await runtime.frame_information(video_id, frame_id)
    return None if info is None else getattr(info, "model_dump", getattr(info, "dict"))()


@tool(args_schema=HistoryGetArgs)
async def history_get_tool(user_id: str = "anonymous", limit: int = 10):
    """Return recent query history for a user."""
    if runtime is None:
        raise RuntimeError("Agent runtime is not initialized.")
    
    return await runtime.history(user_id=user_id, limit=limit)