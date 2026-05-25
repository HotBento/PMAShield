"""
Core abstract base classes for MCPEval framework
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading
import time


class RateLimitError(Exception):
    """Raised when an API rate limit is exceeded."""


class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Limits the number of requests sent within any 60-second window to *rpm*.
    Callers block inside ``wait_if_needed()`` until a slot opens.
    """

    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self._lock = threading.Lock()
        self._timestamps: List[float] = []

    def wait_if_needed(self) -> None:
        """Block until a request slot is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                # Discard timestamps older than 60 s
                self._timestamps = [t for t in self._timestamps if now - t < 60.0]
                if len(self._timestamps) < self.rpm:
                    self._timestamps.append(now)
                    return
                # The oldest timestamp releases a slot after 60 s
                sleep_until = self._timestamps[0] + 60.0

            wait = sleep_until - time.monotonic()
            if wait > 0:
                time.sleep(wait + 0.05)  # small buffer to avoid off-by-one


@dataclass
class ToolFunction:
    """Tool function definition"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema format
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


@dataclass
class SelectionResult:
    """Selection result"""
    selected_tools: List[str]  # Selected tool names
    prompt: str  # Prompt used
    timestamp: datetime = None  # Timestamp when result was generated
    prompt_tokens: int = 0  # Input tokens used
    completion_tokens: int = 0  # Output tokens used
    total_tokens: int = 0  # Total tokens used
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(
        self,
        rpm_limit: Optional[int] = None,
        max_retries: int = 3,
    ):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_requests = 0
        self._rate_limiter = RateLimiter(rpm_limit) if rpm_limit else None
        self._max_retries = max_retries

    def select_tools(self, tools: List[Dict[str, Any]], prompt: str) -> SelectionResult:
        """
        Select tools based on prompt, with optional rate limiting and retry.

        Subclasses implement ``_select_tools_impl``; this wrapper handles:
        * Proactive rate limiting (``rpm_limit``)
        * Exponential-backoff retry on ``RateLimitError``
        """
        from pma_shield.logger import logger

        if self._rate_limiter:
            self._rate_limiter.wait_if_needed()

        for attempt in range(self._max_retries + 1):
            try:
                return self._select_tools_impl(tools, prompt)
            except RateLimitError:
                if attempt >= self._max_retries:
                    raise
                wait = min(30 * (2 ** attempt), 300)  # 30 s, 60 s, 120 s, …, max 300 s
                logger.warning(
                    "Rate limit hit (attempt {}/{}), retrying in {} s …",
                    attempt + 1, self._max_retries, wait,
                )
                time.sleep(wait)

        # Unreachable, but satisfies type checkers
        raise RateLimitError("Max retries exceeded")

    @abstractmethod
    def _select_tools_impl(self, tools: List[Dict[str, Any]], prompt: str) -> SelectionResult:
        """
        Actual API call — implemented by each provider subclass.

        Subclasses should catch their API-specific rate-limit exception and
        re-raise it as ``RateLimitError``.
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name"""
        pass
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get total token usage statistics"""
        return {
            "total_requests": self.total_requests,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }


class ToolSource(ABC):
    """Abstract base class for tool sources"""
    
    @abstractmethod
    def fetch_all_tools(self) -> List[Dict[str, Any]]:
        """
        Fetch all tools
        
        Returns:
            List[Dict]: List of tools
        """
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Get source name"""
        pass
    
    @abstractmethod
    def get_source_category(self) -> str:
        """Get source category (e.g., 'weather', 'web_search', etc.)"""
        pass
    
    def count_tools(self) -> int:
        """
        Count the number of tools
        
        Returns:
            int: Number of tools
        """
        tools = self.fetch_all_tools()
        return len(tools)


class ToolFilter(ABC):
    """Abstract base class for tool filters"""
    
    @abstractmethod
    def filter(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter tool list
        
        Args:
            tools: Original tool list
            
        Returns:
            List[Dict]: Filtered tool list
        """
        pass
    
    @abstractmethod
    def get_filter_name(self) -> str:
        """Get filter name"""
        pass


class ToolSelector(ABC):
    """Abstract base class for tool selectors"""
    
    @abstractmethod
    def choose(self, tools: List[Dict[str, Any]], prompt: str) -> List[str]:
        """
        Choose tools
        
        Args:
            tools: Tool list
            prompt: Prompt
            
        Returns:
            List[str]: List of selected tool names
        """
        pass


class RankingSystem(ABC):
    """Abstract base class for ranking systems"""
    
    @abstractmethod
    def register_tool(self, tool_id: str, initial_rating: float = 1500.0) -> None:
        """Register a tool"""
        pass
    
    @abstractmethod
    def complete_match(self, match_id: str, tool_ids: List[str], winners: List[str]) -> Dict[str, float]:
        """
        Complete a match
        
        Args:
            match_id: Match ID
            tool_ids: List of participating tool IDs
            winners: List of winner IDs
            
        Returns:
            Dict[str, float]: Rating changes for each tool
        """
        pass
    
    @abstractmethod
    def get_ranking(self, top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get ranking"""
        pass
    
    @abstractmethod
    def get_tool_stats(self, tool_id: str) -> Dict[str, Any]:
        """Get tool statistics"""
        pass
