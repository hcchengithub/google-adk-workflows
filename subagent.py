"""
Agent Factories
---------------
Consolidates common travel agents and session-state demo agents for the ADK demos.
"""

import logging
import os
from collections.abc import Mapping
from typing import Any, Callable, Optional

from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai.types import Content, Part

# Load environment variables once so factories can rely on MODEL_NAME overrides.
load_dotenv()

# Route logs to INFO so the demo agents surface detailed execution insights.
logging.basicConfig(level=logging.INFO)

# Maximum preview length for logging Content parts.
MAX_PART_PREVIEW = 200
MAX_PART_COUNT = 3


# ---------------------------------------------------------------------------
# Generic helpers shared by the session-state demo agents
# ---------------------------------------------------------------------------

def _snapshot_content(content: Any) -> Any:
    """Return a readable preview of an ADK Content object."""
    try:
        role = getattr(content, "role", None)
        parts_summary = []
        parts = getattr(content, "parts", None) or []
        for index, part in enumerate(parts):
            if index >= MAX_PART_COUNT:
                parts_summary.append("… (truncated)")
                break
            text = getattr(part, "text", None)
            if isinstance(text, str):
                if len(text) > MAX_PART_PREVIEW:
                    parts_summary.append(text[:MAX_PART_PREVIEW] + "…")
                else:
                    parts_summary.append(text)
            else:
                parts_summary.append(str(part))
        return {"role": role, "parts": parts_summary}
    except Exception as error:  # pylint: disable=broad-except
        logging.debug("⚠️ [snapshot_content] Failed: %s", error, exc_info=True)
        return "<content unavailable>"


def _snapshot_state(state: Any) -> str:
    """Best-effort serialization for session state objects."""
    try:
        if state is None:
            return "{}"

        if isinstance(state, Mapping):
            snapshot = dict(state)
        elif hasattr(state, "to_dict") and callable(getattr(state, "to_dict")):
            snapshot = state.to_dict()
        elif hasattr(state, "__dict__"):
            snapshot = {k: v for k, v in vars(state).items() if not k.startswith("_")}
        else:
            snapshot = state

        if isinstance(snapshot, (dict, list, tuple, set)):
            return repr(snapshot)
        return str(snapshot)
    except Exception as error:  # pylint: disable=broad-except
        logging.debug("⚠️ [snapshot_state] Failed: %s", error, exc_info=True)
        return "<state unavailable>"


def _snapshot_context(ctx: CallbackContext) -> str:
    """Provide a concise snapshot of the callback context for logging."""
    try:
        if ctx is None:
            return "{}"

        snapshot: dict[str, Any] = {}
        agent_name = getattr(ctx, "agent_name", None)
        if agent_name:
            snapshot["agent_name"] = agent_name

        invocation_id = getattr(ctx, "invocation_id", None)
        if invocation_id:
            snapshot["invocation_id"] = invocation_id

        if hasattr(ctx, "state"):
            snapshot["state"] = _snapshot_state(ctx.state)

        user_content = getattr(ctx, "user_content", None)
        if user_content is not None:
            snapshot["user_content"] = _snapshot_content(user_content)

        return repr(snapshot)
    except Exception as error:  # pylint: disable=broad-except
        logging.debug("⚠️ [snapshot_context] Failed: %s", error, exc_info=True)
        return "<context unavailable>"


class FunctionAgent:
    """Decorator helper that wraps a before_model_callback into a fresh LlmAgent."""

    def __init__(self, name: str, model: str, **kwargs: Any):
        self.name = name
        self.model = model
        self.kwargs = kwargs

    def __call__(self, func: Callable[[CallbackContext, LlmRequest], Optional[LlmResponse]]) -> LlmAgent:
        return LlmAgent(
            name=self.name,
            model=self.model,
            before_model_callback=func,
            **self.kwargs,
        )


# ---------------------------------------------------------------------------
# Session-state demo agents (formerly example_agents.py)
# ---------------------------------------------------------------------------

SESSION_STATE_MODEL = os.getenv("SESSION_STATE_MODEL", "gemini-1.5-flash")


async def _process_initial_data(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Convert user text to an integer, persist it in session state, and echo the result."""
    logging.info("🌟 [process_initial_data] Starting…")
    logging.info("🧾 [process_initial_data] Context snapshot: %s", _snapshot_context(callback_context))
    logging.info("🔍 [process_initial_data] Raw request object: %r", llm_request)
    logging.info("🗃️ [process_initial_data] Session state before: %s", _snapshot_state(callback_context.state))

    try:
        latest_content = llm_request.contents[-1]
        latest_part = latest_content.parts[0]
        raw_text = latest_part.text
        logging.info("📝 [process_initial_data] Raw text from request: %s", raw_text)

        input_value = int(raw_text)
        logging.info("🌟 [process_initial_data] Parsed integer value: %s", input_value)

        processed_value = input_value * 2
        logging.info("✅ [process_initial_data] Processed value (input * 2): %s", processed_value)

        callback_context.state["processed_data"] = processed_value
        logging.info("💾 [process_initial_data] Stored processed_data in session state: %s", processed_value)
        logging.info("🗃️ [process_initial_data] Session state after: %s", _snapshot_state(callback_context.state))

        response_text = f"輸入值 {input_value} 已處理，結果 {processed_value} 已寫入會話狀態。"
        logging.info("🎉 [process_initial_data] Done. Response: %s", response_text)
        return LlmResponse(content=Content(role="model", parts=[Part(text=response_text)]))

    except (ValueError, IndexError, AttributeError) as error:
        logging.error("⚠️ [process_initial_data] Error processing input: %s", error)
        return LlmResponse(
            content=Content(
                role="model",
                parts=[Part(text="錯誤：請提供可以解析成整數的文字。")],
            )
        )
    except Exception as error:  # pylint: disable=broad-except
        logging.error("❌ [process_initial_data] Unexpected error: %s", error)
        return LlmResponse(
            content=Content(
                role="model",
                parts=[Part(text=f"處理輸入資料時發生錯誤: {error}")],
            )
        )


async def _use_and_finalize_data(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Read processed data from session state, finalize it, and respond."""
    logging.info("🌟 [use_and_finalize_data] Starting…")
    logging.info("🧾 [use_and_finalize_data] Context snapshot: %s", _snapshot_context(callback_context))
    logging.info("🔍 [use_and_finalize_data] Raw request object: %r", llm_request)
    logging.info("🗃️ [use_and_finalize_data] Session state snapshot: %s", _snapshot_state(callback_context.state))

    processed_data = callback_context.state.get("processed_data")
    logging.info("📦 [use_and_finalize_data] Retrieved processed_data: %s", processed_data)

    if processed_data is not None:
        final_result = processed_data + 10
        logging.info("✨ [use_and_finalize_data] Final result (processed_data + 10): %s", final_result)

        response_text = f"從會話狀態讀取資料 {processed_data}，最終結果為 {final_result}。"
        logging.info("🎉 [use_and_finalize_data] Done. Response: %s", response_text)
        return LlmResponse(content=Content(role="model", parts=[Part(text=response_text)]))

    logging.warning("⚠️ [use_and_finalize_data] 'processed_data' not found in session state.")
    return LlmResponse(
        content=Content(
            role="model",
            parts=[Part(text="錯誤：會話狀態中找不到 'processed_data'。")],
        )
    )


def create_process_initial_data_agent() -> LlmAgent:
    """Factory: returns the session-state processor agent."""

    async def handler(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
        return await _process_initial_data(callback_context, llm_request)

    return FunctionAgent(name="process_initial_data", model=SESSION_STATE_MODEL)(handler)


def create_use_and_finalize_data_agent() -> LlmAgent:
    """Factory: returns the finalizer agent that consumes session state."""

    async def handler(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
        return await _use_and_finalize_data(callback_context, llm_request)

    return FunctionAgent(name="use_and_finalize_data", model=SESSION_STATE_MODEL)(handler)


# ---------------------------------------------------------------------------
# Core travel agents used across the other demos (formerly subagent.py)
# ---------------------------------------------------------------------------


def create_flight_agent() -> LlmAgent:
    return LlmAgent(
        model=os.getenv("MODEL_NAME", "gemini-2.0-flash"),
        name="FlightAgent",
        description="Flight booking agent",
        instruction="""You are a flight booking agent.
        - You take any flight booking or confirmation request
        - You check for available flights based on user preferences
        - You return a valid JSON with flight booking and confirmation details, including flight number, departure and arrival times, airline, price, and status based on user request.
        - If the user does not provide specific details, make reasonable assumptions about the flight and booking details.
        """
    )


def create_hotel_agent() -> LlmAgent:
    return LlmAgent(
        model=os.getenv("MODEL_NAME", "gemini-2.0-flash"),
        name="HotelAgent",
        description="Hotel booking agent",
        instruction="""You are a hotel booking agent.
        - You take any hotel booking or confirmation request
        - Always return a valid JSON with hotel booking and confirmation details, including hotel name, check-in and check-out dates, room type, price, and status based on user request.
        - If the user does not provide specific details, make reasonable assumptions about the hotel and booking details.
        """
    )


def create_sightseeing_agent() -> LlmAgent:
    return LlmAgent(
        model=os.getenv("MODEL_NAME", "gemini-2.0-flash"),
        name="SightseeingAgent",
        description="Sightseeing information agent",
        instruction="""You are a sightseeing information agent.
        - You take any sightseeing request and suggest only the top 2 best places to visit, timings, and any other relevant details.
        - Always return a valid JSON with sightseeing information, including places to visit, timings, and any other relevant details based on user request.
        - If the user does not provide specific details, make reasonable assumptions about the sightseeing options available.
        """
    )


def create_trip_summary_agent() -> LlmAgent:
    return LlmAgent(
        model=os.getenv("MODEL_NAME", "gemini-2.0-flash"),
        name="TripSummaryAgent",
        instruction="Summarize the trip details from the flight, hotel, and sightseeing agents. Summarise JSON responses into a single summary document with all trip information like a travel itinerary. The summary should be well-structured and clearly present all trip details in an organized manner using text format only like a travel itinerary.",
        output_key="trip_summary"
    )


__all__ = [
    "FunctionAgent",
    "create_flight_agent",
    "create_hotel_agent",
    "create_sightseeing_agent",
    "create_trip_summary_agent",
    "create_process_initial_data_agent",
    "create_use_and_finalize_data_agent",
]
