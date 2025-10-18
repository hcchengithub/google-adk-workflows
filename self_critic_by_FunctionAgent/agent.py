"""
Self Critic Agent by FunctionAgent
Provides intelligent critique and quality assurance for travel planning outputs.
"""

import os
from dotenv import load_dotenv
from typing import Optional, Any

from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent, ParallelAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
import google.genai.types as types


# Load environment variables
load_dotenv()

MODEL = os.getenv('MODEL_NAME', 'gemini-2.0-flash')

# Import all agents from common subagent file
from subagent import (
    FunctionAgent,
    create_flight_agent, 
    create_hotel_agent, 
    create_sightseeing_agent, 
    create_trip_summary_agent,
    create_research_probe_agent
)

# Create new instances of the agents
flight_agent = create_flight_agent()
hotel_agent = create_hotel_agent()
sightseeing_agent = create_sightseeing_agent()
trip_summary_agent = create_trip_summary_agent()
research_probe_agent = create_research_probe_agent()

# Trip Summary Reviewer - specific to self-critic workflow
trip_summary_reviewer = LlmAgent(
    model=MODEL,
    name="TripSummaryReviewer",
    instruction="""Review the trip summary in {trip_summary}.
    - Check if the trip summary includes all necessary details such as flight information, hotel booking, sightseeing options, and any other relevant trip details.
    - Ensure the summary is well-structured and clearly presents all trip details in an organized manner.
    - If the summary meets quality standards, output 'pass'. If it does not meet the standards, output 'fail'""",
    output_key="review_status",
)


plan_parallel = ParallelAgent(
    name="ParallelTripPlanner",
    sub_agents=[flight_agent, hotel_agent],
    description="Fetch flight and hotel information parallely. Each sub-agent will return a JSON response with their respective details."
)

@FunctionAgent(
    name="ValidateTripSummary",
    model=MODEL,
    description="Validates the trip summary review status and provides feedback based on the review outcome.",
)
async def validate_trip_summary(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    status = callback_context.state.get("review_status", None)
    review = callback_context.state.get("trip_summary", None)
    print(f"Review Status: {status}")
    print(f"Trip Summary: {review}")
    
    if status == "pass":
        text_content = f"Trip summary review passed: {review}"
    else:
        text_content = f"Trip summary review failed: {review}"
    
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text=text_content)],
        )
    )

root_agent = SequentialAgent(
    name="PlanTripWorkflow",
    description="Orchestrates the trip planning process by first fetching flight, hotel, and sightseeing information concurrently, then summarizing the trip details into a single document.",
    # Run parallel fetch, then synthesize
    sub_agents=[sightseeing_agent, plan_parallel,
                trip_summary_agent, trip_summary_reviewer, validate_trip_summary, research_probe_agent]
)
