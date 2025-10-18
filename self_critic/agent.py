"""
Self Critic Agent
Provides intelligent critique and quality assurance for travel planning outputs.
"""

from google.genai.types import Content, Part
from typing import AsyncGenerator
from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent, ParallelAgent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all agents from common subagent file
from subagent import (
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
    model=os.getenv('MODEL_NAME', 'gemini-2.0-flash'),
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


# ---------------------------------------------------------------------------------------------------
# Custom validation agent - specific to self-critic workflow
#     這個地方展現的就是不同於前後的那些 Agent 的定義。
#     以下它用的是 BaseAgent，也不是 ParallelAgent，也不是 SequentialAgent，也不是單純一個 Agent or LlmAgent。
#     當用到 BaseAgent 的時候，它是從頭定義出一個 Class，然後再用這個 Class 來定義出我們要的 special agent。這時候
#     就可以 access 得到底層的東西。 前述那那些已經包裝好的 agents 就看不到底層的東西。 
#     "底層的東西" 就是 input 的 InvocationContext 以及 output 的 Event. 
#     其實看看也明白了，event 就是 messages 裡面的一個 cell, 有 role, 有 parts, 有 text.

class ValidateTripSummary(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        ctx is InvocationContext or 「調用上下文」或「呼叫上下文」
        impl 是 implementation 的縮寫，中文通常翻譯為「實現」或「實作」
        """
        status = ctx.session.state.get("review_status", None)
        review = ctx.session.state.get("trip_summary", None)
        print(f"Review Status: {status}")
        print(f"Trip Summary: {review}")
        
        if status == "pass":
            yield Event(author=self.name, content=Content(parts=[Part(text=f"Trip summary review passed: {review}")]))
        else:
            yield Event(author=self.name, content=Content(parts=[Part(text=f"Trip summary review failed: {review}")]))

validate_trip_summary_agent = ValidateTripSummary(
    name="ValidateTripSummary",
    description="Validates the trip summary review status and provides feedback based on the review outcome.",
)

# ---------------------------------------------------------------------------------------------------


root_agent = SequentialAgent(
    name="PlanTripWorkflow",
    description="Orchestrates the trip planning process by first fetching flight, hotel, and sightseeing information concurrently, then summarizing the trip details into a single document.",
    # Run parallel fetch, then synthesize
    sub_agents=[sightseeing_agent, plan_parallel,
                trip_summary_agent, trip_summary_reviewer, validate_trip_summary_agent, research_probe_agent]
)
