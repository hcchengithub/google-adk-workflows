"""
Parallel Agent
Orchestrates parallel execution of travel agents for maximum efficiency.
"""

from google.adk.agents import  ParallelAgent, SequentialAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all agents from common subagent file
from subagent import create_flight_agent, create_hotel_agent, create_sightseeing_agent, create_trip_summary_agent, create_research_probe_agent

# Create new instances of the agents
flight_agent = create_flight_agent()
hotel_agent = create_hotel_agent()
sightseeing_agent = create_sightseeing_agent()
trip_summary_agent = create_trip_summary_agent()
research_probe_agent = create_research_probe_agent()

plan_parallel = ParallelAgent(
    name="ParallelTripPlanner",
    sub_agents=[flight_agent, hotel_agent],
    description="Fetch flight and hotel information parallely. Each sub-agent will return a JSON response with their respective details."
)


# Main parallel workflow
root_agent = SequentialAgent(
    name="ParallelWorkflow",
    description="Orchestrates parallel execution of travel planning tasks",
    sub_agents=[
        sightseeing_agent,  
        plan_parallel,
        trip_summary_agent,
        research_probe_agent
    ]
) 
