"""
Dispatcher Agent
Routes travel requests to appropriate specialized agents.
"""

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import agent_tool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all agents from common subagent file
from subagent import create_flight_agent, create_hotel_agent, create_sightseeing_agent, create_research_probe_agent

# Create new instances of the agents
flight_agent = create_flight_agent()
hotel_agent = create_hotel_agent()
sightseeing_agent = create_sightseeing_agent()
research_probe_agent = create_research_probe_agent()

# Convert specialized agents into tools
flight_tool = agent_tool.AgentTool(agent=flight_agent)
hotel_tool = agent_tool.AgentTool(agent=hotel_agent)
sightseeing_tool = agent_tool.AgentTool(agent=sightseeing_agent)


dispatcher_core_agent = LlmAgent(
    model=os.getenv('MODEL_NAME', 'gemini-2.0-flash'),
    name="TripPlanner",
    instruction=f"""
   Acts as a comprehensive trip planner.
   - Use the FlightAgent to find and book flights
   - Use the HotelAgent to find and book accommodation
   - Use the SightSeeingAgent to find information on places to visit

   Based on the user request, sequentially invoke the sub-agents to gather all necessary trip details.:
   - Flight details (from FlightAgent)
   - Hotel booking confirmations (from HotelAgent)
   - Sightseeing information (from SightSeeingAgent)

   Ensure the final output is structured and clearly presents all trip details in an organized manner.
   You will generate customer preferences and complete the task without asking too many questions, making reasonable assumptions when necessary.
   """,
    tools=[flight_tool, hotel_tool, sightseeing_tool]
)

# 這個 root agent 好像是 Codex CLI 幫我加的。我記得原版只到 Dispatcher Agent 為止。那加了這個 root agent 的目的
# 是為了讓 research probe agent 在 sequential 裡在最後面上手。 其實我們如果 root agent 做成 sequential agent，
# 其實也可能沒必要。 如果我先前是很希望能夠跟它用交談的方式來做 research probe，那如果是那個目的的話，這個地方，root 
# agent 就可以做成 LlmAgent。普通的交談就可以了，所以他做完 Dispatcher Core Agent 的工作以後，停下來繼續跟 user 交
# 談。我就可以叫它去 call research probe agent 甚至還讓它進 Break Point。 

root_agent = SequentialAgent(
    name="DispatcherWorkflow",
    description="Runs the dispatcher core agent then executes the research probe for deeper inspection.",
    sub_agents=[dispatcher_core_agent, research_probe_agent],
)

# Hi, Please suggest me places to visit in paris in july for honeymoon and book flight from Delhi and hotel for 5 nights from 15th July 2025 