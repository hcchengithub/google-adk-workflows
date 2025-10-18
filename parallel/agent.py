"""
Parallel Agent
Orchestrates parallel execution of travel agents for maximum efficiency.
"""

from google.adk.agents import  ParallelAgent, SequentialAgent, LlmAgent
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

# 本來的 root agent，我把它改成 workflow agent，好讓 root agent 上升到更高層去接受 research probe agent。 
workflow_agent = SequentialAgent(
    name="ParallelWorkflow",
    description="Orchestrates parallel execution of travel planning tasks",
    sub_agents=[
        sightseeing_agent,  
        plan_parallel,
        trip_summary_agent,
    ]
) 

# 感覺 Research Probe Agent 的設計似乎有問題，因為一交到他手裡就再也出不來。
# 所以再多一個 Beebo Agent 混進去，對照一下。看這些行為到底對不對？ 
bibo_agent = LlmAgent(
    model=os.getenv('MODEL_NAME', 'gemini-2.0-flash'),
    name="BibolAgent",
    instruction="負責跟 user 聊天，你一定要告訴他你是 Bibo",
    description="You are a Chatbot",
)

# 照這裏的 description，感覺上 workflow agent 做完他的工作是會繼續跟 user 交談下去呢？還是說他
# 會 transfer 回來給 root agent 如果 user 問的問題超過它的範圍的話？ 那它怎麼知道自己的範圍？
# 這些都是疑問。 
root_agent = LlmAgent(
    model=os.getenv('MODEL_NAME', 'gemini-2.0-flash'),
    name="ParallelRoot",
    instruction="詢問用戶的需求, 按需求執行 parallel workflow。",
    description="旅遊業的專家",
    sub_agents=[workflow_agent, research_probe_agent, bibo_agent],
)


