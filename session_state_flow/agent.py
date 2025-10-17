"""
Session State Flow Agent
Demonstrates how to pass session state between agents in a sequential workflow.
"""

from google.adk.agents import SequentialAgent

from example_agents import (
    create_process_initial_data_agent,
    create_use_and_finalize_data_agent,
)

# Instantiate the sub-agents from example_agents.py just like other demos do.
process_initial_data_agent = create_process_initial_data_agent()
use_and_finalize_data_agent = create_use_and_finalize_data_agent()

# Create a sequential agent that chains the two agents from example_agents.py
root_agent = SequentialAgent(
    name="SessionStateFlow",
    description="Demonstrates passing session state between agents.",
    sub_agents=[
        process_initial_data_agent,
        use_and_finalize_data_agent,
    ],
)
