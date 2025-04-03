# We have to apply nest_asyncio because crewai is not compatible with async
import nest_asyncio

nest_asyncio.apply()

from typing import AsyncGenerator

from blaxel.models import bl_model
from blaxel.tools import bl_tools
from crewai import Agent, Crew, Task
from crewai.tools import tool


@tool("Weather")
def weather(city: str) -> str:
    """Get the weather in a given city"""
    return f"The weather in {city} is sunny"

async def agent(input: str) -> AsyncGenerator[str, None]:
    tools = await bl_tools(["blaxel-search"]).to_crewai() + [weather]
    model = await bl_model("sandbox-openai").to_crewai()

    agent = Agent(
        role="Weather Researcher",
        goal="Find the weather in a city",
        backstory="You are an experienced weather researcher with attention to detail",
        llm=model,
        tools=tools,
        verbose=True,
    )
    crew = Crew(
        agents=[agent],
        tasks=[Task(description="Find weather", expected_output=input, agent=agent)],
        verbose=True,
    )
    result = crew.kickoff()
    yield result.raw