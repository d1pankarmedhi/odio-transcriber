import json
from typing import Any, Dict

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function

from core.agent.model import Model
from core.agent.tools import build_tools


class AgentOutput(BaseModel):
    result: Dict[str, Any] = Field(
        ..., description="Dictionary containing the output of the agent"
    )


class Agent:
    def __init__(self) -> ChatOpenAI:
        self.tools = build_tools()
        self.llm = Model.openai_chat_llm()

    def _bind_llm_with_tools(self):
        functions = [format_tool_to_openai_function(f) for f in self.tools]
        return self.llm.bind(functions=functions)

    def _prompt(self):
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a smart chatbot. You help the user by fulfilling the requests using the set of tools. Make sure to use atleast one tool.\n\nMake sure to answer on json format.",
                ),
                # MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

    def _agent(self):
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | self._prompt()
            | self._bind_llm_with_tools()
            | OpenAIFunctionsAgentOutputParser()
        )
        return agent

    def _agent_executor(self):
        agent_executor = AgentExecutor(
            agent=self._agent(),
            tools=self.tools,
            verbose=True,
        )
        return agent_executor

    def invoke_agent(self, input: str):
        """
        Invokes the agent with the given input and returns the agent's output.

        Args:
        - input (str): The input string to be passed to the agent.

        Returns:
        - AgentOutput: An object representing the agent's output, parsed from the JSON string.
        """
        output = self._agent_executor().invoke({"input": str(input)})["output"]
        return AgentOutput(result=json.loads(output))
