import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
#from langchain_tavily import TavilySearchResults
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from typing import Literal

from langchain_tavily import TavilySearch

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# Ensure consistency for libraries
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY or ""
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY or ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY or ""
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT or ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Debug
print("GOOGLE_API_KEY:", GOOGLE_API_KEY[:8] + "..." if GOOGLE_API_KEY else "❌ MISSING")
print("TAVILY_API_KEY:", TAVILY_API_KEY[:8] + "..." if TAVILY_API_KEY else "❌ MISSING")
print("GROQ_API_KEY:", GROQ_API_KEY[:8] + "..." if GROQ_API_KEY else "❌ MISSING")
print("LANGCHAIN_API_KEY:", LANGCHAIN_API_KEY[:8] + "..." if LANGCHAIN_API_KEY else "❌ MISSING")
print("LANGCHAIN_PROJECT:", LANGCHAIN_PROJECT if LANGCHAIN_PROJECT else "❌ MISSING")


# ------------------------------
# ChatBot Class
# ------------------------------
class ChatBot:
    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(model_name="Gemma2-9b-It", api_key=groq_api_key)

    def call_tool(self):
        tool = TavilySearch(max_results=2)
        self.tool_node = ToolNode(tools=[tool])
        self.llm_with_tool = self.llm.bind_tools([tool])

    def call_model(self, state: MessagesState):
        messages = state["messages"]
        response = self.llm_with_tool.invoke(messages)
        return {"messages": [response]}

    def summarize_tool_result(self, state: MessagesState):
        """Take tool output and summarize it into natural text."""
        messages = state["messages"]
        last_message = messages[-1]

        # Convert tool result into a prompt for the LLM
        summary_prompt = f"""
        You are an assistant. Convert the following tool result into a short human-readable answer:

        {last_message.content}
        """
        response = self.llm.invoke(summary_prompt)
        return {"messages": [response]}

    def router_function(self, state: MessagesState) -> Literal["tools", "summarize", END]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        if last_message.type == "tool":
            return "summarize"
        return END

    def __call__(self):
        self.call_tool()
        workflow = StateGraph(MessagesState)

        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("summarize", self.summarize_tool_result)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", self.router_function,
            {"tools": "tools", "summarize": "summarize", END: END}
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("summarize", END)

        self.app = workflow.compile()
        return self.app



if __name__ == "__main__":
    mybot = ChatBot()
    workflow = mybot()
    response = workflow.invoke({"messages": ["who is the president of USA?"]})
    print(response["messages"][-1].content)


