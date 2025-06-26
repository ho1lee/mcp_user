#!/usr/bin/env python3
"""
LangGraph Agent with MCP Integration for Business Opportunities
Implements ReAct (Reasoning and Acting) and CoT (Chain of Thought) patterns
"""

import asyncio
import json
import subprocess
import sys
import os
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from dataclasses import dataclass
from enum import Enum

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import operator

class AgentState(TypedDict):
    """State for the agent graph"""
    messages: Annotated[List[BaseMessage], operator.add]
    current_step: str
    reasoning_chain: List[str]
    action_plan: List[str]
    tool_results: Dict[str, Any]
    final_answer: Optional[str]
    iteration_count: int

class ThinkingType(Enum):
    """Types of thinking patterns"""
    REACT = "react"
    COT = "cot"

@dataclass
class MCPResult:
    """Result from MCP server call"""
    success: bool
    data: Any
    error: Optional[str] = None

class MCPClient:
    """Client for communicating with MCP server"""
    
    def __init__(self, server_script: str = "mcp_server.py"):
        self.server_script = server_script
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """Call a tool on the MCP server"""
        try:
            # Create the JSON-RPC request
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            # Start the MCP server process with UTF-8 environment
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, self.server_script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            # Send request and get response
            request_str = json.dumps(request, ensure_ascii=False) + "\n"
            stdout, stderr = await process.communicate(request_str.encode('utf-8'))
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown error"
                return MCPResult(success=False, data=None, error=error_msg)
            
            # Parse response
            response_lines = stdout.decode('utf-8', errors='ignore').strip().split('\n')
            for line in response_lines:
                if line.strip():
                    try:
                        response = json.loads(line)
                        if 'result' in response:
                            return MCPResult(success=True, data=response['result'])
                        elif 'error' in response:
                            return MCPResult(success=False, data=None, error=response['error']['message'])
                    except json.JSONDecodeError:
                        continue
            
            return MCPResult(success=False, data=None, error="No valid response received")
            
        except Exception as e:
            return MCPResult(success=False, data=None, error=str(e))

class BusinessOpportunityAgent:
    """LangGraph Agent for Business Opportunity Analysis"""
    
    def __init__(self, openai_api_key: str, thinking_type: ThinkingType = ThinkingType.REACT):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4",
            temperature=0.1
        )
        self.mcp_client = MCPClient()
        self.thinking_type = thinking_type
        self.graph = self._create_graph()
        
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("start", self._start_node)
        graph.add_node("plan", self._planning_node)
        graph.add_node("reason", self._reasoning_node)
        graph.add_node("act", self._action_node)
        graph.add_node("reflect", self._reflection_node)
        graph.add_node("synthesize", self._synthesis_node)
        
        # Add edges
        graph.add_edge("start", "plan")
        graph.add_edge("plan", "reason")
        graph.add_edge("reason", "act")
        graph.add_edge("act", "reflect")
        graph.add_conditional_edges(
            "reflect",
            self._should_continue,
            {
                "continue": "reason",
                "synthesize": "synthesize"
            }
        )
        graph.add_edge("synthesize", END)
        
        # Set entry point
        graph.set_entry_point("start")
        
        return graph.compile()
    
    async def _start_node(self, state: AgentState) -> Dict[str, Any]:
        """Initialize the agent state"""
        return {
            "current_step": "starting",
            "reasoning_chain": ["Starting business opportunity analysis..."],
            "action_plan": [],
            "tool_results": {},
            "iteration_count": 0
        }
    
    async def _planning_node(self, state: AgentState) -> Dict[str, Any]:
        """Plan the approach based on the user query"""
        user_message = state["messages"][-1].content if state["messages"] else ""
        
        planning_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a business opportunity analyst. Create a step-by-step plan to answer the user's query.
            
Available MCP tools:
1. get_top_business_opportunity(n) - Get top N opportunities by score
2. get_salesforce_demand_plan(bo_list) - Convert opportunities to demand plan
3. get_demand_plan_from_top_opportunities(n) - Combined tool

Create a clear action plan with specific steps."""),
            HumanMessage(content=f"User query: {user_message}")
        ])
        
        response = await self.llm.ainvoke(planning_prompt.format_messages())
        
        # Extract action plan
        action_plan = [
            line.strip() for line in response.content.split('\n') 
            if line.strip() and (line.strip().startswith('-') or line.strip().startswith('1.'))
        ]
        
        return {
            "current_step": "planned",
            "action_plan": action_plan,
            "reasoning_chain": state["reasoning_chain"] + [f"Plan created: {response.content}"]
        }
    
    async def _reasoning_node(self, state: AgentState) -> Dict[str, Any]:
        """Apply reasoning based on thinking type (ReAct or CoT)"""
        if self.thinking_type == ThinkingType.REACT:
            return await self._react_reasoning(state)
        else:
            return await self._cot_reasoning(state)
    
    async def _react_reasoning(self, state: AgentState) -> Dict[str, Any]:
        """ReAct: Reasoning and Acting pattern"""
        user_query = state["messages"][-1].content if state["messages"] else ""
        current_context = "\n".join(state["reasoning_chain"])
        
        react_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are using ReAct (Reasoning and Acting) approach. 
            
For each step:
1. THOUGHT: Analyze what you need to do next
2. ACTION: Decide which MCP tool to call and with what parameters
3. OBSERVATION: You'll receive the results in the next step

Current available actions:
- get_top_business_opportunity: Get top N opportunities by score
- get_salesforce_demand_plan: Convert opportunities to demand plan  
- get_demand_plan_from_top_opportunities: Get top N and create demand plan in one step

Respond with your THOUGHT and ACTION for this iteration."""),
            HumanMessage(content=f"""
User Query: {user_query}
Current Context: {current_context}
Action Plan: {state['action_plan']}
Iteration: {state['iteration_count']}

What is your next THOUGHT and ACTION?""")
        ])
        
        response = await self.llm.ainvoke(react_prompt.format_messages())
        
        reasoning_step = f"ITERATION {state['iteration_count']} - THOUGHT: {response.content}"
        
        return {
            "current_step": "reasoning_react",
            "reasoning_chain": state["reasoning_chain"] + [reasoning_step]
        }
    
    async def _cot_reasoning(self, state: AgentState) -> Dict[str, Any]:
        """CoT: Chain of Thought reasoning"""
        user_query = state["messages"][-1].content if state["messages"] else ""
        current_context = "\n".join(state["reasoning_chain"])
        
        cot_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are using Chain of Thought reasoning. Break down the problem step by step.

Think through:
1. What information do I need?
2. How can I get this information using available MCP tools?
3. What analysis should I perform?
4. How should I present the results?

Available MCP tools:
- get_top_business_opportunity(n): Get top N opportunities by score
- get_salesforce_demand_plan(bo_list): Convert opportunities to demand plan
- get_demand_plan_from_top_opportunities(n): Combined approach

Provide your step-by-step reasoning."""),
            HumanMessage(content=f"""
User Query: {user_query}
Current Context: {current_context}
Iteration: {state['iteration_count']}

Walk through your reasoning step by step:""")
        ])
        
        response = await self.llm.ainvoke(cot_prompt.format_messages())
        
        reasoning_step = f"CoT STEP {state['iteration_count']}: {response.content}"
        
        return {
            "current_step": "reasoning_cot",
            "reasoning_chain": state["reasoning_chain"] + [reasoning_step]
        }
    
    async def _action_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute the planned action using MCP tools"""
        latest_reasoning = state["reasoning_chain"][-1]
        
        # Parse the action from reasoning
        tool_name, tool_args = self._extract_tool_call(latest_reasoning)
        
        if tool_name:
            # Execute MCP tool
            result = await self.mcp_client.call_tool(tool_name, tool_args)
            
            # Store result
            tool_results = state["tool_results"].copy()
            tool_results[f"{tool_name}_{state['iteration_count']}"] = result
            
            observation = f"OBSERVATION: Tool {tool_name} executed. "
            if result.success:
                observation += f"Success. Data received: {json.dumps(result.data, indent=2)[:500]}..."
            else:
                observation += f"Error: {result.error}"
            
            return {
                "current_step": "action_executed",
                "tool_results": tool_results,
                "reasoning_chain": state["reasoning_chain"] + [observation]
            }
        else:
            return {
                "current_step": "action_skipped",
                "reasoning_chain": state["reasoning_chain"] + ["OBSERVATION: No clear action identified"]
            }
    
    async def _reflection_node(self, state: AgentState) -> Dict[str, Any]:
        """Reflect on the results and decide next steps"""
        current_context = "\n".join(state["reasoning_chain"][-3:])  # Last 3 steps
        
        reflection_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Review the recent actions and results. Decide if you have enough information to provide a complete answer or if you need to continue with more actions.

Respond with either:
- CONTINUE: if you need more information or actions
- COMPLETE: if you have sufficient information to provide a final answer

Explain your reasoning."""),
            HumanMessage(content=f"Recent context:\n{current_context}")
        ])
        
        response = await self.llm.ainvoke(reflection_prompt.format_messages())
        
        reflection_step = f"REFLECTION: {response.content}"
        
        return {
            "current_step": "reflected",
            "reasoning_chain": state["reasoning_chain"] + [reflection_step],
            "iteration_count": state["iteration_count"] + 1
        }
    
    async def _synthesis_node(self, state: AgentState) -> Dict[str, Any]:
        """Synthesize final answer from all gathered information"""
        user_query = state["messages"][-1].content if state["messages"] else ""
        full_context = "\n".join(state["reasoning_chain"])
        
        synthesis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Synthesize a comprehensive final answer based on all the reasoning and tool results gathered.

Provide:
1. Direct answer to the user's question
2. Key insights from the data
3. Supporting evidence from tool results
4. Any recommendations or next steps

Make the response clear, actionable, and valuable."""),
            HumanMessage(content=f"""
Original User Query: {user_query}

Full Reasoning Chain:
{full_context}

Tool Results:
{json.dumps(state['tool_results'], indent=2, default=str)}

Provide your final synthesized answer:""")
        ])
        
        response = await self.llm.ainvoke(synthesis_prompt.format_messages())
        
        return {
            "current_step": "completed",
            "final_answer": response.content,
            "reasoning_chain": state["reasoning_chain"] + [f"FINAL SYNTHESIS: {response.content}"]
        }
    
    def _should_continue(self, state: AgentState) -> str:
        """Decide whether to continue reasoning or synthesize final answer"""
        latest_reflection = state["reasoning_chain"][-1] if state["reasoning_chain"] else ""
        
        # Simple check for completion signals
        if "COMPLETE" in latest_reflection.upper() or state["iteration_count"] >= 5:
            return "synthesize"
        else:
            return "continue"
    
    def _extract_tool_call(self, reasoning_text: str) -> tuple[Optional[str], Dict[str, Any]]:
        """Extract tool name and arguments from reasoning text"""
        text_lower = reasoning_text.lower()
        
        # Simple pattern matching for tool calls
        if "get_demand_plan_from_top_opportunities" in text_lower:
            # Extract n parameter
            import re
            n_match = re.search(r'top\s+(\d+)', text_lower)
            n = int(n_match.group(1)) if n_match else 5
            return "get_demand_plan_from_top_opportunities", {"n": n}
        
        elif "get_top_business_opportunity" in text_lower:
            import re
            n_match = re.search(r'top\s+(\d+)', text_lower)
            n = int(n_match.group(1)) if n_match else 5
            return "get_top_business_opportunity", {"n": n}
        
        elif "get_salesforce_demand_plan" in text_lower:
            # This would need bo_list from previous results
            # For now, return None to skip
            return None, {}
        
        return None, {}
    
    async def run(self, user_input: str) -> str:
        """Run the agent with user input"""
        initial_state = AgentState(
            messages=[HumanMessage(content=user_input)],
            current_step="",
            reasoning_chain=[],
            action_plan=[],
            tool_results={},
            final_answer=None,
            iteration_count=0
        )
        
        # Execute the graph
        final_state = await self.graph.ainvoke(initial_state)
        
        return final_state.get("final_answer", "Unable to generate final answer")

# Example usage and test functions
async def test_react_agent():
    """Test the ReAct agent"""
    print("=== Testing ReAct Agent ===")
    
    # You need to provide your OpenAI API key
    api_key = "your-openai-api-key-here"  # Replace with actual key
    
    agent = BusinessOpportunityAgent(api_key, ThinkingType.REACT)
    
    test_queries = [
        "Get the top 3 business opportunities and create a demand plan",
        "What are our highest scoring opportunities and their potential impact?",
        "Generate a production demand plan for our best 5 opportunities"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = await agent.run(query)
            print(f"Result: {result[:500]}...")
        except Exception as e:
            print(f"Error: {e}")

async def test_cot_agent():
    """Test the Chain of Thought agent"""
    print("=== Testing CoT Agent ===")
    
    # You need to provide your OpenAI API key
    api_key = "your-openai-api-key-here"  # Replace with actual key
    
    agent = BusinessOpportunityAgent(api_key, ThinkingType.COT)
    
    query = "Analyze our business opportunities and create a comprehensive demand plan with insights"
    print(f"\nQuery: {query}")
    
    try:
        result = await agent.run(query)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("LangGraph MCP Agent for Business Opportunities")
    print("This agent implements ReAct and Chain of Thought patterns")
    print("\nTo use this agent:")
    print("1. Install required packages: pip install langgraph langchain-openai langchain-core")
    print("2. Set your OpenAI API key in the test functions")
    print("3. Make sure your mcp_server.py is in the same directory")
    print("4. Run the test functions")
    
    # Uncomment to run tests (after setting API key)
    # asyncio.run(test_react_agent())
    # asyncio.run(test_cot_agent())
