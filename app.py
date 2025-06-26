import asyncio
import json
import subprocess
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class AgentMode(Enum):
    REACT = "react"
    COT = "cot"


@dataclass
class MCPTool:
    name: str
    description: str
    parameters: Dict[str, Any]


class AgentState(TypedDict):
    messages: List[Any]
    current_task: str
    mode: str
    thoughts: List[str]
    tools_used: List[str]
    iteration: int
    max_iterations: int
    final_answer: Optional[str]


class MCPClient:
    """MCP 서버와 통신하는 클라이언트"""
    
    def __init__(self, server_command: str = "python mcp_server.py"):
        self.server_command = server_command
        self.available_tools = []
    
    async def get_available_tools(self) -> List[MCPTool]:
        """사용 가능한 도구 목록 가져오기"""
        try:
            # MCP 서버에서 도구 목록 요청
            result = subprocess.run(
                [*self.server_command.split(), "--list-tools"], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                tools_data = json.loads(result.stdout)
                tools = []
                for tool_data in tools_data.get('tools', []):
                    tools.append(MCPTool(
                        name=tool_data['name'],
                        description=tool_data['description'],
                        parameters=tool_data.get('parameters', {})
                    ))
                self.available_tools = tools
                return tools
            else:
                print(f"도구 목록 가져오기 실패: {result.stderr}")
                return []
        except Exception as e:
            print(f"MCP 서버 통신 오류: {e}")
            return []
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """도구 실행"""
        try:
            # MCP 서버에 도구 실행 요청
            command_data = {
                "tool": tool_name,
                "parameters": parameters
            }
            
            result = subprocess.run(
                [*self.server_command.split(), "--execute", json.dumps(command_data)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"도구 실행 오류: {result.stderr}"
        except Exception as e:
            return f"도구 실행 중 예외 발생: {e}"


class LangGraphMCPAgent:
    """LangGraph 기반 MCP Agent"""
    
    def __init__(self, llm_model: str = "gpt-4", max_iterations: int = 10):
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.mcp_client = MCPClient()
        self.max_iterations = max_iterations
        
        # 프롬프트 템플릿 설정
        self.react_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 ReAct (Reasoning and Acting) 방식으로 작업하는 AI 에이전트입니다.

사용 가능한 도구들:
{tools}

다음 형식으로 응답하세요:
Thought: [현재 상황과 다음 행동에 대한 추론]
Action: [사용할 도구 이름]
Action Input: [도구에 전달할 매개변수 (JSON 형식)]

도구 실행 결과를 받은 후:
Observation: [도구 실행 결과]
Thought: [결과에 대한 분석과 다음 단계]

최종 답변이 준비되면:
Final Answer: [최종 답변]

현재 반복: {iteration}/{max_iterations}"""),
            ("human", "{input}")
        ])
        
        self.cot_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 Chain of Thought (CoT) 방식으로 작업하는 AI 에이전트입니다.

사용 가능한 도구들:
{tools}

단계별로 추론하며 문제를 해결하세요:
1. 문제 이해 및 분석
2. 해결 방법 계획
3. 필요한 도구 사용
4. 결과 분석 및 검증
5. 최종 답변 도출

각 단계에서 명확한 추론 과정을 보여주세요.
도구가 필요한 경우 다음 형식을 사용하세요:
Tool: [도구 이름]
Input: [입력값]

현재 반복: {iteration}/{max_iterations}"""),
            ("human", "{input}")
        ])
    
    async def initialize(self):
        """에이전트 초기화 - 사용 가능한 도구 로드"""
        tools = await self.mcp_client.get_available_tools()
        self.tools_description = self._format_tools_description(tools)
        print(f"사용 가능한 도구 {len(tools)}개 로드됨")
    
    def _format_tools_description(self, tools: List[MCPTool]) -> str:
        """도구 설명을 포맷팅"""
        if not tools:
            return "사용 가능한 도구가 없습니다."
        
        descriptions = []
        for tool in tools:
            desc = f"- {tool.name}: {tool.description}"
            if tool.parameters:
                desc += f" (매개변수: {tool.parameters})"
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    async def reasoning_node(self, state: AgentState) -> AgentState:
        """추론 노드 - 현재 상황 분석 및 다음 행동 결정"""
        current_prompt = self.react_prompt if state["mode"] == "react" else self.cot_prompt
        
        # 이전 메시지들을 컨텍스트로 포함
        context = "\n".join([
            f"이전 생각: {thought}" for thought in state["thoughts"]
        ])
        
        messages = current_prompt.format_messages(
            tools=self.tools_description,
            input=f"{state['current_task']}\n\n컨텍스트:\n{context}",
            iteration=state["iteration"],
            max_iterations=state["max_iterations"]
        )
        
        response = await self.llm.ainvoke(messages)
        
        # 응답 파싱
        parsed_response = self._parse_response(response.content, state["mode"])
        
        # 상태 업데이트
        state["thoughts"].append(parsed_response.get("thought", ""))
        state["messages"].append(AIMessage(content=response.content))
        
        return state
    
    async def action_node(self, state: AgentState) -> AgentState:
        """행동 노드 - 도구 실행"""
        last_message = state["messages"][-1].content
        action_info = self._extract_action_from_message(last_message)
        
        if action_info:
            tool_name = action_info["tool"]
            parameters = action_info["parameters"]
            
            # 도구 실행
            result = await self.mcp_client.execute_tool(tool_name, parameters)
            
            # 결과를 상태에 추가
            observation = f"Observation: {result}"
            state["messages"].append(HumanMessage(content=observation))
            state["tools_used"].append(f"{tool_name}({parameters})")
        
        state["iteration"] += 1
        return state
    
    def _parse_response(self, response: str, mode: str) -> Dict[str, Any]:
        """응답 파싱"""
        parsed = {}
        
        if mode == "react":
            # ReAct 형식 파싱
            lines = response.split('\n')
            for line in lines:
                if line.startswith("Thought:"):
                    parsed["thought"] = line.replace("Thought:", "").strip()
                elif line.startswith("Action:"):
                    parsed["action"] = line.replace("Action:", "").strip()
                elif line.startswith("Action Input:"):
                    parsed["action_input"] = line.replace("Action Input:", "").strip()
                elif line.startswith("Final Answer:"):
                    parsed["final_answer"] = line.replace("Final Answer:", "").strip()
        else:
            # CoT 형식 파싱
            parsed["thought"] = response
            if "Tool:" in response and "Input:" in response:
                # 도구 사용 패턴 추출
                tool_start = response.find("Tool:")
                input_start = response.find("Input:", tool_start)
                if tool_start != -1 and input_start != -1:
                    tool_line = response[tool_start:input_start].replace("Tool:", "").strip()
                    input_line = response[input_start:].split('\n')[0].replace("Input:", "").strip()
                    parsed["action"] = tool_line
                    parsed["action_input"] = input_line
        
        return parsed
    
    def _extract_action_from_message(self, message: str) -> Optional[Dict[str, Any]]:
        """메시지에서 행동 정보 추출"""
        try:
            if "Action:" in message and "Action Input:" in message:
                # ReAct 형식
                action_start = message.find("Action:")
                input_start = message.find("Action Input:")
                
                if action_start != -1 and input_start != -1:
                    action = message[action_start:input_start].replace("Action:", "").strip()
                    action_input = message[input_start:].split('\n')[0].replace("Action Input:", "").strip()
                    
                    try:
                        parameters = json.loads(action_input)
                    except json.JSONDecodeError:
                        parameters = {"input": action_input}
                    
                    return {"tool": action, "parameters": parameters}
            
            elif "Tool:" in message and "Input:" in message:
                # CoT 형식
                tool_start = message.find("Tool:")
                input_start = message.find("Input:", tool_start)
                
                if tool_start != -1 and input_start != -1:
                    tool = message[tool_start:input_start].replace("Tool:", "").strip()
                    tool_input = message[input_start:].split('\n')[0].replace("Input:", "").strip()
                    
                    try:
                        parameters = json.loads(tool_input)
                    except json.JSONDecodeError:
                        parameters = {"input": tool_input}
                    
                    return {"tool": tool, "parameters": parameters}
            
            return None
        except Exception as e:
            print(f"행동 추출 오류: {e}")
            return None
    
    def should_continue(self, state: AgentState) -> str:
        """계속 진행할지 결정"""
        last_message = state["messages"][-1].content if state["messages"] else ""
        
        # 최종 답변이 있거나 최대 반복 횟수에 도달한 경우 종료
        if ("Final Answer:" in last_message or 
            state["iteration"] >= state["max_iterations"]):
            return "end"
        
        # 도구 사용이 필요한 경우 action 노드로
        if ("Action:" in last_message or "Tool:" in last_message):
            return "action"
        
        # 계속 추론
        return "reasoning"
    
    def create_graph(self) -> StateGraph:
        """LangGraph 생성"""
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("reasoning", self.reasoning_node)
        workflow.add_node("action", self.action_node)
        
        # 엣지 추가
        workflow.set_entry_point("reasoning")
        
        workflow.add_conditional_edges(
            "reasoning",
            self.should_continue,
            {
                "action": "action",
                "reasoning": "reasoning",
                "end": END
            }
        )
        
        workflow.add_edge("action", "reasoning")
        
        return workflow.compile()
    
    async def run(self, task: str, mode: AgentMode = AgentMode.REACT) -> Dict[str, Any]:
        """에이전트 실행"""
        await self.initialize()
        
        # 초기 상태 설정
        initial_state = AgentState(
            messages=[],
            current_task=task,
            mode=mode.value,
            thoughts=[],
            tools_used=[],
            iteration=0,
            max_iterations=self.max_iterations,
            final_answer=None
        )
        
        # 그래프 생성 및 실행
        app = self.create_graph()
        
        final_state = await app.ainvoke(initial_state)
        
        # 결과 정리
        result = {
            "task": task,
            "mode": mode.value,
            "final_state": final_state,
            "thoughts": final_state["thoughts"],
            "tools_used": final_state["tools_used"],
            "iterations": final_state["iteration"],
            "messages": final_state["messages"]
        }
        
        return result


# 사용 예제
async def main():
    """메인 실행 함수"""
    # 에이전트 생성
    agent = LangGraphMCPAgent(max_iterations=5)
    
    # ReAct 모드로 실행
    print("=== ReAct 모드 실행 ===")
    task1 = "현재 날씨를 확인하고, 그에 따른 옷차림 추천을 해주세요."
    result1 = await agent.run(task1, AgentMode.REACT)
    
    print(f"작업: {result1['task']}")
    print(f"사용된 도구: {', '.join(result1['tools_used'])}")
    print(f"반복 횟수: {result1['iterations']}")
    print("\n")
    
    # CoT 모드로 실행
    print("=== CoT 모드 실행 ===")
    task2 = "주어진 데이터를 분석하여 트렌드를 파악하고 예측을 제시해주세요."
    result2 = await agent.run(task2, AgentMode.COT)
    
    print(f"작업: {result2['task']}")
    print(f"사용된 도구: {', '.join(result2['tools_used'])}")
    print(f"반복 횟수: {result2['iterations']}")


if __name__ == "__main__":
    # 필요한 라이브러리 설치 안내
    print("필요한 라이브러리:")
    print("pip install langgraph langchain-openai langchain-core")
    print("OpenAI API 키를 환경변수 OPENAI_API_KEY에 설정하세요.")
    print("\n")
    
    # 실행
    asyncio.run(main())
