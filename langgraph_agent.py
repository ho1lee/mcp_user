from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
import asyncio

# MCP 서버 임포트 (mcp_server.py가 동일 경로에 있다고 가정)
from mcp_server import server as mcp_server_instance

# 1. 상태 정의
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_action: str | None # 다음 액션 (도구 호출 또는 사용자 응답)
    tool_name: str | None
    tool_input: dict | None
    tool_output: str | None # 도구 실행 결과

# 2. 핵심 노드 함수들

async def call_model_node(state: AgentState):
    """
    사용자 입력 또는 이전 도구 결과를 바탕으로 LLM을 호출하여 다음 단계를 결정합니다.
    (ReAct/CoT의 Thought + Action 결정 부분)
    """
    print(f"---LLM 호출 (ReAct/CoT)---")
    current_messages = state["messages"]
    last_message = current_messages[-1]
    prompt = f"Previous conversation:\n"
    for msg in current_messages:
        prompt += f"{msg.type}: {msg.content}\n"

    # === 실제 LLM 연동을 위한 ReAct/CoT 프롬프팅 로직 (개념) ===
    # 1. LLM 클라이언트 초기화 (예: OpenAI, Anthropic)
    #    - from langchain_openai import ChatOpenAI
    #    - llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
    #    - 또는 다른 LLM 제공자 사용

    # 2. ReAct 프롬프트 템플릿 구성
    #    - 사용 가능한 도구 목록 가져오기
    available_tools = await mcp_server_instance.list_tools() # 비동기 호출로 수정
    tool_descriptions = "\n".join([f"- {tool}" for tool in available_tools]) # 실제로는 각 도구에 대한 설명도 포함해야 함
                                                                           # 예: "- search_tool: query에 대한 정보를 검색합니다."
                                                                           # 예: "- calculator_tool: expression을 평가하여 수학적 계산을 수행합니다."

    react_prompt_template = f"""
You are a helpful assistant that can use tools to answer user questions.
Your goal is to arrive at a final answer for the user.
You have access to the following tools:
{tool_descriptions}

To use a tool, you must respond in the following JSON format:
```json
{{
  "thought": "Your reasoning process and plan to use the tool.",
  "action": "tool_call",
  "tool_name": "Name of the tool to use (e.g., search_tool, calculator_tool)",
  "tool_input": {{ "arg1": "value1", "arg2": "value2" }}
}}
```

If you have enough information to answer the user directly, or if the tool execution provides the answer,
respond in the following JSON format:
```json
{{
  "thought": "Your reasoning process for why you can answer now.",
  "action": "final_answer",
  "answer": "Your final answer to the user."
}}
```

Conversation history:
{{chat_history}}

User question: {{user_question}}

Your turn (respond in the JSON format described above):
"""

    # 3. 프롬프트 채우기
    chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in current_messages[:-1]]) # 마지막 메시지는 현재 처리 중
    user_question_str = ""
    if isinstance(last_message, HumanMessage):
        user_question_str = str(last_message.content)
    elif isinstance(last_message, ToolMessage): # 이전이 도구 실행 결과인 경우
        user_question_str = f"Tool {last_message.additional_kwargs.get('tool_name')} executed with output: {last_message.content}. Now what is the next step or final answer?"

    # 만약 current_messages가 비어있다면 (이론상 발생 안함, HumanMessage로 시작하므로) user_question_str은 비어있을 수 있음
    # 이 경우, 또는 다른 경우에 대한 예외 처리 필요

    prompt_filled = react_prompt_template.format(
        chat_history=chat_history_str,
        user_question=user_question_str
    )
    print(f"LLM 프롬프트 (개념):\n{prompt_filled}")

    # 4. LLM 호출 (실제로는 비동기 호출을 사용해야 할 수 있음)
    #    llm_response_json_str = await llm.ainvoke(prompt_filled) # 예시: Langchain LCEL 사용
    #    import json
    #    try:
    #        llm_response_data = json.loads(llm_response_json_str.content) # .content는 LLM 응답 객체에 따라 다름
    #    except json.JSONDecodeError:
    #        print("Error: LLM response is not valid JSON")
    #        # 오류 처리: 재시도 또는 사용자에게 오류 알림
    #        return {"messages": [AIMessage(content="Error: Could not process LLM response.")], "next_action": "respond_to_user"}


    # === 아래는 이전의 규칙 기반 로직을 LLM 호출 결과(가상)에 따라 대체하는 부분 ===
    # 가상 LLM 응답 (실제 LLM 호출 결과를 파싱한 후의 데이터라고 가정)
    # 이 부분을 실제 LLM 호출 및 응답 파싱 로직으로 대체해야 합니다.

    llm_response_data = {} # 실제 LLM 호출 결과가 여기에 할당됨

    if isinstance(last_message, HumanMessage):
        if "날씨" in str(last_message.content) or "search" in str(last_message.content).lower():
            llm_response_data = {
                "thought": f"사용자가 '{last_message.content}'에 대해 질문했습니다. search_tool을 사용해서 정보를 찾아보겠습니다.",
                "action": "tool_call",
                "tool_name": "search_tool",
                "tool_input": {"query": str(last_message.content)}
            }
        elif "계산" in str(last_message.content) or "+" in str(last_message.content) or "-" in str(last_message.content):
             llm_response_data = {
                "thought": f"사용자가 '{last_message.content}' 계산을 요청했습니다. calculator_tool을 사용하겠습니다.",
                "action": "tool_call",
                "tool_name": "calculator_tool",
                "tool_input": {"expression": str(last_message.content).replace("계산해줘","").replace("계산","").strip()} # 간단한 입력 정제
            }
        else:
            llm_response_data = {
                "thought": "사용자의 질문에 대해 현재 정보만으로는 답변하기 어렵습니다. 일반적인 답변을 제공합니다.",
                "action": "final_answer",
                "answer": f"'{last_message.content}'에 대한 요청을 접수했습니다. 하지만 현재는 구체적인 답변을 드리기 어렵습니다."
            }
    elif isinstance(last_message, ToolMessage):
        tool_name_from_msg = last_message.additional_kwargs.get('tool_name')
        tool_content_from_msg = last_message.content
        llm_response_data = {
            "thought": f"Tool '{tool_name_from_msg}' 실행 결과: '{tool_content_from_msg}'. 이 정보를 바탕으로 사용자에게 최종 답변을 생성합니다.",
            "action": "final_answer",
            "answer": f"Tool '{tool_name_from_msg}' 실행 결과: {tool_content_from_msg}"
        }
    else: # 기타 상황 (예: AIMessage인데 tool_calls가 없는 경우 등) - 기본적으로 종료
        llm_response_data = {
            "thought": "대화의 현재 상태를 기반으로 최종 응답을 생성합니다.",
            "action": "final_answer",
            "answer": f"요청하신 내용에 대해 처리를 완료했습니다." # 더 구체적인 메시지 필요
        }

    print(f"LLM 응답 (모의): {llm_response_data}")

    # 5. LLM 응답 파싱 및 다음 상태 결정
    action = llm_response_data.get("action")

    if action == "tool_call":
        tool_name = llm_response_data.get("tool_name")
        tool_input = llm_response_data.get("tool_input")
        print(f"LLM Thought: {llm_response_data.get('thought')}")
        print(f"LLM Action: Call tool '{tool_name}' with input {tool_input}")
        # AIMessage에 tool_calls 정보를 포함하여 LangGraph가 내부적으로 처리하도록 할 수도 있습니다.
        # 여기서는 명시적으로 next_action을 설정합니다.
        return {"next_action": "tool_call", "tool_name": tool_name, "tool_input": tool_input}
    elif action == "final_answer":
        final_answer = llm_response_data.get("answer", "죄송합니다, 답변을 생성하는 데 문제가 발생했습니다.")
        print(f"LLM Thought: {llm_response_data.get('thought')}")
        print(f"LLM Action: Final Answer: {final_answer}")
        ai_message = AIMessage(content=final_answer)
        # 이전 메시지들과 함께 새로운 AIMessage를 추가
        return {"messages": [ai_message], "next_action": "respond_to_user"}
    else:
        # 잘못된 action 또는 응답 형식 오류 처리
        error_message = "LLM이 잘못된 action을 반환했거나 응답 형식이 올바르지 않습니다."
        print(f"Error: {error_message}")
        ai_message = AIMessage(content=error_message)
        return {"messages": [ai_message], "next_action": "respond_to_user"}

async def call_tool_node(state: AgentState):
    """MCP 서버의 도구를 호출합니다."""
    tool_name = state["tool_name"]
    tool_input = state["tool_input"]
    print(f"---MCP 도구 호출: {tool_name}, 입력: {tool_input}---")

    if not tool_name:
        # 이 경우는 발생하면 안 되지만, 방어적으로 코딩
        return {"tool_output": "Error: No tool name specified."}

    # mcp_server_instance를 직접 사용
    # 실제 mcp 라이브러리가 비동기를 어떻게 지원하는지에 따라 await 키워드 사용이 달라질 수 있음
    # 여기서는 mcp_server.py의 call_tool이 async def로 정의되어 있다고 가정
    try:
        # result = mcp_server_instance.call_tool(tool_name, tool_input) # 동기 호출 방식
        result = await mcp_server_instance.call_tool(tool_name, tool_input) # 비동기 호출 방식

        # ToolMessage 생성 시 content는 문자열이어야 함
        tool_output_str = str(result)
        tool_message = ToolMessage(
            content=tool_output_str,
            tool_call_id="N/A", # Langchain은 tool_call_id를 추적하지만, 여기서는 간단히 처리
            additional_kwargs={"tool_name": tool_name}
        )
        print(f"MCP 도구 결과: {result}")
        return {"messages": [tool_message], "tool_output": tool_output_str, "next_action": "model_call"}
    except Exception as e:
        error_message = f"Error calling tool {tool_name}: {e}"
        print(error_message)
        tool_message = ToolMessage(
            content=error_message,
            tool_call_id="N/A",
            additional_kwargs={"tool_name": tool_name, "error": True}
        )
        return {"messages": [tool_message], "tool_output": error_message, "next_action": "model_call"}


# 3. 조건부 엣지 함수
def should_continue_or_end(state: AgentState):
    """LLM 호출 후 다음 액션(도구 호출 또는 종료)을 결정합니다."""
    if state.get("next_action") == "tool_call":
        return "continue_tool"
    elif state.get("next_action") == "respond_to_user":
        return END
    # 기본적으로는 모델을 다시 호출 (예: 오류 발생 후 재시도 또는 추가 생각)
    # 실제 ReAct에서는 LLM이 명시적으로 "Final Answer"를 생성하면 END로 감
    # 여기서는 call_model_node에서 AIMessage를 생성하면 END로 가도록 단순화
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and not last_message.tool_calls: # AIMessage이고 tool_calls가 없으면 종료
        return END
    return "continue_model" # 기본적으로 모델 재호출 (ReAct 루프)

# 4. 그래프 빌드
workflow = StateGraph(AgentState)

workflow.add_node("llm_agent", call_model_node)
workflow.add_node("mcp_tool_executor", call_tool_node)

# 그래프 흐름 정의
workflow.set_entry_point("llm_agent")

# LLM 에이전트가 도구 호출을 결정하면 mcp_tool_executor 노드로 이동
workflow.add_conditional_edges(
    "llm_agent",
    should_continue_or_end,
    {
        "continue_tool": "mcp_tool_executor",
        END: END
    }
)

# MCP 도구 실행 후에는 다시 LLM 에이전트로 돌아가서 결과를 처리하고 다음 단계를 결정
workflow.add_edge("mcp_tool_executor", "llm_agent")

# 그래프 컴파일
app = workflow.compile()

# 테스트 실행 함수
async def run_agent(user_input: str):
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    # app.invoke를 사용하면 동기적으로 실행됨
    # 비동기 실행을 위해서는 ainvoke 사용
    # final_state = app.invoke(initial_state)

    print("\n---Agent 실행 시작---")
    async for event in app.astream(initial_state):
        for key, value in event.items():
            print(f"노드: {key}")
            # print(f"출력: {value}") # 너무 길어서 주석 처리
            if "messages" in value:
                print(f"  메시지: {value['messages'][-1].type} - {value['messages'][-1].content}")
            if "next_action" in value:
                 print(f"  다음 액션: {value['next_action']}")
            if "tool_name" in value and value["tool_name"]:
                 print(f"  호출 도구: {value['tool_name']}")
        print("---")

    # 최종 결과는 마지막 상태의 메시지에서 가져올 수 있음
    # astream 사용 시 최종 상태를 직접 반환하지 않으므로,
    # 필요하다면 마지막 event의 상태를 저장해야 함.
    # 여기서는 간단히 마지막 AIMessage를 찾아 출력하는 것으로 가정
    # (실제로는 Gradio와 통합 시 다르게 처리될 것임)

    # 비동기 실행의 최종 상태를 얻으려면:
    final_state_events = []
    async for event in app.astream(initial_state):
        final_state_events.append(event)

    final_ai_message = None
    if final_state_events:
        last_event_values = list(final_state_events[-1].values())
        if last_event_values and "messages" in last_event_values[-1]:
            for msg in reversed(last_event_values[-1]["messages"]):
                if isinstance(msg, AIMessage):
                    final_ai_message = msg
                    break

    if final_ai_message:
        print(f"\n최종 AI 응답: {final_ai_message.content}")
        return final_ai_message.content
    else:
        print("\n최종 AI 응답을 찾을 수 없습니다.")
        return "No final response generated."


if __name__ == "__main__":
    async def main_async():
        # MCP 서버에서 사용 가능한 도구 목록 확인 (옵션)
        available_tools = await mcp_server_instance.list_tools()
        print(f"MCP 서버에서 사용 가능한 도구: {available_tools}")

        # 에이전트 실행 예시
        user_query = "오늘 날씨 어때?"
        response = await run_agent(user_query)
        print(f"\n사용자 질문: {user_query}")
        print(f"에이전트 답변: {response}")

        user_query_calc = "5+7 계산해줘"
        response_calc = await run_agent(user_query_calc)
        print(f"\n사용자 질문: {user_query_calc}")
        print(f"에이전트 답변: {response_calc}")

    asyncio.run(main_async())
