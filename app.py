import gradio as gr
import asyncio
from langgraph_agent import run_agent # langgraph_agent.py의 run_agent 함수 임포트

async def chat_interface(user_input_text):
    """
    Gradio 인터페이스에서 호출될 함수.
    사용자 입력을 받아 LangGraph 에이전트를 실행하고 결과를 반환합니다.
    """
    if not user_input_text:
        return "입력된 메시지가 없습니다. 질문을 입력해주세요."

    print(f"Gradio 입력: {user_input_text}")
    # run_agent는 비동기 함수이므로 await 사용
    # Gradio는 asyncio 이벤트 루프를 이미 실행 중일 수 있으므로,
    # asyncio.run()을 직접 호출하는 대신 await를 사용합니다.
    # 만약 Gradio 환경에서 직접 await가 안된다면, asyncio.run_coroutine_threadsafe 등을 고려해야 할 수 있습니다.
    # 하지만 대부분의 최신 Gradio는 async 함수를 잘 지원합니다.
    try:
        agent_response = await run_agent(user_input_text)
        print(f"Gradio 출력 (에이전트 응답): {agent_response}")
        return agent_response
    except Exception as e:
        print(f"Gradio 인터페이스 오류: {e}")
        return f"오류가 발생했습니다: {e}"

# Gradio 인터페이스 정의
iface = gr.Interface(
    fn=chat_interface, # 호출할 함수
    inputs=gr.Textbox(lines=2, placeholder="여기에 메시지를 입력하세요..."), # 입력 컴포넌트
    outputs=gr.Textbox(label="에이전트 응답", lines=5), # 출력 컴포넌트
    title="MCP LangGraph 에이전트",
    description="LangGraph, ReAct/CoT, MCP 도구를 사용하는 에이전트입니다. 질문을 입력하고 'Submit'을 누르세요.",
    allow_flagging="never" # 예제이므로 플래깅 비활성화
)

# Gradio 앱 실행 (app.py를 직접 실행할 경우)
if __name__ == "__main__":
    # mcp_server.py나 langgraph_agent.py의 main_async()는 여기서 직접 실행하지 않습니다.
    # Gradio 앱이 langgraph_agent.run_agent를 통해 필요에 따라 호출합니다.
    # langgraph_agent는 mcp_server_instance를 직접 임포트하여 사용합니다.

    # Gradio 앱을 실행합니다.
    # `share=True`를 사용하면 외부에서 접속 가능한 링크를 생성합니다 (필요시 사용).
    print("Gradio 앱을 시작합니다. http://127.0.0.1:7860 (또는 다른 포트)에서 확인하세요.")
    iface.launch()
    # asyncio.run(iface.launch(show_error=True)) # 만약 launch()가 코루틴이라면

# 참고: Gradio를 실행하기 전에 필요한 라이브러리(gradio, langchain, langgraph 등)가
# 설치되어 있어야 합니다. 예를 들어, requirements.txt 파일을 만들고
# gradio
# langchain
# langgraph
# langchain_openai (실제 LLM 사용 시)
# ... 등을 명시한 후 `pip install -r requirements.txt`로 설치할 수 있습니다.
# 현재는 `mcp_server.py`와 `langgraph_agent.py`가 로컬 임포트로 동작하므로
# 별도의 `mcp` 라이브러리 설치는 필요하지 않습니다.
# (실제 `mcp` 라이브러리가 있다면 해당 라이브러리도 설치 필요)
