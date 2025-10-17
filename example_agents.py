import logging
from typing import Any, Callable, Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
import google.genai.types as types

# 啟用詳細日誌輸出，方便在 ADK 網頁環境觀察流程
logging.basicConfig(level=logging.INFO)

# 預設的 Gemini 模型設定
MODEL = "gemini-1.5-flash"


def _snapshot_state(state: Any) -> str:
    """盡可能把 session state 序列化成可讀字串。"""
    try:
        return repr(dict(state))
    except TypeError:
        return repr(state)
    except Exception as error:  # pylint: disable=broad-except
        logging.debug("⚠️ [snapshot_state] Failed to snapshot state: %s", error)
        return "<state unavailable>"


class FunctionAgent:
    def __init__(self, name: str, model: str, **kwargs: Any):
        self.name = name
        self.model = model
        self.kwargs = kwargs

    def __call__(self, func: Callable[[CallbackContext, LlmRequest], Optional[LlmResponse]]) -> LlmAgent:
        return LlmAgent(
            name=self.name,
            model=self.model,
            before_model_callback=func,
            **self.kwargs,
        )


async def _process_initial_data(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    第一個 FunctionAgent：把輸入文字轉成整數、進行計算，並把結果寫進 session state。
    """
    logging.info("🚀🚀 [process_initial_data] Starting…")
    logging.info("🔍 [process_initial_data] Raw request object: %r", llm_request)
    logging.info("🗃️ [process_initial_data] Session state before: %s", _snapshot_state(callback_context.state))

    try:
        latest_content = llm_request.contents[-1]
        latest_part = latest_content.parts[0]
        raw_text = latest_part.text
        logging.info("📝 [process_initial_data] Raw text from request: %s", raw_text)

        input_value = int(raw_text)
        logging.info("🌟 [process_initial_data] Parsed integer value: %s", input_value)

        processed_value = input_value * 2
        logging.info("✅ [process_initial_data] Processed value (input * 2): %s", processed_value)

        callback_context.state["processed_data"] = processed_value
        logging.info(
            "💾 [process_initial_data] Stored processed_data in session state: %s",
            processed_value,
        )
        logging.info("🗃️ [process_initial_data] Session state after: %s", _snapshot_state(callback_context.state))

        response_text = f"輸入值 {input_value} 已處理，結果 {processed_value} 已寫入會話狀態。"
        logging.info("🎉 [process_initial_data] Done. Response: %s", response_text)
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=response_text)],
            )
        )

    except (ValueError, IndexError, AttributeError) as error:
        logging.error("⚠️ [process_initial_data] Error processing input: %s", error)
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="錯誤：請提供可以解析成整數的文字。")],
            )
        )
    except Exception as error:  # pylint: disable=broad-except
        logging.error("❌ [process_initial_data] Unexpected error: %s", error)
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"處理輸入資料時發生錯誤: {error}")],
            )
        )


async def _use_and_finalize_data(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    第二個 FunctionAgent：讀取 session state，延伸計算並輸出最終結果。
    """
    logging.info("🚀🚀 [use_and_finalize_data] Starting…")
    logging.info("🔍 [use_and_finalize_data] Raw request object: %r", llm_request)
    logging.info("🗃️ [use_and_finalize_data] Session state snapshot: %s", _snapshot_state(callback_context.state))

    processed_data = callback_context.state.get("processed_data")
    logging.info("📦 [use_and_finalize_data] Retrieved processed_data: %s", processed_data)

    if processed_data is not None:
        final_result = processed_data + 10
        logging.info(
            "✨ [use_and_finalize_data] Final result (processed_data + 10): %s",
            final_result,
        )

        response_text = f"從會話狀態讀取資料 {processed_data}，最終結果為 {final_result}。"
        logging.info("🎉 [use_and_finalize_data] Done. Response: %s", response_text)
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=response_text)],
            )
        )

    logging.warning("⚠️ [use_and_finalize_data] 'processed_data' not found in session state.")
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text="錯誤：會話狀態中找不到 'processed_data'。")],
        )
    )


def create_process_initial_data_agent() -> LlmAgent:
    """回傳新的 process_initial_data 代理實例。"""

    async def handler(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
        return await _process_initial_data(callback_context, llm_request)

    return FunctionAgent(name="process_initial_data", model=MODEL)(handler)


def create_use_and_finalize_data_agent() -> LlmAgent:
    """回傳新的 use_and_finalize_data 代理實例。"""

    async def handler(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
        return await _use_and_finalize_data(callback_context, llm_request)

    return FunctionAgent(name="use_and_finalize_data", model=MODEL)(handler)


# 參考：若想自行串起整個流程，可這樣使用：
# root_agent = SequentialAgent(
#     name="data_flow_example",
#     sub_agents=[
#         create_process_initial_data_agent(),
#         create_use_and_finalize_data_agent(),
#     ],
# )

# 範例操作步驟（如在 notebook 測試）：
# request = LlmRequest(text="5")
# context = CallbackContext()
# response1 = await create_process_initial_data_agent()(context, request)
# response2 = await create_use_and_finalize_data_agent()(context, request)
# print(response2, context.state)
