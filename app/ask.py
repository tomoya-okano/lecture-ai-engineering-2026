"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話履歴を保ったまま続けて質問する（空行か exit で終了）

ツールを持たない「素の」呼び出し。--chat なしでは会話履歴も持たない。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def _print_usage(response) -> None:
    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[tokens] input={usage.prompt_token_count}, "
            f"output={usage.candidates_token_count}"
        )


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)
    _print_usage(response)
    return response.text


def chat() -> None:
    """会話履歴を保ったまま、続けて質問する。履歴は毎回モデルに送り直される。"""
    # client を変数で保持する。一時オブジェクトだと GC で HTTP クライアントが閉じられる
    client = genai.Client(api_key=get_api_key())
    session = client.chats.create(model=MODEL)
    while True:
        try:
            question = input("\n[質問] ").strip()
        except EOFError:
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        response = session.send_message(question)
        _print_usage(response)
        print(response.text)


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保ったまま続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat:
            if args.question:
                parser.error("--chat では質問を引数に指定せず、プロンプトに入力してください")
            chat()
            return
        question = args.question or DEFAULT_QUESTION
        print(f"[質問] {question}\n")
        print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
