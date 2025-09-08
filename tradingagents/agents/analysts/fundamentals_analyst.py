from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_fundamentals_analyst(llm, toolkit):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_fundamentals_ark]
        else:
            tools = [
                toolkit.get_finnhub_company_insider_sentiment,
                toolkit.get_finnhub_company_insider_transactions,
                toolkit.get_simfin_balance_sheet,
                toolkit.get_simfin_cashflow,
                toolkit.get_simfin_income_stmt,
            ]

        system_message = (
            "你是一名研究员，负责分析某家公司过去一周的基本面信息。请撰写一份关于该公司基本面信息的综合报告，内容包括但不限于：财务文件、公司概况、基础财务数据、公司财务历史、内部人士情绪以及内部人士交易记录，以便全面了解该公司的基本面情况，为交易者提供决策依据。"
            +"请确保报告包含尽可能详实的信息。不要笼统地描述趋势为“好坏参半'，而要提供详细、细致、粒度精细的分析与洞察，这些内容应有助于交易者做出更为明智的决策。"
            +"请务必在报告的末尾附上一份 Markdown 格式的表格，用于归纳和整理报告中的关键要点，要求表格结构清晰、内容条理分明、便于阅读。"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个乐于助人的 AI 助手，正在与其他助手协同工作。"
                    + "请利用所提供的工具，逐步推进问题的解答。"
                    + "如果你无法完全回答问题，没有关系；其他拥有不同工具的助手"
                    + "将会接替你未完成的部分。请尽力执行你所能做的，以推动任务进展。"
                    + "如果你或任何其他助手得出了最终交易建议：**买入/持有/卖出（BUY/HOLD/SELL）** 或交付成果，"
                    + "请在你的回复开头加上：FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**，以便团队知道可以停止推进。"
                    + "你可使用的工具包括：{tool_names}。"
                    + "{system_message}"
                    + "供你参考，当前日期是 {current_date}。我们要分析的公司是 {ticker}。"
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
