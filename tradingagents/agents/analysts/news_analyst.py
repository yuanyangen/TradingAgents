from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_news_analyst(llm, toolkit):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_global_news_ark, toolkit.get_google_news]
        else:
            tools = [
                toolkit.get_finnhub_news,
                toolkit.get_reddit_news,
                toolkit.get_google_news,
            ]

        system_message = (
           "你是一名新闻研究员，负责分析过去一周内的最新新闻与趋势。请撰写一份全面的报告，阐述当前与交易和宏观经济相关的全球形势。请查阅来自 EODHD 和 finnhub 的新闻，以确保信息全面。不要仅仅笼统地说趋势是“混合的”，而要提供详细、细粒度化的分析与洞察，这些分析应有助于交易者做出决策。"
           +"请务必在报告末尾附上一张 **Markdown 表格**，用于整理报告中的关键要点，要求条理清晰、易于阅读。"
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
                    + "请在你的回复开头加上：FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**，以便团队知道可以停止。"
                    + "你可使用的工具包括：{tool_names}。"
                    + "{system_message}"
                    + "供你参考，当前日期是 {current_date}。我们正在分析的公司是 {ticker}。"
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
            "news_report": report,
        }

    return news_analyst_node
