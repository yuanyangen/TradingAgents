from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_social_media_analyst(llm, toolkit):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_stock_news_ark]
        else:
            tools = [
                toolkit.get_reddit_stock_info,
            ]

        system_message = (
                "你是一名专注于社交媒体与公司特定新闻的研究员/分析师，负责分析过去一周内，针对某家特定公司的社交媒体帖子、近期公司新闻以及公众情绪。你将获得一家公司的名称，你的目标是撰写一份详尽的长篇报告，通过分析该公司在社交媒体上的讨论内容、大众对该公司言论的看法、每日情绪数据，以及近期的公司新闻，深入挖掘该公司当前的状态，并为交易者和投资者提供详细的分析、洞察与潜在影响。"
                + "请尽可能查阅所有可用信息来源，包括社交媒体、情绪数据、新闻报道等。不要只是笼统地指出“趋势复杂”或“好坏参半”，而要提供**详细且细粒度（fine-grained）的分析与洞察**，这些分析应有助于交易者做出更明智的决策。"
                + "请务必在报告末尾附上一份 **Markdown 表格**，用于整理报告中的关键要点，使内容条理清晰、易于阅读。"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个乐于助人的 AI 助手，正在与其他助手协同工作。"
                    +"请利用所提供的工具，逐步推进问题的解答。"
                    +"如果你无法完全回答问题，没有关系；其他拥有不同工具的助手"
                    +"将会接替你未完成的部分。请尽力执行你所能做的，以推动任务进展。"
                    +"如果你或任何其他助手得出了最终交易建议：**买入/持有/卖出（BUY/HOLD/SELL）** 或交付成果，"
                    +"请在你的回复开头加上：FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**，以便团队知道可以停止。"
                    +"你可使用的工具包括：{tool_names}。"
                    +"{system_message}"
                    +"供你参考，当前日期是 {current_date}。当前我们要分析的公司是 {ticker}。",
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
            "sentiment_report": report,
        }

    return social_media_analyst_node
