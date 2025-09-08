from langchain_core.messages import AIMessage
import time
import json


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

#         prompt = f"""You are a Bull Analyst advocating for investing in the stock. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.
#
# Key points to focus on:
# - Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
# - Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
# - Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
# - Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
# - Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.
#
# Resources available:
# Market research report: {market_research_report}
# Social media sentiment report: {sentiment_report}
# Latest world affairs news: {news_report}
# Company fundamentals report: {fundamentals_report}
# Conversation history of the debate: {history}
# Last bear argument: {current_response}
# Reflections from similar situations and lessons learned: {past_memory_str}
# Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position. You must also address reflections and learn from lessons and mistakes you made in the past.
# """

        prompt = f"""你是一位看多分析师（Bull Analyst），致力于倡导投资该股票。你的任务是构建一个强有力、基于证据的论证，重点突出该公司的增长潜力、竞争优势以及积极的市场信号。充分利用所提供的研究和数据，有效回应市场担忧，并有力反驳看空观点。

        重点聚焦以下几个方面：
        - 增长潜力：强调该公司面临的市场机会、收入预期以及业务可扩展性。
        - 竞争优势：突出诸如独特产品、强大品牌、或市场主导地位等因素。
        - 积极信号：以财务健康状况、行业趋势以及近期正面新闻作为论据支撑。
        - 反驳看空观点：以具体数据和严密逻辑，深入分析看空方的论点，全面回应其担忧，并清晰说明为何看多立场更具说服力。
        - 互动表达：以对话的方式呈现你的论点，直接回应看空分析师的观点，进行有效辩论，而不仅仅是罗列数据。

        可供使用的资源包括：
        市场研究报告：{market_research_report}
        社交媒体情绪报告：{sentiment_report}
        最新全球要闻：{news_report}
        公司基本面报告：{fundamentals_report}
        辩论历史对话记录：{history}
        看空方最新论点：{current_response}
        来自类似情况的经验总结与教训反思：{past_memory_str}

        利用上述信息，构建一个令人信服的看多论述，有效反驳看空方的担忧，并展开一场富有动态性的辩论，充分展现看多立场的优势。你还必须回应经验总结，吸取过去所犯的错误与教训。
        """
        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
