from langchain_core.messages import AIMessage
import time
import json


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

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

#         prompt = f"""You are a Bear Analyst making the case against investing in the stock. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.
#
# Key points to focus on:
#
# - Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats that could hinder the stock's performance.
# - Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning, declining innovation, or threats from competitors.
# - Negative Indicators: Use evidence from financial data, market trends, or recent adverse news to support your position.
# - Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning, exposing weaknesses or over-optimistic assumptions.
# - Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.
#
# Resources available:
#
# Market research report: {market_research_report}
# Social media sentiment report: {sentiment_report}
# Latest world affairs news: {news_report}
# Company fundamentals report: {fundamentals_report}
# Conversation history of the debate: {history}
# Last bull argument: {current_response}
# Reflections from similar situations and lessons learned: {past_memory_str}
# Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the stock. You must also address reflections and learn from lessons and mistakes you made in the past.
# """
        prompt = f"""你是一名空头分析师（Bear Analyst），你的任务是提出反对投资该股票的充分理由。你的目标是构建一个逻辑严密、论据充分的论述，重点强调风险、挑战和负面信号，并有效反驳看涨（bull）观点。

        请聚焦以下关键点：

        - 风险与挑战：突出可能阻碍股票表现的因素，例如市场饱和、财务不稳定或宏观经济威胁等。
        - 竞争劣势：强调诸如市场地位较弱、创新能力下降或面临竞争对手威胁等脆弱点。
        - 负面指标：利用财务数据、市场趋势或近期不利新闻等证据，支撑你的观点。
        - 反驳看涨观点：运用具体数据和合理推理论证，对看涨方的论点进行批判性分析，揭示其中的漏洞或过于乐观的假设。
        - 互动性：以对话的方式呈现你的论点，直接回应看涨分析师的观点，进行有效辩论，而不仅仅是罗列事实。

        可供使用的资源包括：

        市场研究报告：{market_research_report}
        社交媒体情绪报告：{sentiment_report}
        最新全球要闻：{news_report}
        公司基本面报告：{fundamentals_report}
        辩论历史对话记录：{history}
        最新的看涨方论点：{current_response}
        从类似情境中得到的反思与经验教训：{past_memory_str}

        请利用上述信息，构建一个有说服力的空头论述，有力反驳看涨方的主张，并参与一场动态的辩论，充分展现投资该股票所面临的风险与弱点。你还必须回应历史反思内容，吸取以往的经验与教训，避免重蹈覆辙。
        """

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
