import functools
import time
import json


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            # "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
            "content": f"基于分析师团队的全面分析，以下是为 {company_name} 量身定制的投资计划。该计划综合了当前技术市场趋势、宏观经济指标以及社交媒体情绪的洞察。请将此计划作为评估你下一步交易决策的基础。\n\n推荐投资计划：{investment_plan}\n\n利用这些洞察，做出明智且具有战略性的决策。"
        }


        messages = [
            {
                "role": "system",
                # "content": f"""You are a trading agent analyzing market data to make investment decisions. Based on your analysis, provide a specific recommendation to buy, sell, or hold. End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation. Do not forget to utilize lessons from past decisions to learn from your mistakes. Here is some reflections from similar situatiosn you traded in and the lessons learned: {past_memory_str}""",
                "content": f"""你是一个交易代理，负责分析市场数据并做出投资决策。根据你的分析，给出具体的买入、卖出或持有建议。请以明确的决策作为结尾，并始终以 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' 来确认你的推荐。请务必利用以往决策的经验教训，从错误中学习。以下是一些你在类似交易情境中的反思与总结出的经验教训：{past_memory_str}"""
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
