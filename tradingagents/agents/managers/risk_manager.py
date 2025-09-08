import time
import json


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["news_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

#         prompt = f"""As the Risk Management Judge and Debate Facilitator, your goal is to evaluate the debate between three risk analysts—Risky, Neutral, and Safe/Conservative—and determine the best course of action for the trader. Your decision must result in a clear recommendation: Buy, Sell, or Hold. Choose Hold only if strongly justified by specific arguments, not as a fallback when all sides seem valid. Strive for clarity and decisiveness.
#
# Guidelines for Decision-Making:
# 1. **Summarize Key Arguments**: Extract the strongest points from each analyst, focusing on relevance to the context.
# 2. **Provide Rationale**: Support your recommendation with direct quotes and counterarguments from the debate.
# 3. **Refine the Trader's Plan**: Start with the trader's original plan, **{trader_plan}**, and adjust it based on the analysts' insights.
# 4. **Learn from Past Mistakes**: Use lessons from **{past_memory_str}** to address prior misjudgments and improve the decision you are making now to make sure you don't make a wrong BUY/SELL/HOLD call that loses money.
#
# Deliverables:
# - A clear and actionable recommendation: Buy, Sell, or Hold.
# - Detailed reasoning anchored in the debate and past reflections.
#
# ---
#
# **Analysts Debate History:**
# {history}
#
# ---
#
# Focus on actionable insights and continuous improvement. Build on past lessons, critically evaluate all perspectives, and ensure each decision advances better outcomes."""
        prompt = f"""作为风险管理裁判与辩论协调人，你的目标是评估三位风险分析师——Risky（激进型）、Neutral（中立型）和 Safe/Conservative（保守型/安全型）——之间的辩论，并为交易员确定最佳行动方案。你的决策必须得出一个明确的建议：买入（Buy）、卖出（Sell）或持有（Hold）。仅当有非常充分的特定论据支持时才选择“持有”，不要仅仅因为各方观点看似都有道理就将其作为默认选项。请力求清晰与果断。

        决策制定的指导原则：
        1. **总结关键论点**：从每位分析师的观点中提取最具说服力的论据，重点关注其与当前情境的相关性。
        2. **提供决策依据**：用辩论中的直接引述与反驳论点来支撑你的推荐意见。
        3. **优化交易员的计划**：从交易员的原计划 **{trader_plan}** 出发，根据分析师们的见解对其进行调整。
        4. **吸取过往教训**：利用 **{past_memory_str}** 中的经验，反思以往的误判，改进你当前正在做出的决策，确保不会因错误的买入/卖出/持有决策而造成亏损。

        交付内容：
        - 一个清晰且可执行的建议：买入（Buy）、卖出（Sell）或持有（Hold）。
        - 基于辩论内容和历史反思的详细推理过程。

        ---

        **分析师辩论历史记录：**  
        {history}

        ---

        专注于可落地的洞察与持续改进。充分借鉴历史经验，批判性地评估所有观点，确保每一项决策都能推动更优的交易结果。"""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
