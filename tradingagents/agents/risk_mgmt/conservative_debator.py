from langchain_core.messages import AIMessage
import time
import json


def create_safe_debator(llm):
    def safe_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        safe_history = risk_debate_state.get("safe_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]
#
#         prompt = f"""As the Safe/Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth. You prioritize stability, security, and risk mitigation, carefully assessing potential losses, economic downturns, and market volatility. When evaluating the trader's decision or plan, critically examine high-risk elements, pointing out where the decision may expose the firm to undue risk and where more cautious alternatives could secure long-term gains. Here is the trader's decision:
#
# {trader_decision}
#
# Your task is to actively counter the arguments of the Risky and Neutral Analysts, highlighting where their views may overlook potential threats or fail to prioritize sustainability. Respond directly to their points, drawing from the following data sources to build a convincing case for a low-risk approach adjustment to the trader's decision:
#
# Market Research Report: {market_research_report}
# Social Media Sentiment Report: {sentiment_report}
# Latest World Affairs Report: {news_report}
# Company Fundamentals Report: {fundamentals_report}
# Here is the current conversation history: {history} Here is the last response from the risky analyst: {current_risky_response} Here is the last response from the neutral analyst: {current_neutral_response}. If there are no responses from the other viewpoints, do not halluncinate and just present your point.
#
# Engage by questioning their optimism and emphasizing the potential downsides they may have overlooked. Address each of their counterpoints to showcase why a conservative stance is ultimately the safest path for the firm's assets. Focus on debating and critiquing their arguments to demonstrate the strength of a low-risk strategy over their approaches. Output conversationally as if you are speaking without any special formatting."""

        prompt = f"""作为安全/保守型风险分析师，你的首要目标是保护资产、降低波动性，并确保稳健、可靠的增长。你优先考虑稳定性、安全性以及风险缓释，仔细评估潜在损失、经济下行与市场波动风险。在评估交易员的决策或计划时，你要批判性地审视其中的高风险因素，指出该决策可能在哪些方面使公司面临过度风险，以及在哪些地方采取更谨慎的替代方案能够保障长期收益。

        以下是交易员的决策：

        {trader_decision}

        你的任务是积极反驳“激进型分析师”和“中立型分析师”的观点，重点指出他们的看法可能忽略了哪些潜在威胁，或未能优先考虑可持续性。针对他们的论点进行直接回应，并利用以下数据来源，构建一个有说服力的、主张低风险调整的论证，以优化交易员的决策：

        市场研究报告：{market_research_report}
        社交媒体情绪报告：{sentiment_report}
        最新全球事务报告：{news_report}
        公司基本面报告：{fundamentals_report}

        以下是对话历史记录：{history}
        以下是激进型分析师的最新回应：{current_risky_response}
        以下是中立型分析师的最新回应：{current_neutral_response}

        如果其他视角没有提供回应，请不要虚构内容，只需陈述你自己的观点。

        通过质疑他们的乐观态度，强调他们可能忽略的潜在负面影响。针对他们的每一个反驳点，阐述为什么保守立场最终才是保障公司资产安全的最稳妥路径。重点在于通过辩论和批驳他们的论点，来证明低风险策略相比他们所持立场的优势。以对话的方式输出内容，就像你在自然交谈一样，无需任何特殊格式。"""
        response = llm.invoke(prompt)

        argument = f"Safe Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": safe_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Safe",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return safe_node
