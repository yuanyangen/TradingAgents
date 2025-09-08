# TradingAgents/graph/reflection.py

from typing import Dict, Any
from langchain_openai import ChatOpenAI


class Reflector:
    """Handles reflection on decisions and updating memory."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize the reflector with an LLM."""
        self.quick_thinking_llm = quick_thinking_llm
        self.reflection_system_prompt = self._get_reflection_prompt()

    def _get_reflection_prompt(self) -> str:
        """Get the system prompt for reflection."""
        return """
你是一名专业的金融分析师，负责审查交易决策/分析，并提供全面、逐步深入的分析报告。
你的目标是针对投资决策给出详细的洞察，指出其中的机会与改进空间，并严格遵循以下指导原则：

1. 决策推理（Reasoning）：
 - 针对每一项交易决策，判断其是否正确。正确的决策 应带来收益的提升，错误的决策 则导致收益下降或亏损。
 - 针对每一次成功或失误，分析其影响因素，考虑以下维度：
 - 市场情报（Market intelligence）。
 - 技术指标（Technical indicators）。
 - 技术信号（Technical signals）。
 - 价格走势分析（Price movement analysis）。
 - 整体市场数据分析（Overall market data analysis）。
 - 新闻分析（News analysis）。
 - 社交媒体与情绪分析（Social media and sentiment analysis）。
 - 基本面数据分析（Fundamental data analysis）。
 - 对每一个因素在决策过程中的重要性进行权重分析与评估。

2. 改进方案（Improvement）：
 - 对于任何错误的决策，提出能够最大化收益的修正建议。
 - 提供一份详细的改进措施或优化行动清单，包括具体建议（例如：将某日持仓状态从 HOLD 修改为 BUY）。

3. 总结反思（Summary）：
 - 总结从成功与失误中获得的经验教训。
 - 强调这些经验如何能够应用于未来的交易场景，找出相似情形之间的关联性，以便复用已有的知识与判断。

4. 信息提炼（Query）：从上述总结中提取核心洞见，精炼成一句不超过 1000 个 token 的简洁语句。
 - 确保该浓缩语句准确抓住经验教训与推理逻辑的精髓，便于后续快速参考与复用。
 - 请严格遵循以上指令，确保你的输出内容详细、准确、具备可操作性。

此外，你还将获得来自客观市场数据的描述，包括：价格走势（price movements），技术指标（technical indicators），新闻资讯（news），市场情绪（sentiment）。这些附加信息将为你提供更全面的分析背景与上下文支持。
"""

    def _extract_current_situation(self, current_state: Dict[str, Any]) -> str:
        """Extract the current market situation from the state."""
        curr_market_report = current_state["market_report"]
        curr_sentiment_report = current_state["sentiment_report"]
        curr_news_report = current_state["news_report"]
        curr_fundamentals_report = current_state["fundamentals_report"]

        return f"{curr_market_report}\n\n{curr_sentiment_report}\n\n{curr_news_report}\n\n{curr_fundamentals_report}"

    def _reflect_on_component(
        self, component_type: str, report: str, situation: str, returns_losses
    ) -> str:
        """Generate reflection for a component."""
        messages = [
            ("system", self.reflection_system_prompt),
            (
                "human",
                f"Returns: {returns_losses}\n\nAnalysis/Decision: {report}\n\nObjective Market Reports for Reference: {situation}",
            ),
        ]

        result = self.quick_thinking_llm.invoke(messages).content
        return result

    def reflect_bull_researcher(self, current_state, returns_losses, bull_memory):
        """Reflect on bull researcher's analysis and update memory."""
        situation = self._extract_current_situation(current_state)
        bull_debate_history = current_state["investment_debate_state"]["bull_history"]

        result = self._reflect_on_component(
            "BULL", bull_debate_history, situation, returns_losses
        )
        bull_memory.add_situations([(situation, result)])

    def reflect_bear_researcher(self, current_state, returns_losses, bear_memory):
        """Reflect on bear researcher's analysis and update memory."""
        situation = self._extract_current_situation(current_state)
        bear_debate_history = current_state["investment_debate_state"]["bear_history"]

        result = self._reflect_on_component(
            "BEAR", bear_debate_history, situation, returns_losses
        )
        bear_memory.add_situations([(situation, result)])

    def reflect_trader(self, current_state, returns_losses, trader_memory):
        """Reflect on trader's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        trader_decision = current_state["trader_investment_plan"]

        result = self._reflect_on_component(
            "TRADER", trader_decision, situation, returns_losses
        )
        trader_memory.add_situations([(situation, result)])

    def reflect_invest_judge(self, current_state, returns_losses, invest_judge_memory):
        """Reflect on investment judge's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["investment_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "INVEST JUDGE", judge_decision, situation, returns_losses
        )
        invest_judge_memory.add_situations([(situation, result)])

    def reflect_risk_manager(self, current_state, returns_losses, risk_manager_memory):
        """Reflect on risk manager's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["risk_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "RISK JUDGE", judge_decision, situation, returns_losses
        )
        risk_manager_memory.add_situations([(situation, result)])
