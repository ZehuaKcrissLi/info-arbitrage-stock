import logging
import asyncio
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import openai
import anthropic

from ..models.analysis import AnalysisBundle
from ..models.trading import DebateResult, TradingRecommendation, TradingAction
from ..config.settings import settings

logger = logging.getLogger(__name__)

class DebateCoordinator:
    """Coordinates debate between bull and bear agents"""
    
    def __init__(self):
        self.bull_agent = BullResearchAgent()
        self.bear_agent = BearResearchAgent()
        self.max_rounds = 3
        self.llm_client = self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize LLM client based on configuration"""
        if settings.llm.provider == "openai":
            openai.api_key = settings.openai_api_key
            return openai
        elif settings.llm.provider == "anthropic":
            return anthropic.Anthropic(api_key=settings.anthropic_api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.llm.provider}")
    
    async def conduct_debate(self, analyses: AnalysisBundle) -> DebateResult:
        """Orchestrate debate between bull and bear agents"""
        
        logger.info(f"Starting debate for event {analyses.event_id}")
        
        debate_history = []
        
        try:
            # Initialize positions
            bull_position = await self.bull_agent.formulate_position(analyses)
            bear_position = await self.bear_agent.formulate_position(analyses)
            
            debate_history.append(("BULL_OPENING", bull_position))
            debate_history.append(("BEAR_OPENING", bear_position))
            
            # Conduct debate rounds
            for round_num in range(self.max_rounds):
                logger.info(f"Debate round {round_num + 1}")
                
                # Bear responds to bull
                bear_response = await self.bear_agent.respond_to_bull(
                    bull_position, debate_history, analyses
                )
                debate_history.append((f"BEAR_ROUND_{round_num}", bear_response))
                
                # Bull responds to bear
                bull_response = await self.bull_agent.respond_to_bear(
                    bear_response, debate_history, analyses
                )
                debate_history.append((f"BULL_ROUND_{round_num}", bull_response))
                
                # Check for convergence
                if self._check_convergence(debate_history):
                    logger.info("Debate converged early")
                    break
            
            # Generate final recommendation
            final_recommendation = await self._synthesize_debate(debate_history, analyses)
            
            # Calculate metrics
            confidence_score = self._calculate_debate_confidence(debate_history)
            consensus_level = self._measure_consensus(debate_history)
            
            return DebateResult(
                event_id=analyses.event_id,
                debate_history=debate_history,
                final_recommendation=final_recommendation,
                confidence_score=confidence_score,
                consensus_level=consensus_level
            )
            
        except Exception as e:
            logger.error(f"Error in debate: {e}")
            # Return default recommendation on error
            return DebateResult(
                event_id=analyses.event_id,
                debate_history=debate_history,
                final_recommendation=TradingRecommendation(
                    ticker=analyses.fundamental.ticker_analyses[0].ticker if analyses.fundamental and analyses.fundamental.ticker_analyses else "UNKNOWN",
                    action=TradingAction.HOLD,
                    rationale="Debate failed, defaulting to HOLD",
                    confidence=0.1
                ),
                confidence_score=0.1,
                consensus_level=0.0
            )
    
    def _check_convergence(self, debate_history: List[Tuple[str, str]]) -> bool:
        """Check if debate has converged to a conclusion"""
        if len(debate_history) < 4:
            return False
        
        # Simple convergence check - if last bull and bear positions are similar
        last_bull = None
        last_bear = None
        
        for role, position in reversed(debate_history):
            if role.startswith("BULL") and last_bull is None:
                last_bull = position
            elif role.startswith("BEAR") and last_bear is None:
                last_bear = position
            
            if last_bull and last_bear:
                break
        
        if last_bull and last_bear:
            # Check for convergence keywords
            convergence_keywords = ["agree", "consensus", "similar", "aligned"]
            bull_lower = last_bull.lower()
            bear_lower = last_bear.lower()
            
            return any(keyword in bull_lower or keyword in bear_lower for keyword in convergence_keywords)
        
        return False
    
    async def _synthesize_debate(self, debate_history: List[Tuple[str, str]], 
                               analyses: AnalysisBundle) -> TradingRecommendation:
        """Synthesize debate into final trading recommendation"""
        
        try:
            # Create synthesis prompt
            synthesis_prompt = self._create_synthesis_prompt(debate_history, analyses)
            
            # Get LLM response
            if settings.llm.provider == "openai":
                response = await self._get_openai_response(synthesis_prompt)
            else:
                response = await self._get_anthropic_response(synthesis_prompt)
            
            # Parse the response
            return self._parse_trading_recommendation(response, analyses)
            
        except Exception as e:
            logger.error(f"Error synthesizing debate: {e}")
            # Return default recommendation
            return TradingRecommendation(
                ticker=analyses.fundamental.ticker_analyses[0].ticker if analyses.fundamental and analyses.fundamental.ticker_analyses else "UNKNOWN",
                action=TradingAction.HOLD,
                rationale="Error in synthesis, defaulting to HOLD",
                confidence=0.1
            )
    
    def _create_synthesis_prompt(self, debate_history: List[Tuple[str, str]], 
                               analyses: AnalysisBundle) -> str:
        """Create prompt for debate synthesis"""
        
        # Format debate history
        debate_text = "\n\n".join([f"{role}: {position}" for role, position in debate_history])
        
        # Format analysis summary
        analysis_summary = self._format_analyses_summary(analyses)
        
        prompt = f"""
You are a senior trading strategist tasked with making a final trading decision based on a debate between bull and bear analysts.

ANALYSIS DATA:
{analysis_summary}

DEBATE TRANSCRIPT:
{debate_text}

Based on the analysis data and debate, provide a final trading recommendation in the following format:

ACTION: [BUY/SELL/HOLD]
TICKER: [Primary ticker symbol]
CONFIDENCE: [0.0-1.0]
RATIONALE: [Brief explanation of reasoning]

Consider:
1. Which side presented stronger arguments
2. Quality of supporting evidence
3. Risk vs reward potential
4. Market timing factors
5. Overall consensus or lack thereof

Provide your recommendation:
"""
        
        return prompt
    
    def _format_analyses_summary(self, analyses: AnalysisBundle) -> str:
        """Format analyses into readable summary"""
        summary_parts = []
        
        if analyses.fundamental:
            summary_parts.append(f"Fundamental: Overall sentiment {analyses.fundamental.overall_sentiment:.2f}")
        
        if analyses.technical:
            summary_parts.append(f"Technical: Direction {analyses.technical.overall_direction}")
        
        if analyses.sentiment:
            summary_parts.append(f"Sentiment: Aggregate {analyses.sentiment.aggregate_sentiment:.2f}")
        
        if analyses.macro:
            summary_parts.append(f"Macro: Risk assessment {analyses.macro.risk_assessment}")
        
        return " | ".join(summary_parts)
    
    async def _get_openai_response(self, prompt: str) -> str:
        """Get response from OpenAI"""
        try:
            response = await openai.ChatCompletion.acreate(
                model=settings.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "ERROR: Could not get LLM response"
    
    async def _get_anthropic_response(self, prompt: str) -> str:
        """Get response from Anthropic"""
        try:
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            response = client.messages.create(
                model=settings.llm.model,
                max_tokens=settings.llm.max_tokens,
                temperature=settings.llm.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return "ERROR: Could not get LLM response"
    
    def _parse_trading_recommendation(self, response: str, analyses: AnalysisBundle) -> TradingRecommendation:
        """Parse LLM response into TradingRecommendation"""
        
        # Default values
        action = TradingAction.HOLD
        ticker = "UNKNOWN"
        confidence = 0.5
        rationale = "Unable to parse recommendation"
        
        try:
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                if line.startswith("ACTION:"):
                    action_str = line.replace("ACTION:", "").strip()
                    if action_str.upper() in ["BUY", "SELL", "HOLD"]:
                        action = TradingAction(action_str.upper())
                
                elif line.startswith("TICKER:"):
                    ticker = line.replace("TICKER:", "").strip()
                
                elif line.startswith("CONFIDENCE:"):
                    confidence_str = line.replace("CONFIDENCE:", "").strip()
                    try:
                        confidence = float(confidence_str)
                        confidence = max(0.0, min(1.0, confidence))
                    except:
                        confidence = 0.5
                
                elif line.startswith("RATIONALE:"):
                    rationale = line.replace("RATIONALE:", "").strip()
            
            # Use first ticker from analyses if parsing failed
            if ticker == "UNKNOWN" and analyses.fundamental and analyses.fundamental.ticker_analyses:
                ticker = analyses.fundamental.ticker_analyses[0].ticker
            
        except Exception as e:
            logger.error(f"Error parsing recommendation: {e}")
        
        return TradingRecommendation(
            ticker=ticker,
            action=action,
            rationale=rationale,
            confidence=confidence
        )
    
    def _calculate_debate_confidence(self, debate_history: List[Tuple[str, str]]) -> float:
        """Calculate confidence score based on debate quality"""
        
        if not debate_history:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # More rounds = more thorough analysis
        num_rounds = len([h for h in debate_history if "ROUND" in h[0]])
        confidence += min(0.2, num_rounds * 0.05)
        
        # Check for evidence-based arguments
        evidence_keywords = ["data", "analysis", "metric", "indicator", "trend", "performance"]
        evidence_count = 0
        
        for role, position in debate_history:
            for keyword in evidence_keywords:
                if keyword in position.lower():
                    evidence_count += 1
        
        # Boost confidence based on evidence
        confidence += min(0.2, evidence_count * 0.02)
        
        # Check for balanced discussion
        bull_count = len([h for h in debate_history if "BULL" in h[0]])
        bear_count = len([h for h in debate_history if "BEAR" in h[0]])
        
        if bull_count > 0 and bear_count > 0:
            balance = min(bull_count, bear_count) / max(bull_count, bear_count)
            confidence += balance * 0.1
        
        return min(1.0, confidence)
    
    def _measure_consensus(self, debate_history: List[Tuple[str, str]]) -> float:
        """Measure level of consensus in debate"""
        
        if len(debate_history) < 2:
            return 0.0
        
        # Look for agreement/disagreement keywords
        agreement_keywords = ["agree", "correct", "right", "valid", "good point"]
        disagreement_keywords = ["disagree", "wrong", "incorrect", "invalid", "however", "but"]
        
        agreement_count = 0
        disagreement_count = 0
        
        for role, position in debate_history:
            position_lower = position.lower()
            
            for keyword in agreement_keywords:
                if keyword in position_lower:
                    agreement_count += 1
            
            for keyword in disagreement_keywords:
                if keyword in position_lower:
                    disagreement_count += 1
        
        total_interactions = agreement_count + disagreement_count
        
        if total_interactions == 0:
            return 0.5  # Neutral consensus
        
        consensus_level = agreement_count / total_interactions
        return consensus_level

class BullResearchAgent:
    """Agent that advocates for bullish positions"""
    
    def __init__(self):
        self.llm_client = self._initialize_llm()
        self.system_prompt = self._load_bull_system_prompt()
    
    def _initialize_llm(self):
        """Initialize LLM client"""
        if settings.llm.provider == "openai":
            openai.api_key = settings.openai_api_key
            return openai
        elif settings.llm.provider == "anthropic":
            return anthropic.Anthropic(api_key=settings.anthropic_api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.llm.provider}")
    
    def _load_bull_system_prompt(self) -> str:
        """Load system prompt for bull agent"""
        return """
You are a bullish stock analyst whose job is to find and present the strongest possible case for buying stocks. 
You should:
- Focus on positive fundamentals, growth prospects, and favorable technicals
- Highlight opportunities and upside potential
- Address bearish concerns by showing why they are overblown or temporary
- Use data and analysis to support your bullish thesis
- Be optimistic but grounded in facts
- Present compelling arguments for why this is a good buying opportunity
"""
    
    async def formulate_position(self, analyses: AnalysisBundle) -> str:
        """Formulate initial bullish position"""
        
        prompt = f"""
{self.system_prompt}

Based on the following analysis data, formulate a strong bullish argument:

Fundamental Analysis: {self._format_fundamental_analysis(analyses.fundamental)}
Technical Analysis: {self._format_technical_analysis(analyses.technical)}
Sentiment Analysis: {self._format_sentiment_analysis(analyses.sentiment)}
Macro Analysis: {self._format_macro_analysis(analyses.macro)}

Focus on:
1. Strongest positive factors
2. Why negative factors are overblown or temporary
3. Specific catalysts for upward movement
4. Risk/reward justification

Provide a clear, compelling bullish case in 2-3 paragraphs.
"""
        
        return await self._get_llm_response(prompt)
    
    async def respond_to_bear(self, bear_argument: str, debate_history: List[Tuple[str, str]],
                            analyses: AnalysisBundle) -> str:
        """Respond to bear's arguments"""
        
        prompt = f"""
{self.system_prompt}

The bear analyst has made the following argument:
{bear_argument}

Counter this argument while maintaining your bullish stance. Use data from:
{self._format_analyses_summary(analyses)}

Address their concerns directly but show why the bullish case still holds.
Provide a strong rebuttal in 1-2 paragraphs.
"""
        
        return await self._get_llm_response(prompt)
    
    def _format_fundamental_analysis(self, analysis) -> str:
        """Format fundamental analysis for prompt"""
        if not analysis:
            return "No fundamental analysis available"
        
        return f"Overall sentiment: {analysis.overall_sentiment:.2f}, {len(analysis.ticker_analyses)} tickers analyzed"
    
    def _format_technical_analysis(self, analysis) -> str:
        """Format technical analysis for prompt"""
        if not analysis:
            return "No technical analysis available"
        
        return f"Overall direction: {analysis.overall_direction}, {len(analysis.ticker_analyses)} tickers analyzed"
    
    def _format_sentiment_analysis(self, analysis) -> str:
        """Format sentiment analysis for prompt"""
        if not analysis:
            return "No sentiment analysis available"
        
        return f"Aggregate sentiment: {analysis.aggregate_sentiment:.2f}, News sentiment: {analysis.news_sentiment:.2f}"
    
    def _format_macro_analysis(self, analysis) -> str:
        """Format macro analysis for prompt"""
        if not analysis:
            return "No macro analysis available"
        
        return f"Risk assessment: {analysis.risk_assessment}"
    
    def _format_analyses_summary(self, analyses: AnalysisBundle) -> str:
        """Format all analyses for prompt"""
        parts = []
        
        if analyses.fundamental:
            parts.append(f"Fundamental: {analyses.fundamental.overall_sentiment:.2f}")
        if analyses.technical:
            parts.append(f"Technical: {analyses.technical.overall_direction}")
        if analyses.sentiment:
            parts.append(f"Sentiment: {analyses.sentiment.aggregate_sentiment:.2f}")
        if analyses.macro:
            parts.append(f"Macro Risk: {analyses.macro.risk_assessment}")
        
        return " | ".join(parts)
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM"""
        try:
            if settings.llm.provider == "openai":
                response = await openai.ChatCompletion.acreate(
                    model=settings.llm.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings.llm.temperature,
                    max_tokens=settings.llm.max_tokens
                )
                return response.choices[0].message.content
            else:
                client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
                response = client.messages.create(
                    model=settings.llm.model,
                    max_tokens=settings.llm.max_tokens,
                    temperature=settings.llm.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return "I strongly believe this presents a compelling buying opportunity based on the available data."

class BearResearchAgent:
    """Agent that advocates for bearish/cautious positions"""
    
    def __init__(self):
        self.llm_client = self._initialize_llm()
        self.system_prompt = self._load_bear_system_prompt()
    
    def _initialize_llm(self):
        """Initialize LLM client"""
        if settings.llm.provider == "openai":
            openai.api_key = settings.openai_api_key
            return openai
        elif settings.llm.provider == "anthropic":
            return anthropic.Anthropic(api_key=settings.anthropic_api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.llm.provider}")
    
    def _load_bear_system_prompt(self) -> str:
        """Load system prompt for bear agent"""
        return """
You are a bearish/cautious stock analyst whose job is to identify risks and present the case for avoiding or selling stocks.
You should:
- Focus on negative fundamentals, risks, and unfavorable technicals
- Highlight potential downside and risk factors
- Show why positive factors may be temporary or already priced in
- Use data and analysis to support your bearish/cautious thesis
- Be skeptical but grounded in facts
- Present compelling arguments for why this is not a good buying opportunity
"""
    
    async def formulate_position(self, analyses: AnalysisBundle) -> str:
        """Formulate initial bearish position"""
        
        prompt = f"""
{self.system_prompt}

Based on the following analysis data, formulate a strong bearish/cautious argument:

Fundamental Analysis: {self._format_fundamental_analysis(analyses.fundamental)}
Technical Analysis: {self._format_technical_analysis(analyses.technical)}
Sentiment Analysis: {self._format_sentiment_analysis(analyses.sentiment)}
Macro Analysis: {self._format_macro_analysis(analyses.macro)}

Focus on:
1. Key risk factors and red flags
2. Why positive factors are already priced in or temporary
3. Potential catalysts for downward movement
4. Market timing concerns

Provide a clear, compelling bearish case in 2-3 paragraphs.
"""
        
        return await self._get_llm_response(prompt)
    
    async def respond_to_bull(self, bull_argument: str, debate_history: List[Tuple[str, str]],
                            analyses: AnalysisBundle) -> str:
        """Respond to bull's arguments"""
        
        prompt = f"""
{self.system_prompt}

The bull analyst has made the following argument:
{bull_argument}

Counter this argument while maintaining your bearish/cautious stance. Use data from:
{self._format_analyses_summary(analyses)}

Address their points directly but show why caution or bearishness is warranted.
Provide a strong rebuttal in 1-2 paragraphs.
"""
        
        return await self._get_llm_response(prompt)
    
    def _format_fundamental_analysis(self, analysis) -> str:
        """Format fundamental analysis for prompt"""
        if not analysis:
            return "No fundamental analysis available"
        
        return f"Overall sentiment: {analysis.overall_sentiment:.2f}, {len(analysis.ticker_analyses)} tickers analyzed"
    
    def _format_technical_analysis(self, analysis) -> str:
        """Format technical analysis for prompt"""
        if not analysis:
            return "No technical analysis available"
        
        return f"Overall direction: {analysis.overall_direction}, {len(analysis.ticker_analyses)} tickers analyzed"
    
    def _format_sentiment_analysis(self, analysis) -> str:
        """Format sentiment analysis for prompt"""
        if not analysis:
            return "No sentiment analysis available"
        
        return f"Aggregate sentiment: {analysis.aggregate_sentiment:.2f}, News sentiment: {analysis.news_sentiment:.2f}"
    
    def _format_macro_analysis(self, analysis) -> str:
        """Format macro analysis for prompt"""
        if not analysis:
            return "No macro analysis available"
        
        return f"Risk assessment: {analysis.risk_assessment}"
    
    def _format_analyses_summary(self, analyses: AnalysisBundle) -> str:
        """Format all analyses for prompt"""
        parts = []
        
        if analyses.fundamental:
            parts.append(f"Fundamental: {analyses.fundamental.overall_sentiment:.2f}")
        if analyses.technical:
            parts.append(f"Technical: {analyses.technical.overall_direction}")
        if analyses.sentiment:
            parts.append(f"Sentiment: {analyses.sentiment.aggregate_sentiment:.2f}")
        if analyses.macro:
            parts.append(f"Macro Risk: {analyses.macro.risk_assessment}")
        
        return " | ".join(parts)
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM"""
        try:
            if settings.llm.provider == "openai":
                response = await openai.ChatCompletion.acreate(
                    model=settings.llm.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings.llm.temperature,
                    max_tokens=settings.llm.max_tokens
                )
                return response.choices[0].message.content
            else:
                client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
                response = client.messages.create(
                    model=settings.llm.model,
                    max_tokens=settings.llm.max_tokens,
                    temperature=settings.llm.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return "I believe caution is warranted given the current risk factors and market conditions."