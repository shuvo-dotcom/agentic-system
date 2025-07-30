"""
Human Simulator

This module simulates human responses to agent queries, providing realistic
interactions for testing and demonstration purposes.

Expected response_templates.json structure:
{
  "time_period": ["response1", "response2", ...],
  "parameter": ["response1", ...],
  "clarification": ["response1", ...],
  "forecasting": ["response1", ...],
  "general": ["response1", ...]
}
"""

import json
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from utils.llm_provider import get_llm_response, get_openai_client
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class HumanResponse:
    """Represents a human response to an agent query"""
    response: str
    confidence: float
    reasoning: str
    context: Dict[str, Any]

class HumanSimulator:
    """
    Simulates human responses to agent queries with realistic behavior patterns
    """
    
    def __init__(self, api_key: Optional[str] = None, template_path: Optional[str] = None):
        self.client = get_openai_client() if api_key else None
        self.conversation_history = []
        self.personality_traits = {
            "technical_knowledge": 0.7,  # 0-1 scale
            "patience": 0.8,
            "detail_oriented": 0.6,
            "preference_for_simple": 0.5
        }
        self.template_path = template_path or os.path.join(os.path.dirname(__file__), 'response_templates.json')
        self.response_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, List[str]]:
        if not os.path.exists(self.template_path):
            raise FileNotFoundError(f"Response templates file not found: {self.template_path}")
        with open(self.template_path, 'r') as f:
            templates = json.load(f)
        logger.info(f"Loaded response templates from {self.template_path}")
        return templates

    def reload_templates(self):
        self.response_templates = self._load_templates()
        logger.info("Response templates reloaded.")
        
    def generate_response(self, agent_query: str, context: Dict[str, Any] = None) -> HumanResponse:
        """
        Generate a realistic human response to an agent query
        """
        try:
            if self.client:
                return self._llm_generated_response(agent_query, context)
            else:
                return self._rule_based_response(agent_query, context)
        except Exception as e:
            logger.error(f"Error generating human response: {e}")
            return self._fallback_response(agent_query)

    def generate_parameter_value_response(self, param_name: str, param_info: Dict[str, Any], context: Dict[str, Any] = None) -> HumanResponse:
        """
        Generate a direct value response for a parameter using the LLM.
        """
        if not self.client:
            # Fallback to rule-based or generic
            return self._fallback_response(f"Parameter value for {param_name}")
        param_type = param_info.get("type", "string")
        description = param_info.get("description", "")
        prompt = (
            f"You are a user providing a value for a required parameter in an energy analysis system.\n"
            f"Parameter: {param_name}\n"
            f"Type: {param_type}\n"
            f"Description: {description}\n"
            f"Context: {context or {}}\n"
            f"Please provide ONLY the value for '{param_name}' as a valid {param_type}. Do not include any explanation, units, or extra text. Just the value."
        )
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt}
            ],
            max_tokens=20,
            temperature=0.0
        )
        value = response.choices[0].message.content.strip()
        return HumanResponse(
            response=value,
            confidence=1.0,
            reasoning="LLM direct value response for parameter",
            context={"method": "llm_param_value", "param": param_name, "type": param_type}
        )
    
    def _llm_generated_response(self, agent_query: str, context: Dict[str, Any] = None) -> HumanResponse:
        """Generate response using LLM for more realistic interactions"""
        system_prompt = f"""
        You are a human user interacting with an AI energy analysis system. 
        Respond naturally and realistically to agent queries.
        
        Your personality traits:
        - Technical knowledge level: {self.personality_traits['technical_knowledge']}
        - Patience level: {self.personality_traits['patience']}
        - Detail orientation: {self.personality_traits['detail_oriented']}
        - Preference for simple explanations: {self.personality_traits['preference_for_simple']}
        
        Context: {context or {}}
        
        Respond as a real human would - sometimes confused, sometimes helpful, 
        sometimes asking for clarification, sometimes providing detailed information.
        Keep responses conversational and natural.
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Agent asks: {agent_query}"}
            ],
            max_tokens=200,
            temperature=0.8
        )
        human_response = response.choices[0].message.content.strip()
        return HumanResponse(
            response=human_response,
            confidence=random.uniform(0.6, 0.9),
            reasoning="LLM-generated realistic human response",
            context={"method": "llm", "personality": self.personality_traits}
        )
    
    def _rule_based_response(self, agent_query: str, context: Dict[str, Any] = None) -> HumanResponse:
        """Generate response using dynamically loaded templates"""
        query_lower = agent_query.lower()
        if any(word in query_lower for word in ["time", "period", "years", "months", "duration"]):
            key = "time_period"
        elif any(word in query_lower for word in ["parameter", "value", "input", "data"]):
            key = "parameter"
        elif any(word in query_lower for word in ["clarify", "explain", "what do you mean"]):
            key = "clarification"
        elif any(word in query_lower for word in ["forecast", "prediction", "trend", "future"]):
            key = "forecasting"
        else:
            key = "general"
        responses = self.response_templates.get(key, self.response_templates.get("general", ["I'm not sure."]))
        response = random.choice(responses)
        return HumanResponse(
            response=response,
            confidence=random.uniform(0.5, 0.8),
            reasoning="Dynamic template-based response",
            context={"method": "dynamic_template", "query_type": key}
        )
    
    def _fallback_response(self, agent_query: str) -> HumanResponse:
        """Fallback response when other methods fail"""
        return HumanResponse(
            response="I'm not sure how to respond to that. Can you help me understand?",
            confidence=0.3,
            reasoning="Fallback response due to error",
            context={"method": "fallback", "error": True}
        )
    
    def update_personality(self, traits: Dict[str, float]):
        """Update personality traits for more varied responses"""
        self.personality_traits.update(traits)
        logger.info(f"Updated personality traits: {self.personality_traits}")
    
    def add_conversation_context(self, context: Dict[str, Any]):
        """Add context to conversation history"""
        self.conversation_history.append(context)
        if len(self.conversation_history) > 10:  # Keep last 10 interactions
            self.conversation_history.pop(0)
