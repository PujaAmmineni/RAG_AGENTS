# agents.py
from autogen import AssistantAgent, GroupChat, GroupChatManager
from typing import Dict
import os

class RAGAgents:
    def __init__(self, config: Dict):
        self.base_config = {
            "config_list": [{
                "model": config["chat_deployment"],
                "api_type": "azure",
                "api_key": config["api_key"],
                "base_url": config["azure_endpoint"],
                "api_version": config["api_version"]
            }]
        }

        os.environ.update({
            "OPENAI_API_KEY": config["api_key"],
            "OPENAI_API_TYPE": "azure",
            "OPENAI_API_VERSION": config["api_version"],
            "OPENAI_API_BASE": config["azure_endpoint"]
        })

        
        self.commander = AssistantAgent(
            name="Commander",
            system_message="""You are the Research Commander. Your role is to:
            1. Analyze the question thoroughly
            2. Identify key concepts and requirements
            3. Plan the research approach
            Start your response with 'ANALYSIS:'""",
            llm_config=self.base_config
        )

        self.prover = AssistantAgent(
            name="Prover",
            system_message="""You are the Evidence Prover. Your role is to:
            1. Search through the provided context
            2. Find relevant evidence and citations
            3. Support or challenge claims with specific examples
            Start your response with 'EVIDENCE:'""",
            llm_config=self.base_config
        )

        self.verifier = AssistantAgent(
            name="Verifier",
            system_message="""You are the Final Verifier. Your role is to:
            1. Review the analysis and evidence
            2. Verify the accuracy and completeness
            3. Provide a clear, concise summary
            Start your response with 'FINAL ANSWER:'""",
            llm_config=self.base_config
        )

        self.generalist = AssistantAgent(
            name="Generalist",
            system_message="""You are a fallback AI agent. If the context is insufficient or unrelated to the user query, generate a general but accurate response to help the user. 
            Start your response with 'GENERAL ANSWER:'""",
            llm_config=self.base_config
        )

        self.group_chat = GroupChat(
            agents=[self.commander, self.prover, self.verifier],
            messages=[],
            max_round=3
        )

        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.base_config
        )

    def process_query(self, query: str, context: str) -> str:
        try:
            prompt = f"""
            Question: {query}

            Context from documents:
            {context}

            Please analyze this in three steps:
            1. First, analyze the question and context (Commander)
            2. Then, provide specific evidence from the documents (Prover)
            3. Finally, verify and summarize the findings (Verifier)

            Each response must start with the appropriate marker:
            ANALYSIS:
            EVIDENCE:
            FINAL ANSWER:
            """

            if not context.strip():
                general_prompt = f"The user asked: '{query}'\n\nThere is no information found in the documents. Please provide a general answer."
                result = self.generalist.run(general_prompt)
                return "GENERAL ANSWER: " + result

            result = self.manager.initiate_chat(sender=self.commander, recipient=self.prover, message=prompt)

            if hasattr(result, 'messages'):
                responses = [str(msg.content) for msg in result.messages if hasattr(msg, 'content')]
                return "\n\n".join(responses)

            return str(result)

        except Exception as e:
            return f"Error in agent processing: {str(e)}"