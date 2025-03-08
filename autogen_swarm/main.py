import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from autogen_swarm.hosting import container
from autogen_swarm.protocols.i_azure_openai_service import IAzureOpenAIService
from autogen_swarm.tools import (
    get_bank_account_id,
    get_investment_account_balance,
    get_saving_account_balance,
)


def get_banking_teller_agent(
    model_client: AzureOpenAIChatCompletionClient,
) -> AssistantAgent:
    return AssistantAgent(
        "banking_teller_agent",
        model_client=model_client,
        tools=[get_bank_account_id],
        handoffs=["saving_account_agent", "investment_account_agent", "user"],
        system_message="""You are a customer service representative, helps customers
with financial transactions. You need to get the customer bank account ID first and then
handoff to the other agents for assistance. The account ID for saving and investment
account are the same. When customer's request is completed, you can respond with
TERMINATE.""",
    )


def get_saving_account_agent(
    model_client: AzureOpenAIChatCompletionClient,
) -> AssistantAgent:
    return AssistantAgent(
        "saving_account_agent",
        model_client=model_client,
        handoffs=["investment_account_agent", "banking_teller_agent"],
        tools=[get_saving_account_balance],
        system_message="""You are a saving account banking agent.
You have information about the user's saving account balance. You shall provide the
saving account balance when customer ask for it. You handoff to the investment account
agent only if customer asks for investment account balance, otherwise handoff to
banking teller.""",
    )


def get_investment_account_agent(
    model_client: AzureOpenAIChatCompletionClient,
) -> AssistantAgent:
    return AssistantAgent(
        "investment_account_agent",
        model_client=model_client,
        handoffs=["banking_teller_agent"],
        tools=[get_investment_account_balance],
        system_message="""You are a investment account banking agent.
You have information about the user's investment account balance. You shall provide
the investment account balance and then you can handoff to the banking teller.""",
    )


async def main(task: str):
    openai_service = container[IAzureOpenAIService]

    model_client = openai_service.get_model()
    termination = HandoffTermination(target="user") | TextMentionTermination(
        "TERMINATE"
    )
    team = Swarm(
        [
            get_banking_teller_agent(model_client),
            get_saving_account_agent(model_client),
            get_investment_account_agent(model_client),
        ],
        termination_condition=termination,
    )

    task_result = await Console(team.run_stream(task=task))
    last_message = task_result.messages[-1]

    while isinstance(last_message, HandoffMessage) and last_message.target == "user":
        user_message = input("User: ")

        task_result = await Console(
            team.run_stream(
                task=HandoffMessage(
                    source="user", target=last_message.source, content=user_message
                )
            )
        )
        last_message = task_result.messages[-1]


if __name__ == "__main__":
    asyncio.run(main("I want to know my saving and investment account balances."))
