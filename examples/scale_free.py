# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import asyncio
import os
import random

from camel.models import ModelFactory
from camel.types import ModelPlatformType

import oasis
from oasis import ActionType, EnvAction, SingleAction


async def main():
    # NOTE: You need to deploy the vllm server first
    vllm_model_1 = ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type="qwen2.5-72b-instruct",
    )
    # Define the models for agents. Agents will select models based on
    # pre-defined scheduling strategies
    models = [vllm_model_1]

    # Define the available actions for the agents
    available_actions = [
        ActionType.CREATE_POST,
        ActionType.LIKE_POST,
        ActionType.DISLIKE_POST,
        ActionType.REPOST,
        ActionType.FOLLOW,
        ActionType.DO_NOTHING,
    ]

    # Define the path to the database
    db_path = "./data/twitter_simulation_scale_free.db"

    # Delete the old database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Make the environment
    env = oasis.make(
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
        agent_profile_path=("tmp/scale_free_network.csv"),
        agent_models=models,
        available_actions=available_actions,
    )

    # Run the environment
    await env.reset()

    # inject truth and misinformation across different topics
    business_action_truth = SingleAction(
        agent_id=0,
        action=ActionType.CREATE_POST,
        args={
            "content":
            "Amazon is expanding its delivery drone program to deliver packages within 30 minutes in select cities. This initiative aims to improve efficiency and reduce delivery times."
        })
    business_action_misinfo = SingleAction(
        agent_id=0,
        action=ActionType.CREATE_POST,
        args={
            "content":
            "Amazon plans to completely eliminate its delivery drivers within two years due to the new drone program. #Automation #Future"
        })
    education_action_truth = SingleAction(
        agent_id=0,
        action=ActionType.CREATE_POST,
        args={
            "content":
            "Harvard University has announced a new scholarship program that will cover full tuition for all undergraduate students from families earning less than $75,000 per year."
        })
    education_action_misinfo = SingleAction(
        agent_id=0,
        action=ActionType.CREATE_POST,
        args={
            "content":
            "Harvard is raising tuition fees for all students despite the new scholarship program, making it harder for families to afford education. #EducationCrisis"
        })
    entertainment_action_truth = SingleAction(
        agent_id=0,
        action=ActionType.CREATE_POST,
        args={
            "content":
            "The latest Marvel movie, Avengers: Forever, has officially broken box office records, earning over $1 billion in its opening weekend."
        })
    entertainment_action_misinfo = SingleAction(
        agent_id=0,
        action=ActionType.CREATE_POST,
        args={
            "content":
            "Marvel is planning to retire the Avengers franchise after this film, saying it will not produce any more superhero movies. #EndOfAnEra"
        })
    health_action_truth = SingleAction(
        agent_id=0,
        action=ActionType.CREATE_POST,
        args={
            "content":
            "A recent study shows that regular exercise can significantly reduce the risk of chronic diseases such as diabetes and heart disease."
        })
    health_action_misinfo = SingleAction(
        agent_id=0,
        action=ActionType.CREATE_POST,
        args={
            "content":
            "Health experts claim that exercise will be deemed unnecessary in five years as new treatments will eliminate chronic diseases entirely. #HealthRevolution"
        })

    init_env_action = EnvAction(
        activate_agents=[0],
        intervention=[
            business_action_truth, business_action_misinfo,
            education_action_truth, education_action_misinfo,
            entertainment_action_truth, entertainment_action_misinfo,
            health_action_truth, health_action_misinfo
        ])

    env_simulation_actions = [init_env_action]
    for timestep in range(60):
        # Randomly select 1% of agents to activate. This is the active probability in the paper.
        total_agents = env.agent_graph.get_num_nodes()
        num_agents_to_activate = max(1, int(
            total_agents * 0.1))  # Ensure at least 1 agent is activated
        agents_to_activate = random.sample(range(total_agents),
                                           num_agents_to_activate)

        # Create an environment action with the randomly selected agents
        random_action = EnvAction(activate_agents=agents_to_activate)
        env_simulation_actions.append(random_action)

    # Simulate 3 timesteps
    for i in range(61):
        env_actions = env_simulation_actions[i]
        # Perform the actions
        await env.step(env_actions)

    # Close the environment
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
