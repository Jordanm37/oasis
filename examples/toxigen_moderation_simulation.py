#!/usr/bin/env python
"""
Example: Content moderation simulation using ToxiGen datasets.

This example demonstrates how to use ToxiGen toxic content datasets
to create a social media simulation that studies content moderation,
user reactions to toxic content, and information spread patterns.
"""

import asyncio
import os
from pathlib import Path
import logging
from typing import Dict, List, Optional

# OASIS imports
import oasis
from oasis import ActionType, ManualAction, LLMAction
from oasis.social_agent import agents_generator
from oasis.social_platform.platform import Platform
from orchestrator.model_factory import ModelFactory
from orchestrator.model_provider import ModelPlatformType, ModelType

# ToxiGen loader
from oasis.datasets.toxigen_loader import ToxiGenLoader, create_content_moderation_simulation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def create_moderation_aware_agents(
    num_agents: int = 10,
    model=None,
    moderation_focus: bool = True
) -> oasis.AgentGraph:
    """
    Create agents with content moderation awareness.

    Args:
        num_agents: Number of agents to create
        model: LLM model to use
        moderation_focus: Whether to give agents moderation-focused personalities

    Returns:
        AgentGraph with configured agents
    """
    # Create diverse agent profiles
    profiles = []

    if moderation_focus:
        # Create agents with varying tolerance for toxic content
        agent_types = [
            {
                "type": "strict_moderator",
                "description": "Reports and blocks toxic content immediately",
                "tolerance": 0.1
            },
            {
                "type": "concerned_user",
                "description": "Worries about toxic content but engages constructively",
                "tolerance": 0.3
            },
            {
                "type": "free_speech_advocate",
                "description": "Believes in minimal moderation",
                "tolerance": 0.8
            },
            {
                "type": "neutral_observer",
                "description": "Observes but rarely engages with controversial content",
                "tolerance": 0.5
            },
            {
                "type": "community_builder",
                "description": "Tries to maintain positive community atmosphere",
                "tolerance": 0.2
            }
        ]

        for i in range(num_agents):
            agent_type = agent_types[i % len(agent_types)]
            profile = {
                "user_id": i,
                "name": f"Agent_{agent_type['type']}_{i}",
                "bio": agent_type["description"],
                "tolerance_level": agent_type["tolerance"],
                "followers_count": 100 + (i * 10),
                "following_count": 50 + (i * 5)
            }
            profiles.append(profile)
    else:
        # Create generic agents
        for i in range(num_agents):
            profile = {
                "user_id": i,
                "name": f"Agent_{i}",
                "bio": f"Regular user interested in various topics",
                "followers_count": 100 + (i * 10),
                "following_count": 50 + (i * 5)
            }
            profiles.append(profile)

    # Available actions for content moderation scenario
    available_actions = [
        ActionType.CREATE_POST,
        ActionType.CREATE_COMMENT,
        ActionType.LIKE_POST,
        ActionType.DISLIKE_POST,
        ActionType.REPORT_POST,
        ActionType.FOLLOW,
        ActionType.UNFOLLOW,
        ActionType.MUTE,
        ActionType.UNMUTE,
        ActionType.REFRESH,
        ActionType.SEARCH_POSTS,
        ActionType.DO_NOTHING
    ]

    # Create agent graph
    agent_graph = await agents_generator.generate_agent_graph_from_profiles(
        profiles=profiles,
        model=model,
        available_actions=available_actions
    )

    return agent_graph


async def seed_toxic_content(
    env: oasis.OasisEnv,
    toxic_posts: List[Dict],
    posting_agent_ids: Optional[List[int]] = None
):
    """
    Seed the environment with toxic and neutral content from ToxiGen.

    Args:
        env: OASIS environment
        toxic_posts: List of posts from ToxiGen
        posting_agent_ids: Which agents should post content
    """
    if posting_agent_ids is None:
        # Use first few agents as content seeders
        num_seeders = min(5, len(toxic_posts))
        posting_agent_ids = list(range(num_seeders))

    logger.info(f"Seeding {len(toxic_posts)} posts from {len(posting_agent_ids)} agents")

    for i, post_data in enumerate(toxic_posts):
        agent_id = posting_agent_ids[i % len(posting_agent_ids)]
        agent = env.agent_graph.get_agent(agent_id)

        # Create post with metadata
        content = post_data["content"]

        # Add context about toxicity for logging (not visible to other agents)
        metadata = f" [Type: {post_data['prompt_type']}, Demo: {post_data['demographic']}]"

        action = ManualAction(
            action_type=ActionType.CREATE_POST,
            action_args={"content": content}
        )

        # Post content
        await env.step({agent: action})

        logger.info(f"Agent {agent_id} posted: {content[:50]}...{metadata}")

    logger.info("Content seeding complete")


async def run_moderation_simulation(
    num_agents: int = 20,
    simulation_steps: int = 50,
    toxic_ratio: float = 0.2,
    focus_demographics: Optional[List[str]] = None,
    database_path: str = "./data/toxigen_simulation.db"
):
    """
    Run a complete content moderation simulation.

    Args:
        num_agents: Number of agents in simulation
        simulation_steps: Number of simulation steps
        toxic_ratio: Ratio of toxic content
        focus_demographics: Specific demographics to focus on
        database_path: Path to store simulation database
    """
    # Clean up old database if exists
    if os.path.exists(database_path):
        os.remove(database_path)
        logger.info(f"Removed existing database: {database_path}")

    # Create model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    # Create agents with moderation awareness
    logger.info("Creating moderation-aware agents...")
    agent_graph = await create_moderation_aware_agents(
        num_agents=num_agents,
        model=model,
        moderation_focus=True
    )

    # Create ToxiGen scenario
    logger.info("Loading ToxiGen content...")
    posts, config = create_content_moderation_simulation(
        agent_count=30,  # More posts than agents
        toxic_content_ratio=toxic_ratio,
        focus_demographics=focus_demographics,
        seed=42
    )

    logger.info(f"Loaded {len(posts)} posts with {toxic_ratio*100:.0f}% toxic content")
    logger.info(f"Demographics focus: {focus_demographics or 'all groups'}")

    # Create custom platform with moderation features
    platform_config = {
        "name": "ContentModPlatform",
        "max_post_length": 500,
        "enable_content_moderation": True,
        "moderation_threshold": 0.7,
        "show_reported_content": False,
        "recsys_type": "random",
        "refresh_rec_post_count": 5,
        "allow_self_rating": False
    }

    platform = Platform(platform_config)

    # Create environment
    env = oasis.make(
        agent_graph=agent_graph,
        platform=platform,
        database_path=database_path,
    )

    # Initialize environment
    await env.reset()

    # Seed toxic content
    logger.info("Seeding initial content...")
    await seed_toxic_content(env, posts[:20])  # Use first 20 posts

    # Run simulation steps
    logger.info(f"Running {simulation_steps} simulation steps...")

    for step in range(simulation_steps):
        logger.info(f"\n=== Step {step + 1}/{simulation_steps} ===")

        # All agents take LLM-driven actions
        actions = {}
        for agent_id, agent in env.agent_graph.get_agents():
            actions[agent] = LLMAction()

        # Execute step
        observations = await env.step(actions)

        # Log interesting events
        for agent, obs in observations.items():
            if obs and "reported" in str(obs).lower():
                logger.info(f"⚠️ Agent {agent.user_id} reported content")
            if obs and "toxic" in str(obs).lower():
                logger.info(f"🚫 Agent {agent.user_id} encountered toxic content")

        # Periodically add more content
        if step % 10 == 0 and step > 0:
            remaining_posts = posts[20 + step//10*2:20 + step//10*2 + 2]
            if remaining_posts:
                logger.info("Adding more content to simulation...")
                await seed_toxic_content(env, remaining_posts)

    # Close environment
    await env.close()

    # Analyze results
    logger.info("\n=== Simulation Complete ===")
    logger.info(f"Database saved to: {database_path}")
    logger.info("Use oasis.print_db_contents() to analyze results")

    # Print basic stats
    from oasis import print_db_contents
    print("\n=== Database Contents ===")
    print_db_contents(database_path)


async def analyze_moderation_patterns(database_path: str):
    """
    Analyze content moderation patterns from simulation results.

    Args:
        database_path: Path to simulation database
    """
    import sqlite3
    import pandas as pd

    conn = sqlite3.connect(database_path)

    # Analyze reported posts
    reported_posts = pd.read_sql_query(
        "SELECT * FROM post WHERE post_id IN (SELECT post_id FROM report)",
        conn
    )

    # Analyze engagement with toxic content
    all_posts = pd.read_sql_query("SELECT * FROM post", conn)

    # Get report counts
    report_counts = pd.read_sql_query(
        """
        SELECT p.post_id, p.content, COUNT(r.report_id) as report_count
        FROM post p
        LEFT JOIN report r ON p.post_id = r.post_id
        GROUP BY p.post_id
        ORDER BY report_count DESC
        """,
        conn
    )

    conn.close()

    logger.info("\n=== Moderation Analysis ===")
    logger.info(f"Total posts: {len(all_posts)}")
    logger.info(f"Reported posts: {len(reported_posts)}")
    logger.info(f"Report rate: {len(reported_posts)/len(all_posts)*100:.1f}%")

    if len(report_counts) > 0:
        logger.info("\nMost reported posts:")
        for _, row in report_counts.head(5).iterrows():
            if row['report_count'] > 0:
                logger.info(f"  - Reports: {row['report_count']}, "
                          f"Content: {row['content'][:50]}...")

    return {
        "total_posts": len(all_posts),
        "reported_posts": len(reported_posts),
        "report_rate": len(reported_posts) / len(all_posts) if len(all_posts) > 0 else 0
    }


async def main():
    """Main function to run the simulation."""
    # Check if ToxiGen data exists
    toxigen_path = Path("./data/toxigen_datasets/")
    if not toxigen_path.exists():
        logger.error("ToxiGen datasets not found!")
        logger.error("Please run: python scripts/download_toxigen_datasets.py")
        return

    # Run simulation focusing on specific demographics
    await run_moderation_simulation(
        num_agents=15,
        simulation_steps=30,
        toxic_ratio=0.25,
        focus_demographics=["women", "lgbtq", "immigrant"],
        database_path="./data/toxigen_moderation_sim.db"
    )

    # Analyze results
    await analyze_moderation_patterns("./data/toxigen_moderation_sim.db")


if __name__ == "__main__":
    asyncio.run(main())