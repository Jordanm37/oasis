#!/usr/bin/env python
"""
Example: Unified toxicity simulation using both ToxiGen and ImplicitHate datasets.

This example demonstrates how to create a comprehensive content moderation
simulation that includes both explicit and implicit forms of hate speech,
testing agent responses to different types of toxic content.
"""

import asyncio
import os
from pathlib import Path
import logging
from typing import Dict, List, Optional
import json

# OASIS imports
import oasis
from oasis import ActionType, ManualAction, LLMAction
from oasis.social_agent import agents_generator
from oasis.social_platform.platform import Platform
from orchestrator.model_factory import ModelFactory
from orchestrator.model_provider import ModelPlatformType, ModelType

# Toxicity dataset loaders
from oasis.datasets.unified_toxicity_loader import UnifiedToxicityLoader, ToxicityType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def create_diverse_moderation_agents(
    num_agents: int = 15,
    model=None
) -> oasis.AgentGraph:
    """
    Create agents with diverse perspectives on content moderation.

    Args:
        num_agents: Number of agents to create
        model: LLM model to use

    Returns:
        AgentGraph with configured agents
    """
    # Define agent archetypes for content moderation
    archetypes = [
        {
            "name": "strict_moderator",
            "bio": "Believes in zero tolerance for any form of hate speech, including subtle implications",
            "traits": "vigilant, protective, cautious",
            "detection_sensitivity": "high"
        },
        {
            "name": "nuance_aware",
            "bio": "Understands the difference between explicit and implicit hate, considers context",
            "traits": "analytical, thoughtful, balanced",
            "detection_sensitivity": "medium"
        },
        {
            "name": "free_speech_advocate",
            "bio": "Prioritizes freedom of expression, only flags the most egregious content",
            "traits": "libertarian, permissive, principled",
            "detection_sensitivity": "low"
        },
        {
            "name": "community_guardian",
            "bio": "Focuses on maintaining a positive community atmosphere and protecting vulnerable groups",
            "traits": "empathetic, protective, community-focused",
            "detection_sensitivity": "medium-high"
        },
        {
            "name": "context_analyzer",
            "bio": "Examines posts for dog whistles and coded language often used in implicit hate",
            "traits": "detail-oriented, knowledgeable, perceptive",
            "detection_sensitivity": "high"
        }
    ]

    profiles = []
    for i in range(num_agents):
        archetype = archetypes[i % len(archetypes)]
        profile = {
            "user_id": i,
            "name": f"Agent_{archetype['name']}_{i}",
            "bio": archetype["bio"],
            "traits": archetype["traits"],
            "detection_sensitivity": archetype["detection_sensitivity"],
            "followers_count": 100 + (i * 10),
            "following_count": 50 + (i * 5)
        }
        profiles.append(profile)

    # Available actions for moderation scenario
    available_actions = [
        ActionType.CREATE_POST,
        ActionType.CREATE_COMMENT,
        ActionType.LIKE_POST,
        ActionType.DISLIKE_POST,
        ActionType.REPORT_POST,  # Key action for moderation
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
        available_actions=available_actions,
        custom_instructions="""
        You are participating in a content moderation study. Your role is to:
        1. Browse content and identify potentially harmful posts
        2. React appropriately based on your detection sensitivity
        3. Report content that violates community guidelines
        4. Consider both explicit hate and subtle, implicit forms of toxicity
        """
    )

    return agent_graph


async def seed_mixed_toxicity_content(
    env: oasis.OasisEnv,
    loader: UnifiedToxicityLoader,
    num_explicit: int = 5,
    num_implicit: int = 10,
    num_neutral: int = 15
):
    """
    Seed the environment with a mix of explicit hate, implicit hate, and neutral content.

    Args:
        env: OASIS environment
        loader: Unified toxicity loader
        num_explicit: Number of explicit hate posts
        num_implicit: Number of implicit hate posts
        num_neutral: Number of neutral posts
    """
    # Create mixed content
    posts = loader.create_mixed_toxicity_scenario(
        num_explicit_hate=num_explicit,
        num_implicit_hate=num_implicit,
        num_neutral=num_neutral,
        demographics=["women", "immigrant", "lgbtq"],
        implicit_classes=["white_grievance", "irony", "stereotypical"],
        random_seed=42
    )

    logger.info(f"Seeding {len(posts)} posts with mixed toxicity levels")

    # Track seeded content for analysis
    seeded_content = []

    # Have different agents post different content types
    agent_ids = list(range(min(5, env.agent_graph.num_agents)))

    for i, post_data in enumerate(posts):
        agent_id = agent_ids[i % len(agent_ids)]
        agent = env.agent_graph.get_agent(agent_id)

        content = post_data["content"]
        toxicity_type = post_data["toxicity_type"]

        # Create post
        action = ManualAction(
            action_type=ActionType.CREATE_POST,
            action_args={"content": content}
        )

        await env.step({agent: action})

        # Track for analysis
        seeded_content.append({
            "agent_id": agent_id,
            "content": content[:100] + "..." if len(content) > 100 else content,
            "toxicity_type": toxicity_type,
            "source": post_data.get("source", "unknown")
        })

        # Log based on toxicity type
        if toxicity_type == ToxicityType.EXPLICIT_HATE.value:
            logger.info(f"🔴 Agent {agent_id} posted EXPLICIT hate content")
        elif toxicity_type == ToxicityType.IMPLICIT_HATE.value:
            logger.info(f"🟡 Agent {agent_id} posted IMPLICIT hate content")
        else:
            logger.info(f"🟢 Agent {agent_id} posted NEUTRAL content")

    return seeded_content


async def run_unified_toxicity_simulation(
    num_agents: int = 15,
    simulation_steps: int = 40,
    database_path: str = "./data/unified_toxicity_sim.db",
    output_analysis: bool = True
):
    """
    Run a comprehensive toxicity detection simulation with both datasets.

    Args:
        num_agents: Number of agents
        simulation_steps: Number of simulation steps
        database_path: Path to store simulation database
        output_analysis: Whether to output analysis at the end
    """
    # Clean up old database if exists
    if os.path.exists(database_path):
        os.remove(database_path)
        logger.info(f"Removed existing database: {database_path}")

    # Initialize unified loader
    try:
        loader = UnifiedToxicityLoader()
        logger.info(f"Loaded datasets: {loader.available_datasets}")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        logger.error("Please run download scripts first:")
        logger.error("  python scripts/download_toxigen_datasets.py")
        logger.error("  python scripts/download_implicithate_dataset.py")
        return

    # Create model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    # Create diverse moderation agents
    logger.info("Creating diverse moderation agents...")
    agent_graph = await create_diverse_moderation_agents(
        num_agents=num_agents,
        model=model
    )

    # Create platform with enhanced moderation features
    platform_config = {
        "name": "UnifiedToxicityTestPlatform",
        "max_post_length": 500,
        "enable_content_moderation": True,
        "track_reports": True,
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

    # Seed mixed toxicity content
    logger.info("\n=== Seeding Mixed Toxicity Content ===")
    seeded_content = await seed_mixed_toxicity_content(
        env=env,
        loader=loader,
        num_explicit=5,
        num_implicit=10,
        num_neutral=15
    )

    # Track simulation metrics
    metrics = {
        "reports_by_type": {"explicit": 0, "implicit": 0, "neutral": 0},
        "agent_actions": {},
        "detection_patterns": []
    }

    # Run simulation steps
    logger.info(f"\n=== Running {simulation_steps} Simulation Steps ===")

    for step in range(simulation_steps):
        logger.info(f"\nStep {step + 1}/{simulation_steps}")

        # All agents take LLM-driven actions
        actions = {}
        for agent_id, agent in env.agent_graph.get_agents():
            actions[agent] = LLMAction()

            # Track agent actions
            if agent_id not in metrics["agent_actions"]:
                metrics["agent_actions"][agent_id] = {
                    "reports": 0,
                    "likes": 0,
                    "dislikes": 0,
                    "posts": 0
                }

        # Execute step
        observations = await env.step(actions)

        # Analyze observations
        for agent, obs in observations.items():
            if obs:
                obs_str = str(obs).lower()

                # Track reporting behavior
                if "reported" in obs_str:
                    logger.warning(f"⚠️ Agent {agent.user_id} reported content")
                    metrics["agent_actions"][agent.user_id]["reports"] += 1

                    # Try to determine what type was reported
                    if "explicit" in obs_str or "hate" in obs_str:
                        metrics["reports_by_type"]["explicit"] += 1
                    elif "implicit" in obs_str or "subtle" in obs_str:
                        metrics["reports_by_type"]["implicit"] += 1
                    else:
                        metrics["reports_by_type"]["neutral"] += 1

                # Track engagement
                if "liked" in obs_str:
                    metrics["agent_actions"][agent.user_id]["likes"] += 1
                if "disliked" in obs_str:
                    metrics["agent_actions"][agent.user_id]["dislikes"] += 1

        # Add more diverse content periodically
        if step % 15 == 0 and step > 0:
            logger.info("Adding fresh content to simulation...")
            await seed_mixed_toxicity_content(
                env=env,
                loader=loader,
                num_explicit=2,
                num_implicit=3,
                num_neutral=5
            )

    # Close environment
    await env.close()

    # Output analysis if requested
    if output_analysis:
        await analyze_simulation_results(
            database_path=database_path,
            metrics=metrics,
            seeded_content=seeded_content
        )

    return metrics


async def analyze_simulation_results(
    database_path: str,
    metrics: Dict,
    seeded_content: List[Dict]
):
    """
    Analyze and report on simulation results.

    Args:
        database_path: Path to simulation database
        metrics: Collected metrics during simulation
        seeded_content: Information about seeded content
    """
    import sqlite3
    import pandas as pd

    logger.info("\n" + "="*60)
    logger.info("SIMULATION ANALYSIS REPORT")
    logger.info("="*60)

    # Content statistics
    logger.info("\n📊 CONTENT STATISTICS:")
    toxicity_counts = {}
    for content in seeded_content:
        t_type = content["toxicity_type"]
        toxicity_counts[t_type] = toxicity_counts.get(t_type, 0) + 1

    for t_type, count in toxicity_counts.items():
        logger.info(f"  {t_type:20s}: {count:3d} posts")

    # Reporting patterns
    logger.info("\n🚨 REPORTING PATTERNS:")
    total_reports = sum(metrics["reports_by_type"].values())
    logger.info(f"  Total reports: {total_reports}")
    for content_type, count in metrics["reports_by_type"].items():
        percentage = (count / total_reports * 100) if total_reports > 0 else 0
        logger.info(f"  {content_type:20s}: {count:3d} ({percentage:.1f}%)")

    # Agent behavior analysis
    logger.info("\n👥 AGENT BEHAVIOR ANALYSIS:")

    # Find most active reporters
    top_reporters = sorted(
        metrics["agent_actions"].items(),
        key=lambda x: x[1]["reports"],
        reverse=True
    )[:5]

    logger.info("  Top content reporters:")
    for agent_id, actions in top_reporters:
        if actions["reports"] > 0:
            logger.info(f"    Agent {agent_id}: {actions['reports']} reports")

    # Database analysis
    conn = sqlite3.connect(database_path)

    # Get post engagement metrics
    post_engagement = pd.read_sql_query(
        """
        SELECT
            p.post_id,
            p.content,
            COUNT(DISTINCT l.user_id) as likes,
            COUNT(DISTINCT d.user_id) as dislikes,
            COUNT(DISTINCT r.user_id) as reports
        FROM post p
        LEFT JOIN like_post l ON p.post_id = l.post_id
        LEFT JOIN dislike_post d ON p.post_id = d.post_id
        LEFT JOIN report r ON p.post_id = r.post_id
        GROUP BY p.post_id
        ORDER BY reports DESC, dislikes DESC
        """,
        conn
    )

    conn.close()

    # Most reported posts
    logger.info("\n📝 MOST REPORTED CONTENT:")
    reported_posts = post_engagement[post_engagement['reports'] > 0].head(5)
    for _, row in reported_posts.iterrows():
        preview = row['content'][:50] + "..." if len(row['content']) > 50 else row['content']
        logger.info(f"  Reports: {row['reports']}, Content: {preview}")

    # Detection effectiveness
    logger.info("\n✅ DETECTION EFFECTIVENESS:")

    # Calculate theoretical detection rates
    if toxicity_counts.get("explicit_hate", 0) > 0:
        explicit_detection = metrics["reports_by_type"]["explicit"] / toxicity_counts["explicit_hate"]
        logger.info(f"  Explicit hate detection rate: {explicit_detection*100:.1f}%")

    if toxicity_counts.get("implicit_hate", 0) > 0:
        implicit_detection = metrics["reports_by_type"]["implicit"] / toxicity_counts["implicit_hate"]
        logger.info(f"  Implicit hate detection rate: {implicit_detection*100:.1f}%")

    # Save detailed report
    report_path = Path(database_path).parent / "simulation_report.json"
    with open(report_path, 'w') as f:
        report_data = {
            "metrics": metrics,
            "content_statistics": toxicity_counts,
            "total_reports": total_reports,
            "database_path": str(database_path)
        }
        json.dump(report_data, f, indent=2)

    logger.info(f"\n📄 Detailed report saved to: {report_path}")
    logger.info(f"🗃️ Database saved to: {database_path}")
    logger.info("\nUse oasis.print_db_contents() for detailed database analysis")


async def main():
    """Main function to run the unified toxicity simulation."""
    await run_unified_toxicity_simulation(
        num_agents=12,
        simulation_steps=30,
        database_path="./data/unified_toxicity_sim.db",
        output_analysis=True
    )


if __name__ == "__main__":
    asyncio.run(main())