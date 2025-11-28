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
import logging
import os
from datetime import datetime
from typing import List, Union

from configs.adaptive_rate_limiter import AdaptiveRateLimiter
from configs.llm_settings import SEMAPHORE_LIMIT, get_effective_semaphore_limit
from oasis.environment.env_action import LLMAction, ManualAction
from oasis.social_agent.agent import SocialAgent
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_agent.agents_generator import generate_custom_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import (ActionType, DefaultPlatformType,
                                          RecsysType)

# Create log directory if it doesn't exist
log_dir = "./log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logger
env_log = logging.getLogger("oasis.env")
env_log.setLevel("INFO")

# Add file handler to save logs to file
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_handler = logging.FileHandler(f"{log_dir}/oasis-{current_time}.log",
                                   encoding="utf-8")
file_handler.setLevel("INFO")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
env_log.addHandler(file_handler)


class OasisEnv:

    def __init__(
        self,
        agent_graph: AgentGraph,
        platform: Union[DefaultPlatformType, Platform],
        database_path: str = None,
        semaphore: int = None,  # None = auto-calculate based on TPM
        use_adaptive_rate_limiter: bool = True,  # Use adaptive rate limiting
    ) -> None:
        r"""Init the oasis environment.

        Args:
            agent_graph: The AgentGraph to use in the simulation.
            platform: The platform type to use. Including
                `DefaultPlatformType.TWITTER` or `DefaultPlatformType.REDDIT`.
                Or you can pass a custom `Platform` instance.
            database_path: The path to create a sqlite3 database. The file
                extension must be `.db` such as `twitter_simulation.db`.
            semaphore: Max concurrent LLM requests. If None, auto-calculated
                based on TPM limits and available API keys.
            use_adaptive_rate_limiter: If True, use adaptive rate limiting that
                adjusts concurrency based on actual API responses.
        """
        # Initialize the agent graph
        self.agent_graph = agent_graph
        
        # Use adaptive rate limiter or static semaphore
        self.use_adaptive_rate_limiter = use_adaptive_rate_limiter
        if use_adaptive_rate_limiter:
            self.rate_limiter = AdaptiveRateLimiter(
                initial_concurrency=semaphore,  # None = auto from TPM
            )
            self.llm_semaphore = None  # Not used with adaptive limiter
            env_log.info(
                f"Initialized with ADAPTIVE rate limiter: "
                f"concurrency={self.rate_limiter.current_concurrency}, "
                f"tpm={self.rate_limiter.tpm_limit:,}"
            )
        else:
            # Fallback to static semaphore
            self.rate_limiter = None
            if semaphore is None:
                semaphore = get_effective_semaphore_limit()
            self.llm_semaphore = asyncio.Semaphore(semaphore)
            env_log.info(f"Initialized with static semaphore: {semaphore}")
        if isinstance(platform, DefaultPlatformType):
            if database_path is None:
                raise ValueError(
                    "database_path is required for DefaultPlatformType")
            self.platform = platform
            if platform == DefaultPlatformType.TWITTER:
                self.channel = Channel()
                self.platform = Platform(
                    db_path=database_path,
                    channel=self.channel,
                    recsys_type="twhin-bert",
                    refresh_rec_post_count=2,
                    max_rec_post_len=2,
                    following_post_count=3,
                )
                self.platform_type = DefaultPlatformType.TWITTER
            elif platform == DefaultPlatformType.REDDIT:
                self.channel = Channel()
                self.platform = Platform(
                    db_path=database_path,
                    channel=self.channel,
                    recsys_type="reddit",
                    allow_self_rating=True,
                    show_score=True,
                    max_rec_post_len=100,
                    refresh_rec_post_count=5,
                )
                self.platform_type = DefaultPlatformType.REDDIT
            else:
                raise ValueError(f"Invalid platform: {platform}. Only "
                                 "DefaultPlatformType.TWITTER or "
                                 "DefaultPlatformType.REDDIT are supported.")
        elif isinstance(platform, Platform):
            if database_path != platform.db_path:
                env_log.warning("database_path is not the same as the "
                                "platform.db_path, using the platform.db_path")
            self.platform = platform
            self.channel = platform.channel
            if platform.recsys_type == RecsysType.REDDIT:
                self.platform_type = DefaultPlatformType.REDDIT
            else:
                self.platform_type = DefaultPlatformType.TWITTER
        else:
            raise ValueError(
                f"Invalid platform: {platform}. You should pass a "
                "DefaultPlatformType or a Platform instance.")

    async def reset(self) -> None:
        r"""Start the platform and sign up the agents."""
        self.platform_task = asyncio.create_task(self.platform.running())
        self.agent_graph = await generate_custom_agents(
            channel=self.channel, agent_graph=self.agent_graph)

    async def _perform_llm_action(self, agent):
        r"""Send the request to the llm model and execute the action."""
        if self.use_adaptive_rate_limiter and self.rate_limiter:
            async with await self.rate_limiter.acquire():
                try:
                    result = await agent.perform_action_by_llm()
                    # Estimate tokens used (rough: 500 prompt + response length)
                    tokens_used = 500 + len(str(result)) // 4 if result else 500
                    self.rate_limiter.record_success(tokens_used=tokens_used)
                    return result
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = "429" in error_str or "rate limit" in error_str
                    self.rate_limiter.record_error(is_rate_limit=is_rate_limit)
                    raise
        else:
            async with self.llm_semaphore:
                return await agent.perform_action_by_llm()

    async def _perform_interview_action(self, agent, interview_prompt: str):
        r"""Send the request to the llm model and execute the interview."""
        if self.use_adaptive_rate_limiter and self.rate_limiter:
            async with await self.rate_limiter.acquire():
                try:
                    result = await agent.perform_interview(interview_prompt)
                    tokens_used = 500 + len(str(result)) // 4 if result else 500
                    self.rate_limiter.record_success(tokens_used=tokens_used)
                    return result
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = "429" in error_str or "rate limit" in error_str
                    self.rate_limiter.record_error(is_rate_limit=is_rate_limit)
                    raise
        else:
            async with self.llm_semaphore:
                return await agent.perform_interview(interview_prompt)

    async def step(
        self, actions: dict[SocialAgent, Union[ManualAction, LLMAction,
                                               List[Union[ManualAction,
                                                          LLMAction]]]]
    ) -> None:
        r"""Update the recommendation system and perform the actions.

        Args:
            actions(dict[SocialAgent, Union[ManualAction, LLMAction,
                List[Union[ManualAction, LLMAction]]]]): The actions to
                perform, including the manual(pre-defined) actions and llm
                actions.
        Returns:
            None
        """
        import time as _time
        step_timing = {}
        
        # Update the recommendation system (potential bottleneck with large post counts)
        t_recsys = _time.perf_counter()
        await self.platform.update_rec_table()
        step_timing["recsys_update"] = _time.perf_counter() - t_recsys
        env_log.info(f"update rec table in {step_timing['recsys_update']:.3f}s")
        
        # OPTIMIZATION: Pre-compute recommendations for all agents in batch
        # This eliminates 5000+ sequential channel round-trips per step.
        # Instead of each agent calling refresh() through the channel one at a time,
        # we compute all recommendations upfront in a single batch operation.
        t_batch_rec = _time.perf_counter()
        agent_ids = [agent.social_agent_id for agent, _ in actions.items()]
        if hasattr(self.platform, 'precompute_agent_recommendations'):
            needs_refresh = True
            cache_valid = getattr(self.platform, "_batch_rec_cache_valid", False)
            if cache_valid:
                cache_map = getattr(self.platform, "_batch_rec_cache", {})
                needs_refresh = any(agent_id not in cache_map for agent_id in agent_ids)
            if needs_refresh:
                self.platform.precompute_agent_recommendations(agent_ids)
                step_timing["batch_recommendations"] = _time.perf_counter() - t_batch_rec
                env_log.info(f"batch recommendations in {step_timing['batch_recommendations']:.3f}s")
            else:
                step_timing["batch_recommendations"] = 0.0
                env_log.info("batch recommendations skipped (cache still valid)")

            # OPTIMIZATION: Inject cached posts directly into each agent's environment
            # This completely bypasses the channel round-trip for refresh().
            # Without this, each agent's to_text_prompt() → refresh() still goes
            # through the channel sequentially, even though the data is cached.
            t_inject = _time.perf_counter()
            injected_count = 0
            for agent, _ in actions.items():
                cached = self.platform.get_cached_recommendation(agent.social_agent_id)
                if cached is not None and hasattr(agent, 'env') and hasattr(agent.env, 'inject_posts'):
                    agent.env.inject_posts(cached)
                    injected_count += 1
            step_timing["inject_posts"] = _time.perf_counter() - t_inject
            env_log.info(f"Injected cached posts into {injected_count} agents in {step_timing['inject_posts']:.3f}s")

        # Create tasks for both manual and LLM actions
        tasks = []
        agent_count = 0
        for agent, action in actions.items():
            agent_count += 1
            if isinstance(action, list):
                for single_action in action:
                    if isinstance(single_action, ManualAction):
                        if single_action.action_type == ActionType.INTERVIEW:
                            # Use the agent's perform_interview method for
                            # interview actions
                            interview_prompt = single_action.action_args.get(
                                "prompt", "")
                            tasks.append(
                                self._perform_interview_action(
                                    agent, interview_prompt))
                        else:
                            tasks.append(
                                agent.perform_action_by_data(
                                    single_action.action_type,
                                    **single_action.action_args))
                    elif isinstance(single_action, LLMAction):
                        tasks.append(self._perform_llm_action(agent))
            else:
                if isinstance(action, ManualAction):
                    if action.action_type == ActionType.INTERVIEW:
                        # Use the agent's perform_interview method for
                        # interview actions
                        interview_prompt = action.action_args.get("prompt", "")
                        tasks.append(
                            self._perform_interview_action(
                                agent, interview_prompt))
                    else:
                        tasks.append(
                            agent.perform_action_by_data(
                                action.action_type, **action.action_args))
                elif isinstance(action, LLMAction):
                    tasks.append(self._perform_llm_action(agent))

        # Track task creation time
        step_timing["task_creation"] = _time.perf_counter() - t_recsys - step_timing["recsys_update"]
        
        # Log concurrency info
        if self.use_adaptive_rate_limiter and self.rate_limiter:
            concurrency_info = f"Adaptive concurrency: {self.rate_limiter.current_concurrency}"
        else:
            concurrency_info = f"Semaphore limit: {self.llm_semaphore._value}"

        env_log.info(f"Created {len(tasks)} tasks for {agent_count} agents. Starting asyncio.gather()...")
        print(f"[env.step] Created {len(tasks)} tasks. {concurrency_info}. Starting...", flush=True)
        
        # Execute all tasks concurrently with progress logging
        import sys
        completed = 0
        total = len(tasks)
        start_time = _time.perf_counter()
        
        async def task_wrapper(idx, task):
            nonlocal completed
            result = None
            try:
                result = await task
            except Exception as e:
                env_log.warning(f"Task {idx} failed: {e}")
                print(f"[env.step] Task {idx} FAILED: {e}", file=sys.stderr, flush=True)
                result = None
            completed += 1
            elapsed = _time.perf_counter() - start_time
            # Log more frequently for visibility
            if completed % 10 == 0 or completed == total or completed <= 3:
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                
                # Include rate limiter stats if adaptive
                if self.use_adaptive_rate_limiter and self.rate_limiter:
                    stats = self.rate_limiter.get_stats_summary()
                    extra = f" | RPM:{stats['current_rpm']:.0f} TPM:{stats['current_tpm']:.0f} Conc:{stats['current_concurrency']}"
                else:
                    extra = ""
                
                env_log.info(f"Progress: {completed}/{total} ({rate:.2f}/s, ETA: {eta:.0f}s){extra}")
                print(f"[env.step] Progress: {completed}/{total} ({rate:.2f}/s, ETA: {eta:.0f}s){extra}", flush=True)
            return result
        
        wrapped_tasks = [task_wrapper(i, t) for i, t in enumerate(tasks)]
        await asyncio.gather(*wrapped_tasks)
        
        # Print final rate limiter stats for this step
        if self.use_adaptive_rate_limiter and self.rate_limiter:
            stats = self.rate_limiter.get_stats_summary()
            env_log.info(
                f"Step complete. Rate limiter: concurrency={stats['current_concurrency']}, "
                f"success_rate={stats['success_rate']}%, rate_limited={stats['rate_limited']}"
            )
        
        total_time = _time.perf_counter() - start_time
        step_timing["llm_execution"] = total_time
        
        # Log timing breakdown
        env_log.info(
            f"Step timing breakdown: recsys={step_timing['recsys_update']:.3f}s, "
            f"task_creation={step_timing.get('task_creation', 0):.3f}s, "
            f"llm_execution={step_timing['llm_execution']:.3f}s"
        )
        print(f"[env.step] Completed {total} tasks in {total_time:.1f}s ({total/total_time:.2f}/s)", flush=True)
        env_log.info("performed all actions.")
        # # Control some agents to perform actions
        # Update the clock
        if self.platform_type == DefaultPlatformType.TWITTER:
            self.platform.sandbox_clock.time_step += 1

    async def close(self) -> None:
        r"""Stop the platform and close the environment.
        """
        await self.channel.write_to_receive_queue(
            (None, None, ActionType.EXIT))
        await self.platform_task
        env_log.info("Simulation finished! Please check the results in the "
                     f"database: {self.platform.db_path}. Note that the trace "
                     "table stored all the actions of the agents.")
