# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from string import Template

from oasis.social_agent.agent_action import SocialAction
from oasis.social_platform.database import get_db_path


class Environment(ABC):

    @abstractmethod
    def to_text_prompt(self) -> str:
        r"""Convert the environment to text prompt."""
        raise NotImplementedError


class SocialEnvironment(Environment):
    followers_env_template = Template("I have $num_followers followers.")
    follows_env_template = Template("I have $num_follows follows.")

    posts_env_template = Template(
        "After refreshing, you see some posts $posts")

    groups_env_template = Template(
        "And there are many group chat channels $all_groups\n"
        "And You are already in some groups $joined_groups\n"
        "You receive some messages from them $messages\n"
        "You can join the groups you are interested, "
        "leave the groups you already in, send messages to the group "
        "you already in.\n"
        "You must make sure you can only send messages to the group you "
        "are already in")
    env_template = Template(
        "$groups_env\n"
        "$posts_env\npick one you want to perform action that best "
        "reflects your current inclination based on your profile and "
        "posts content. Do not limit your action in just `like` to like posts")

    def __init__(self, action: SocialAction):
        self.action = action
        # OPTIMIZATION: Pre-injected posts cache to skip channel round-trip
        # Set by env.step() before agents run, populated from Platform's batch cache.
        # This eliminates 5000+ sequential channel round-trips per step.
        self._injected_posts: dict | None = None

    def inject_posts(self, posts_response: dict) -> None:
        """Inject pre-fetched posts to skip channel round-trip.
        
        Args:
            posts_response: The response dict from Platform.refresh() or batch cache.
                           Format: {"success": bool, "posts": [...]} or {"success": False, "error": ...}
        """
        self._injected_posts = posts_response

    def clear_injected_posts(self) -> None:
        """Clear injected posts after use (called after to_text_prompt)."""
        self._injected_posts = None

    async def get_posts_env(self) -> str:
        # OPTIMIZATION: Use injected posts if available (skips channel round-trip)
        # The posts are pre-fetched in batch by env.step() and injected here.
        if self._injected_posts is not None:
            posts = self._injected_posts
            # Clear after use to avoid stale data
            self._injected_posts = None
        else:
            # Fallback to channel-based refresh (slower, sequential)
            posts = await self.action.refresh()
        
        # TODO: Replace posts json format string to other formats
        if posts.get("success"):
            posts_env = json.dumps(posts["posts"], indent=4)
            posts_env = self.posts_env_template.substitute(posts=posts_env)
        else:
            posts_env = "After refreshing, there are no existing posts."
        return posts_env

    async def get_followers_env(self) -> str:
        # OPTIMIZATION: Skip DB lookup - follower count is just cosmetic context
        # for the LLM prompt and doesn't affect available actions.
        # This eliminates 5000+ blocking DB connections per step.
        # 
        # Original implementation created a new sqlite connection per agent,
        # causing massive contention with 5000 concurrent agents.
        return self.followers_env_template.substitute({"num_followers": "many"})

    async def get_follows_env(self) -> str:
        # OPTIMIZATION: Skip DB lookup - following count is just cosmetic context
        # for the LLM prompt and doesn't affect available actions.
        # This eliminates 5000+ blocking DB connections per step.
        return self.follows_env_template.substitute({"num_follows": "several"})

    async def get_group_env(self) -> str:
        # NOTE: Cannot cache - listen_from_group() returns personalized data
        # per agent (groups they've joined and messages in those groups).
        groups = await self.action.listen_from_group()
        if groups["success"]:
            all_groups = json.dumps(groups["all_groups"])
            joined_groups = json.dumps(groups["joined_groups"])
            messages = json.dumps(groups["messages"])
            groups_env = self.groups_env_template.substitute(
                all_groups=all_groups,
                joined_groups=joined_groups,
                messages=messages,
            )
        else:
            groups_env = "No groups."
        return groups_env

    async def to_text_prompt(
        self,
        include_posts: bool = True,
        include_followers: bool = True,
        include_follows: bool = True,
        include_groups: bool = True,  # OPTIMIZATION: Skip group lookup for Twitter sims
    ) -> str:
        followers_env = (await self.get_followers_env()
                         if include_follows else "No followers.")
        follows_env = (await self.get_follows_env()
                       if include_followers else "No follows.")
        posts_env = await self.get_posts_env() if include_posts else ""
        
        # OPTIMIZATION: Skip group channel call for Twitter simulations
        # This eliminates 5000 channel round-trips per step for sims that don't use groups.
        # The platform processes channel messages sequentially, so each round-trip
        # adds latency. With 5000 agents, skipping groups saves ~5000 * 1-10ms = 5-50s/step.
        if include_groups:
            groups_env = await self.get_group_env()
        else:
            groups_env = "No groups."

        return self.env_template.substitute(
            followers_env=followers_env,
            follows_env=follows_env,
            posts_env=posts_env,
            groups_env=groups_env,
        )
