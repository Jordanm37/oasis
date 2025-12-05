#!/usr/bin/env python3
"""Comprehensive test script for new simulation features.

This script tests:
1. Trajectory Stage Distribution (llm_settings.py)
2. PersonaConfig with trajectory fields (emission_policy.py)
3. EmissionPolicy trajectory modifiers
4. EventManager loading and stepping
5. Event modifiers for agents
6. Integration with generate_personas_llm.py
7. Integration with agent_factory.py

Run with: poetry run python3 scripts/test_new_features.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Track test results
TESTS_RUN = 0
TESTS_PASSED = 0
TESTS_FAILED = 0
FAILURES: List[Tuple[str, str]] = []


def test(name: str):
    """Decorator to register and run a test."""
    def decorator(func):
        def wrapper():
            global TESTS_RUN, TESTS_PASSED, TESTS_FAILED
            TESTS_RUN += 1
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print('='*60)
            try:
                func()
                TESTS_PASSED += 1
                print(f"✅ PASSED: {name}")
                return True
            except Exception as e:
                TESTS_FAILED += 1
                error_msg = f"{type(e).__name__}: {e}"
                FAILURES.append((name, error_msg))
                print(f"❌ FAILED: {name}")
                print(f"   Error: {error_msg}")
                traceback.print_exc()
                return False
        return wrapper
    return decorator


# =============================================================================
# TEST 1: Trajectory Stage Configuration
# =============================================================================

@test("Trajectory Stage Distribution Import")
def test_trajectory_import():
    """Test that trajectory configuration can be imported."""
    from configs.llm_settings import (
        TRAJECTORY_STAGE_DISTRIBUTION,
        TRAJECTORY_STAGE_MODIFIERS,
        SLANG_FLUENCY_TIERS,
    )
    
    assert TRAJECTORY_STAGE_DISTRIBUTION is not None, "TRAJECTORY_STAGE_DISTRIBUTION is None"
    assert TRAJECTORY_STAGE_MODIFIERS is not None, "TRAJECTORY_STAGE_MODIFIERS is None"
    assert SLANG_FLUENCY_TIERS is not None, "SLANG_FLUENCY_TIERS is None"
    
    print(f"  - Loaded {len(TRAJECTORY_STAGE_DISTRIBUTION)} archetype distributions")
    print(f"  - Loaded {len(TRAJECTORY_STAGE_MODIFIERS)} stage modifiers")
    print(f"  - Loaded {len(SLANG_FLUENCY_TIERS)} fluency tiers")


@test("Trajectory Distribution Validation")
def test_trajectory_distribution_valid():
    """Test that all trajectory distributions sum to 1.0."""
    from configs.llm_settings import TRAJECTORY_STAGE_DISTRIBUTION
    
    for archetype, dist in TRAJECTORY_STAGE_DISTRIBUTION.items():
        total = sum(dist.values())
        assert abs(total - 1.0) < 0.001, f"{archetype} distribution sums to {total}, not 1.0"
        print(f"  - {archetype}: curious={dist.get('curious', 0)*100:.0f}% active={dist.get('active', 0)*100:.0f}% entrenched={dist.get('entrenched', 0)*100:.0f}%")


@test("Trajectory Modifiers Have Required Fields")
def test_trajectory_modifiers_fields():
    """Test that all trajectory modifiers have required fields."""
    from configs.llm_settings import TRAJECTORY_STAGE_MODIFIERS
    
    required_fields = [
        "label_frequency_multiplier",
        "aggression_level",
        "slang_fluency",
        "prompt_hint",
    ]
    
    for stage, mods in TRAJECTORY_STAGE_MODIFIERS.items():
        for field in required_fields:
            assert field in mods, f"Stage '{stage}' missing field '{field}'"
        print(f"  - {stage}: freq_mult={mods['label_frequency_multiplier']}, aggression={mods['aggression_level']}")


# =============================================================================
# TEST 2: PersonaConfig with Trajectory Fields
# =============================================================================

@test("PersonaConfig Trajectory Fields")
def test_persona_config_trajectory():
    """Test that PersonaConfig accepts trajectory fields."""
    from generation.emission_policy import PersonaConfig
    
    # Create with default trajectory values
    persona_default = PersonaConfig(
        persona_id="test_default",
        primary_label="incel_misogyny",
        allowed_labels=["incel_misogyny"],
        label_mode_cap="single",
    )
    assert persona_default.trajectory_stage == "active", f"Default trajectory_stage should be 'active', got '{persona_default.trajectory_stage}'"
    assert persona_default.slang_fluency == "fluent", f"Default slang_fluency should be 'fluent', got '{persona_default.slang_fluency}'"
    print(f"  - Default persona: stage={persona_default.trajectory_stage}, fluency={persona_default.slang_fluency}")
    
    # Create with explicit trajectory values
    persona_explicit = PersonaConfig(
        persona_id="test_explicit",
        primary_label="extremist",
        allowed_labels=["extremist", "hate_speech"],
        label_mode_cap="double",
        trajectory_stage="entrenched",
        slang_fluency="native",
    )
    assert persona_explicit.trajectory_stage == "entrenched"
    assert persona_explicit.slang_fluency == "native"
    print(f"  - Explicit persona: stage={persona_explicit.trajectory_stage}, fluency={persona_explicit.slang_fluency}")


# =============================================================================
# TEST 3: EmissionPolicy Trajectory Modifiers
# =============================================================================

@test("EmissionPolicy Trajectory Modifier Application")
def test_emission_policy_trajectory():
    """Test that EmissionPolicy applies trajectory modifiers correctly."""
    from generation.emission_policy import EmissionPolicy, PersonaConfig
    
    policy = EmissionPolicy(
        run_seed=42,
        post_label_mode_probs={"none": 0.5, "single": 0.4, "double": 0.1},
    )
    
    # Test curious persona (should have LESS label frequency)
    curious_persona = PersonaConfig(
        persona_id="curious",
        primary_label="incel_misogyny",
        allowed_labels=["incel_misogyny"],
        label_mode_cap="single",
        trajectory_stage="curious",
    )
    curious_probs = policy._apply_trajectory_modifiers(curious_persona)
    print(f"  - Curious probs: {curious_probs}")
    assert curious_probs["none"] > 0.5, f"Curious 'none' prob should be > 0.5, got {curious_probs['none']}"
    
    # Test entrenched persona (should have MORE label frequency)
    entrenched_persona = PersonaConfig(
        persona_id="entrenched",
        primary_label="incel_misogyny",
        allowed_labels=["incel_misogyny"],
        label_mode_cap="single",
        trajectory_stage="entrenched",
    )
    entrenched_probs = policy._apply_trajectory_modifiers(entrenched_persona)
    print(f"  - Entrenched probs: {entrenched_probs}")
    assert entrenched_probs["none"] < 0.5, f"Entrenched 'none' prob should be < 0.5, got {entrenched_probs['none']}"
    
    # Verify entrenched has more label tokens than curious
    assert entrenched_probs["single"] > curious_probs["single"], "Entrenched should have higher 'single' prob than curious"


# =============================================================================
# TEST 4: EventManager
# =============================================================================

@test("EventManager Import")
def test_event_manager_import():
    """Test that EventManager can be imported."""
    from orchestrator.event_manager import EventManager, SimulationEvent, EventModifiers
    print("  - EventManager imported successfully")
    print("  - SimulationEvent imported successfully")
    print("  - EventModifiers imported successfully")


@test("EventManager Load Events")
def test_event_manager_load():
    """Test that EventManager loads events from YAML."""
    from orchestrator.event_manager import EventManager
    
    config_path = Path("configs/simulation_events.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Event config not found: {config_path}")
    
    em = EventManager(config_path=config_path, seed=42)
    
    assert len(em.events) > 0, "No events loaded"
    print(f"  - Loaded {len(em.events)} events")
    
    # Check event structure
    for event in em.events[:3]:
        assert event.id, "Event missing id"
        assert event.event_type, "Event missing event_type"
        assert event.trigger_step >= 0, "Event has invalid trigger_step"
        assert event.duration_steps > 0, "Event has invalid duration_steps"
        assert len(event.affects_archetypes) > 0, "Event has no affects_archetypes"
        print(f"  - Event '{event.id}': triggers at step {event.trigger_step}, lasts {event.duration_steps} steps")


@test("EventManager Step and Activation")
def test_event_manager_step():
    """Test that EventManager correctly activates/deactivates events."""
    from orchestrator.event_manager import EventManager
    
    config_path = Path("configs/simulation_events.yaml")
    em = EventManager(config_path=config_path, seed=42)
    
    # Find an event that triggers early
    early_events = [e for e in em.events if e.trigger_step <= 10]
    if not early_events:
        print("  - WARNING: No events trigger in first 10 steps, skipping activation test")
        return
    
    first_event = min(early_events, key=lambda e: e.trigger_step)
    print(f"  - First event '{first_event.id}' triggers at step {first_event.trigger_step}")
    
    # Step through until event triggers
    active_found = False
    for step in range(first_event.trigger_step + 5):
        active = em.step(step)
        if active:
            print(f"  - Step {step}: {len(active)} active events: {[e.id for e in active]}")
            active_found = True
    
    assert active_found, "No events became active during stepping"


@test("EventManager Agent Modifiers")
def test_event_manager_modifiers():
    """Test that EventManager returns correct modifiers for archetypes."""
    from orchestrator.event_manager import EventManager
    
    config_path = Path("configs/simulation_events.yaml")
    em = EventManager(config_path=config_path, seed=42)
    
    # Step until we have active events
    for step in range(50):
        active = em.step(step)
        if active:
            # Get an affected archetype
            affected_archetype = active[0].affects_archetypes[0]
            mods = em.get_agent_modifiers(affected_archetype)
            
            assert mods.has_active_event, "Modifier should indicate active event"
            assert len(mods.active_event_ids) > 0, "Should have active event IDs"
            print(f"  - Step {step}: {affected_archetype} modifiers:")
            print(f"    aggression_boost={mods.aggression_boost:.2f}")
            print(f"    inject_narratives={mods.inject_narratives[:1]}")
            return
    
    print("  - WARNING: No events activated in 50 steps")


# =============================================================================
# TEST 5: Extended Agent Trajectory/Event Hints
# =============================================================================

@test("ExtendedAgent Trajectory Hint Method")
def test_extended_agent_trajectory_hint():
    """Test that ExtendedSocialAgent has trajectory hint method."""
    from generation.extended_agent import ExtendedSocialAgent
    
    # Check method exists
    assert hasattr(ExtendedSocialAgent, "_format_trajectory_hint"), "Missing _format_trajectory_hint method"
    print("  - _format_trajectory_hint method exists")


@test("ExtendedAgent Event Hint Method")
def test_extended_agent_event_hint():
    """Test that ExtendedSocialAgent has event hint method."""
    from generation.extended_agent import ExtendedSocialAgent
    
    # Check method exists
    assert hasattr(ExtendedSocialAgent, "_format_event_hint"), "Missing _format_event_hint method"
    print("  - _format_event_hint method exists")


# =============================================================================
# TEST 6: Generate Personas Trajectory Assignment
# =============================================================================

@test("Generate Personas Trajectory Function")
def test_generate_personas_trajectory():
    """Test that generate_personas_llm has trajectory assignment function."""
    import random
    
    # Import the function
    try:
        from scripts.generate_personas_llm import _assign_trajectory_stage
    except ImportError:
        # Try alternative import path
        import sys
        sys.path.insert(0, str(Path.cwd()))
        from scripts.generate_personas_llm import _assign_trajectory_stage
    
    rng = random.Random(42)
    
    # Test for different archetypes
    test_archetypes = ["incel_misogyny", "benign", "extremist", "pro_ana"]
    
    for archetype in test_archetypes:
        stage, fluency, hint = _assign_trajectory_stage(rng, archetype)
        assert stage in ["curious", "active", "entrenched"], f"Invalid stage: {stage}"
        assert fluency in ["outsider", "learning", "fluent", "native"], f"Invalid fluency: {fluency}"
        print(f"  - {archetype}: stage={stage}, fluency={fluency}")


# =============================================================================
# TEST 7: Run Production Sim Event Integration
# =============================================================================

@test("Run Production Sim Event Manager Import")
def test_run_production_sim_event():
    """Test that run_production_sim has EventManager integration."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location(
        "run_production_sim",
        Path("scripts/run_production_sim.py")
    )
    
    # Just check the file contains the right imports
    with open("scripts/run_production_sim.py", "r") as f:
        content = f.read()
    
    assert "from orchestrator.event_manager import EventManager" in content, "Missing EventManager import"
    assert "_apply_event_modifiers" in content, "Missing _apply_event_modifiers function"
    assert "event_manager.step" in content, "Missing event_manager.step call"
    
    print("  - EventManager import found")
    print("  - _apply_event_modifiers function found")
    print("  - event_manager.step call found")


# =============================================================================
# TEST 8: SimulationCoordinator (existing feature)
# =============================================================================

@test("SimulationCoordinator Import")
def test_simulation_coordinator():
    """Test that SimulationCoordinator works."""
    from orchestrator.simulation_coordinator import (
        SimulationCoordinator,
        SimulationCoordinatorConfig,
    )
    
    config = SimulationCoordinatorConfig(
        enable_thread_dynamics=True,
        pile_on_probability=0.1,
    )
    
    coord = SimulationCoordinator(config=config, seed=42)
    
    # Register some agents
    for i in range(10):
        archetype = ["incel_misogyny", "extremist", "conspiracy", "bullying"][i % 4]
        coord.register_agent(i, archetype)
    
    print(f"  - Registered {len(coord._agents)} agents")
    
    # Step a few times
    for step in range(10):
        coord.step()
    
    stats = coord.get_stats()
    print(f"  - After 10 steps: {stats['active_events']} active events")


# =============================================================================
# TEST 9: Full Integration Smoke Test
# =============================================================================

@test("Full Integration Smoke Test")
def test_full_integration():
    """Test that all components can work together."""
    from configs.llm_settings import TRAJECTORY_STAGE_DISTRIBUTION, TRAJECTORY_STAGE_MODIFIERS
    from generation.emission_policy import EmissionPolicy, PersonaConfig
    from orchestrator.event_manager import EventManager
    from orchestrator.simulation_coordinator import SimulationCoordinator, SimulationCoordinatorConfig
    import random
    
    # Create components
    rng = random.Random(42)
    
    # 1. Create personas with trajectory stages
    personas = []
    archetypes = ["incel_misogyny", "extremist", "conspiracy", "pro_ana", "benign"]
    for i, arch in enumerate(archetypes):
        dist = TRAJECTORY_STAGE_DISTRIBUTION.get(arch, {"active": 1.0})
        stages = list(dist.keys())
        weights = list(dist.values())
        stage = rng.choices(stages, weights=weights, k=1)[0]
        fluency = TRAJECTORY_STAGE_MODIFIERS.get(stage, {}).get("slang_fluency", "fluent")
        
        persona = PersonaConfig(
            persona_id=f"user_{i}",
            primary_label=arch,
            allowed_labels=[arch],
            label_mode_cap="single",
            trajectory_stage=stage,
            slang_fluency=fluency,
        )
        personas.append(persona)
        print(f"  - Created persona {i}: {arch} ({stage})")
    
    # 2. Create emission policy
    policy = EmissionPolicy(
        run_seed=42,
        post_label_mode_probs={"none": 0.5, "single": 0.4, "double": 0.1},
    )
    
    # 3. Create event manager
    em = EventManager(config_path=Path("configs/simulation_events.yaml"), seed=42)
    print(f"  - EventManager: {len(em.events)} events loaded")
    
    # 4. Create simulation coordinator
    coord = SimulationCoordinator(
        config=SimulationCoordinatorConfig(enable_thread_dynamics=True),
        seed=42,
    )
    for i, p in enumerate(personas):
        coord.register_agent(i, p.primary_label)
    
    # 5. Simulate a few steps
    for step in range(20):
        # Step event manager
        active_events = em.step(step)
        
        # Step coordinator
        coord.step()
        
        # For each persona, get modifiers and make a decision
        for i, persona in enumerate(personas):
            # Get event modifiers
            event_mods = em.get_agent_modifiers(persona.primary_label)
            
            # Get coordination modifiers
            coord_mods = coord.get_agent_modifiers(i)
            
            # Get emission decision
            decision = policy.decide(
                user_id=i,
                thread_id=f"thread_{step}",
                step_idx=step,
                persona=persona,
            )
        
        if active_events:
            print(f"  - Step {step}: {len(active_events)} events active")
    
    print("  - Full integration completed successfully!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TESTING NEW SIMULATION FEATURES")
    print("="*60)
    
    # Run all tests
    tests = [
        test_trajectory_import,
        test_trajectory_distribution_valid,
        test_trajectory_modifiers_fields,
        test_persona_config_trajectory,
        test_emission_policy_trajectory,
        test_event_manager_import,
        test_event_manager_load,
        test_event_manager_step,
        test_event_manager_modifiers,
        test_extended_agent_trajectory_hint,
        test_extended_agent_event_hint,
        test_generate_personas_trajectory,
        test_run_production_sim_event,
        test_simulation_coordinator,
        test_full_integration,
    ]
    
    for test_func in tests:
        test_func()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total tests: {TESTS_RUN}")
    print(f"Passed:      {TESTS_PASSED} ✅")
    print(f"Failed:      {TESTS_FAILED} ❌")
    
    if FAILURES:
        print("\nFailed tests:")
        for name, error in FAILURES:
            print(f"  - {name}: {error}")
    
    print("="*60)
    
    # Exit with appropriate code
    sys.exit(0 if TESTS_FAILED == 0 else 1)


if __name__ == "__main__":
    main()

