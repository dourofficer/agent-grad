from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re
import json


class EventType(Enum):
    TASK = "task"
    INITIAL_PLAN = "initial_plan"
    SYNTHESIZED_PROMPT = "synthesized_prompt"
    LEDGER_UPDATE = "ledger_update"
    INSTRUCTION = "instruction"
    WORKER_ACTION = "worker_action"
    NEXT_SPEAKER = "next_speaker"
    REPLAN_TRIGGER = "replan_trigger"
    NEW_PLAN = "new_plan"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class TrajectoryEvent:
    index: int
    role: str
    content: str
    event_type: EventType
    agent_name: Optional[str] = None
    epoch: int = 0
    
    def __repr__(self):
        return f"Event({self.index}, {self.event_type.value}, epoch={self.epoch}, agent={self.agent_name})"


class MagenticOneTrajectoryParser:
    """
    Parse Magentic-One trajectories and extract dependency graphs.
    
    Dependency Rules (based on codebase analysis):
    1. Within an epoch, worker actions depend on full chat history (all broadcasts)
    2. Ledger updates depend on all previous broadcasts in the epoch
    3. Instructions depend on the immediately preceding ledger update
    4. Replan events create epoch boundaries
    5. After replan, new epoch starts with dependencies on:
       - Original task
       - New synthesized prompt (which encapsulates updated facts/plan)
    """
    
    WORKER_AGENTS = {"WebSurfer", "Coder", "Executor", "FileSurfer", "Assistant", "UserProxy"}
    
    def __init__(self, dependency_mode: str = "full"):
        """
        Args:
            dependency_mode: One of:
                - "full": Each step depends on ALL previous steps in epoch
                - "immediate": Each step depends only on immediately relevant predecessors
                - "structural": Dependencies based on structural relationships (ledger->instruction->action)
        """
        self.dependency_mode = dependency_mode
    
    def parse_trajectory(self, trajectory: List[Dict[str, Any]]) -> List[TrajectoryEvent]:
        """Parse raw trajectory into structured events."""
        events = []
        current_epoch = 0
        
        for idx, entry in enumerate(trajectory):
            role = entry.get("role", "")
            content = entry.get("content", "")
            
            event_type, agent_name = self._classify_event(role, content)
            
            # Detect epoch boundaries (replan)
            if event_type == EventType.REPLAN_TRIGGER:
                current_epoch += 1
            
            event = TrajectoryEvent(
                index=idx,
                role=role,
                content=content,
                event_type=event_type,
                agent_name=agent_name,
                epoch=current_epoch
            )
            events.append(event)
        
        return events
    
    def _classify_event(self, role: str, content: str) -> Tuple[EventType, Optional[str]]:
        """Classify an event based on role and content."""
        
        # Human task
        if role == "human":
            return EventType.TASK, None
        
        # Orchestrator thoughts
        if role == "Orchestrator (thought)":
            if "Initial plan:" in content or content.startswith("Initial plan:"):
                return EventType.INITIAL_PLAN, "Orchestrator"
            if "Updated Ledger:" in content:
                return EventType.LEDGER_UPDATE, "Orchestrator"
            if "Next speaker" in content:
                return EventType.NEXT_SPEAKER, "Orchestrator"
            if "Stalled" in content or "Replanning" in content:
                return EventType.REPLAN_TRIGGER, "Orchestrator"
            if "New plan:" in content:
                return EventType.NEW_PLAN, "Orchestrator"
            return EventType.UNKNOWN, "Orchestrator"
        
        # Orchestrator instructions to specific agents
        instruction_match = re.match(r"Orchestrator \(-> (\w+)\)", role)
        if instruction_match:
            agent_name = instruction_match.group(1)
            return EventType.INSTRUCTION, agent_name
        
        # Orchestrator final answer
        if role == "Orchestrator (final answer)":
            return EventType.FINAL_ANSWER, "Orchestrator"
        
        # Worker agent actions
        if role in self.WORKER_AGENTS:
            return EventType.WORKER_ACTION, role
        
        # Check for errors in content
        if "Error" in content or "Traceback" in content:
            return EventType.ERROR, None
        
        return EventType.UNKNOWN, None
    
    def build_dependency_graph(self, events: List[TrajectoryEvent]) -> Dict[int, List[int]]:
        """
        Build dependency graph based on the selected mode.
        
        Returns:
            Dict mapping event index to list of dependency indices
        """
        if self.dependency_mode == "full":
            return self._build_full_dependencies(events)
        elif self.dependency_mode == "immediate":
            return self._build_immediate_dependencies(events)
        elif self.dependency_mode == "structural":
            return self._build_structural_dependencies(events)
        else:
            raise ValueError(f"Unknown dependency mode: {self.dependency_mode}")
    
    def _build_full_dependencies(self, events: List[TrajectoryEvent]) -> Dict[int, List[int]]:
        """
        Full dependency mode: Each event depends on all previous events in the same epoch.
        Cross-epoch dependencies only on task and synthesized prompts.
        """
        dependencies = {}
        
        # Find task index (usually 0)
        task_idx = None
        for e in events:
            if e.event_type == EventType.TASK:
                task_idx = e.index
                break
        
        # Track epoch start indices and synthesized prompts
        epoch_starts = {0: 0}  # epoch -> first index in that epoch
        synthesized_prompts = {}  # epoch -> index of synthesized prompt/new plan
        
        for e in events:
            if e.event_type in [EventType.INITIAL_PLAN, EventType.NEW_PLAN]:
                synthesized_prompts[e.epoch] = e.index
            if e.epoch not in epoch_starts:
                epoch_starts[e.epoch] = e.index
        
        for event in events:
            deps = []
            
            if event.index == 0:
                dependencies[event.index] = []
                continue
            
            epoch_start = epoch_starts.get(event.epoch, 0)
            
            # All events in the same epoch before this one
            for i in range(epoch_start, event.index):
                if events[i].epoch == event.epoch:
                    deps.append(i)
            
            # Cross-epoch: depend on task
            if event.epoch > 0 and task_idx is not None and task_idx not in deps:
                deps.append(task_idx)
            
            # Cross-epoch: depend on previous epoch's synthesized prompt for context
            # (This represents the facts/plan being carried forward)
            if event.epoch > 0:
                prev_epoch = event.epoch - 1
                if prev_epoch in synthesized_prompts:
                    sp_idx = synthesized_prompts[prev_epoch]
                    if sp_idx not in deps:
                        deps.append(sp_idx)
            
            dependencies[event.index] = sorted(deps)
        
        return dependencies
    
    def _build_immediate_dependencies(self, events: List[TrajectoryEvent]) -> Dict[int, List[int]]:
        """
        Immediate dependency mode: Only direct predecessors.
        - Ledger depends on previous ledger + intervening broadcasts
        - Instruction depends on ledger
        - Worker action depends on instruction
        """
        dependencies = {}
        
        last_ledger_idx = None
        last_instruction_idx = None
        last_broadcast_indices = []  # Broadcasts since last ledger
        task_idx = None
        
        for event in events:
            deps = []
            
            if event.event_type == EventType.TASK:
                task_idx = event.index
                dependencies[event.index] = []
                continue
            
            if event.event_type == EventType.INITIAL_PLAN:
                if task_idx is not None:
                    deps.append(task_idx)
            
            elif event.event_type == EventType.LEDGER_UPDATE:
                # Depends on previous ledger and all broadcasts since then
                if last_ledger_idx is not None:
                    deps.append(last_ledger_idx)
                deps.extend(last_broadcast_indices)
                last_ledger_idx = event.index
                last_broadcast_indices = []
            
            elif event.event_type == EventType.INSTRUCTION:
                # Depends on the ledger update
                if last_ledger_idx is not None:
                    deps.append(last_ledger_idx)
                last_instruction_idx = event.index
            
            elif event.event_type == EventType.WORKER_ACTION:
                # Depends on instruction
                if last_instruction_idx is not None:
                    deps.append(last_instruction_idx)
                last_broadcast_indices.append(event.index)
            
            elif event.event_type == EventType.REPLAN_TRIGGER:
                # Depends on recent history showing stall
                if last_ledger_idx is not None:
                    deps.append(last_ledger_idx)
                # Reset tracking for new epoch
                last_broadcast_indices = []
            
            elif event.event_type == EventType.NEW_PLAN:
                if task_idx is not None:
                    deps.append(task_idx)
                # Depends on replan trigger
                for i in range(event.index - 1, -1, -1):
                    if events[i].event_type == EventType.REPLAN_TRIGGER:
                        deps.append(i)
                        break
            
            else:
                # Default: depend on immediate predecessor
                if event.index > 0:
                    deps.append(event.index - 1)
            
            dependencies[event.index] = sorted(set(deps))
        
        return dependencies
    
    def _build_structural_dependencies(self, events: List[TrajectoryEvent]) -> Dict[int, List[int]]:
        """
        Structural dependency mode: Based on the agent communication pattern.
        """
        dependencies = {}
        
        task_idx = None
        plan_idx = None
        current_epoch = 0
        epoch_worker_actions = []
        last_ledger_in_epoch = None
        
        # Track last ledger BEFORE epoch change for REPLAN_TRIGGER
        last_ledger_before_replan = None
        
        for event in events:
            deps = []
            
            # Handle epoch changes AFTER processing REPLAN_TRIGGER
            # First, check if THIS event triggers a new epoch
            is_epoch_boundary = (event.event_type == EventType.REPLAN_TRIGGER)
            
            # If we've moved to a new epoch (and it's not the triggering event itself)
            if event.epoch != current_epoch and not is_epoch_boundary:
                current_epoch = event.epoch
                epoch_worker_actions = []
                last_ledger_in_epoch = None
            
            if event.event_type == EventType.TASK:
                task_idx = event.index
                dependencies[event.index] = []
                continue
            
            elif event.event_type in [EventType.INITIAL_PLAN, EventType.NEW_PLAN]:
                plan_idx = event.index
                if task_idx is not None:
                    deps.append(task_idx)
                # New plan depends on the replan trigger
                if event.event_type == EventType.NEW_PLAN:
                    for i in range(event.index - 1, -1, -1):
                        if events[i].event_type == EventType.REPLAN_TRIGGER:
                            deps.append(i)
                            break
            
            elif event.event_type == EventType.LEDGER_UPDATE:
                if task_idx is not None:
                    deps.append(task_idx)
                if plan_idx is not None:
                    deps.append(plan_idx)
                if last_ledger_in_epoch is not None:
                    deps.append(last_ledger_in_epoch)
                deps.extend(epoch_worker_actions)
                
                # Track this ledger for potential REPLAN_TRIGGER
                last_ledger_before_replan = event.index
                last_ledger_in_epoch = event.index
                epoch_worker_actions = []
            
            elif event.event_type == EventType.INSTRUCTION:
                if last_ledger_in_epoch is not None:
                    deps.append(last_ledger_in_epoch)
            
            elif event.event_type == EventType.WORKER_ACTION:
                for i in range(event.index - 1, -1, -1):
                    if events[i].event_type == EventType.INSTRUCTION:
                        deps.append(i)
                        break
                epoch_worker_actions.append(event.index)
            
            elif event.event_type == EventType.NEXT_SPEAKER:
                if last_ledger_in_epoch is not None:
                    deps.append(last_ledger_in_epoch)
            
            elif event.event_type == EventType.REPLAN_TRIGGER:
                # CRITICAL: Use the ledger from BEFORE the epoch reset
                if last_ledger_before_replan is not None:
                    deps.append(last_ledger_before_replan)
                
                # Now trigger the epoch change for subsequent events
                current_epoch = event.epoch
                epoch_worker_actions = []
                last_ledger_in_epoch = None
            
            elif event.event_type == EventType.FINAL_ANSWER:
                if task_idx is not None:
                    deps.append(task_idx)
                if epoch_worker_actions:
                    deps.append(epoch_worker_actions[-1])
                elif last_ledger_in_epoch is not None:
                    deps.append(last_ledger_in_epoch)
            
            else:
                if event.index > 0 and not deps:
                    deps.append(event.index - 1)
            
            dependencies[event.index] = sorted(set(deps))
        
        return dependencies
    
    def get_event_summary(self, events: List[TrajectoryEvent]) -> List[Dict[str, Any]]:
        """Get a summary of events for debugging/visualization."""
        return [
            {
                "index": e.index,
                "type": e.event_type.value,
                "agent": e.agent_name,
                "epoch": e.epoch,
                "content_preview": e.content[:100] + "..." if len(e.content) > 100 else e.content
            }
            for e in events
        ]


def parse_trajectory_dependencies(
    trajectory: List[Dict[str, Any]], 
    mode: str = "structural"
) -> Dict[int, List[int]]:
    """
    Convenience function to parse trajectory and return dependency graph.
    
    Args:
        trajectory: List of trajectory entries with 'role' and 'content' keys
        mode: Dependency mode - "full", "immediate", or "structural"
    
    Returns:
        Dictionary mapping step index to list of dependency indices
    """
    parser = MagenticOneTrajectoryParser(dependency_mode=mode)
    events = parser.parse_trajectory(trajectory)
    return parser.build_dependency_graph(events)


# ============== Visualization & Analysis Utilities ==============

def visualize_dependencies(
    dependencies: Dict[int, List[int]], 
    events: Optional[List[TrajectoryEvent]] = None,
    max_display: int = 50
) -> str:
    """Create ASCII visualization of dependency graph."""
    lines = ["Dependency Graph:", "=" * 50]
    
    for idx in sorted(dependencies.keys())[:max_display]:
        deps = dependencies[idx]
        event_info = ""
        if events and idx < len(events):
            e = events[idx]
            event_info = f" [{e.event_type.value}]"
            if e.agent_name:
                event_info += f" ({e.agent_name})"
        
        if deps:
            deps_str = ", ".join(map(str, deps))
            lines.append(f"  {idx}{event_info} <- [{deps_str}]")
        else:
            lines.append(f"  {idx}{event_info} <- []  (root)")
    
    if len(dependencies) > max_display:
        lines.append(f"  ... ({len(dependencies) - max_display} more events)")
    
    return "\n".join(lines)


def compute_dependency_stats(dependencies: Dict[int, List[int]]) -> Dict[str, Any]:
    """Compute statistics about the dependency graph."""
    num_events = len(dependencies)
    total_deps = sum(len(deps) for deps in dependencies.values())
    max_deps = max(len(deps) for deps in dependencies.values()) if dependencies else 0
    
    # Compute depth (longest path to root)
    depths = {}
    def get_depth(idx):
        if idx in depths:
            return depths[idx]
        if not dependencies.get(idx, []):
            depths[idx] = 0
            return 0
        depths[idx] = 1 + max(get_depth(d) for d in dependencies[idx])
        return depths[idx]
    
    for idx in dependencies:
        get_depth(idx)
    
    return {
        "num_events": num_events,
        "total_dependencies": total_deps,
        "avg_dependencies": total_deps / num_events if num_events > 0 else 0,
        "max_dependencies": max_deps,
        "max_depth": max(depths.values()) if depths else 0,
        "roots": [idx for idx, deps in dependencies.items() if not deps]
    }

# if __name__ == "__main__":
#     # Example trajectory (simplified)
#     # Parse and build dependencies with different modes
#     parser = MagenticOneTrajectoryParser(dependency_mode="structural")
#     events = parser.parse_trajectory(trajectory)
#     deps = parser.build_dependency_graph(events)