#!/usr/bin/env python3
"""Unified data loading for RAS-Eval, ATBench, and cx-cmu/mcpbench.

Converts all datasets to a common session format:
    {
        "calls": [{"tool_name", "args_str", "response_str", "response_length", "params_hash"}],
        "label": 0 or 1,
        "task_id": str,
        "source": "ras_eval" | "atbench" | "mcpbench",
        "attack_type": str (for attacks),
        "agent": str (domain/category),
    }
"""
import hashlib
import json
import os
from collections import Counter


# ---- RAS-Eval ----

def _extract_tool_calls_raseval(record):
    calls = []
    pending = []
    for msg in record.get("response", []):
        if not isinstance(msg, dict):
            continue
        if msg.get("type") == "AIMessage" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                pending.append({
                    "name": tc.get("name", "unknown"),
                    "args_str": json.dumps(tc.get("args", {})),
                })
        elif msg.get("type") == "ToolMessage" and pending:
            ci = pending.pop(0)
            content = str(msg.get("content", ""))
            calls.append({
                "tool_name": ci["name"],
                "args_str": ci["args_str"],
                "response_str": content,
                "response_length": len(content),
                "params_hash": int(hashlib.md5(ci["args_str"].encode()).hexdigest(), 16) % 10000,
            })
    return calls


def load_ras_eval(raw_dir="data/raw/ras_eval", pooled=False):
    """Load RAS-Eval benign + attacked sessions."""
    # Attack mode map
    attack_tasks_path = os.path.join(raw_dir, "_repo", "data", "tasks", "attack_tasks.json")
    with open(attack_tasks_path) as f:
        attack_tasks = json.load(f)
    attack_mode_map = {}
    for t in attack_tasks:
        modes = sorted(set(a["mode"] for a in t["attack"]))
        attack_mode_map[t["index"]] = "+".join(modes)

    # Agent map
    tasks_path = os.path.join(raw_dir, "_repo", "data", "tasks", "tasks.json")
    with open(tasks_path) as f:
        tasks = json.load(f)
    agent_map = {t["index"]: t["agent"] for t in tasks}
    target_map = {t["index"]: t["target_index"] for t in attack_tasks}

    # Benign sessions
    benign = []
    logs_dir = os.path.join(raw_dir, "logs")
    exclude = {"guard_response.jsonl"}
    for fname in sorted(os.listdir(logs_dir)):
        if not fname.endswith(".jsonl") or fname in exclude:
            continue
        if not pooled and fname != "glm-4-flash.jsonl":
            continue
        with open(os.path.join(logs_dir, fname)) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                calls = _extract_tool_calls_raseval(r)
                if not calls:
                    continue
                task_id = r.get("index", r.get("id", -1))
                benign.append({
                    "calls": calls,
                    "label": 0,
                    "task_id": f"ras_{task_id}",
                    "source": "ras_eval",
                    "attack_type": "benign",
                    "agent": agent_map.get(task_id, "unknown"),
                })

    # Attacked sessions
    attacked = []
    attacked_path = os.path.join(raw_dir, "attacked", "glm-4-flash.jsonl")
    with open(attacked_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            calls = _extract_tool_calls_raseval(r)
            if not calls:
                continue
            atk_idx = r["index"]
            target = target_map.get(atk_idx, -1)
            attacked.append({
                "calls": calls,
                "label": 1,
                "task_id": f"ras_{target}",
                "source": "ras_eval",
                "attack_type": attack_mode_map.get(atk_idx, "unknown"),
                "agent": agent_map.get(target, "unknown"),
            })

    print(f"[RAS-Eval] {len(benign)} benign + {len(attacked)} attacked")
    return benign, attacked


# ---- ATBench ----

def _extract_tool_calls_atbench(entry):
    calls = []
    messages = entry["contents"][0]
    actions = []
    env_responses = []

    for msg in messages:
        if msg.get("role") == "agent" and msg.get("action"):
            action = msg["action"]
            if action.startswith("Complete"):
                continue
            try:
                parsed = json.loads(action)
                actions.append(parsed)
            except json.JSONDecodeError:
                continue
        elif msg.get("role") == "environment":
            env_responses.append(str(msg.get("content", "")))

    for i, action in enumerate(actions):
        args_str = json.dumps(action.get("arguments", {}))
        response_str = env_responses[i] if i < len(env_responses) else ""
        calls.append({
            "tool_name": action.get("name", "unknown"),
            "args_str": args_str,
            "response_str": response_str,
            "response_length": len(response_str),
            "params_hash": int(hashlib.md5(args_str.encode()).hexdigest(), 16) % 10000,
        })
    return calls


def load_atbench(raw_dir="data/raw/atbench"):
    """Load ATBench sessions."""
    path = os.path.join(raw_dir, "atbench.json")
    with open(path) as f:
        data = json.load(f)

    benign, attacked = [], []
    for entry in data:
        calls = _extract_tool_calls_atbench(entry)
        if not calls:
            continue
        session = {
            "calls": calls,
            "label": entry["label"],
            "task_id": f"atb_{entry['id']}",
            "source": "atbench",
            "attack_type": entry.get("risk_source", "benign") if entry["label"] == 1 else "benign",
            "agent": entry.get("failure_mode", "unknown") if entry["label"] == 1 else "safe",
        }
        if entry["label"] == 0:
            benign.append(session)
        else:
            attacked.append(session)

    print(f"[ATBench] {len(benign)} benign + {len(attacked)} attacked")
    return benign, attacked


# ---- cx-cmu/mcpbench ----

def _extract_tool_calls_mcpbench(record):
    calls = []
    messages = record.get("messages", [])
    pending = []

    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                pending.append({
                    "id": tc.get("id", ""),
                    "name": func.get("name", "unknown"),
                    "args_str": func.get("arguments", "{}"),
                })
        elif msg.get("role") == "tool" and pending:
            tc_id = msg.get("tool_call_id", "")
            matched = None
            for i, p in enumerate(pending):
                if p["id"] == tc_id:
                    matched = pending.pop(i)
                    break
            if matched is None and pending:
                matched = pending.pop(0)
            if matched:
                content = str(msg.get("content", ""))
                calls.append({
                    "tool_name": matched["name"],
                    "args_str": matched["args_str"],
                    "response_str": content,
                    "response_length": len(content),
                    "params_hash": int(hashlib.md5(matched["args_str"].encode()).hexdigest(), 16) % 10000,
                })
    return calls


def load_mcpbench(raw_dir="data/raw/cx_cmu"):
    """Load cx-cmu/mcpbench benign sessions."""
    path = os.path.join(raw_dir, "mcpbench.jsonl")
    sessions = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            calls = _extract_tool_calls_mcpbench(r)
            if not calls:
                continue
            sessions.append({
                "calls": calls,
                "label": 0,
                "task_id": f"mcp_{r.get('task_id', r.get('id', 'unk'))}",
                "source": "mcpbench",
                "attack_type": "benign",
                "agent": r.get("domain", "unknown"),
            })

    print(f"[mcpbench] {len(sessions)} benign sessions")
    return sessions


# ---- Combined loading ----

def load_all(ras_dir="data/raw/ras_eval", atb_dir="data/raw/atbench",
             mcp_dir="data/raw/cx_cmu", pooled=False):
    """Load all datasets and return combined benign + attacked."""
    ras_b, ras_a = load_ras_eval(ras_dir, pooled=pooled)
    atb_b, atb_a = load_atbench(atb_dir)
    mcp_b = load_mcpbench(mcp_dir)

    all_benign = ras_b + atb_b + mcp_b
    all_attacked = ras_a + atb_a

    print(f"\n[Combined] {len(all_benign)} benign + {len(all_attacked)} attacked")
    print(f"  Sources: RAS-Eval({len(ras_b)}b/{len(ras_a)}a), "
          f"ATBench({len(atb_b)}b/{len(atb_a)}a), mcpbench({len(mcp_b)}b)")

    # Attack type distribution
    atk_types = Counter(s["attack_type"] for s in all_attacked)
    print(f"  Attack types: {dict(atk_types.most_common())}")

    return all_benign, all_attacked


if __name__ == "__main__":
    load_all(pooled=True)
