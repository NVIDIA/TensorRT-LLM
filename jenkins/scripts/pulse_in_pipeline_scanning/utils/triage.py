"""Utilities for calling the PLC risk triage agent and recording ticket results in OpenSearch."""

import json
import os
import sys

import requests

TRIAGE_AGENT_URL = os.environ.get(
    "TRTLLM_TRIAGE_AGENT_URL",
    "https://plc-risk-triage-agent-deploy-backend.stg.astra.nvidia.com/v1/workflow",
)
TRIAGE_AGENT_TIMEOUT = int(os.environ.get("TRTLLM_TRIAGE_AGENT_TIMEOUT", "1800"))

# Cap items sent to avoid overwhelming the agent on large scans
MAX_TRIAGE_ITEMS = int(os.environ.get("TRTLLM_TRIAGE_MAX_ITEMS", "20"))


def _vuln_doc_to_agent_item(doc: dict) -> dict:
    cve = doc.get("s_cve") or doc.get("s_bdsa") or ""
    pkg = doc.get("s_package_name", "unknown")
    ver = doc.get("s_package_version", "unknown")
    sev = doc.get("s_severity", "")
    scan_type = doc.get("s_type", "")
    fix_version = doc.get("s_package_fix_version") or "N/A"
    detail_parts = [f"scan_type={scan_type}", f"severity={sev}"]
    if cve:
        detail_parts.append(f"CVE={cve}")
    detail_parts.append(f"fixVer={fix_version}")
    return {
        "dependency_name": pkg,
        "action_type": "bump_version",
        "current_version": ver,
        "action_detail": "; ".join(detail_parts),
    }


def _license_doc_to_agent_item(doc: dict) -> dict:
    pkg = doc.get("s_package_name", "unknown")
    ver = doc.get("s_package_version", "unknown")
    scan_type = doc.get("s_type", "")
    is_container = scan_type == "container_license"
    artifact_type = "container" if is_container else "source"
    container = doc.get("s_release_image") or None if is_container else None
    license_ids = doc.get("s_license_ids", "")
    ticket_url = doc.get("s_ticket_url", "")
    action_detail = f"license: {license_ids}" if license_ids and license_ids != "N/A" else None
    return {
        "dependency_name": pkg,
        "action_type": "license_correction",
        "current_version": ver,
        "artifact_type": artifact_type,
        "container": container,
        "license": license_ids if license_ids and license_ids != "N/A" else None,
        "action_detail": action_detail,
        "ticket_url": ticket_url,
    }


def format_risks_for_agent(vuln_docs: list, license_docs: list) -> list:
    """Deduplicate and format risk documents into the agent's Form-B input format."""
    items = []
    seen: set = set()
    for doc in vuln_docs:
        item = _vuln_doc_to_agent_item(doc)
        key = item["dependency_name"]
        if key not in seen:
            seen.add(key)
            items.append(item)
    for doc in license_docs:
        item = _license_doc_to_agent_item(doc)
        key = (
            item["dependency_name"],
            item["current_version"],
            item["action_type"],
            item.get("container"),
        )
        if key not in seen:
            seen.add(key)
            items.append(item)
    return items[:MAX_TRIAGE_ITEMS]


def call_triage_agent(risk_items: list) -> dict:
    """POST risk items to the deployed triage agent; returns the parsed JSON response or {}."""
    if not risk_items:
        return {}
    # The agent API expects input_message as a JSON string (same as query_agent.sh)
    body = {"input_message": json.dumps(risk_items)}
    try:
        resp = requests.post(TRIAGE_AGENT_URL, json=body, timeout=TRIAGE_AGENT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"Triage agent call failed: {exc}", file=sys.stderr)
        return {}


def _ref_from_url(url: str) -> str:
    """Extract Jira key or NVBug ID from a ticket URL."""
    return url.rstrip("/").split("/")[-1]


def extract_ticket_refs(agent_response: dict) -> dict:
    """Parse the structured Form-B agent response into a flat list of ticket refs.

    Expected agent output shape::
    {
        "value": {
          "license_correction_ticket": {"link": "<jira-url>", "description": "..."},
          "version_bump_tickets": [
              {"dependency_name": "<pkg>", "link": "<nvbugs-url>", "description": "..."},
              ...
          ]
        }
    }

    Returns a list of dicts with keys:
        dependency_name, ticket_reference, ticket_url, status, notes
    ``dependency_name`` is None for the license-correction ticket (covers all deps).
    """
    refs = {}

    print(f"[Triage agent raw response] {agent_response}", file=sys.stderr)
    agent_resp_value = json.loads(agent_response.get("value", "{}"))
    license_ticket = agent_resp_value.get("license_correction_ticket")
    if license_ticket:
        link = license_ticket.get("link") or ""
        deps = license_ticket.get("dependencies") or []
        license_info = {
            dep["name"]: {
                "is_permissive": dep.get("is_permissive"),
                "corrected_license": dep.get("corrected_license"),
                "is_nvidia_proprietary": dep.get("is_nvidia_proprietary"),
            }
            for dep in deps
            if dep.get("name")
        }
        refs["license"] = {
            "ticket_url": link,
            "dependencies": deps,
            "license_info": license_info,
            "status": "CREATED" if link else "FAILED",
        }

    refs["vulnerability"] = []
    for item in agent_resp_value.get("version_bump_tickets") or []:
        print(f"[Triage agent version_bump_ticket] {item}", file=sys.stderr)
        link = item.get("link", "")
        if not link:
            continue
        refs["vulnerability"].append(
            {
                "dependency_name": item.get("dependency_name", ""),
                "ticket_reference": _ref_from_url(link),
                "ticket_url": link,
                "status": "CREATED",
                "notes": item.get("description", ""),
            }
        )

    return refs
