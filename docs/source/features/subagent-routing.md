# Sub-agent Routing

Sub-agent routing lets a sub-agent reuse its parent's placement when conversation-aware
affinity routing is enabled. Co-locating their requests on the same context instance and
attention data-parallel (ADP) rank improves the opportunity to reuse shared prompt prefixes.
Each agent retains its own conversation ID for KV-cache and conversation management,
which track a linear history per conversation.

## Configuration

Add the following settings to an existing [disaggregated serving](disagg-serving.md)
configuration. Merge the router settings into the corresponding server sections:

```yaml
conversation_affinity_header_for_subagents: X-Dynamo-Parent-Session-ID
subagent_affinity_scope: context

context_servers:
  router:
    type: conversation
```

The feature is opt-in: leaving `conversation_affinity_header_for_subagents` unset
preserves ordinary conversation routing. Choose a dedicated parent-session header
supplied by your agent gateway, separate from the headers identifying each agent's
own conversation.

On the context workers, enable conversation affinity in your existing attention-DP
configuration:

```yaml
enable_attention_dp: true
attention_dp_config:
  kv_cache_routing_conversation_affinity: true
```

Use `/v1/chat/completions` for both instance and ADP-rank affinity. The
`/v1/completions` endpoint supports instance affinity only.

## Request headers

Send a stable, distinct `x-session-id` for each agent. On sub-agent requests, also
send the configured parent header with the parent's conversation ID. Main-agent
requests carry their own session ID only. An explicit body
`conversation_params.conversation_id` takes precedence over session headers.

For example, with `X-Dynamo-Parent-Session-ID` configured:

| Request | `x-session-id` | `X-Dynamo-Parent-Session-ID` | Context routing key | Conversation ID |
| --- | --- | --- | --- | --- |
| Parent | `parent-1` | Omitted | `parent-1` | `parent-1` |
| Sub-agent A | `child-a` | `parent-1` | `parent-1` | `child-a` |
| Sub-agent B | `child-b` | `parent-1` | `parent-1` | `child-b` |

Keep these headers stable across each sub-agent's turns. This feature supports
one level of sub-agents; nested sub-agent routing is not supported.

## Routing scope

`subagent_affinity_scope` controls which fleets use the parent routing key:

| Scope | Context fleet | Generation fleet |
| --- | --- | --- |
| `context` (default) | Parent affinity | Child's own conversation routing |
| `both` | Parent affinity | Parent affinity |

The default scope targets shared prefill cache reuse while allowing children to
spread across generation workers. To use `both`, configure the generation fleet
with `router.type: conversation` and enable conversation-aware ADP affinity on its
workers as well. Use `context` scope with conditional disaggregation, which requires
a KV-cache-aware generation router.

Placement follows the existing conversation routers' behavior. Affinity is best
effort: a saturated ADP rank can overflow to another rank, and affinity bindings can
be evicted. Cache reuse depends on matching prompt prefixes and cache availability.

## How it works

The disagg edge reads the configured parent header into an internal routing key.
The conversation router uses that key to select the parent's instance, and the HTTP
client forwards it as `x-trtllm-subagent-affinity-id`. The worker places the key in
`SchedulingParams.subagent_affinity_id` for the conversation-aware ADP router.

The child's conversation ID continues to identify its own history throughout this
path. The internal affinity field is excluded from the serialized request body;
older workers can ignore the new header during a rolling deployment.
