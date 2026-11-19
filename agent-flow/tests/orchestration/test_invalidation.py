from agent_flow.orchestration.graph import ExecutionGraph, Node
from agent_flow.orchestration.invalidation import invalidation_set


def _g(*nodes):
    return ExecutionGraph(nodes=tuple(nodes))


def test_identical_graphs_invalidate_nothing():
    g = _g(Node(id="a", type="s"), Node(id="b", type="s", depends_on=("a",)))
    assert invalidation_set(g, g) == set()


def test_added_node_is_changed():
    old = _g(Node(id="a", type="s"))
    new = _g(Node(id="a", type="s"), Node(id="b", type="s"))
    assert invalidation_set(old, new) == {"b"}


def test_changed_depends_on_invalidates_node_and_dependents():
    # b depends_on a; c depends_on b. In new, b's depends_on changed → b, c invalidated (a untouched).
    old = _g(
        Node(id="a", type="s"),
        Node(id="b", type="s", depends_on=("a",)),
        Node(id="c", type="s", depends_on=("b",)),
    )
    new = _g(
        Node(id="a", type="s"),
        Node(id="b", type="s"),  # b no longer depends_on a
        Node(id="c", type="s", depends_on=("b",)),
    )
    assert invalidation_set(old, new) == {"b", "c"}


def test_changed_kind_invalidates_node_and_transitive_dependents():
    old = _g(
        Node(id="a", type="s"),
        Node(id="b", type="s", depends_on=("a",)),
        Node(id="c", type="s", depends_on=("b",)),
    )
    new = _g(
        Node(id="a", type="s", kind="merge"),  # a's kind changed
        Node(id="b", type="s", depends_on=("a",)),
        Node(id="c", type="s", depends_on=("b",)),
    )
    assert invalidation_set(old, new) == {"a", "b", "c"}


def test_ancestor_stage_dep_change_invalidates_nested_goals():
    # s2 depends_on s1; s2 has a child goal g with empty depends_on. Changing s1 must
    # invalidate s2 AND its child g (g inherits s2's cross-stage dep).
    old = _g(
        Node(id="s1", type="stage"),
        Node(id="s2", type="stage", depends_on=("s1",), children=(Node(id="g", type="goal"),)),
    )
    new = _g(
        Node(id="s1", type="stage", kind="merge"),  # s1 changed
        Node(id="s2", type="stage", depends_on=("s1",), children=(Node(id="g", type="goal"),)),
    )
    assert invalidation_set(old, new) == {"s1", "s2", "g"}


def test_content_changed_callback_marks_node_and_dependents():
    old = _g(Node(id="a", type="s"), Node(id="b", type="s", depends_on=("a",)))
    new = _g(Node(id="a", type="s"), Node(id="b", type="s", depends_on=("a",)))
    # structurally identical, but a's prose changed
    got = invalidation_set(old, new, content_changed=lambda nid: nid == "a")
    assert got == {"a", "b"}
