"""Tests for reading/writing .lp encodings as ASPTerm objects."""

import io

from aspplanners.plasp.planner import PLASPPlanner, ENCODERS
from aspplanners.lp_io import (
    ASPConstraint,
    ASPDirective,
    ASPFact,
    ASPRule,
    ASPStatement,
    ASPWeakConstraint,
    dump_lp,
    parse_lp,
    parse_lp_file,
)

from test_planner import (
    numeric_counter_problem,
    robot_line_problem,
    rover_recharge_problem,
)

SEQ_ENCODING_PATH = ENCODERS["seq"][1]


def test_parse_lp_classifies_statements():
    terms = parse_lp("""
        p(1..3).
        q(X) :- p(X), not r(X).
        :- q(X), X > 2.
        :~ q(X). [1@0, X]
        #show q/1.
        #program step(t).
        occ(t) :- q(t).
    """)
    kinds = [type(t) for t in terms]
    assert kinds == [ASPFact, ASPRule, ASPConstraint, ASPWeakConstraint,
                     ASPDirective, ASPDirective, ASPRule]
    rule = terms[1]
    assert rule.head == "q(X)"
    assert rule.body == ["p(X)", "not r(X)"]
    constraint = terms[2]
    assert str(constraint).startswith(":- ")  # not clingo's "#false :- ..."


def test_parse_dump_parse_is_a_fixpoint():
    first = parse_lp_file(SEQ_ENCODING_PATH)
    assert first, "the bundled encoding should parse to statements"

    buffer = io.StringIO()
    dump_lp(first, buffer)
    second = parse_lp(buffer.getvalue())
    assert [str(t) for t in first] == [str(t) for t in second]

    # the bundled encoding exercises every category except weak constraints
    kinds = {type(t) for t in first}
    assert {ASPFact, ASPRule, ASPConstraint, ASPDirective} <= kinds


def test_dumped_encoding_still_solves(tmp_path):
    """parse -> dump must preserve the encoding's semantics end to end."""
    problem = robot_line_problem()
    planner = PLASPPlanner(problem, "seq")

    dumped = tmp_path / "roundtripped.lp"
    dump_lp(parse_lp_file(SEQ_ENCODING_PATH), dumped)
    planner.encoding_path = str(dumped)

    plan = planner.plan(max_horizon=6)
    assert len(plan.actions) == 3


def test_dump_accepts_fact_builders_and_strings(tmp_path):
    planner = PLASPPlanner(robot_line_problem(), "seq")
    out = tmp_path / "facts.lp"
    dump_lp(sorted(planner.compiled_task.fact_lines), out)
    reparsed = parse_lp_file(out)
    assert all(isinstance(t, ASPFact) or isinstance(t, ASPRule) for t in reparsed)
    assert any("initialState" in str(t) for t in reparsed)


def test_dump_reparses_numeric_fact_builders(tmp_path):
    """A numeric task's fact builders (numVariable/numPrecondition/numEffect)
    must render to valid clingo that dumps, reparses, and round-trips.

    ``robot_line_problem`` has no numeric fluents, so this is the only place
    the numeric ASPTerm rendering (ASPNumFluent, ASPNumComparison, numeric
    ASPAction effects) is exercised through the .lp I/O path.
    """
    # finish needs `counter >= 3`; tick increments counter -> a `<=` numeric
    # precondition and an increase numEffect on an int-valued fluent.
    planner = PLASPPlanner(numeric_counter_problem(threshold=3), "seq")
    out = tmp_path / "numeric_facts.lp"
    dump_lp(sorted(planner.compiled_task.fact_lines), out)

    reparsed = parse_lp_file(out)
    assert all(isinstance(t, (ASPFact, ASPRule)) for t in reparsed)
    rendered = [str(t) for t in reparsed]

    # the int-valued fluent is declared as a numVariable (fact)...
    assert any(r.startswith("numVariable(") for r in rendered)
    # ...the `counter >= 3` precondition renders as a `le` numComparison...
    assert any("numPrecondition(" in r and ",le," in r for r in rendered)
    # ...and the increase effect renders as a numEffect rule.
    assert any(r.startswith("numEffect(") for r in rendered)

    # reparse -> dump -> reparse is a fixpoint (numeric rendering is stable).
    buffer = io.StringIO()
    dump_lp(reparsed, buffer)
    assert [str(t) for t in parse_lp(buffer.getvalue())] == rendered


def test_numeric_goal_renders_as_numgoal_fact(tmp_path):
    """A numeric comparison *goal* (`battery >= 2`) must compile to a numGoal
    fact -- not a boolean goal/2 -- and survive dump + reparse."""
    planner = PLASPPlanner(rover_recharge_problem(target=2), "seq")
    out = tmp_path / "numeric_goal.lp"
    dump_lp(sorted(planner.compiled_task.fact_lines), out)

    rendered = [str(t) for t in parse_lp_file(out)]
    # the `battery >= 2` goal folds to `2 <= battery` -> a `le` numGoal fact...
    assert any(r.startswith("numGoal(le,") for r in rendered)
    # ...and it is NOT emitted as a boolean goal/2 atom.
    assert not any(r.startswith("goal(") for r in rendered)


def test_parse_keeps_script_blocks():
    terms = parse_lp("""
#script (python)
def helper(x):
    return x
#end.
p(1).
""")
    assert isinstance(terms[0], ASPDirective)
    assert str(terms[0]).startswith("#script (python)")
    assert isinstance(terms[1], ASPFact)


def test_planner_encoding_terms():
    planner = PLASPPlanner(robot_line_problem(), "seq")
    terms = planner.encoding_terms()
    assert all(isinstance(t, ASPStatement) for t in terms)
    assert any(isinstance(t, ASPConstraint) for t in terms)  # the goal check


def test_dump_lp_program(tmp_path):
    planner = PLASPPlanner(robot_line_problem(), "seq")
    out = tmp_path / "program.lp"
    planner.dump_lp_program(out)
    text = out.read_text()
    assert "initialState(" in text and "#program step(t)." in text
