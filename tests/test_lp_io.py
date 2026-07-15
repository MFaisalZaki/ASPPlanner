"""Tests for reading/writing .lp encodings as ASPTerm objects."""

import io

from aspplanner.asp_planner import ASPPlanner, ENCODERS
from aspplanner.compilers.asp_facts import (
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

from test_planner import robot_line_problem

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
    planner = ASPPlanner(problem, "seq")

    dumped = tmp_path / "roundtripped.lp"
    dump_lp(parse_lp_file(SEQ_ENCODING_PATH), dumped)
    planner.encoding_path = str(dumped)

    plan = planner.plan(max_horizon=6)
    assert len(plan.actions) == 3


def test_dump_accepts_fact_builders_and_strings(tmp_path):
    planner = ASPPlanner(robot_line_problem(), "seq")
    out = tmp_path / "facts.lp"
    dump_lp(sorted(planner.compiled_task.fact_lines), out)
    reparsed = parse_lp_file(out)
    assert all(isinstance(t, ASPFact) or isinstance(t, ASPRule) for t in reparsed)
    assert any("initialState" in str(t) for t in reparsed)


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
    planner = ASPPlanner(robot_line_problem(), "seq")
    terms = planner.encoding_terms()
    assert all(isinstance(t, ASPStatement) for t in terms)
    assert any(isinstance(t, ASPConstraint) for t in terms)  # the goal check


def test_dump_lp_program(tmp_path):
    planner = ASPPlanner(robot_line_problem(), "seq")
    out = tmp_path / "program.lp"
    planner.dump_lp_program(out)
    text = out.read_text()
    assert "initialState(" in text and "#program step(t)." in text
