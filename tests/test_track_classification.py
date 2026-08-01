"""Tests for the track a domain file is filed under.

`classify_domain` decides, by reading the PDDL, whether a task belongs to the
classical, numeric, or temporal track -- and the runner dispatches on that, so a
domain filed under the wrong track is not merely mislabelled in the tables, it
is handed to a different set of planners (or to none).

The trap pinned here is action costs. IPC's standard STRIPS encoding declares
``(:functions (total-cost) - number)`` purely so plans can be scored, and
treating that declaration as the signal filed 69 classical domains -- sokoban,
woodworking, transport and the rest of the cost-annotated suite -- under
numeric. The rule is that a fluent must be *used as state*, not just declared.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'benchmarks'))

aspbench_tasks = pytest.importorskip(
    'aspbench.tasks', reason='the benchmark harness is a separate package')
classify_domain = aspbench_tasks.classify_domain


def write(tmp_path, text):
    path = tmp_path / 'domain.pddl'
    path.write_text(text)
    return str(path)


# ----------------------------------------------------------------------
# Cost annotations are not numeric planning
# ----------------------------------------------------------------------

def test_a_bare_strips_domain_is_classical(tmp_path):
    domain = write(tmp_path, """
        (define (domain d)
          (:requirements :strips)
          (:predicates (at ?x) (clear ?x))
          (:action move :parameters (?a ?b)
            :precondition (at ?a) :effect (and (not (at ?a)) (at ?b))))
    """)
    assert classify_domain(domain) == 'classical'


def test_a_total_cost_accumulator_stays_classical(tmp_path):
    """The sokoban-opt08-strips shape: one function, only ever increased."""
    domain = write(tmp_path, """
        (define (domain d)
          (:requirements :strips :action-costs)
          (:functions (total-cost) - number)
          (:predicates (at ?x))
          (:action move :parameters (?a ?b)
            :precondition (at ?a)
            :effect (and (at ?b) (increase (total-cost) 1))))
    """)
    assert classify_domain(domain) == 'classical'


def test_a_static_cost_table_stays_classical(tmp_path):
    """The woodworking-opt08-strips shape: per-object costs, read only as costs.

    `(glaze-cost ?x)` is a declared function that never changes and is never
    compared -- it is a constant the cost expression looks up.
    """
    domain = write(tmp_path, """
        (define (domain d)
          (:requirements :strips :action-costs)
          (:functions (total-cost) - number
                      (glaze-cost ?obj - part) - number)
          (:predicates (surface-condition ?x ?y))
          (:action glaze :parameters (?x - part)
            :precondition (surface-condition ?x rough)
            :effect (and (surface-condition ?x smooth)
                         (increase (total-cost) (glaze-cost ?x)))))
    """)
    assert classify_domain(domain) == 'classical'


def test_a_fluent_declared_but_never_used_stays_classical(tmp_path):
    """Names inside `(:functions ...)` are declarations, not uses.

    The block is excised before the domain body is searched, so a function
    called `fuel` does not read as a numeric use of `fuel`.
    """
    domain = write(tmp_path, """
        (define (domain d)
          (:requirements :strips :action-costs)
          (:functions (total-cost) - number (fuel ?v - vehicle) - number)
          (:predicates (at ?x))
          (:action move :parameters (?a ?b)
            :precondition (at ?a)
            :effect (and (at ?b) (increase (total-cost) 1))))
    """)
    assert classify_domain(domain) == 'classical'


# ----------------------------------------------------------------------
# Real numeric state
# ----------------------------------------------------------------------

@pytest.mark.parametrize('comparison', ['(< (fuel ?v) 10)', '(> (fuel ?v) 10)',
                                        '(<= (fuel ?v) 10)', '(>= (fuel ?v) 10)'])
def test_a_numeric_precondition_is_numeric(tmp_path, comparison):
    """Every comparison operator counts, `<=` and `>=` included.

    Ordering `<|<=` in the alternation matches the `<` of `(<= ...)` and then
    fails on the `=`, which silently passed every `<=` domain through as
    classical -- hence all four are pinned.
    """
    domain = write(tmp_path, f"""
        (define (domain d)
          (:requirements :strips :numeric-fluents)
          (:functions (total-cost) - number (fuel ?v - vehicle) - number)
          (:predicates (at ?x))
          (:action move :parameters (?v - vehicle ?a ?b)
            :precondition (and (at ?a) {comparison})
            :effect (and (at ?b) (increase (total-cost) 1))))
    """)
    assert classify_domain(domain) == 'numeric'


@pytest.mark.parametrize('effect', ['(decrease (fuel ?v) 3)', '(increase (fuel ?v) 3)',
                                    '(assign (fuel ?v) 3)', '(scale-up (fuel ?v) 3)'])
def test_assigning_a_non_cost_fluent_is_numeric(tmp_path, effect):
    domain = write(tmp_path, f"""
        (define (domain d)
          (:requirements :strips :numeric-fluents)
          (:functions (total-cost) - number (fuel ?v - vehicle) - number)
          (:predicates (at ?x))
          (:action move :parameters (?v - vehicle ?a ?b)
            :precondition (at ?a)
            :effect (and (at ?b) {effect} (increase (total-cost) 1))))
    """)
    assert classify_domain(domain) == 'numeric'


# ----------------------------------------------------------------------
# Temporal wins, and comments never decide
# ----------------------------------------------------------------------

def test_a_durative_action_is_temporal(tmp_path):
    domain = write(tmp_path, """
        (define (domain d)
          (:requirements :durative-actions :numeric-fluents)
          (:functions (fuel ?v - vehicle) - number)
          (:durative-action move :parameters (?v - vehicle)
            :duration (= ?duration 5)
            :condition (over all (>= (fuel ?v) 1))
            :effect (at end (decrease (fuel ?v) 1))))
    """)
    assert classify_domain(domain) == 'temporal'


def test_a_commented_out_numeric_use_does_not_decide(tmp_path):
    domain = write(tmp_path, """
        (define (domain d)
          (:requirements :strips :action-costs)
          (:functions (total-cost) - number)
          ; (>= (fuel ?v) 1) and (decrease (fuel ?v) 1) live in a comment
          (:predicates (at ?x))
          (:action move :parameters (?a ?b)
            :precondition (at ?a)
            :effect (and (at ?b) (increase (total-cost) 1))))
    """)
    assert classify_domain(domain) == 'classical'


def test_a_missing_domain_file_falls_back_to_classical(tmp_path):
    assert classify_domain(str(tmp_path / 'nope.pddl')) == 'classical'
