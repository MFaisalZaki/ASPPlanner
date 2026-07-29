;; Forall effects, the ADL shape the encoding expands in the grounder rather
;; than compiling away. `move` carries the hard case -- a `forall` over a
;; `when`, whose bindings must fire independently: an object travels with the
;; briefcase iff *it* is inside, not iff every object is. `sweep` is the plain
;; case, a `forall` with no condition, and `tax` the numeric one.
(define (domain briefcase)
  (:requirements :strips :typing :adl :fluents)
  (:types obj loc)
  (:predicates (at-b ?l - loc) (in ?o - obj) (glued ?o - obj)
               (at-o ?o - obj ?l - loc))
  (:functions (weight ?o - obj))

  (:action move
    :parameters (?from - loc ?to - loc)
    :precondition (and (at-b ?from) (not (= ?from ?to)))
    :effect (and (at-b ?to) (not (at-b ?from))
                 ;; a disjunction under the quantifier, so the effect's
                 ;; orGroup facts are keyed by the per-binding effect term too
                 (forall (?o - obj)
                    (when (or (in ?o) (glued ?o))
                       (and (at-o ?o ?to) (not (at-o ?o ?from)))))))

  ;; unconditional forall effect
  (:action dump
    :parameters ()
    :effect (forall (?o - obj) (not (in ?o))))

  ;; unconditional forall effect on a numeric fluent
  (:action tax
    :parameters ()
    :effect (forall (?o - obj) (increase (weight ?o) 2)))

  (:action put-in
    :parameters (?o - obj ?l - loc)
    :precondition (and (at-o ?o ?l) (at-b ?l) (not (in ?o)))
    :effect (in ?o)))
