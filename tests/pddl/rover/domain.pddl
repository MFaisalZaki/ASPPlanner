;; Numeric fluents on the lifted path (no pre-grounding): constant-delta
;; increase/decrease effects, linear comparison preconditions, and a
;; parameter inequality that folds into the action signature.
(define (domain rover-charge)
  (:requirements :strips :typing :fluents :equality)
  (:types rover location)
  (:predicates (at ?r - rover ?l - location)
               (path ?a - location ?b - location)
               (sampled ?l - location))
  (:functions (charge ?r - rover))

  (:action move
    :parameters (?r - rover ?a - location ?b - location)
    :precondition (and (at ?r ?a) (path ?a ?b) (not (= ?a ?b))
                       (>= (charge ?r) 1))
    :effect (and (at ?r ?b) (not (at ?r ?a)) (decrease (charge ?r) 1)))

  (:action recharge
    :parameters (?r - rover)
    :precondition (<= (charge ?r) 0)
    :effect (increase (charge ?r) 3))

  (:action sample
    :parameters (?r - rover ?l - location)
    :precondition (at ?r ?l)
    :effect (sampled ?l)))
