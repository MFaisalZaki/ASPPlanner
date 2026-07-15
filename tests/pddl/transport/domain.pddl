;; Hierarchical typing: truck/package are subtypes of locatable, and the
;; `at` predicate ranges over the supertype -- exercises the has/2 ancestor
;; chain (an object typed as a leaf must satisfy the supertype binding).
(define (domain transport)
  (:requirements :strips :typing)
  (:types truck package - locatable
          location)
  (:predicates (at ?x - locatable ?l - location)
               (in ?p - package ?t - truck)
               (link ?a - location ?b - location))

  (:action drive
    :parameters (?t - truck ?from - location ?to - location)
    :precondition (and (at ?t ?from) (link ?from ?to))
    :effect (and (at ?t ?to) (not (at ?t ?from))))

  (:action load
    :parameters (?p - package ?t - truck ?l - location)
    :precondition (and (at ?p ?l) (at ?t ?l))
    :effect (and (in ?p ?t) (not (at ?p ?l))))

  (:action unload
    :parameters (?p - package ?t - truck ?l - location)
    :precondition (and (in ?p ?t) (at ?t ?l))
    :effect (and (at ?p ?l) (not (in ?p ?t)))))
