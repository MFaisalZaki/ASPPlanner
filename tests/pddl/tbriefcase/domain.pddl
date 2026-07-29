;; The briefcase, durative. A `forall` effect on a durative action travels
;; into the snap actions the encoder splits it into, so the at-start and
;; at-end halves each carry their own quantified conditional effect.
(define (domain tbriefcase)
  (:requirements :typing :adl :durative-actions)
  (:types obj loc)
  (:predicates (at-b ?l - loc) (in ?o - obj) (at-o ?o - obj ?l - loc))

  (:durative-action move
    :parameters (?from - loc ?to - loc)
    :duration (= ?duration 2)
    :condition (at start (at-b ?from))
    :effect (and (at start (not (at-b ?from))) (at end (at-b ?to))
                 (forall (?o - obj)
                    (and (at end (when (in ?o) (at-o ?o ?to)))
                         (at start (when (in ?o) (not (at-o ?o ?from))))))))

  (:durative-action put-in
    :parameters (?o - obj ?l - loc)
    :duration (= ?duration 1)
    :condition (and (at start (at-o ?o ?l)) (at start (at-b ?l)))
    :effect (at end (in ?o))))
