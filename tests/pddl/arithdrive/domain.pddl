;; A duration that is arithmetic over two static fluents rather than a lone one:
;; `(/ (distance ?a ?b) (speed ?v))`, the shape depots, pipesworld, map-analyzer
;; and road-traffic all state their move actions with. The division has to stay
;; exact -- clingo's `/` truncates, so an encoding that divides before it scales
;; turns a 5/2 leg into 2 -- and a leg whose distance is 0 is not an action at
;; all, however well its other preconditions hold.
(define (domain arithdrive)
  (:requirements :typing :durative-actions :numeric-fluents)
  (:types location vehicle)
  (:predicates
    (at ?v - vehicle ?l - location)
    (link ?a - location ?b - location))
  (:functions
    (distance ?a - location ?b - location)
    (speed ?v - vehicle))

  (:durative-action drive
    :parameters (?v - vehicle ?a - location ?b - location)
    :duration (= ?duration (/ (distance ?a ?b) (speed ?v)))
    :condition (and (at start (at ?v ?a))
                    (at start (link ?a ?b)))
    :effect (and (at start (not (at ?v ?a)))
                 (at end (at ?v ?b)))))
