;; Zenotravel's fuel accounting, cut down to the shape that used to be refused:
;; the cost of a leg is `(* (distance ?c1 ?c2) (burn ?a))`, a product of two
;; fluents. Neither is state -- no action writes either -- so the grounder can
;; read both out of the initial state and the product is a coefficient by the
;; time the rule grounds.
;;
;; `capacity` is static too and read linearly, so the same task exercises both
;; readings of a static fluent side by side: one folded into a lookup because a
;; product asked for it, one left as an ordinary numTerm over its numval.
(define (domain zenoflight)
  (:requirements :typing :fluents)
  (:types aircraft city - object)

  (:predicates (at ?a - aircraft ?c - city)
               (delivered ?c - city))

  (:functions (fuel ?a - aircraft)
              (distance ?c1 - city ?c2 - city)
              (burn ?a - aircraft)
              (capacity ?a - aircraft)
              (total-burn))

  (:action fly
    :parameters (?a - aircraft ?c1 - city ?c2 - city)
    :precondition (and (at ?a ?c1)
                       (>= (fuel ?a) (* (distance ?c1 ?c2) (burn ?a))))
    :effect (and (not (at ?a ?c1))
                 (at ?a ?c2)
                 (delivered ?c2)
                 (decrease (fuel ?a) (* (distance ?c1 ?c2) (burn ?a)))
                 (increase (total-burn) (* (distance ?c1 ?c2) (burn ?a)))))

  (:action refuel
    :parameters (?a - aircraft)
    :precondition (< (fuel ?a) (capacity ?a))
    :effect (assign (fuel ?a) (capacity ?a)))
)
