;; The two numeric shapes zenoflight does not reach, both parameterised so the
;; encoding has to instantiate them per binding:
;;
;;   `buy`  -- `(* (- (request ?g) (bought ?g)) (price ?g ?m))`, TPP-Metric's
;;             cost. `price` and `request` are static and `bought` is not, so
;;             this is a static value multiplying a *dynamic* fluent: the
;;             grounder resolves the coefficient, the fluent stays a numTerm
;;             over its numval.
;;
;;   `haul` -- `(increase (moved ?t) (* (rate ?t) 1.5))`, fo-sailing's step. The
;;             3/2 is on a fluent an action writes, so nothing can be looked up;
;;             what clears it is storing `moved` twice as fine, which is a
;;             factor on that fluent alone.
(define (domain tradepost)
  (:requirements :typing :fluents)
  (:types goods market truck - object)

  (:predicates (at ?t - truck ?m - market))

  (:functions (price ?g - goods ?m - market)
              (request ?g - goods)
              (bought ?g - goods)
              (spent)
              (rate ?t - truck)
              (moved ?t - truck))

  ;; Buy the whole outstanding order at one market, and pay for it.
  (:action buy
    :parameters (?t - truck ?g - goods ?m - market)
    :precondition (and (at ?t ?m) (< (bought ?g) (request ?g)))
    :effect (and (increase (spent) (* (- (request ?g) (bought ?g)) (price ?g ?m)))
                 (increase (bought ?g) (- (request ?g) (bought ?g)))))

  (:action drive
    :parameters (?t - truck ?from - market ?to - market)
    :precondition (at ?t ?from)
    :effect (and (not (at ?t ?from)) (at ?t ?to)))

  (:action haul
    :parameters (?t - truck)
    :effect (increase (moved ?t) (* (rate ?t) 1.5)))

  (:action speed-up
    :parameters (?t - truck)
    :effect (increase (rate ?t) 1))
)
