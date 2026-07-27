;; A fractional value in a fluent that is read as a duration: satellite states
;; `(= (slew_time ?a ?b) 5.9)` and clingo has no rationals, so those values go
;; onto an integer grid of their own that the duration arithmetic divides back
;; out. 5.9 / 2 is 2.95, and the leg has to last exactly that.
(define (problem drive-a-fractional-leg)
  (:domain arithdrive)
  (:objects l1 l2 l3 - location  car - vehicle)
  (:init (at car l1)
         (link l1 l2) (link l2 l3)
         (= (speed car) 2)
         (= (distance l1 l2) 5.9)
         (= (distance l2 l3) 3))
  (:goal (and (at car l3))))
