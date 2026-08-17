;; The same task with fractional values, so the fluents that need one get a
;; storage factor and the product has to survive it. `burn` is 1/2 and the
;; distances are halves as well, so a leg costs a quarter of what the integer
;; problem charges -- and `fuel` starts at 2.5, which the encoding cannot hold
;; as it stands.
;;
;; The trap this pins is the one a task-wide factor falls into: with every value
;; multiplied by the same k, a *product* of two of them comes out k times too
;; large, so the cheap leg would look unaffordable (or the dear one free).
(define (problem zenoflight-fractional)
  (:domain zenoflight)
  (:objects plane - aircraft
            c1 c2 c3 - city)
  (:init
    (at plane c1)
    (= (fuel plane) 2.5)
    (= (capacity plane) 5)
    (= (burn plane) 0.5)
    (= (distance c1 c2) 2)
    (= (distance c2 c3) 8)
    (= (distance c1 c3) 9)
    (= (distance c2 c1) 2)
    (= (distance c3 c2) 8)
    (= (distance c3 c1) 9)
    (= (distance c1 c1) 0)
    (= (distance c2 c2) 0)
    (= (distance c3 c3) 0)
    (= (total-burn) 0))
  (:goal (and (delivered c2) (delivered c3)))
)
