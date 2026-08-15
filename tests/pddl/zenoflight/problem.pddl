;; c1 -> c2 -> c3, and the plane cannot make both legs on one tank:
;;   fly(c1,c2) burns 2*3 = 6 of the 10 it starts with,
;;   fly(c2,c3) needs   4*3 = 12, so it has to refuel first.
;; A plan that got the product wrong by a factor -- say by reading `burn`'s
;; stored value in the wrong units -- either skips the refuel or cannot fly at
;; all, and neither validates against this problem.
(define (problem zenoflight-2legs)
  (:domain zenoflight)
  (:objects plane - aircraft
            c1 c2 c3 - city)
  (:init
    (at plane c1)
    (= (fuel plane) 10)
    (= (capacity plane) 20)
    (= (burn plane) 3)
    (= (distance c1 c2) 2)
    (= (distance c2 c3) 4)
    (= (distance c1 c3) 9)
    (= (distance c2 c1) 2)
    (= (distance c3 c2) 4)
    (= (distance c3 c1) 9)
    (= (distance c1 c1) 0)
    (= (distance c2 c2) 0)
    (= (distance c3 c3) 0)
    (= (total-burn) 0))
  (:goal (and (delivered c2) (delivered c3)))
)
