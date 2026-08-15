;; The order is 4 units and the two markets price them differently, so `spent`
;; pins which one the truck bought at: 4*3 = 12 at m1, 4*5 = 20 at m2. Asking
;; for `spent <= 12` rules out m2 and rules out any reading of the coefficient
;; that is off by a factor.
;;
;; `moved` needs 3, which is two hauls at rate 1 (2 * 1.5 = 3) -- and `rate` is
;; written, so the 3/2 cannot be looked up. A plan that rounds the coefficient
;; down to 1 never reaches 3 in two hauls; one that rounds it up to 2 reaches it
;; and then fails validation against the task's own arithmetic.
(define (problem tradepost-order)
  (:domain tradepost)
  (:objects widget - goods
            m1 m2 - market
            van - truck)
  (:init
    (at van m2)
    (= (price widget m1) 3)
    (= (price widget m2) 5)
    (= (request widget) 4)
    (= (bought widget) 0)
    (= (spent) 0)
    (= (rate van) 1)
    (= (moved van) 0))
  (:goal (and (>= (bought widget) 4)
              (<= (spent) 12)
              (>= (moved van) 3))))
