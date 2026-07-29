;; `p1` rides up from f0 to f2 and `p2` rides down from f2 to f0. The second
;; leg is only reachable if `p1` actually leaves the lift at f2 -- with the
;; clear dropped, `p1` stays aboard and `going-down` blocks nothing on the way
;; up but everything after, so the task reads as unsolvable.
(define (problem lift-1)
  (:domain lift)
  (:objects p1 p2 - passenger f0 f1 f2 - floor)
  (:init (lift-at f0)
         (above f0 f1) (above f1 f2) (above f0 f2)
         (origin p1 f0) (destin p1 f2)
         (origin p2 f2) (destin p2 f0) (going-down p2))
  (:goal (and (served p1) (served p2))))
