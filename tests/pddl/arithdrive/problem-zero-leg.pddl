;; The only route to l3 runs over a link of distance 0, so every plan would need
;; a zero-duration action and there is none.
(define (problem drive-over-a-zero-leg)
  (:domain arithdrive)
  (:objects l1 l2 l3 - location  car - vehicle)
  (:init (at car l1)
         (link l1 l2) (link l2 l3)
         (= (speed car) 2)
         (= (distance l1 l2) 5)
         (= (distance l2 l3) 0))
  (:goal (and (at car l3))))
