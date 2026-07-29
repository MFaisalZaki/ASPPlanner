;; The numeric forall effect: one `tax` has to raise *every* object's weight,
;; including the one that never enters the briefcase.
(define (problem briefcase-tax)
  (:domain briefcase)
  (:objects d k - obj home office - loc)
  (:init (at-b home) (at-o d home) (at-o k home)
         (= (weight d) 0) (= (weight k) 0))
  (:goal (and (at-o d office) (at-o k home)
              (>= (weight d) 2) (>= (weight k) 2))))
