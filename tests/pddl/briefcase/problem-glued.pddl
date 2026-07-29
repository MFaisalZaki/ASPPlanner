;; `k` is glued to the briefcase, so it travels without ever being put in --
;; the disjunctive arm of the effect's condition. `d` is put in as usual.
(define (problem briefcase-glued)
  (:domain briefcase)
  (:objects d k - obj home office - loc)
  (:init (at-b home) (at-o d home) (at-o k home) (glued k)
         (= (weight d) 0) (= (weight k) 0))
  (:goal (and (at-o d office) (at-o k office))))
