;; Only `d` is ever put in, and `k` has to stay home. If the bindings of the
;; `forall` effect shared one effect term -- so that it fired only when *every*
;; object satisfied the condition -- nothing here would move at all.
(define (problem briefcase-carry)
  (:domain briefcase)
  (:objects d k - obj home office - loc)
  (:init (at-b home) (at-o d home) (at-o k home)
         (= (weight d) 0) (= (weight k) 0))
  (:goal (and (at-o d office) (at-o k home))))
