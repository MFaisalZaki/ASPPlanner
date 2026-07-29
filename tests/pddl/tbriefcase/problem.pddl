;; `k` never goes in, so it must still be home when `d` reaches the office.
(define (problem tbriefcase-1)
  (:domain tbriefcase)
  (:objects d k - obj home office - loc)
  (:init (at-b home) (at-o d home) (at-o k home))
  (:goal (and (at-o d office) (at-o k home))))
