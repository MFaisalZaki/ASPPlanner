;; The shape every IPC temporal instance has, which the engines used to refuse:
;; a `(:metric minimize (total-time))` (ProblemKind MAKESPAN) and a
;; `(:functions ...)` cross product that `(:init ...)` only fills in for the
;; pairs a `drive` can actually use (ProblemKind UNDEFINED_INITIAL_NUMERIC).
(define (problem drive-to-l3-and-top-up-ipc-shaped)
  (:domain tempdrive)
  (:objects l1 l2 l3 - location)
  (:init (at l1) (charged)
         (link l1 l2) (link l2 l3)
         (= (travel-time l1 l2) 3) (= (travel-time l2 l3) 2))
  (:goal (and (at l3) (topped_up)))
  (:metric minimize (total-time)))
