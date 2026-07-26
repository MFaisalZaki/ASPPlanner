;; problem.pddl plus the `(:metric minimize (total-time))` every IPC temporal
;; instance carries, and nothing else: the initial state still fills the whole
;; `travel-time` cross product, so this is the makespan feature on its own.
(define (problem drive-to-l3-and-top-up-metric)
  (:domain tempdrive)
  (:objects l1 l2 l3 - location)
  (:init (at l1) (charged)
         (link l1 l2) (link l2 l3)
         (= (travel-time l1 l1) 1) (= (travel-time l1 l2) 3) (= (travel-time l1 l3) 1)
         (= (travel-time l2 l1) 1) (= (travel-time l2 l2) 1) (= (travel-time l2 l3) 2)
         (= (travel-time l3 l1) 1) (= (travel-time l3 l2) 1) (= (travel-time l3 l3) 1))
  (:goal (and (at l3) (topped_up)))
  (:metric minimize (total-time)))
