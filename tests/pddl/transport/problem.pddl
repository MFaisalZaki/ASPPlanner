(define (problem transport-1)
  (:domain transport)
  (:objects truck-1 - truck
            pkg-1 - package
            loc-1 loc-2 loc-3 - location)
  (:init (at truck-1 loc-1) (at pkg-1 loc-1)
         (link loc-1 loc-2) (link loc-2 loc-3))
  (:goal (at pkg-1 loc-3)))
