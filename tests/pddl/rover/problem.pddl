(define (problem rover-1)
  (:domain rover-charge)
  (:objects rover-1 - rover
            way-1 way-2 way-3 - location)
  (:init (at rover-1 way-1)
         (path way-1 way-2) (path way-2 way-3)
         (= (charge rover-1) 1))
  (:goal (sampled way-3)))
