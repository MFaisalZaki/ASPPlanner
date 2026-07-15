(define (problem alarm-trigger)
  (:domain alarm)
  (:init (armed) (at-door))
  (:goal (and (door-open) (alarm-on))))
