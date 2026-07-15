(define (problem alarm-quiet)
  (:domain alarm)
  (:init (armed) (at-door))
  (:goal (and (door-open) (not (alarm-on)))))
