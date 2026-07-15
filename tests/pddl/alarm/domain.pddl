;; Conditional effects and zero-parameter actions: opening the door while
;; the system is armed also triggers the alarm. Zero-parameter actions
;; exercise the action(("name")) singleton-term extraction path.
(define (domain alarm)
  (:requirements :strips :conditional-effects)
  (:predicates (armed) (door-open) (alarm-on) (at-door))

  (:action open-door
    :parameters ()
    :precondition (at-door)
    :effect (and (door-open)
                 (when (armed) (alarm-on))))

  (:action disarm
    :parameters ()
    :precondition (armed)
    :effect (not (armed))))
