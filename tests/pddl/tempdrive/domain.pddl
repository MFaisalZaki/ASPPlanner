;; Durations that are neither constant nor all equal: `drive` reads its duration
;; off a static function, `recharge` is stated as a duration inequality, and
;; driving carries an over-all condition that `recharge` breaks -- so the two
;; may never overlap, however much shorter that would make the schedule.
(define (domain tempdrive)
  (:requirements :typing :durative-actions :numeric-fluents :duration-inequalities)
  (:types location)
  (:predicates
    (at ?l - location)
    (link ?a - location ?b - location)
    (charged)
    (topped_up))
  (:functions (travel-time ?a - location ?b - location))

  (:durative-action drive
    :parameters (?a - location ?b - location)
    :duration (= ?duration (travel-time ?a ?b))
    :condition (and (at start (at ?a))
                    (at start (link ?a ?b))
                    (over all (charged)))
    :effect (and (at start (not (at ?a)))
                 (at end (at ?b))))

  (:durative-action recharge
    :parameters ()
    :duration (and (>= ?duration 2) (<= ?duration 5))
    :condition (at start (charged))
    :effect (and (at start (not (charged)))
                 (at end (charged))
                 (at end (topped_up)))))
