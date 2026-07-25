;; The canonical required-concurrency temporal domain: a fuse can only be
;; mended while a match is burning, and no match burns long enough to be lit
;; before and still be alight after. mend_fuse therefore has to run strictly
;; inside light_match, which a planner that cannot overlap durative actions
;; -- or whose happenings are spaced too coarsely to fit two extra ones inside
;; a six-unit interval -- cannot solve.
(define (domain matchcellar)
  (:requirements :typing :durative-actions)
  (:types match fuse)
  (:predicates
    (handfree)
    (light)
    (unused ?m - match)
    (mended ?f - fuse))

  (:durative-action light_match
    :parameters (?m - match)
    :duration (= ?duration 6)
    :condition (at start (unused ?m))
    :effect (and (at start (not (unused ?m)))
                 (at start (light))
                 (at end (not (light)))))

  (:durative-action mend_fuse
    :parameters (?f - fuse)
    :duration (= ?duration 5)
    :condition (and (at start (handfree))
                    (over all (light)))
    :effect (and (at start (not (handfree)))
                 (at end (handfree))
                 (at end (mended ?f)))))
