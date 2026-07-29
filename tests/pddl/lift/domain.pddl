;; The miconic shape, reduced: one action carries two `forall`/`when` effects
;; on the *same* fluent -- `stop` clears `boarded` for the passengers who have
;; arrived and sets it for the ones getting on.
;;
;; Neither may be dropped for the other. Dropping the clear (because some other
;; effect adds the same fluent) leaves every passenger aboard forever, and the
;; lift cannot move again: `up`/`down` require nobody aboard who is going the
;; other way. What remains -- both firing at once for one passenger -- is
;; settled in the encoding by PDDL's rule that the add wins.
(define (domain lift)
  (:requirements :typing :adl)
  (:types passenger floor)
  (:predicates (lift-at ?f - floor) (above ?f1 ?f2 - floor)
               (origin ?p - passenger ?f - floor)
               (destin ?p - passenger ?f - floor)
               (going-down ?p - passenger)
               (boarded ?p - passenger) (served ?p - passenger))

  (:action stop
    :parameters (?f - floor)
    :precondition (lift-at ?f)
    :effect (and
       (forall (?p - passenger)
          (when (and (boarded ?p) (destin ?p ?f))
                (and (not (boarded ?p)) (served ?p))))
       (forall (?p - passenger)
          (when (and (origin ?p ?f) (not (served ?p)))
                (boarded ?p)))))

  (:action up
    :parameters (?f1 ?f2 - floor)
    :precondition (and (lift-at ?f1) (above ?f1 ?f2)
                       (forall (?p - passenger)
                          (imply (going-down ?p) (not (boarded ?p)))))
    :effect (and (lift-at ?f2) (not (lift-at ?f1))))

  (:action down
    :parameters (?f1 ?f2 - floor)
    :precondition (and (lift-at ?f1) (above ?f2 ?f1))
    :effect (and (lift-at ?f2) (not (lift-at ?f1)))))
