(define (problem gripper-2)
  (:domain gripper)
  (:objects room-a room-b - room
            ball-1 ball-2 - ball
            left-gripper right-gripper - gripper)
  (:init (at-robby room-a)
         (at ball-1 room-a) (at ball-2 room-a)
         (free left-gripper) (free right-gripper))
  (:goal (and (at ball-1 room-b) (at ball-2 room-b))))
