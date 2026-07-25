%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential horizon encoding, multi-shot (iclingo-style).
%
% Parts:
%   base     -- the task facts (added by the planner) and the initial state.
%   step(t)  -- exactly one action at timestep t, its effects, and inertia.
%   check(t) -- the goal test at timestep t, gated on the external query(t)
%               so a deepening search can retire the test of horizon t and
%               move on to t+1 without regrounding earlier steps.
%
% The planner grounds base + check(0), then incrementally step(t) + check(t)
% for t = 1, 2, ... releasing query(t-1) each time. A fixed-horizon solve
% grounds base, step(1..h), check(h) in one go and sets query(h).
%
% Timesteps double as SMTPlan-style *happenings* when the task has durative
% actions; see the temporal layer at the bottom of this file.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Predicates that may legitimately be absent from a task's facts.
% occurs/2 is absent until the first step(t) part is grounded (horizon 0).
#defined occurs/2.
#defined action/1.
#defined goal/2.
#defined numGoal/3.
#defined precondition/3.
#defined postcondition/4.
#defined numVariable/1.
#defined numPrecondition/4.
#defined numEffect/3.
#defined numAssign/3.
#defined orGroup/2.
#defined orDisjunct/5.
#defined goalOrGroup/1.
#defined goalOrDisjunct/4.

#show occurs/2.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Establish initial state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program base.

holds(Variable, Value, 0) :- initialState(Variable, Value).

% Numeric fluents: numval projects a numeric variable's holds atom to its
% integer; the pseudo-variable `none` carries constant-only comparison sides.
numval(none, 0, 0).
numval(V, X, 0) :- holds(V, value(V, X), 0), numVariable(V).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform actions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program step(t).

1 {occurs(Action, t) : action(Action)} 1.

% Check preconditions
:- occurs(Action, t), precondition(Action, Variable, Value), not holds(Variable, Value, t - 1).

% Disjunctive preconditions. precondition/3 facts are read conjunctively, which
% is what lets a `forall` be emitted as one rule with its variable ranged by the
% grounder. A disjunction cannot be stated that way, so it gets its own
% vocabulary: an action carries one orGroup/2 per disjunctive condition, each
% holding orDisjunct/5 literals grouped by disjunct. At least one disjunct has
% to hold, and a disjunct holds when every literal in it does. An `exists` is
% the same thing with the disjuncts indexed by the quantified variable's
% binding, so the grounder enumerates them.
orDisjunctFails(A, G, D, t) :- orDisjunct(A, G, D, Variable, Value), not holds(Variable, Value, t - 1).
orGroupHolds(A, G, t) :- orDisjunct(A, G, D, _, _), not orDisjunctFails(A, G, D, t).
:- occurs(A, t), orGroup(A, G), not orGroupHolds(A, G, t).

% Numeric preconditions: linear comparisons of value(V)+C sides at t-1.
:- occurs(A, t), numPrecondition(A, lt, expr(V1, C1), expr(V2, C2)), numval(V1, X1, t - 1), numval(V2, X2, t - 1), X1 + C1 >= X2 + C2.
:- occurs(A, t), numPrecondition(A, le, expr(V1, C1), expr(V2, C2)), numval(V1, X1, t - 1), numval(V2, X2, t - 1), X1 + C1 > X2 + C2.
:- occurs(A, t), numPrecondition(A, eq, expr(V1, C1), expr(V2, C2)), numval(V1, X1, t - 1), numval(V2, X2, t - 1), X1 + C1 != X2 + C2.

% Apply effects
caused(Variable, Value, t) :- occurs(Action, t), postcondition(Action, Effect, Variable, Value), holds(VariablePre, ValuePre, t - 1) : precondition(Effect, VariablePre, ValuePre).

% Numeric effects: constant-delta updates and constant assignments.
caused(V, value(V, X + D), t) :- occurs(A, t), numEffect(A, V, D), holds(V, value(V, X), t - 1).
caused(V, value(V, K), t) :- occurs(A, t), numAssign(A, V, K).

% Inertia
modified(Variable, t) :- caused(Variable, Value, t).

holds(Variable, Value, t) :- caused(Variable, Value, t).
holds(variable(V), Value, t) :- holds(variable(V), Value, t - 1), not modified(variable(V), t).

numval(none, 0, t).
numval(V, X, t) :- holds(V, value(V, X), t), numVariable(V).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Verify that goal is met
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program check(t).

#external query(t).

:- query(t), goal(Variable, Value), not holds(Variable, Value, t).

% Disjunctive goals, read exactly as disjunctive preconditions are but against
% the state at the goal step rather than the one before the action.
goalDisjunctFails(G, D, t) :- goalOrDisjunct(G, D, Variable, Value), not holds(Variable, Value, t).
goalOrGroupHolds(G, t) :- goalOrDisjunct(G, D, _, _), not goalDisjunctFails(G, D, t).
:- query(t), goalOrGroup(G), not goalOrGroupHolds(G, t).

% Numeric goals: linear comparisons of value(V)+C sides at the goal step,
% evaluated against numval(_, _, t) exactly as numeric preconditions are at t-1.
:- query(t), numGoal(lt, expr(V1, C1), expr(V2, C2)), numval(V1, X1, t), numval(V2, X2, t), X1 + C1 >= X2 + C2.
:- query(t), numGoal(le, expr(V1, C1), expr(V2, C2)), numval(V1, X1, t), numval(V2, X2, t), X1 + C1 > X2 + C2.
:- query(t), numGoal(eq, expr(V1, C1), expr(V2, C2)), numval(V1, X1, t), numval(V2, X2, t), X1 + C1 != X2 + C2.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Temporal layer -- SMTPlan's happening encoding on top of the same horizon
%
% (Constraint names in the comments below are the ones used in "A Compilation
% of the Full PDDL+ Language into SMT" / KCL-Planning/SMTPlan's
% EncoderHappening.)
%
% A timestep is a *happening*: happening t sits at absolute time
% time(t) = time(t-1) + delta(t), with time(0) = 0 and delta(t) >= epsilon
% (P3--P4). SMTPlan's happening times are reals and its epsilon is 1/10; clingo
% terms are integers, so the encoder scales every duration in the task by the
% least common denominator and epsilon becomes one scaled unit (minSeparation).
% The discretisation is therefore exact for rational durations, not an
% approximation. Only delta/2 is in the model -- the absolute time of happening
% t is the prefix sum delta(1) + ... + delta(t).
%
% A durative action DA is split into two ordinary actions, its start and end
% *snap* actions, tied back to DA by snap(DA, start|end, A) (H14--H15). At-start
% and at-end conditions/effects are the snaps' own precondition/postcondition
% facts, so every rule above applies to them verbatim; this section adds only
% what couples the two snaps:
%
%   running(DA, t)      SMTPlan's run(a, h): DA is open across the interval that
%                       opens at happening t. True at exactly the happenings
%                       s..e-1 of a start at s and its matching end at e, so
%                       holds(_, s) .. holds(_, e-1) are precisely the states the
%                       open interval (t_s, t_e) spans -- the ones PDDL 2.1 makes
%                       the over-all conditions responsible for.
%   remaining(DA, R, t) SMTPlan's dur(a, h): duration left to run at happening t.
%                       Set when DA starts, decremented by delta at every
%                       happening, and required to be exactly 0 at the end snap
%                       (H9--H10).
%
% Unlike SMTPlan a happening carries exactly one snap action (the 1{occurs}1
% choice above), which makes SMTPlan's mutex constraints between snap actions at
% the same happening vacuous, so they are not encoded. The price is that snap
% actions that must be *simultaneous* cannot be expressed -- they have to be
% sequentialised epsilon apart. Continuous change (PDDL+ processes and events)
% is out of scope: this is the PDDL 2.1 durative-action fragment of SMTPlan.
%
% Cost: the remaining-duration recursion grounds one rule per (remaining value,
% gap) pair per durative action per step, so it is quadratic in the largest
% scaled duration. That is what the encoder's `time_scale` trades against
% resolution -- normalising the durations by their gcd first is what keeps the
% integers small enough for the default of 10 to be affordable.
%
% Every rule here is inert on a task whose facts declare no durativeAction.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#defined durativeAction/1.
#defined snap/3.
% Derived here, but a classical task grounds check(0) before any step(t) part
% exists to define them; declaring them keeps gringo from reporting that.
#defined running/2.
#defined delta/2.
#defined durationValue/2.
#defined overall/3.
#defined numOverall/4.
#defined minSeparation/1.

#show delta/2.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The happening timeline
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program base.

temporal :- durativeAction(_).

% Minimum separation between two happenings (SMTPlan's epsilon) in the encoder's
% scaled time unit; one unit unless the task facts override it.
hasMinSeparation :- minSeparation(_).
separation(E) :- minSeparation(E).
separation(1) :- not hasMinSeparation.

% Candidate gaps. No gap ever has to exceed the longest duration in the task: a
% longer one would step past the end of every action open across it, which the
% remaining-duration constraints below reject anyway.
delay(E) :- separation(E).
delay(D) :- durationValue(_, B), separation(E), D = E..B.

#program step(t).

% t_t = t_{t-1} + delta(t), delta(t) >= epsilon (P3--P4).
1 { delta(t, D) : delay(D) } 1 :- temporal.

% Nothing is open across this gap, so no duration equation spans it and the
% happening may be pulled forward to the earliest legal time. Pruning only.
open(t - 1) :- running(DA, t - 1).
:- delta(t, D), separation(E), D != E, not open(t - 1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Snap actions and open intervals (H14--H15)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

starts(DA, t) :- occurs(A, t), snap(DA, start, A).
ends(DA, t)   :- occurs(A, t), snap(DA, end, A).

% run(a, h): open from the start snap until the matching end snap fires.
running(DA, t) :- starts(DA, t).
running(DA, t) :- running(DA, t - 1), not ends(DA, t).

% sta -> not run(h-1): a durative action may not overlap itself.
:- starts(DA, t), running(DA, t - 1).
% end -> run(h-1): an end snap has to close an interval that is open.
:- ends(DA, t), not running(DA, t - 1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Remaining duration (H9--H10)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% sta -> dur = D. The choice is SMTPlan's duration *constraint*: a fixed
% duration leaves one candidate, `lo <= ?duration <= hi` leaves lo..hi.
1 { remaining(DA, D, t) : durationValue(DA, D) } 1 :- starts(DA, t).

% dur(h) = dur(h-1) - delta(h). A value of 0 does not propagate: it is consumed
% by the end snap at this very happening, and a restart re-draws the choice.
remaining(DA, R - D, t) :- remaining(DA, R, t - 1), delta(t, D), R >= D.
hasRemaining(DA, t) :- remaining(DA, _, t).

% end -> dur = 0: the end snap fires exactly when the duration has elapsed.
:- ends(DA, t), not remaining(DA, 0, t).
% run <-> dur > 0: an interval that is still open has time left to run, and a
% gap may not step over an open action's remaining duration.
:- running(DA, t), remaining(DA, 0, t).
:- running(DA, t), not hasRemaining(DA, t).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Over-all conditions: run(a, h) -> condition holds at h
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- running(DA, t), overall(DA, Variable, Value), not holds(Variable, Value, t).

% The numeric analogue, read off numval at t exactly as numPrecondition is at t-1.
:- running(DA, t), numOverall(DA, lt, expr(V1, C1), expr(V2, C2)), numval(V1, X1, t), numval(V2, X2, t), X1 + C1 >= X2 + C2.
:- running(DA, t), numOverall(DA, le, expr(V1, C1), expr(V2, C2)), numval(V1, X1, t), numval(V2, X2, t), X1 + C1 > X2 + C2.
:- running(DA, t), numOverall(DA, eq, expr(V1, C1), expr(V2, C2)), numval(V1, X1, t), numval(V2, X2, t), X1 + C1 != X2 + C2.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Goal: no durative action left open (SMTPlan puts !run(a, H-1) in its goal)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program check(t).

:- query(t), running(DA, t).
