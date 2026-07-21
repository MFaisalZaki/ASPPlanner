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

% Numeric goals: linear comparisons of value(V)+C sides at the goal step,
% evaluated against numval(_, _, t) exactly as numeric preconditions are at t-1.
:- query(t), numGoal(lt, expr(V1, C1), expr(V2, C2)), numval(V1, X1, t), numval(V2, X2, t), X1 + C1 >= X2 + C2.
:- query(t), numGoal(le, expr(V1, C1), expr(V2, C2)), numval(V1, X1, t), numval(V2, X2, t), X1 + C1 > X2 + C2.
:- query(t), numGoal(eq, expr(V1, C1), expr(V2, C2)), numval(V1, X1, t), numval(V2, X2, t), X1 + C1 != X2 + C2.
