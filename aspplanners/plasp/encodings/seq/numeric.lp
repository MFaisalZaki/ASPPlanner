%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential horizon encoding -- NUMERIC layer.
%
%   owns:     numVariable/1 numConst/2 numTerm/3 numPrecondition/4 numEffect/3
%             numAssign/3 numGoal/3
%   requires: core (holds/3, occurs/2, the caused/modified inertia)
%
% Numeric fluents ride on core's holds/3 chain -- a numeric variable's value is
% carried as `holds(V, value(V, X), t)`, so core's inertia rule propagates it and
% this layer adds no frame axiom of its own. numval/3 projects that atom to its
% integer and numValue/3 evaluates linear expressions over it.
%
% numValue/3 is the interface other layers read: the disjunctive layer's numeric
% literals and the temporal layer's over-all comparisons both go through it, each
% guarded by its own `#defined numValue/3` so they stay inert without this layer.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Every layer file opens by pinning the current program part, so that loading the
% layers as separate files and concatenating them into one program (what
% PLASPPlanner.lp_program does) are the same program. Without it the statements
% above a file's first `#program` would inherit the part the previous file
% happened to end in.
#program base.

#defined numGoal/3.
#defined numVariable/1.
#defined numConst/2.
#defined numTerm/3.
#defined numPrecondition/4.
#defined numEffect/3.
#defined numAssign/3.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program base.

% numval projects a numeric variable's holds atom to its integer.
numval(V, X, 0) :- holds(V, value(V, X), 0), numVariable(V).

% A side of a numeric comparison is the linear expression `K1*V1 + ... + C`.
% numConst/2 states its constant and declares it (so a constant-only side, whose
% sum is empty, still gets a value); one numTerm/3 states each K*V. Keeping the
% terms in their own facts is what lets a side hold any number of fluents
% without the comparison's arity depending on how many -- `(+ (x ?b) (y ?b))`
% and `(* 2 (x ?f))` are ordinary expressions here, not special cases.
numValue(E, S + C, 0) :- numConst(E, C), S = #sum { X * K, V : numTerm(E, K, V), numval(V, X, 0) }.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preconditions, effects, and the value of every expression at t
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program step(t).

% Numeric preconditions: linear comparisons of two numValue sides at t-1.
:- occurs(A, t), numPrecondition(A, lt, E1, E2), numValue(E1, S1, t - 1), numValue(E2, S2, t - 1), S1 >= S2.
:- occurs(A, t), numPrecondition(A, le, E1, E2), numValue(E1, S1, t - 1), numValue(E2, S2, t - 1), S1 > S2.
:- occurs(A, t), numPrecondition(A, eq, E1, E2), numValue(E1, S1, t - 1), numValue(E2, S2, t - 1), S1 != S2.
:- occurs(A, t), numPrecondition(A, neq, E1, E2), numValue(E1, S1, t - 1), numValue(E2, S2, t - 1), S1 = S2.

% Numeric effects: constant-delta updates and constant assignments. They feed
% core's caused/3, so core's inertia and modified/2 handle them like any effect.
caused(V, value(V, X + D), t) :- occurs(A, t), numEffect(A, V, D), holds(V, value(V, X), t - 1).
caused(V, value(V, K), t) :- occurs(A, t), numAssign(A, V, K).

numval(V, X, t) :- holds(V, value(V, X), t), numVariable(V).
numValue(E, S + C, t) :- numConst(E, C), S = #sum { X * K, V : numTerm(E, K, V), numval(V, X, t) }.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Numeric goals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program check(t).

% Linear comparisons at the goal step, evaluated against numValue(_, _, t)
% exactly as numeric preconditions are at t-1.
:- query(t), numGoal(lt, E1, E2), numValue(E1, S1, t), numValue(E2, S2, t), S1 >= S2.
:- query(t), numGoal(le, E1, E2), numValue(E1, S1, t), numValue(E2, S2, t), S1 > S2.
:- query(t), numGoal(eq, E1, E2), numValue(E1, S1, t), numValue(E2, S2, t), S1 != S2.
:- query(t), numGoal(neq, E1, E2), numValue(E1, S1, t), numValue(E2, S2, t), S1 = S2.
