

import os
import clingo

from unified_planning.plans import SequentialPlan, ActionInstance

from aspplanner.compilers.asp_seq_encoder import ASPSeqEncoder
from aspplanner.compilers.asp_facts import ASPOccursFluent, ASPConstraint, ASPRule, ASPCmd, ASPFact

from aspplanner.utilities import AspPlanParser, validate

encoder_map = {
    'seq': ASPSeqEncoder,
}

encoder_file_map = {
    'seq': os.path.join(os.path.dirname(__file__), 'encodings', 'sequential-horizon.lp'),
}


class ASPPlanner:
    def __init__(self, problem, encoder_type):
        self.compiled_task = encoder_map[encoder_type]().compile(problem)
        self.task          = self.compiled_task.problem
        self.plan_parser   = AspPlanParser()
        self.base_formula  = self.__load_asp_encoding_formula__(encoder_type)
        self.logs          = []
    
    def __load_asp_encoding_formula__(self, encodingname):
        assert encodingname in encoder_file_map.keys(), f"Unsupported encoding name: {encodingname}"
        model = set(open(encoder_file_map[encodingname], 'r').readlines())
        model = set(filter(lambda l: l != '' and not '%' in l, map(str.strip, model)))
        return model
    
    def __construct_action__(self, action_tuple):
        # first step get the action from the task.
        up_action = next(filter(lambda a: a.name == action_tuple[0], self.task.actions), None)
        assert up_action is not None, f"Action {action_tuple[0]} not found in the task."
        up_args   = []
        for arg in action_tuple[1:]:
            up_args.append(next(filter(lambda p: p.name == arg, self.task.all_objects)))
        return ActionInstance(up_action, up_args)
    
    # This will be multiple plans.
    def __extract_plan__(self, answer):
        self.actions = sorted(map(lambda a: ASPOccursFluent(a), filter(lambda a: 'occurs' in str(a), answer)), key=lambda a:a.timestep)
        _plan = SequentialPlan(list(map(self.__construct_action__, map(lambda e: (e['action'], *[eval(a)['value'] for a in e['arguments']]), map(lambda a: self.plan_parser.parse_plan_fact(a.fact_str), self.actions)))))
        _lifted_plan = _plan.replace_action_instances(self.compiled_task.map_back_action_instance)
        return _lifted_plan
    
    def plan(self):
        _plan = SequentialPlan([])
        for n in range(0, 1000):
            if len(_plan.actions) > 0: break
            ctl = clingo.Control(arguments=['-n', '1', '-c', f'horizon={n}'])
            # debug lp program.
            ctl.add("base", [], '\n'.join(set.union(self.base_formula, set.union(*list(self.task.asp_encoding_str.values())))))
            ctl.ground([("base", [])])
            with ctl.solve(yield_=True) as solution_iterator:
                if ctl.is_conflicting: continue
                for solution in solution_iterator:
                    _plan = self.__extract_plan__(set(solution.symbols(shown=True)))
                    if len(_plan.actions) > 0: break
                    
        validation_result, reason = validate(self.task, _plan)
        
        if not validation_result:
            self.logs.append(f'Plan validation failed: {reason}')
            _plan = SequentialPlan([])
        
        return _plan