from lark import Lark, Tree, Transformer
import os
from typing import List, Dict, Any, Union

from unified_planning.shortcuts import PlanValidator
def validate(task, plan):
    validation_fail_reason = ''
    if plan is None or task is None:
        return False, "No plan or task provided."
    
    with PlanValidator(name='sequential_plan_validator') as validator:
        validationresult = validator.validate(task, plan)
    validation_fail_reason = validationresult.reason
    isvalid = validationresult.status.value == 1 if validationresult else False
    return isvalid, validation_fail_reason

class AspPlanTransformer(Transformer):
    """Transforms parsed ASP facts into structured dictionaries."""
    
    def start(self, facts):
        return list(facts)
    
    def fact(self, items):
        return items[0] if items else None
    
    def predicate(self, items):
        name = str(items[0])
        args = items[1] if len(items) > 1 else []
        return {"predicate": name, "args": args}
    
    def args(self, items):
        return list(items)
    
    def arg(self, items):
        return items[0]
    
    def tuple(self, items):
        return {"type": "tuple", "values": items[0] if items else []}
    
    def constant(self, items):
        value = items[0]
        if hasattr(value, 'value'):
            value = value.value.strip('"')
        return {"type": "constant", "value": str(value)}
    
    def atom(self, items):
        return {"type": "atom", "value": str(items[0])}
    
    def number(self, items):
        return {"type": "number", "value": int(items[0])}
    
    def string(self, items):
        return {"type": "string", "value": str(items[0]).strip('"')}

class AspPlanParser:
    """Parser for ASP plan facts that returns structured dictionaries."""
    
    def __init__(self, grammar_path=None):
        if grammar_path is None:
            grammar_path = os.path.join(os.path.dirname(__file__), 'grammars', 'asp_plan_grammar.lark')
        
        with open(grammar_path, 'r') as f:
            grammar = f.read()
        
        self.parser = Lark(grammar, start='start', transformer=AspPlanTransformer(), parser='lalr')
    
    def parse_plan_fact(self, asp_fact: str) -> Dict[str, Any]:
        """
        Parse a single ASP plan fact and return structured data.
        
        Args:
            asp_fact: ASP fact string like 'occurs(action(("navigate", ...)), 1)'
            
        Returns:
            Dictionary with action, arguments, and timestep information
        """
        try:
            parsed = self.parser.parse(asp_fact)
            
            if not parsed or len(parsed) == 0:
                return None
            
            fact = parsed[0]
            
            # Handle occurs(action(...), timestep) pattern
            if fact["predicate"] == "occurs" and len(fact["args"]) == 2:
                action_data = fact["args"][0]
                timestep_data = fact["args"][1]
                
                if (action_data["predicate"] == "action" and timestep_data["type"] == "number"):
                    
                    # Extract action tuple
                    if (len(action_data["args"]) > 0 and action_data["args"][0]["type"] == "tuple"):
                        
                        tuple_values = action_data["args"][0]["values"]
                        
                        if tuple_values:
                            action_name = tuple_values[0]["value"]
                            action_args = []
                            
                            for arg in tuple_values[1:]:
                                if arg["type"] == "constant":
                                    action_args.append(arg["value"])
                                elif arg["type"] == "string":
                                    action_args.append(arg["value"])
                                elif arg["type"] == "atom":
                                    action_args.append(arg["value"])
                                else:
                                    action_args.append(str(arg["value"]))
                            
                            return {
                                "action": action_name,
                                "arguments": action_args,
                                "timestep": timestep_data["value"]
                            }
                    else:
                        # this is a grounded action
                        return {
                            "action": action_data["args"][0]["value"],
                            "arguments": [],
                            "timestep": timestep_data["value"]
                        }
            
            # For other predicate types, return general structure
            return {
                "predicate": fact["predicate"],
                "args": fact["args"],
                "raw_fact": fact
            }
            
        except Exception as e:
            raise ValueError(f"Failed to parse ASP fact '{asp_fact}': {e}")
    
    def parse_multiple_facts(self, asp_facts: str) -> List[Dict[str, Any]]:
        """
        Parse multiple ASP facts separated by periods.
        
        Args:
            asp_facts: String containing multiple ASP facts
            
        Returns:
            List of dictionaries with parsed fact data
        """
        # Split on periods and clean up
        facts = [fact.strip() for fact in asp_facts.split('.') if fact.strip()]
        
        results = []
        for fact in facts:
            if not fact.endswith('.'):
                fact += '.'  # Ensure period for parsing
            
            try:
                parsed = self.parse_plan_fact(fact)
                if parsed:
                    results.append(parsed)
            except Exception as e:
                print(f"Warning: Failed to parse fact '{fact}': {e}")
                continue
        
        return results