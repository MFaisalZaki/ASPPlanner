# register the planner.
from aspplanner.up_asp_planner import UPASPPlanner
import unified_planning as up


# Register the planner to the UP framework
# This is done once the package is imported so its transparent to the user.
env = up.environment.get_environment()
env.factory.add_engine('ASPPlanner', 'aspplanner.up_asp_planner', 'UPASPPlanner')