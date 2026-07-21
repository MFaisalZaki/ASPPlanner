# register the planner.
from aspplanner.up_asp_planner import UPASPPlanner
import unified_planning as up


# Register the planners with the UP framework on import, so it's transparent to
# the user. add_engine takes the module/class as strings and imports them
# lazily on first use -- the ABA backend is registered this way (and NOT
# eagerly imported) so its optional `aspforaba` dependency is only required
# when the ABAPlanner engine is actually instantiated.
env = up.environment.get_environment()
env.factory.add_engine('ASPPlanner', 'aspplanner.up_asp_planner', 'UPASPPlanner')
env.factory.add_engine('ABAPlanner', 'aspplanner.abaplan.up_engine', 'UPABAPlanner')