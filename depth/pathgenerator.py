from pathplannerlib.path import PathPlannerPath, PathConstraints, GoalEndState
from wpimath.geometry import Pose2d, Rotation2d
import math

# Create a list of bezier points from poses. Each pose represents one waypoint.
# The rotation component of the pose should be the direction of travel. Do not use holonomic rotation.
bezierPoints = PathPlannerPath.bezierFromPoses(
    Pose2d(1.0, 1.0, Rotation2d.fromDegrees(0)),
    Pose2d(3.0, 1.0, Rotation2d.fromDegrees(0)),
    Pose2d(5.0, 3.0, Rotation2d.fromDegrees(90))
)

# Create the path using the bezier points created above
path = new PathPlannerPath(
    bezierPoints,
    PathConstraints(3.0, 3.0, 2 * math.pi, 4 * math.pi), # The constraints for this path. If using a differential drivetrain, the angular constraints have no effect.
    GoalEndState(0.0, Rotation2d.fromDegrees(-90)) # Goal end state. You can set a holonomic rotation here. If using a differential drivetrain, the rotation will have no effect.
)

# Prevent the path from being flipped if the coordinates are already correct
path.preventFlipping = True;