import ntcore
import time

inst = ntcore.NetworkTableInstance.getDefault()
table=inst.getTable("SmartDashboard")
xSub=table.getDoubleTopic("x").publish()
ySub=table.getDoubleTopic("y").publish()
distSub = table.getDoubleTopic("y").publish()
inst.startClient4("example client")
inst.setServerTeam(4169)
inst.startDSClient()

while True:
    time.sleep(0.1)
    xSub.set(257)
    ySub.set(375)
    distSub.set(1004.3427)
