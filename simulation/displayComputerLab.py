import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

# Get Apritag locations
file_path = "/home/jetson/vision/apriltags/maps/computerLab.fmap"

with open(file_path, 'r') as file:
    data = json.load(file)

coordinates = []
for apriltag in data["fiducials"]:
    coordinates.append((apriltag["id"], apriltag["transform"][3], apriltag["transform"][7]))

x = [coord[1] for coord in coordinates]
y = [coord[2] for coord in coordinates]
ids = [coord[0] for coord in coordinates]

# Plot Apriltag locations
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(x, y, color='b')

for i, txt in enumerate(ids): 
    if (txt in [0, 1]):
        ax.annotate(txt, (x[i] - 0.6, y[i] - 0.15), textcoords="offset points", xytext=(0, 0), ha='center')
    else:
        ax.annotate(txt, (x[i] + 0.6, y[i] - 0.15), textcoords="offset points", xytext=(0, 0), ha='center')
    
# Draw Game Field Boundary
rect = patches.Rectangle((0, -2), 7.04215, 4, linewidth=1, edgecolor='b', facecolor='none')
ax.add_patch(rect)

# Plot Robot location
center = (3.5, 0.5)
rotation = 0

width = 0.8255
height = 0.6604

rect = patches.Rectangle((-width / 2, -height / 2), width, height, edgecolor='black', facecolor='white')

transform = Affine2D().rotate_deg(rotation).translate(center[0], center[1])
rect.set_transform(transform + ax.transData)

ax.add_patch(rect)

arrow_length = 0.125
arrow_width = 0.025

arrow = patches.FancyArrow(0, 0, arrow_length, 0, width=arrow_width, edgecolor='black', facecolor='black', head_width=0.3, head_length=0.2)
arrow.set_transform(transform + ax.transData)

ax.add_patch(arrow)

# Adjusting plot limits
plt.xlim(-0.5, 7.54215)
plt.ylim(-2.5, 2.5)

# Display plot
plt.gca().set_aspect('equal', adjustable='box')
plt.title('2024 Game Field Positioning Simulation')
plt.grid(False)
plt.show()
