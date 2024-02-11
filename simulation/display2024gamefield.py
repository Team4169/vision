import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from PIL import Image

# Get Apritag locations
file_path = "/home/team4169/vision/apriltags/maps/2024gamefield.fmap"

with open(file_path, 'r') as file:
    data = json.load(file)

coordinates = []
for apriltag in data["fiducials"]:
    coordinates.append((apriltag["id"], apriltag["transform"][3], apriltag["transform"][7]))

x = [coord[1] for coord in coordinates]
y = [coord[2] for coord in coordinates]
ids = [coord[0] for coord in coordinates]

# Plot Apriltag locations
fig, ax = plt.subplots()
ax.scatter(x, y, color='b')

for i, txt in enumerate(ids):
    if (txt in [3, 4, 13]): # left
        ax.annotate(txt, (x[i] - 0.6, y[i] - 0.15), textcoords="offset points", xytext=(0, 0), ha='center')
    elif (txt in [7, 8, 14]): # right
        ax.annotate(txt, (x[i] + 0.6, y[i] - 0.15), textcoords="offset points", xytext=(0, 0), ha='center')    
    elif (txt in [5, 6, 11, 16]): # bottom
        ax.annotate(txt, (x[i], y[i] - 0.75), textcoords="offset points", xytext=(0, 0), ha='center')
    else:
        ax.annotate(txt, (x[i], y[i] + 0.5), textcoords="offset points", xytext=(0, 0), ha='center')
    
# Draw Game Field Boundary
rect = patches.Rectangle((-8.2423, -4.0513), 16.4846, 8.1026, linewidth=1, edgecolor='b', facecolor='none')
ax.add_patch(rect)

# Plot Robot location
center = (0.5, 0.5)
rotation = 135

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

# Add Field Map Image
field_map_path = "/home/team4169/vision/simulation/fieldmap.png"
field_map = Image.open(field_map_path).convert("RGBA")

opacity = int(225 * 0.5)
field_map = field_map.copy()
field_map.putalpha(opacity)
field_map = field_map.rotate(180)

ax.imshow(field_map, extent=[-8.2423, 8.2423, -4.0513, 4.0513])

# Display plot
plt.gca().set_aspect('equal', adjustable='box')
plt.title('2024 Game Field Positioning Simulation')
plt.grid(False)
plt.show()
