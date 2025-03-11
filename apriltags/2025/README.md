2025 apriltag code
==========

The three files `tags.py`, `displayTags.py`, and `plotAndDisplayTags.py`, contain all the code for finding tags.

All three files have the same functionality of using the cameras to do math and determine the robots position and rotation.

The file `displayRags.py` also incorporates code to display what the camera sees visually.

The fule `plotAndDisplayTags.py` also includes code to plot a simulated map of the 2025 field, and plots the robots estimated position and rotation.

All three files also make use of a `.env` file to control certain features. There are 3 enviornment variables options:

1. `ENABLE_NETWORK_TABLES` - A boolean (0=off or 1=on) for if the code should attempt to initialize Network Tables and send data.
2. `NEW_OS` - A boolean (0=off or 1=on) for if the code is meant to use the new OS stuff. This changes both the path tocertain files and the way the cameras are found.
3. `JETSON_ID` - A string (1 or 2) for the ID of the Jetson, this controls which cameras it will make use of, and the naming of the Network Tables variables.