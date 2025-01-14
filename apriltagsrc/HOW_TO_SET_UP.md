Make desired changes to library in apriltagsrc/core, then run:

    cd /something/something/apriltagsrc
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j4

To install, do the following:

1) Install default `apriltag` module. Run `pip uninstall apriltag` then `pip install apriltag`. Why? We will use the regular apriltag library as a base to insert our own code.
2) Locate the location of the module that we will modify: run `pip show apriltag` and look for the location, for example `Location: /home/robotics4169/.local/lib/python3.6/site-packages`, then navigate to the given directory (in this case, `/home/robotics4169/.local/lib/python3.6/site-packages`)
3) Look for the file that we will replace. The file is called `libapriltag.so`, and should be in the path `/home/robotics4169/.local/lib/python3.6/site-packages/libapriltag.so`
4) Finally, delete the `libapriltag.so` file and replace it with the `libapriltag.so` file found in `vision/apriltagsrc/build/lib/libapriltag.so`
5) Done! The apriltag library should now work with the modified apriltag family. 
6) Bonus: Copy the 3 apriltag module's files into `vision/apriltagsrc/compiledlib` to make instillation easier by simply copy and pasting these 3 files. The 3 files are: `apriltag-0.0.16.dist-info` (a folder actually), `apriltag.py`, `and libapriltag.so`, all of which are found in `/home/robotics4169/.local/lib/python3.6/site-packages`
