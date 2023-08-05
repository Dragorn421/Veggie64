# Veggie64

A wip Blender (https://www.blender.org/) addon model exporter for the opengl implementation of libdragon (https://github.com/DragonMinded/libdragon) for n64 homebrew

In practice it's probably a generic opengl 1.1 exporter but it's meant to be geared for the n64

# dev

```
$ python3 -m venv .venv
$ . .venv/bin/activate
(.venv) $ pip install fake-bpy-module
$ ln -s -t /home/dragorn421/.config/blender/3.6/scripts/addons/ $(realpath veggie64.py)
```