import math
import importlib

def generate_hexagonal_grid(boundingbox, spacing):
    """Generates an hexagonal grid inside a given bounding-box with a given spacing between the vertices"""
    hexheight = spacing
    hexwidth = math.sqrt(3) * spacing / 2
    vertspacing = 0.75 * hexheight
    horizspacing = hexwidth
    sizex = int((boundingbox[1] - boundingbox[0]) / horizspacing) + 2
    sizey = int((boundingbox[3] - boundingbox[2]) / vertspacing) + 2
    if sizey % 2 == 0:
        sizey += 1
    pointsret = []
    for i in range(-2, sizex):
        for j in range(-2, sizey):
            xpos = i * spacing
            ypos = j * spacing
            if j % 2 == 1:
                xpos += spacing * 0.5
            if (j % 2 == 1) and (i == sizex - 1):
                continue
            pointsret.append([int(xpos + boundingbox[0]), int(ypos + boundingbox[2])])
    return pointsret

def load_plugin(class_full_name):
    package, class_name = class_full_name.rsplit('.', 1)
    plugin_module = importlib.import_module(package)
    plugin_class = getattr(plugin_module, class_name)
    return plugin_class
