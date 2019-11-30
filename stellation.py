#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
'''
Stellation extension for Inkscape.

'''

import inkex       # Required
import simplestyle # will be needed here for styles support
import simpletransform
import os          # here for alternative debug method only - so not usually required
import json
# many other useful ones in extensions folder. E.g. simplepath, cubicsuperpath, ...

from math import cos, sin, radians, sqrt, pi
phi = (1+sqrt(5))/2
EPSILON = .00001 # arbitrary small #

__version__ = '0.0.0'

inkex.localize()


### Your helper functions go here
def points_to_svgd(p, close=True):
    """ convert list of points (x,y) pairs
        into a closed SVG path list
    """
    f = p[0]
    p = p[1:]
    svgd = 'M%.4f,%.4f' % f
    for x in p:
        svgd += 'L%.4f,%.4f' % x
    if close:
        svgd += 'z'
    return svgd

# if a -> b -> c a counter-clockwise turn?
# +1 if counter-clockwise, -1 is clockwise, 0 if colinear
# XXX 2D ONLY XXX
def ccw(a, b, c):
    area2 = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
    if area2 < 0:
        return -1
    elif area2 > 0:
        return +1
    else:
        return 0

# Valid on 3d points
def ccw_from_origin(a, b, c, V = None):
    N = normal(a, b, c)
    w = N.dot(a if V is None else (a - V))
    if w < 0:
        return -1
    elif w > 0:
        return +1
    else:
        return 0

def normal(a, b, c):
    return (b - a).cross(c - a)

class Point:
    """Points are three dimensional."""
    x = 0
    y = 0
    z = 0
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __repr__(self):
        return "Point(%r,%r,%r)" % (self.x, self.y, self.z)
    def __str__(self):
        return "(%f,%f,%f)" % (self.x, self.y, self.z)
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)
    def __mul__(self, other):
        return Point(self.x * other, self.y * other, self.z * other)
    def __truediv__(self, other):
        return Point(self.x / other, self.y / other, self.z / other)
    def __neg__(self):
        return Point(-self.x, -self.y, -self.z)
    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self
    def __itruediv__(self, other):
        self.x /= other
        self.y /= other
        self.z /= other
        return self
    def dist(self, other=None):
        return sqrt(self.dist2(other))
    def dist2(self, other=None):
        pt = self if other is None else (self - other)
        return (pt.x*pt.x + pt.y*pt.y + pt.z*pt.z)
    def cross(self, other):
        return Point(
            self.y * other.z - self.z * other.y,
            -(self.x * other.z - self.z * other.x),
            self.x * other.y - self.y * other.x
        )
    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    def normalized(self):
        return self / self.dist()
    def transform(self, matrix):
        return matrix.transform(self)

class Vector(Point):
    pass

class Face:
    """A face is a planar collection of points"""
    points = None
    inside = None
    def __init__(self, *points, **kwargs):
        self.points = points
        self.inside = kwargs.get('inside')
        # check that points wind CW when viewed from origin (inside)
        l = len(points)
        for i in xrange(0, l):
            assert ccw_from_origin(points[i], points[(i+1)%l], points[(i+2)%l], V=self.inside) > 0
    def __repr__(self):
        return "Face(" + (",".join(repr(pt) for pt in self.points)) + ")"
    def __str__(self):
        return "Face(" + (",".join(str(pt) for pt in self.points)) + ")"
    def __mul__(self, other):
        return Face(*[pt*other for pt in self.points])
    def __truediv__(self, other):
        return Face(*[pt/other for pt in self.points])
    def centroid(self):
        sum = Point(0,0,0)
        count = 0
        for pt in self.points:
            sum += pt
            count += 1
        return sum / count
    def plane(self):
        return Plane(
            self.centroid(),
            normal(self.points[0], self.points[1], self.points[2])
        )
    def transform(self, matrix):
        return Face(*[p.transform(matrix) for p in self.points], **{
            'inside': None if self.inside is None else \
                self.inside.transform(matrix)
        })

class Line:
    """A line is represented as a point and a direction vector."""
    point = None
    direction = None
    def __init__(self, point, direction):
        self.point = point
        self.direction = direction.normalized()
    def __repr__(self):
        return "Line(%r,%r)" % (self.point, self.direction)
    def intersect(self, other):
        if isinstance(other, Plane):
            return other.intersectLine(self)
        assert False
    def transform(self, matrix):
        p1 = self.point.transform(matrix)
        p2 = (self.point + self.direction).transform(matrix)
        return Line(p1, p2 - p1)

class Plane:
    """A plane is represented as a point (usually the center point of a face)
    and a normal vector."""
    point = None
    normal = None
    def __init__(self, point, normal):
        # Passes through this point
        self.point = point
        # Unit normal vector; also the direction cosines of the angle n
        # makes with the xyz-axes
        self.normal = normal.normalized()
    def __repr__(self):
        return "Plane(%r,%r)" % (self.point, self.normal)
    def d(self):
        """Perpendicular distance from the origin to the plane."""
        return -(self.normal.dot(self.point))
    def transform(self, matrix):
        p1 = self.point.transform(matrix)
        p2 = (self.point + self.normal).transform(matrix)
        return Plane(p1, p2 - p1)
    def intersect(self, other):
        if isinstance(other, Plane):
            return self.intersectPlane(other)
        if isinstance(other, Line):
            return self.intersectLine(other)
        assert False
    def intersectLine(self, line):
        nu = self.normal.dot(line.direction)
        if abs(nu) < EPSILON:
            return None # parallel
        s = self.normal.dot(self.point - line.point) / nu
        return line.point + (line.direction * s)
    def intersectPlane(self, plane):
        n3 = self.normal.cross(plane.normal)
        if n3.dist() < EPSILON: # arbitrary epsilon
            return None
        # Now find a point on the line
        n1 = self.normal
        d1 = self.d()
        n2 = plane.normal
        d2 = plane.d()
        P0 = (n1*d2 - n2*d1).cross(n3) / n3.dist2()
        return Line(P0, n3)

class TransformMatrix:
    rows = None
    def __init__(self, *rows):
        self.rows = rows
    def __repr__(self):
        return "TransformMatrix(" + ",".join(repr(r) for r in self.rows) + ")"
    def __mul__(self, other):
        a = self
        b = other
        m = len(a.rows)
        n = len(b.rows)
        assert len(a.rows[0]) == n
        p = len(b.rows[0])
        c = [[0 for col in xrange(p)] for row in xrange(m)]
        for i in xrange(m):
            for j in xrange(p):
                for k in xrange(n):
                    c[i][j] += a.rows[i][k] * b.rows[k][j]
        return TransformMatrix(*c)
    def transform(self, point):
        m = self * TransformMatrix([point.x],[point.y],[point.z],[1])
        return Point(m.rows[0][0], m.rows[1][0], m.rows[2][0])
    def toSVG(self):
        # The simpletransform.py code represents 2d affine matrices by
        # omitting the final [0,0,1] row
        return [
            [self.rows[0][0], self.rows[0][1], self.rows[0][3]],
            [self.rows[1][0], self.rows[1][1], self.rows[1][3]],
        ]
    @staticmethod
    def fromSVG(self, mat):
        return TransformMatrix(
            [mat[0][0], mat[0][1], 0, mat[0][2]],
            [mat[1][0], mat[1][1], 0, mat[1][2]],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        )
    @staticmethod
    def identity():
        return TransformMatrix.translate(Point(0,0,0))

    @staticmethod
    def translate(vector):
        return TransformMatrix(
            [1, 0, 0, vector.x],
            [0, 1, 0, vector.y],
            [0, 0, 1, vector.z],
            [0, 0, 0, 1]
        )
    @staticmethod
    def scale(amt):
        if not isinstance(amt, Point):
            amt = Point(amt, amt, amt)
        return TransformMatrix(
            [amt.x, 0, 0, 0],
            [0, amt.y, 0, 0],
            [0, 0, amt.z, 0],
            [0, 0, 0, 1]
        )
    @staticmethod
    def rotateAxis(axis, angle=None, sin=None, cos=None):
        s = sin(angle) if sin is None else sin
        c = cos(angle) if cos is None else cos
        x, y, z = axis.x, axis.y, axis.z
        return TransformMatrix(
            [c+x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c)+y*s, 0],
            [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c) - x*s, 0],
            [z*x*(1-c)+y*s, z*y*(1-c) + x*s, c+z*z*(1-c), 0],
            [0, 0, 0, 1]
        )

class Shape:
    faces = None
    planes = None
    def __init__(self, name, faces):
        self.name = name
        self.faces = faces
        self.planes = [f.plane() for f in self.faces]
    def representativeFace(self):
        """Return one face which will represent this particular symmetry
        group.  Polyhedra with multiple symmetry groups will override
        this method to define one Shape object for each symmetry group."""
        return self.faces[0]
    @staticmethod
    def faceTransform(face1, face2):
        """Transform from face1 to face2"""
        return Shape.planeTransform(face1.plane(), face2.plane())
    @staticmethod
    def planeTransform(plane1, plane2):
        # if the planes are different distances from the origin we may need
        # to scale or translate at the end
        assert abs(plane1.d() - plane2.d()) < EPSILON
        axis = plane1.normal.cross(plane2.normal)
        if axis.dist() < EPSILON: # no rotation necessary
            m = TransformMatrix.identity()
        else:
            axis = axis.normalized()
            rotCos = plane1.normal.dot(plane2.normal)
            rotSin = sqrt(1-(rotCos*rotCos))
            m = TransformMatrix.rotateAxis(axis, sin=rotSin, cos=rotCos)
        return m

class Dodecahedron(Shape):
    """A dodecahedron."""
    def __init__(self, diameter=1):
        def mkTop(i):
            i = i % 5
            return Point(2*cos((2/5)*pi*i), 2*sin((2/5)*pi*i), phi+1)
        def mkBot(i):
            i = i % 5
            return Point(2*phi*cos((2/5)*pi*i), 2*phi*sin((2/5)*pi*i), phi-1)
        faces = [
            Face(*[mkTop(i) for i in xrange(0, 5)])
        ] + [
            Face(mkBot(i+1), mkTop(i+1), mkTop(i), mkBot(i), -mkBot(i+3)) for i in xrange(0, 5)
        ] + [
            Face(-mkBot(i), -mkTop(i), -mkTop(i+1), -mkBot(i+1), mkBot(i+3)) for i in xrange(0, 5)
        ] + [
            Face(*[-mkTop(4 - i) for i in xrange(0,5)])
        ]
        Shape.__init__(self, "Dodecahedron", [f*(diameter/4) for f in faces])

class Icosahedron(Shape):
    """An icosahedron."""
    def __init__(self, diameter=1):
        peaks = [Point(0,0,1), Point(0,0,-1)]
        r = (2/5)*sqrt(5)
        h = (1/5)*sqrt(5)
        top = [Point(r*cos(2*pi*i/5), r*sin(2*pi*i/5), h) for i in xrange(6)]
        bot = [Point(r*cos(2*pi*(i+.5)/5), r*sin(2*pi*(i+.5)/5), -h) for i in xrange(6)]
        faces = [
            Face(peaks[0], top[i], top[i+1]) for i in xrange(0,5)
        ] + [
            Face(top[i], bot[i], top[i+1]) for i in xrange(0,5)
        ] + [
            Face(bot[i+1], top[i+1], bot[i]) for i in xrange(0,5)
        ] + [
            Face(peaks[1], bot[i+1], bot[i]) for i in xrange(0,5)
        ]
        Shape.__init__(self, "Icosahedron", [f*(diameter/2) for f in faces])
    def representativeFace(self):
        return self.faces[7]

def name_to_shape(str, diameter, default="dodecahedron"):
    if str.lower() == 'dodecahedron':
        return Dodecahedron(diameter)
    if str.lower() == 'icosahedron':
        return Icosahedron(diameter)
    # Default
    return name_to_shape(default, diameter)

### Your main function subclasses the inkex.Effect class

class LayerSettings:
    effect = None
    layer = None
    metaLayer = None

    origin = None
    translation = Point(0, 0, 0)

    shape = None
    scaleFactor = 1
    symmetry = 1

    def __init__(self, effect, layer):
        self.effect = effect
        self.layer = layer
        self.metaLayer = effect.ensure_layer(layer, 'Meta')
        self.parse_meta()
        self.parse_origin()

    def toString(self, obj):
        return json.dumps(obj, indent=2)

    def fromString(self, str, default=None):
        if str is None:
            return default
        try:
            return json.loads(str)
        except ValueError:
            return default

    def parse_meta(self):
        DEFAULT_PLANE = {
            'shape': 'dodecahedron',
            'size': '3in',
            'symmetry': '1',
            'frontThick': '.125in',
            'backThick': '0',
        }
        # Look for the text element
        els = self.metaLayer.xpath(
            "./svg:text[@data-stellation]", namespaces=inkex.NSS
        )
        if len(els) > 0:
            textEl = els[0]
        else:
            textEl = inkex.etree.SubElement(
                self.metaLayer, inkex.addNS('text', 'svg'), {
                inkex.addNS('label', 'inkscape'): 'Annotation',
                'style': simplestyle.formatStyle({
                    'font-size': '20px',
                    'font-style': 'normal',
                    'font-weight': 'normal',
                    'fill': '#000',
                    'font-family': 'sans-serif',
                    'text-anchor': 'left',
                    'text-align': 'left',
                }),
                'x': '0',
                'y': '0',
            })
            textEl.text = self.toString(DEFAULT_PLANE)
        newData = self.fromString(textEl.text, DEFAULT_PLANE)
        oldData = self.fromString(textEl.get('data-stellation'), None)
        newDiameter = self.effect.unittouu(
            newData.get('size', DEFAULT_PLANE['size'])
        )
        self.shape = name_to_shape(
            newData.get('shape', DEFAULT_PLANE['shape']), newDiameter
        )
        if oldData is not None and oldData.has_key('size'):
            oldDiameter = self.effect.unittouu(oldData['size'])
            # Rescale objects if size has changed
            self.scaleFactor = newDiameter / oldDiameter
        if newData.get('symmetry') is not None:
            self.symmetry = int(newData['symmetry'])
        textEl.set('data-stellation', self.toString(newData))

    def parse_origin(self):
        # Look for the origin element
        els = self.metaLayer.xpath(
            "./svg:circle[@data-stellation]", namespaces=inkex.NSS
        )
        if len(els) > 0:
            circle = els[0]
        else:
            docHeight = self.effect.unittouu(self.effect.getDocumentHeight())
            docWidth = self.effect.unittouu(self.effect.getDocumentWidth())
            circle = inkex.etree.SubElement(
                self.metaLayer, inkex.addNS('circle','svg'), {
                'style': simplestyle.formatStyle({
                    'stroke': 'none',
                    'fill': 'black',
                }),
                'cx': str(docWidth/2),
                'cy': str(docHeight/2),
                'r': str(self.effect.unittouu("3mm")),
                inkex.addNS('label','inkscape'): 'Origin',
            });
        self.origin = Point(
            float(circle.get('cx')),
            float(circle.get('cy')),
            0
        )
        prev = self.fromString(circle.get('data-stellation'))
        if prev is not None:
            oldOrigin = Point(float(prev.get('x', self.origin.x)),
                              float(prev.get('y', self.origin.y)),
                              0)
            self.translation = self.origin - oldOrigin

        circle.set('data-stellation', self.toString({
            'x': self.origin.x,
            'y': self.origin.y,
        }))

class StellationEffect(inkex.Effect):

    def __init__(self):
        " define how the options are mapped from the inx file "
        inkex.Effect.__init__(self) # initialize the super class
        self.OptionParser.add_option(
            '--add', action="store", type="inkbool", dest="add", default=False,
            help="Add a new plane"
        )

### -------------------------------------------------------------------
### This is your main function and is called when the extension is run.

    def effect(self):
        #self.ensure_base()
        if self.options.add:
            self.add_new_plane()
        for layer in self.stellation_layers():
            self.update_layer(layer)

    def update_layer(self, layer):
        settings = LayerSettings(self, layer)
        self.update_layer_xform(settings)
        self.update_layer_guidelines(settings)
        self.update_layer_symmetry(settings)
        self.update_layer_intersections(settings)

    def update_layer_xform(self, settings):
        scale = (
            TransformMatrix.translate(settings.origin) *
            TransformMatrix.scale(settings.scaleFactor) *
            TransformMatrix.translate(-settings.origin)
        )
        mat = (scale * TransformMatrix.translate(settings.translation)).toSVG()
        for node in self.layer_contents(settings.layer):
            simpletransform.applyTransformToNode(mat, node)

    def update_layer_guidelines(self, settings):
        guidelines = self.ensure_layer(settings.layer, 'Guidelines')
        # save style from existing guidelines (if any)
        els = guidelines.xpath("./svg:path", namespaces=inkex.NSS)
        oldStyle = None if len(els) < 1 else els[0].get('style')
        # delete all existing guidelines
        self.delete_layer_contents(guidelines)
        # transform from page space to layer origin
        xform = TransformMatrix.translate(settings.origin)
        # compute new guidelines
        face = settings.shape.representativeFace()
        xform = xform * Shape.planeTransform(
            face.plane(),
            Plane(Point(0,0,face.centroid().dist()),
                  Point(0,0,1))
        )
        path = ""
        for plane in settings.shape.planes:
            line = face.plane().intersect(plane)
            if line is None: continue # parallel plane
            # Compute points where this line intersects the page bounding planes
            line = line.transform(xform)
            pts = [line.intersect(p) for p in self.pagePlanes()]
            # filter out points outside page face
            pageFace = self.pageFace()
            pts = [p for p in pts if
                   p is not None and
                   p.x + EPSILON >= pageFace.points[0].x and
                   p.y + EPSILON >= pageFace.points[0].y and
                   p.x - EPSILON <= pageFace.points[2].x and
                   p.y - EPSILON <= pageFace.points[2].y]
            if len(pts) == 0: continue # line outside of page bounds
            assert len(pts) == 2
            path += "M %f,%f L %f,%f " % (pts[0].x, pts[0].y, pts[1].x, pts[1].y)
        inkex.etree.SubElement(guidelines, inkex.addNS('path', 'svg'),{
            'd': path,
            'style': simplestyle.formatStyle({
                'opacity': 1,
                'fill': 'none',
                'stroke': '#000',
                'stroke-width': 1,
                'stroke-linecap': 'butt',
            }) if oldStyle is None else oldStyle,
        })

    def update_layer_symmetry(self, settings):
        symmetry = self.ensure_layer(settings.layer, 'Symmetry')
        # delete all existing stuff
        self.delete_layer_contents(symmetry)
        for i in xrange(settings.symmetry - 1):
            g = inkex.etree.SubElement(symmetry, inkex.addNS('g', 'svg'), {
                'transform':'rotate(%f %f %f)' % (
                    360*(i+1)/settings.symmetry,
                    settings.origin.x,
                    settings.origin.y,
                )
            })
            for node in self.layer_contents(settings.layer):
                id = node.get('id')
                if id is None:
                    # ensure there's an id
                    id = self.uniqueId('stella');
                    node.set('id', id)
                inkex.etree.SubElement(g, inkex.addNS('use', 'svg'), {
                    inkex.addNS('href','xlink'): '#' + id,
                })

    def update_layer_intersections(self, settings):
        intersections = self.ensure_layer(settings.layer, 'Intersections')
        # delete all existing intersections
        self.delete_layer_contents(intersections)
        # compute new intersections
        pass

    def delete_layer_contents(self, layer):
        attribs = layer.items()
        layer.clear()
        for key, value in attribs:
            layer.set(key, value)

    def layer_contents(self, layer):
        for child in layer:
            if child.get(inkex.addNS('groupmode', 'inkscape')) == 'layer':
                continue
            yield child

    def add_new_plane(self):
        svg = self.document.getroot()
        layer = inkex.etree.SubElement(svg, inkex.addNS('g', 'svg'), {
            inkex.addNS('label', 'inkscape'): 'Plane',
            inkex.addNS('groupmode', 'inkscape'): 'layer',
            'data-stellation': 'plane',
        })

    def stellation_layers(self):
        return self.document.xpath("//svg:svg/svg:g[@data-stellation='plane']", namespaces=inkex.NSS)

    def ensure_layer(self, parent, name, locked=True):
        for g in parent.xpath('./svg:g', namespaces=inkex.NSS):
            if g.get(inkex.addNS('label', 'inkscape')) == name:
                return g
        layer = inkex.etree.SubElement(parent, inkex.addNS('g', 'svg'), {
            inkex.addNS('label', 'inkscape'): name,
            inkex.addNS('groupmode', 'inkscape'): 'layer',
        })
        if locked:
            layer.set(inkex.addNS('insensitive', 'sodipodi'), 'true')
        return layer

    def pageFace(self):
        """Return a face representing the document."""
        viewbox = [
            float(s) for s in self.document.getroot().get('viewBox').split()
        ]
        return Face(
            Point(viewbox[0],viewbox[1],0),
            Point(viewbox[2],viewbox[1],0),
            Point(viewbox[2],viewbox[3],0),
            Point(viewbox[0],viewbox[3],0),
            inside=Point(0,0,-1)
        )
    def pagePlanes(self):
        """Return four planes representing the edges of the document."""
        pageFace = self.pageFace()
        return [
            Plane(pageFace.points[0], Vector( 0,-1, 0)),
            Plane(pageFace.points[0], Vector(-1, 0, 0)),
            Plane(pageFace.points[2], Vector( 0, 1, 0)),
            Plane(pageFace.points[2], Vector( 1, 0, 0)),
        ]

if __name__ == '__main__':
    e = StellationEffect()
    e.affect()
