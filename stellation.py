#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Stellation extension for Inkscape.

'''

import inkex       # Required
import simplestyle # will be needed here for styles support
import os          # here for alternative debug method only - so not usually required
import json
# many other useful ones in extensions folder. E.g. simplepath, cubicsuperpath, ...

from math import cos, sin, radians

__version__ = '0.0.0'

inkex.localize()

def tolerant_json(str, default=None):
    if str is None:
        return default
    try:
        return json.loads(str)
    except ValueError:
        return default

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

def points_to_bbox(p):
    """ from a list of points (x,y pairs)
        - return the lower-left xy and upper-right xy
    """
    llx = urx = p[0][0]
    lly = ury = p[0][1]
    for x in p[1:]:
        if   x[0] < llx: llx = x[0]
        elif x[0] > urx: urx = x[0]
        if   x[1] < lly: lly = x[1]
        elif x[1] > ury: ury = x[1]
    return (llx, lly, urx, ury)

def points_to_bbox_center(p):
    """ from a list of points (x,y pairs)
        - find midpoint of bounding box around all points
        - return (x,y)
    """
    bbox = points_to_bbox(p)
    return ((bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0)

def point_on_circle(radius, angle):
    " return xy coord of the point at distance radius from origin at angle "
    x = radius * cos(angle)
    y = radius * sin(angle)
    return (x, y)

def draw_SVG_circle(parent, r, cx, cy, name, style):
    " structre an SVG circle entity under parent "
    circ_attribs = {'style': simplestyle.formatStyle(style),
                    'cx': str(cx), 'cy': str(cy), 
                    'r': str(r),
                    inkex.addNS('label','inkscape'): name}
    circle = inkex.etree.SubElement(parent, inkex.addNS('circle','svg'), circ_attribs )



### Your main function subclasses the inkex.Effect class

class StellationEffect(inkex.Effect):

    def __init__(self):
        " define how the options are mapped from the inx file "
        inkex.Effect.__init__(self) # initialize the super class
        self.OptionParser.add_option(
            '--add', action="store", type="inkbool", dest="add", default=False,
            help="Add a new plane"
        )

        # Two ways to get debug info:
        # OR just use inkex.debug(string) instead...
        try:
            self.tty = open("/dev/tty", 'w')
        except:
            self.tty = open(os.devnull, 'w')  # '/dev/null' for POSIX, 'nul' for Windows.
            # print >>self.tty, "gears-dev " + __version__

    def getUnittouu(self, param):
        " for 0.48 and 0.91 compatibility "
        try:
            return inkex.unittouu(param)
        except AttributeError:
            return self.unittouu(param)

    def getColorString(self, longColor, verbose=False):
        """ Convert the long into a #RRGGBB color value
            - verbose=true pops up value for us in defaults
            conversion back is A + B*256^1 + G*256^2 + R*256^3
        """
        if verbose: inkex.debug("%s ="%(longColor))
        longColor = long(longColor)
        if longColor <0: longColor = long(longColor) & 0xFFFFFFFF
        hexColor = hex(longColor)[2:-3]
        hexColor = '#' + hexColor.rjust(6, '0').upper()
        if verbose: inkex.debug("  %s for color default value"%(hexColor))
        return hexColor
    
    def add_text(self, node, text, position, text_height=12):
        """ Create and insert a single line of text into the svg under node.
        """
        line_style = {'font-size': '%dpx' % text_height, 'font-style':'normal', 'font-weight': 'normal',
                     'fill': '#F6921E', 'font-family': 'Bitstream Vera Sans,sans-serif',
                     'text-anchor': 'middle', 'text-align': 'center'}
        line_attribs = {inkex.addNS('label','inkscape'): 'Annotation',
                       'style': simplestyle.formatStyle(line_style),
                       'x': str(position[0]),
                       'y': str((position[1] + text_height) * 1.2)
                       }
        line = inkex.etree.SubElement(node, inkex.addNS('text','svg'), line_attribs)
        line.text = text

           
    def calc_unit_factor(self):
        """ return the scale factor for all dimension conversions.
            - The document units are always irrelevant as
              everything in inkscape is expected to be in 90dpi pixel units
        """
        # namedView = self.document.getroot().find(inkex.addNS('namedview', 'sodipodi'))
        # doc_units = self.getUnittouu(str(1.0) + namedView.get(inkex.addNS('document-units', 'inkscape')))
        unit_factor = self.getUnittouu(str(1.0) + self.options.units)
        return unit_factor


### -------------------------------------------------------------------
### This is your main function and is called when the extension is run.

    def effect(self):
        #self.ensure_base()
        if self.options.add:
            self.add_new_plane()
        for layer in self.stellation_layers():
            self.update_layer(layer)

    def update_layer(self, layer):
        meta = self.ensure_layer(layer, 'Meta')
        settings = self.parse_meta(meta)

        self.update_layer_guidelines(layer, settings)

        self.update_layer_intersections(layer, settings)
        pass

    def update_layer_guidelines(self, layer, settings):
        guidelines = self.ensure_layer(layer, 'Guidelines')
        # delete all existing guidelines
        self.delete_layer_contents(guidelines)
        # compute new guidelines
        pass

    def update_layer_intersections(self, layer, settings):
        intersections = self.ensure_layer(layer, 'Intersections')
        # delete all existing intersections
        self.delete_layer_contents(intersections)
        # compute new intersections
        pass

    def delete_layer_contents(self, layer):
        attribs = layer.items()
        layer.clear()
        for key, value in attribs:
            layer.set(key, value)

    def add_new_plane(self):
        svg = self.document.getroot()
        layer = inkex.etree.SubElement(svg, inkex.addNS('g', 'svg'))
        layer.set(inkex.addNS('label', 'inkscape'), 'Plane');
        layer.set(inkex.addNS('groupmode', 'inkscape'), 'layer')
        layer.set('data-stellation', 'plane')

    def parse_meta(self, meta):
        DEFAULT_PLANE = {
            'type': 'dodecahedron',
            'size': '5',
            'units': 'inch',
        }
        # Look for the text element
        els = meta.xpath("./svg:text[@data-stellation]", namespaces=inkex.NSS)
        if len(els) > 0:
            metaText = els[0].text
            metaWas = els[0].get('data-stellation')
        else:
            metaText = json.dumps(DEFAULT_PLANE)
            metaWas = None
            el = inkex.etree.SubElement(meta, inkex.addNS('text', 'svg'), {
                inkex.addNS('label', 'inkscape'): 'Annotation',
                'data-stellation': '',
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
            el.text = metaText
        # parse metaText
        newData = tolerant_json(metaText)
        oldData = tolerant_json(metaWas)
        inkex.debug(metaText)
        # if metaWas != None and metaWas != "": rescale objects
        meta.set('data-stellation', json.dumps(newData))
        return newData

    def stellation_layers(self):
        return self.document.xpath("//svg:svg/svg:g[@data-stellation='plane']", namespaces=inkex.NSS)

    def ensure_layer(self, parent, name, locked=True):
        for g in parent.xpath('./svg:g', namespaces=inkex.NSS):
            if g.get(inkex.addNS('label', 'inkscape')) == name:
                return g
        layer = inkex.etree.SubElement(parent, 'g');
        layer.set(inkex.addNS('label', 'inkscape'), name);
        layer.set(inkex.addNS('groupmode', 'inkscape'), 'layer')
        if locked:
            layer.set(inkex.addNS('insensitive', 'sodipodi'), 'true')
        return layer

    def old_effect(self):
        """ Calculate Gear factors from inputs.
            - Make list of radii, angles, and centers for each tooth and 
              iterate through them
            - Turn on other visual features e.g. cross, rack, annotations, etc
        """
        
        # check for correct number of selected objects and return a translatable errormessage to the user
        if len(self.options.ids) != 2:
            inkex.errormsg(_("This extension requires two selected objects."))
            exit()
        # Convert color - which comes in as a long into a string like '#FFFFFF'
        self.options.strokeColour = self.getColorString(self.options.strokeColour)
        #
        path_stroke = self.options.strokeColour  # take color from tab3
        path_fill   = 'none'     # no fill - just a line
        path_stroke_width  = 0.6 # can also be in form '0.6mm'
        # gather incoming params and convert
        param1 = self.options.param1
        param2 = self.options.param2
        param3 = self.options.param3
        choice = self.options.achoice
        units2 = self.options.units2
        accuracy = self.options.accuracy # although a string in inx - option parser converts to int.
        # calculate unit factor for units defined in dialog. 
        unit_factor = self.calc_unit_factor()
        # what page are we on
        page_id = self.options.active_tab # sometimes wrong the very first time

        # Do your thing - create some points or a path or whatever...
        points = []
        points.extend( [ (i*2,i*2) for i in range(0, param1) ])
        points.append((param1, param1*2+5))
        #inkex.debug(points)
        path = points_to_svgd( points )
        #inkex.debug(path)
        bbox_center = points_to_bbox_center( points )
        # example debug
        # print >>self.tty, bbox_center
        # or
        # inkex.debug("bbox center %s" % bbox_center)

        
        # Embed the path in a group to make animation easier:
        # Be sure to examine the internal structure by looking in the xml editor inside inkscape
        # This finds center of exisiting document page
        
        # This finds center of current view in inkscape
        t = 'translate(%s,%s)' % (self.view_center[0], self.view_center[1] )
        # Make a nice useful name
        g_attribs = { inkex.addNS('label','inkscape'): 'useful name' + str( param1 ),
                      inkex.addNS('transform-center-x','inkscape'): str(-bbox_center[0]),
                      inkex.addNS('transform-center-y','inkscape'): str(-bbox_center[1]),
                      'transform': t,
                      'info':'N: '+str(param1)+'; with:'+ str(param2) }
        # add the group to the document's current layer
        topgroup = inkex.etree.SubElement(self.current_layer, 'g', g_attribs )

        # Create SVG Path under this top level group
        # define style using basic dictionary
        style = { 'stroke': path_stroke, 'fill': path_fill, 'stroke-width': param2 }
        # convert style into svg form (see import at top of file)
        mypath_attribs = { 'style': simplestyle.formatStyle(style), 'd': path }
        # add path to scene
        squiggle = inkex.etree.SubElement(topgroup, inkex.addNS('path','svg'), mypath_attribs )


        # Add another feature in same group (under it)
        style = { 'stroke': path_stroke, 'fill': path_fill, 'stroke-width': path_stroke_width }
        cs = param1 / 2 # centercross length
        cs2 = str(cs)
        d = 'M-'+cs2+',0L'+cs2+',0M0,-'+cs2+'L0,'+cs2  # 'M-10,0L10,0M0,-10L0,10'
        # or
        d = 'M %s,0 L %s,0 M 0,-%s L 0,%s' % (-cs, cs, cs,cs)
        # or
        d = 'M {0},0 L {1},0 M 0,{0} L 0,{1}'.format(-cs,cs)
        # or
        #d = 'M-10 0L10 0M0 -10L0 10' # commas superfluous, minimise spaces.
        cross_attribs = { inkex.addNS('label','inkscape'): 'Center cross',
                          'style': simplestyle.formatStyle(style), 'd': d }
        cross = inkex.etree.SubElement(topgroup, inkex.addNS('path','svg'), cross_attribs )


        # Add a precalculated svg circle
        style = { 'stroke': path_stroke, 'fill': path_fill, 'stroke-width': self.getUnittouu(str(param2) +self.options.units) }
        draw_SVG_circle(topgroup, param1*4*unit_factor, 0, 0, 'a circle', style)


        # Add some super basic text (e.g. for debug)
        if choice:
            notes = ['a label: %d (%s) ' % (param1*unit_factor, self.options.units),
                     'doc line'
                     ]
            text_height = 12
            # position above
            y = - 22
            for note in notes:
                self.add_text(topgroup, note, [0,y], text_height)
                y += text_height * 1.2
        #
        #more complex text
        font_height = min(32, max( 10, int(self.getUnittouu(str(param1) + self.options.units))))
        text_style = { 'font-size': str(font_height),
                       'font-family': 'arial',
                       'text-anchor': 'middle',
                       'text-align': 'center',
                       'fill': path_stroke }
        text_atts = {'style':simplestyle.formatStyle(text_style),
                     'x': str(44),
                     'y': str(-15) }
        text = inkex.etree.SubElement(topgroup, 'text', text_atts)
        text.text = "%4.3f" %(param1*param2)

if __name__ == '__main__':
    e = StellationEffect()
    e.affect()

# Notes

