import os
import argparse
import pathlib
import xml.etree.ElementTree as ET
import traceback

from fontTools.ttLib import TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.svgLib.path import SVGPath, parse_path
from fontTools.misc.transform import Transform
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib.tables._c_m_a_p import CmapSubtable
from fontTools.ttLib.tables.O_S_2f_2 import Panose

from fontTools.ttLib.tables import (
    _h_e_a_d, _h_h_e_a, O_S_2f_2, _p_o_s_t, _c_m_a_p,
    _g_l_y_f, _h_m_t_x, _l_o_c_a, _m_a_x_p, _n_a_m_e
)

DEFAULT_INPUT_DIR = './output/characters/'
DEFAULT_FAMILY_NAME = "MyMonospaceFont"
DEFAULT_STYLE_NAME = "Regular"
DEFAULT_OUTPUT_FILE = f'./{DEFAULT_FAMILY_NAME}-{DEFAULT_STYLE_NAME}.ttf'
DEFAULT_VERSION = "1.000"
DEFAULT_COPYRIGHT = "Copyright 2025 by Me"

DEFAULT_UNITS_PER_EM = 1000
DEFAULT_ASCENDER = 800       # Y=0 is baseline, positive Y is up
DEFAULT_DESCENDER = -200     # Must be negative
DEFAULT_CAP_HEIGHT = 700     # Typical height of capital letters
DEFAULT_X_HEIGHT = 500       # Typical height of 'x'
DEFAULT_LINE_GAP = 0         # Extra space between lines
DEFAULT_FIXED_ADVANCE = 600  # Fixed width for ALL glyphs (monospace)

# Map filenames (without extension) to internal glyph names
# Use standard names where possible (e.g., 'exclam' not 'exclamation')
FILENAME_TO_GLYPHNAME = {
    'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e', 'f': 'f', 'g': 'g',
    'h': 'h', 'i': 'i', 'j': 'j', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n',
    'o': 'o', 'p': 'p', 'q': 'q', 'r': 'r', 's': 's', 't': 't', 'u': 'u',
    'v': 'v', 'w': 'w', 'x': 'x', 'y': 'y', 'z': 'z',
    'comma': ',',
    'period': '.',
    'question': '?',
    'exclamation': '!',
    'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G',
    'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N',
    'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U',
    'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
}
FILENAME_TO_GLYPHNAME = {f"{ord(char)}_{char}": glyphname for char, glyphname in FILENAME_TO_GLYPHNAME.items() if len(char) == 1 and ('a' <= char <= 'z' or 'A' <= char <= 'Z')}
FILENAME_TO_GLYPHNAME.update({
    'comma': ',',
    'period': '.',
    'question': '?',
    'exclamation': '!',
})

# Map internal glyph names to Unicode code points
GLYPHNAME_TO_UNICODE = {
    glyph_name: ord(glyph_name) for char, glyph_name in FILENAME_TO_GLYPHNAME.items() if len(glyph_name) == 1 and ('a' <= glyph_name <= 'z' or 'A' <= glyph_name <= 'Z')
}
# Add punctuation mappings
GLYPHNAME_TO_UNICODE.update({
    ',': ord(','),
    '.': ord('.'),
    '?': ord('?'),
    '!': ord('!'),
})

UNICODE_TO_GLYPHNAME = {}
for filename, glyph_name in FILENAME_TO_GLYPHNAME.items():
    if len(glyph_name) == 1 and ('a' <= glyph_name <= 'z' or 'A' <= glyph_name <= 'Z'):
        UNICODE_TO_GLYPHNAME[ord(glyph_name)] = glyph_name
UNICODE_TO_GLYPHNAME[ord(',')] = ','
UNICODE_TO_GLYPHNAME[ord('.')] = '.'
UNICODE_TO_GLYPHNAME[ord('?')] = '?'
UNICODE_TO_GLYPHNAME[ord('!')] = '!'
# Map space (special case, no SVG needed)
UNICODE_TO_GLYPHNAME[ord(' ')] = 'space'

SVG_NS = {'svg': 'http://www.w3.org/2000/svg'}


def setup_font_tables(font, units_per_em, ascender, descender, cap_height, x_height, line_gap, fixed_advance, glyph_order):
    """Initializes essential font tables and metrics."""

    font.setGlyphOrder(glyph_order)

    # --- Create and Assign Table Instances ---

    # --- HEAD Table ---
    head = font['head'] = _h_e_a_d.table__h_e_a_d()
    head.tableVersion = 1.0
    head.checkSumAdjustment = 0
    head.magicNumber = 0x5F0F3CF5
    head.created = 0
    head.modified = 0
    head.unitsPerEm = units_per_em
    head.fontRevision = float(DEFAULT_VERSION.split(' ')[0])
    head.macStyle = 0
    head.flags = 3
    head.lowestRecPPEM = 8
    head.fontDirectionHint = 2
    head.indexToLocFormat = 0
    head.glyphDataFormat = 0

    # --- HHEA Table (Horizontal Header) ---
    hhea = font['hhea'] = _h_h_e_a.table__h_h_e_a()
    hhea.tableVersion = 0x00010000
    hhea.ascent = ascender
    hhea.descent = descender
    hhea.lineGap = line_gap
    hhea.advanceWidthMax = 0 # Let font.save() calculate
    hhea.minLeftSideBearing = 0 # Let font.save() calculate
    hhea.minRightSideBearing = 0 # Let font.save() calculate
    hhea.xMaxExtent = 0 # Let font.save() calculate
    hhea.caretSlopeRise = 1
    hhea.caretSlopeRun = 0
    hhea.caretOffset = 0
    hhea.metricDataFormat = 0
    hhea.numberOfHMetrics = 0
    hhea.reserved0 = 0
    hhea.reserved1 = 0
    hhea.reserved2 = 0
    hhea.reserved3 = 0

    # --- OS/2 Table (OS Specific Metrics) ---
    os2 = font['OS/2'] = O_S_2f_2.table_O_S_2f_2()
    os2.version = 4 # Use version 4 for modern compatibility, includes typo/win metrics
    os2.xAvgCharWidth = fixed_advance # Monospace: average = fixed width
    os2.usWeightClass = 400 # Normal
    os2.usWidthClass = 5 # Medium (normal)
    os2.fsType = 1 << 3 # Editable embedding allowed (Bit 3 = 8). Review licensing needs.
    os2.ySubscriptXSize = int(units_per_em * 0.65)
    os2.ySubscriptYSize = int(units_per_em * 0.60)
    os2.ySubscriptXOffset = 0
    os2.ySubscriptYOffset = int(units_per_em * 0.14)
    os2.ySuperscriptXSize = int(units_per_em * 0.65)
    os2.ySuperscriptYSize = int(units_per_em * 0.60)
    os2.ySuperscriptXOffset = 0
    os2.ySuperscriptYOffset = int(units_per_em * 0.48)
    os2.yStrikeoutSize = int(units_per_em * 0.05)
    os2.yStrikeoutPosition = int(units_per_em * 0.26) # From baseline
    os2.sFamilyClass = 0 # No classification
    # Panose: Set basic values for a monospace font
    os2.panose = Panose()
    os2.panose.bFamilyType = 2  # Latin Text
    os2.panose.bSerifStyle = 11 # Sans Serif - Normal
    os2.panose.bWeight = 5 # Book (matches 400 weight class)
    os2.panose.bProportion = 9 # Monospaced
    os2.panose.bContrast = 2 # None
    os2.panose.bStrokeVariation = 1 # No variation
    os2.panose.bArmStyle = 1 # No fitting
    os2.panose.bLetterForm = 1 # No fitting
    os2.panose.bMidline = 1 # Standard
    os2.panose.bXHeight = 1 # Standard
    # Unicode Ranges: Let fontTools calculate based on cmap or set manually if known
    os2.ulUnicodeRange1 = 1 # Basic Latin (Adjust if needed)
    os2.ulUnicodeRange2 = 0
    os2.ulUnicodeRange3 = 0
    os2.ulUnicodeRange4 = 0
    os2.achVendID = "UKWN"
    os2.fsSelection = 1 << 6 # REGULAR bit
    os2.fsSelection |= (1 << 7) # USE_TYPO_METRICS bit (recommended)

    if UNICODE_TO_GLYPHNAME:
        valid_unicodes = [uni for uni in UNICODE_TO_GLYPHNAME.keys() if uni is not None]
        if valid_unicodes:
             # Filter for Basic Multilingual Plane (BMP) for these fields
            bmp_unicodes = [u for u in valid_unicodes if 0 <= u <= 0xFFFF]
            if bmp_unicodes:
                 min_unicode = min(bmp_unicodes)
                 max_unicode = max(bmp_unicodes)
                 os2.usFirstCharIndex = min_unicode
                 os2.usLastCharIndex = max_unicode
            else:
                 os2.usFirstCharIndex = 0xFFFF
                 os2.usLastCharIndex = 0xFFFF
        else:
            os2.usFirstCharIndex = 0xFFFF
            os2.usLastCharIndex = 0xFFFF
    else:
         os2.usFirstCharIndex = 0xFFFF
         os2.usLastCharIndex = 0xFFFF

    # Typographic Metrics (Important for cross-platform consistency)
    os2.sTypoAscender = ascender
    os2.sTypoDescender = descender
    os2.sTypoLineGap = line_gap
    # Windows Metrics (Often set equal to typo metrics, but can differ)
    os2.usWinAscent = ascender
    os2.usWinDescent = abs(descender) # Must be positive
    # Code Page Ranges: Let fontTools calculate or set manually
    os2.ulCodePageRange1 = 1 # Latin 1
    os2.ulCodePageRange2 = 0
    # Font Dimensions
    os2.sxHeight = x_height
    os2.sCapHeight = cap_height
    os2.usDefaultChar = 0 # .notdef glyph index
    space_unicode = ord(' ')
    os2.usBreakChar = space_unicode if 0 <= space_unicode <= 0xFFFF else 0 # Use space if in BMP
    os2.usMaxContext = 0 # Required for version >= 2

    # --- POST Table ---
    post = font['post'] = _p_o_s_t.table__p_o_s_t()
    post.tableVersion = 2.0 # Use version 2 for standard glyph names
    post.italicAngle = 0.0
    post.underlinePosition = int(units_per_em * -0.075) # Below baseline
    post.underlineThickness = int(units_per_em * 0.05)
    post.isFixedPitch = 1 # Monospace!
    post.minMemType42 = 0
    post.maxMemType42 = 0
    post.minMemType1 = 0
    post.maxMemType1 = 0
    post.formatType = 3

    # --- CMAP Table (Character to Glyph Mapping) ---
    cmap = font['cmap'] = _c_m_a_p.table__c_m_a_p()
    cmap.tableVersion = 0
    # Create Unicode BMP subtable (Platform 3, Encoding 1, Format 4)
    subtable_bmp = CmapSubtable.newSubtable(4)
    subtable_bmp.platformID = 3 # Windows
    subtable_bmp.platEncID = 1 # Unicode BMP
    subtable_bmp.language = 0
    subtable_bmp.cmap = {} # Will be populated later
    # Create Unicode Full subtable (Platform 3, Encoding 10, Format 12) - Optional but recommended for > BMP
    # subtable_full = CmapSubtable.newSubtable(12)
    # subtable_full.platformID = 3 # Windows
    # subtable_full.platEncID = 10 # Unicode Full Repertoire
    # subtable_full.language = 0
    # subtable_full.cmap = {} # Will be populated later
    # Create Mac Roman subtable (Platform 1, Encoding 0, Format 0) - Less common now
    # subtable_mac = CmapSubtable.newSubtable(0)
    # subtable_mac.platformID = 1 # Macintosh
    # subtable_mac.platEncID = 0 # Roman
    # subtable_mac.language = 0
    # subtable_mac.cmap = {} # Will be populated later

    cmap.tables = [subtable_bmp] # Add more tables here if created

    glyf = font['glyf'] = _g_l_y_f.table__g_l_y_f()
    glyf.glyphs = {}

    hmtx = font['hmtx'] = _h_m_t_x.table__h_m_t_x()
    hmtx.metrics = {}

    font['loca'] = _l_o_c_a.table__l_o_c_a() # Will be built by font.save()

    maxp = font['maxp'] = _m_a_x_p.table__m_a_x_p()
    maxp.tableVersion = 0x00010000 # Version 1.0 contains more detail
    maxp.numGlyphs = len(glyph_order)
    maxp.maxPoints = 0
    maxp.maxContours = 0
    maxp.maxCompositePoints = 0
    maxp.maxCompositeContours = 0
    maxp.maxZones = 1
    maxp.maxTwilightPoints = 0
    maxp.maxStorage = 0
    maxp.maxFunctionDefs = 0
    maxp.maxInstructionDefs = 0
    maxp.maxStackElements = 0
    maxp.maxSizeOfInstructions = 0
    maxp.maxComponentElements = 0
    maxp.maxComponentDepth = 0

    # --- Initialize NAME Table ---
    name = font['name'] = _n_a_m_e.table__n_a_m_e()
    name.names = []


def set_font_names(font, family_name, style_name, version, copyright_info):
    """Populates the name table with standard naming info."""
    name_table = font['name']

    # Name IDs
    COPYRIGHT_ID = 0
    FAMILY_ID = 1
    STYLE_ID = 2
    UNIQUE_ID = 3
    FULL_NAME_ID = 4
    VERSION_ID = 5
    PS_NAME_ID = 6
    PREFERRED_FAMILY_ID = 16 # Use if Family/Style naming is complex (e.g., Bold Italic)
    PREFERRED_STYLE_ID = 17 # Use if Family/Style naming is complex

    # Format names
    unique_id = f"{version};{family_name}-{style_name}"
    full_name = f"{family_name} {style_name}"
    # PostScript name constraints: max 63 chars, no spaces/brackets/etc., [A-Za-z0-9] and hyphen.
    ps_name_family = "".join(filter(str.isalnum, family_name))
    ps_name_style = "".join(filter(str.isalnum, style_name))
    ps_name = f"{ps_name_family}-{ps_name_style}"[:63]

    # Add names using nameTable.setName(string, nameID, platformID, platEncID, langID)
    # Platform IDs: 1=Mac, 3=Windows
    # Encoding IDs: 1=Unicode BMP (UCS-2) [Windows], 0=Roman [Mac]
    # Language IDs: 1033=English (US) [Windows], 0=English [Mac]

    name_list = [
        (COPYRIGHT_ID, copyright_info),
        (FAMILY_ID, family_name),
        (STYLE_ID, style_name),
        (UNIQUE_ID, unique_id),
        (FULL_NAME_ID, full_name),
        (VERSION_ID, f"Version {version}"),
        (PS_NAME_ID, ps_name),
        # Add Preferred names, usually same as Family/Style for simple cases
        (PREFERRED_FAMILY_ID, family_name),
        (PREFERRED_STYLE_ID, style_name),
    ]

    for nameid, text in name_list:
        # Windows entry (Unicode BMP, US English)
        name_table.setName(text, nameid, 3, 1, 1033)
        # Mac entry (Roman, English) - Still recommended for older Mac apps
        name_table.setName(text, nameid, 1, 0, 0)


def create_empty_glyph(glyph_name, fixed_advance, units_per_em):
    """Creates an empty glyph (like space or .notdef)."""
    pen = TTGlyphPen(glyphSet=None)
    glyph = pen.glyph()
    glyph.coordinates = []
    glyph.endPtsOfContours = []
    glyph.flags = []
    glyph.numberOfContours = 0

    # Metrics: Advance width, Left Side Bearing (usually 0 for empty/space)
    metrics = (fixed_advance, 0) # Use fixed advance for space in monospace
    # .notdef often has advance=0 or advance=fixed_advance/2, but fixed_advance is safe for monospace
    return glyph, metrics


def process_svg_glyph(svg_path, font, units_per_em, ascender, descender, fixed_advance):
    """Parses SVG, creates TTGlyph, calculates metrics for monospace."""
    glyph_name = pathlib.Path(svg_path).stem
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  Error parsing SVG {svg_path}: {e}")
        return None, None
    except FileNotFoundError:
        print(f"  SVG file not found: {svg_path}")
        return None, None

    path_elements = root.findall('.//svg:path', SVG_NS)
    if not path_elements:
        path_elements = root.findall('.//path')
        if not path_elements:
            print(f"  Warning: No <path> elements found in {svg_path}. Creating empty glyph.")
            intended_glyph_name = FILENAME_TO_GLYPHNAME.get(glyph_name, glyph_name)
            return create_empty_glyph(intended_glyph_name, fixed_advance, units_per_em)

    viewbox_str = root.get('viewBox')
    vb_x, vb_y, vb_w, vb_h = 0, 0, 0, 0
    if viewbox_str:
        try:
            # Assuming SVG generated with full viewBox, vb_x/vb_y are 0
            vb_x, vb_y, vb_w, vb_h = map(float, viewbox_str.split())
        except ValueError:
            print(f"  Warning: Invalid viewBox format '{viewbox_str}' in {svg_path}. Cannot process transform.")
            intended_glyph_name = FILENAME_TO_GLYPHNAME.get(glyph_name, glyph_name)
            return create_empty_glyph(intended_glyph_name, fixed_advance, units_per_em)
    else:
        print(f"  Warning: No viewBox found in {svg_path}. Cannot determine scale/position accurately.")
        intended_glyph_name = FILENAME_TO_GLYPHNAME.get(glyph_name, glyph_name)
        return create_empty_glyph(intended_glyph_name, fixed_advance, units_per_em)


    if vb_w <= 0 or vb_h <= 0:
        print(f"  Warning: Zero or negative width/height in viewBox/attributes for {svg_path}")
        intended_glyph_name = FILENAME_TO_GLYPHNAME.get(glyph_name, glyph_name)
        return create_empty_glyph(intended_glyph_name, fixed_advance, units_per_em)

    # --- Calculate Transformation ---
    print(f"    Calculating transform (Align SVG bottom to font descender, scale full range)")
    if vb_h > 0:
        scale = (ascender - descender) / vb_h
    else:
        scale = 1.0
    if scale <= 0: # Should not happen with positive height and asc > desc
        print(f"      Warning: Invalid scale ({scale:.4f}) calculated. vb_h={vb_h}, ascender={ascender}, descender={descender}. Using scale=1.0")
        scale = 1.0

    scaled_glyph_width = vb_w * scale
    # Center the scaled *viewBox* width horizontally
    lsb = max(0, int(round((fixed_advance - scaled_glyph_width) / 2)))

    transform = Transform()
    # 1. Move SVG *bottom-left* (vb_x, vb_y + vb_h) to SVG origin (0,0).
    #    Since using full viewBox from updated SVG generator, vb_x=0, vb_y=0. This simplifies to:
    transform = transform.translate(0, -vb_h)
    # 2. Scale (based on full font range) and flip Y axis (font Y=up)
    #    The original bottom edge (now at Y=0) stays at Y=0 after scaling/flipping.
    transform = transform.scale(scale, -scale)
    # 3. Translate horizontally by LSB and vertically so the original bottom edge
    #    (which is currently at Y=0) lands on the font's descender line.
    transform = transform.translate(lsb, descender)

    print(f"      SVG: {os.path.basename(svg_path)}, ViewBox: ({vb_x:.2f}, {vb_y:.2f}, {vb_w:.2f}, {vb_h:.2f})")
    print(f"      Font Metrics: Asc={ascender}, Desc={descender}, AdvWidth={fixed_advance}, UPM={units_per_em}")
    print(f"      Calculated: Scale={scale:.4f}, ScaledWidth={scaled_glyph_width:.2f}, LSB={lsb}")
    print(f"      Transform Matrix: {transform}")

    pen = TTGlyphPen(glyphSet=None)
    transform_pen = TransformPen(pen, transform)

    try:
        for path_element in path_elements:
            path_d = path_element.get('d')
            if not path_d:
                print(f"    Warning: Found <path> element with empty 'd' attribute in {svg_path}. Skipping this path.")
                continue
            parse_path(path_d, transform_pen)

        glyph = pen.glyph()

    except Exception as e:
        print(f"  Error converting path data for {svg_path}: {e}")
        traceback.print_exc()
        intended_glyph_name = FILENAME_TO_GLYPHNAME.get(glyph_name, glyph_name)
        return create_empty_glyph(intended_glyph_name, fixed_advance, units_per_em)

    # --- Define metrics ---
    try:
        glyph.recalcBounds(None)
    except Exception as e:
        if glyph.numberOfContours > 0 or hasattr(glyph, 'components'):
             print(f"  Warning: Could not recalculate bounds for {svg_path}: {e}")

    metrics = (fixed_advance, lsb)
    return glyph, metrics

# --- Main Function ---
def build_font(input_dir, output_file, family_name, style_name, version, copyright_info,
               units_per_em, ascender, descender, cap_height, x_height, line_gap, fixed_advance):
    """Builds the TTF font from SVG glyphs."""

    # Reminder: Clear OS font cache after generating the font!
    print("-" * 20)
    print("Starting Font Build Process")
    print("Reminder: Clear OS font cache after generation before testing!")
    print("Reminder: Check SVG winding order (outer=CCW, inner=CW) for rendering issues!")
    print("-" * 20)

    print("Initializing font structure...")
    # --- Create Font Object ---
    font = TTFont()

    # --- Define Glyph Order ---
    # Start with required glyphs
    glyph_order = ['.notdef', '.null', 'space'] # .null is often included (same as space usually)

    # Add glyph names from the files we intend to process
    # Ensure uniqueness and preserve order somewhat reasonably
    processed_glyph_names = set()
    for fname, gname in FILENAME_TO_GLYPHNAME.items():
        if gname not in processed_glyph_names and gname not in glyph_order:
             glyph_order.append(gname)
             processed_glyph_names.add(gname)

    print(f"Glyph Order ({len(glyph_order)} glyphs): {glyph_order}")

    # --- Setup Font Tables (This now creates them) ---
    setup_font_tables(font, units_per_em, ascender, descender, cap_height, x_height, line_gap, fixed_advance, glyph_order)

    # --- Set Font Naming ---
    set_font_names(font, family_name, style_name, version, copyright_info)

    # --- Get References to Tables (Now guaranteed to exist) ---
    glyf_table = font['glyf']
    hmtx_table = font['hmtx']
    cmap_subtable_bmp = None
    # Find the BMP cmap subtable we created
    for table in font['cmap'].tables:
        if table.platformID == 3 and table.platEncID == 1 and table.format == 4:
            cmap_subtable_bmp = table
            break
    if cmap_subtable_bmp is None:
         print("CRITICAL ERROR: BMP cmap subtable (3,1,4) not found after setup.")
         return
    cmap_dict = cmap_subtable_bmp.cmap # Get the dictionary to add mappings

    print("Creating special glyphs (.notdef, .null, space)...")
    # --- Add .notdef Glyph (Glyph 0) ---
    # Often uses 0 advance, but fixed_advance safer for mono previews
    glyph_notdef, metrics_notdef = create_empty_glyph('.notdef', fixed_advance, units_per_em)
    glyf_table.glyphs['.notdef'] = glyph_notdef
    hmtx_table.metrics['.notdef'] = metrics_notdef

    # --- Add .null Glyph (Glyph 1 - often same as space) ---
    glyph_null, metrics_null = create_empty_glyph('.null', fixed_advance, units_per_em)
    glyf_table.glyphs['.null'] = glyph_null
    hmtx_table.metrics['.null'] = metrics_null
    # Some systems map U+0000 to .null
    if 0x0000 not in cmap_dict:
        cmap_dict[0x0000] = '.null'


    # --- Add space Glyph ---
    glyph_space, metrics_space = create_empty_glyph('space', fixed_advance, units_per_em)
    glyf_table.glyphs['space'] = glyph_space
    hmtx_table.metrics['space'] = metrics_space
    space_unicode = ord(' ')
    if UNICODE_TO_GLYPHNAME.get(space_unicode) == 'space':
        if 0 <= space_unicode <= 0xFFFF:
            cmap_dict[space_unicode] = 'space'
        else:
            print(f"Warning: Space unicode {space_unicode} out of BMP range for format 4 cmap.")
    else:
         print(f"Warning: Space character U+{space_unicode:04X} not mapped to 'space' glyph in UNICODE_TO_GLYPHNAME.")


    print(f"Processing SVG glyphs from {input_dir}...")
    # --- Process SVG Glyphs ---
    processed_files = 0
    added_glyphs = set(['.notdef', '.null', 'space']) # Track added glyphs

    # Iterate through the filename mapping (SVG file -> glyph name)
    for filename_base, glyph_name in FILENAME_TO_GLYPHNAME.items():

        if glyph_name in added_glyphs:
             # This might happen if multiple filenames map to the same glyph name.
             # We'll only process the first one encountered in the FILENAME_TO_GLYPHNAME dict.
             print(f"  Skipping {filename_base}.svg, glyph '{glyph_name}' already added.")
             continue

        svg_filename = f"{filename_base}.svg"
        svg_path = os.path.join(input_dir, svg_filename)

        # Check if file exists before processing
        if not os.path.exists(svg_path):
            print(f"  Warning: SVG file not found for glyph '{glyph_name}': {svg_path}. Skipping.")
            continue

        print(f"  Processing {svg_filename} -> Glyph '{glyph_name}'")

        glyph, metrics = process_svg_glyph(svg_path, font, units_per_em, ascender, descender, fixed_advance)

        if glyph is None or metrics is None:
            print(f"    -> Failed to process {svg_filename}. Skipping glyph '{glyph_name}'.")
            # Ensure it's removed from glyph order if processing failed
            if glyph_name in glyph_order:
                 glyph_order.remove(glyph_name)
                 print(f"    -> Removed '{glyph_name}' from glyph order due to processing failure.")
            continue # Skip adding this glyph

        # Add the processed glyph and its metrics
        if glyph_name not in glyf_table.glyphs:
            glyf_table.glyphs[glyph_name] = glyph
            hmtx_table.metrics[glyph_name] = metrics
            added_glyphs.add(glyph_name)
            processed_files += 1
        else:
             # Should not happen with current logic, but good to note
             print(f"    Warning: Glyph name '{glyph_name}' seems to be added twice. Check mappings.")


    # --- Map Unicode points to added glyphs ---
    print("Adding Unicode mappings (CMAP)...")
    final_cmap = {}
    mapped_chars = 0
    for unicode_val, glyph_name in UNICODE_TO_GLYPHNAME.items():
        if glyph_name in added_glyphs: # Only map glyphs that were successfully added
            if 0 <= unicode_val <= 0xFFFF: # Check if in BMP for Format 4 table
                final_cmap[unicode_val] = glyph_name
                mapped_chars += 1
            else:
                 print(f"    Warning: Unicode U+{unicode_val:04X} for glyph '{glyph_name}' is outside BMP. Requires Format 12 cmap (not implemented).")
        else:
            # This happens if the SVG for the glyph failed processing or wasn't found
            print(f"  Warning: Cannot map U+{unicode_val:04X}. Glyph '{glyph_name}' was not successfully processed or added.")

    cmap_dict.update(final_cmap)
    print(f"Mapped {mapped_chars} characters to glyphs in BMP cmap.")


    # --- Final Calculations & Cleanup ---
    print("Finalizing font...")

    # Update glyph order based on actually added glyphs
    final_glyph_order = [g for g in glyph_order if g in added_glyphs]
    if '.notdef' not in final_glyph_order:
         final_glyph_order.insert(0, '.notdef')
    else:
         if final_glyph_order[0] != '.notdef':
              final_glyph_order.remove('.notdef')
              final_glyph_order.insert(0, '.notdef')

    font.setGlyphOrder(final_glyph_order)
    print(f"Final Glyph Order ({len(final_glyph_order)} glyphs): {final_glyph_order}")

    font['maxp'].numGlyphs = len(final_glyph_order)

    # --- Save Font ---
    try:
        print(f"Saving font to {output_file}...")
        # reorderTables=None (default) is usually fine. True might optimize slightly.
        font.save(output_file, reorderTables=True) # Try reordering for potentially smaller file
        print("-" * 20)
        print("Font saved successfully!")
        print(f"Output file: {os.path.abspath(output_file)}")
        print("Don't forget to clear your OS font cache before testing!")
        print("-" * 20)
    except Exception as e:
        print(f"Error saving font file: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a monospaced TTF font from SVG glyphs.")
    parser.add_argument("-i", "--input_dir", default=DEFAULT_INPUT_DIR, help=f"Input directory containing SVG glyph files (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("-o", "--output_file", default=DEFAULT_OUTPUT_FILE, help=f"Output TTF font file path (default: {DEFAULT_OUTPUT_FILE})")
    parser.add_argument("--family", default=DEFAULT_FAMILY_NAME, help=f"Font Family Name (default: {DEFAULT_FAMILY_NAME})")
    parser.add_argument("--style", default=DEFAULT_STYLE_NAME, help=f"Font Style Name (e.g., Regular) (default: {DEFAULT_STYLE_NAME})")
    parser.add_argument("--version", default=DEFAULT_VERSION, help=f"Font version string (default: {DEFAULT_VERSION})")
    parser.add_argument("--copyright", default=DEFAULT_COPYRIGHT, help=f"Copyright string (default: {DEFAULT_COPYRIGHT})")
    parser.add_argument("--upm", type=int, default=DEFAULT_UNITS_PER_EM, help=f"Units Per Em (default: {DEFAULT_UNITS_PER_EM})")
    parser.add_argument("--ascender", type=int, default=DEFAULT_ASCENDER, help=f"Font Ascender (positive) (default: {DEFAULT_ASCENDER})")
    parser.add_argument("--descender", type=int, default=DEFAULT_DESCENDER, help=f"Font Descender (negative) (default: {DEFAULT_DESCENDER})")
    parser.add_argument("--capheight", type=int, default=DEFAULT_CAP_HEIGHT, help=f"Capital letter height (default: {DEFAULT_CAP_HEIGHT})")
    parser.add_argument("--xheight", type=int, default=DEFAULT_X_HEIGHT, help=f"x-height (default: {DEFAULT_X_HEIGHT})")
    parser.add_argument("--linegap", type=int, default=DEFAULT_LINE_GAP, help=f"Line Gap (default: {DEFAULT_LINE_GAP})")
    parser.add_argument("--width", type=int, default=DEFAULT_FIXED_ADVANCE, help=f"Fixed Advance Width (for monospace) (default: {DEFAULT_FIXED_ADVANCE})")

    args = parser.parse_args()

    # Basic validation
    if not os.path.isdir(args.input_dir):
         print(f"Error: Input directory not found: {args.input_dir}")
         exit(1)
    if args.descender > 0:
         print(f"Warning: Descender ({args.descender}) should typically be negative.")
    if args.ascender <= 0:
         print(f"Warning: Ascender ({args.ascender}) should typically be positive.")

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
         print(f"Creating output directory: {output_dir}")
         os.makedirs(output_dir)


    build_font(
        input_dir=args.input_dir,
        output_file=args.output_file,
        family_name=args.family,
        style_name=args.style,
        version=args.version,
        copyright_info=args.copyright,
        units_per_em=args.upm,
        ascender=args.ascender,
        descender=args.descender,
        cap_height=args.capheight,
        x_height=args.xheight,
        line_gap=args.linegap,
        fixed_advance=args.width
    )