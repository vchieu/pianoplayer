#
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:22:20 2015

@author: marco musy
"""


def nameof(n):
    a = n.name + str(n.octave)
    if "--" in a:                 # order matters
        b = a.replace("B--", "A")
        b = b.replace("A--", "G") # chain b.replace not a.replace
        b = b.replace("G--", "F")
        b = b.replace("E--", "D")
        b = b.replace("D--", "C")
        return b
    elif "-" in a:
        b = a.replace("C-", "B")
        b = b.replace("D-", "C#")  # chain b.replace not a.replace
        b = b.replace("E-", "D#")
        b = b.replace("F-", "E")
        b = b.replace("G-", "F#")
        b = b.replace("A-", "G#")
        b = b.replace("B-", "A#")
        return b
    elif "##" in a:
        b = a.replace("C##", "D")
        b = b.replace("D##", "E") # chain b.replace not a.replace
        b = b.replace("F##", "G")
        b = b.replace("G##", "A")
        b = b.replace("A##", "B")
        return b
    elif "E#" in a:
        b = a.replace("E#", "F")
        return b
    elif "B#" in a:
        b = a.replace("B#", "C")
        return b
    else:
        return a


def fpress(f, color):
    f.rotate(-20, axis=(1, 0, 0), point=f.pos())
    f.addPos([0, 0, -1])
    f.color(color)


def frelease(f):
    f.addPos([0, 0, 1])
    f.rotate(20, axis=(1, 0, 0), point=f.pos())
    f.color((0.7, 0.3, 0.3))


def kpress(f, color):
    f.rotate(4, axis=(1, 0, 0), point=f.pos())
    f.addPos([0, 0, -0.4])
    f.color(color)


def krelease(f):
    f.addPos([0, 0, 0.4])
    p = f.pos()
    f.rotate(-4, axis=(1, 0, 0), point=p)
    if p[2] > 0.5:
        f.color("k")
    else:
        f.color("w")


_kb_layout = {
    "C"  : 0.5,
    "D"  : 1.5,
    "E"  : 2.5,
    "F"  : 3.5,
    "G"  : 4.5,
    "A"  : 5.5,
    "B"  : 6.5,
    "B#" : 0.5,
    "C#" : 1.0,
    "D#" : 2.0,
    "E#" : 3.5,
    "F#" : 4.0,
    "G#" : 5.0,
    "A#" : 6.0,
    "C-" : 6.5,
    "D-" : 1.0,
    "E-" : 2.0,
    "F-" : 2.5,
    "G-" : 4.0,
    "A-" : 5.0,
    "B-" : 6.0,
    "C##": 1.5,
    "D##": 2.5,
    "F##": 4.5,
    "G##": 5.5,
    "A##": 6.5,
    "D--": 0.5,
    "E--": 1.5,
    "G--": 3.5,
    "A--": 4.5,
    "B--": 5.5,
}


def keypos(n):  # position of notes on keyboard
    step = 0.0
    keybsize = 16.5  # cm
    k = keybsize / 7.0  # 7 notes
    if n.name in _kb_layout.keys():
        step = _kb_layout[n.name] * k
    else:
        print("ERROR note not found", n.name)
    return keybsize * n.octave + step


def handSizeFactor(s):
    f = 0.82
    if s == "XXS":
        f = 0.33
    elif s == "XS":
        f = 0.46
    elif s == "S":
        f = 0.64
    elif s == "M":
        f = 0.82
    elif s == "L":
        f = 1.0
    elif s == "XL":
        f = 1.1
    elif s == "XXL":
        f = 1.2
    return f


#-------------------------------------------------------------------------------
# keyboard geometry shared by the 3D keyboard and by the hand rig
#-------------------------------------------------------------------------------
KEYBSIZE = 16.5          # cm, span of one octave (7 white keys)
KEY_WB = KEYBSIZE / 7.0  # cm, width of a white key
WHITE_TOP = 0.5          # z of the top surface of a white key
BLACK_TOP = 1.5          # z of the top surface of a black key
KEY_PIVOT_Y = 4.0        # y of the back edge: keys hinge around it
KEY_PRESS_ANGLE = 5.0    # deg, white key dip when fully pressed
BLACK_KEY_PRESS_ANGLE = 7.0
KEY_PRESS_DEPTH = 0.05   # sink of the fingertip pad into the key [cm]

# contact point of a fingertip on the key surface (absolute keyboard coords)
CONTACT_Y_WHITE = {"thumb": -5.2, "index": -3.0, "middle": -3.0,
                   "ring": -3.0, "little": -3.4}
CONTACT_Y_BLACK = {"thumb": -3.0, "index": -1.5, "middle": -1.5,
                   "ring": -1.5, "little": -1.9}


def key_is_black(x) -> bool:
    """True if the keyboard position x (cm) falls on a black key."""
    import math
    u = (x % KEYBSIZE) / KEY_WB      # 0..7 inside the octave
    m = int(math.floor(u - 0.7))
    if m < 0:
        return False
    return m in (0, 1, 3, 4, 5) and u < m + 1.3


def key_top(x) -> float:
    """z of the (unpressed) key surface at keyboard position x."""
    return BLACK_TOP if key_is_black(x) else WHITE_TOP


def key_angle(x) -> float:
    """Degrees the key tilts when fully pressed."""
    return BLACK_KEY_PRESS_ANGLE if key_is_black(x) else KEY_PRESS_ANGLE


def key_drop(x, y, amount=1.0) -> float:
    """
    How much the key surface sank at depth `y` when pressed by `amount`
    (0..1).  Keys hinge on their back edge, so the drop grows linearly
    from the pivot towards the player.
    """
    import math
    a = key_angle(x) * float(min(max(amount, 0.0), 1.0))
    return max(KEY_PIVOT_Y - y, 0.0) * math.sin(math.radians(a))


def score_tempo(sf, default=72.0):
    """Tempo (BPM) of a music21 score, or `default` when not found."""
    try:
        from music21.tempo import MetronomeMark, TempoIndication
        for cls in (MetronomeMark, TempoIndication):
            for mark in sf.recurse().getElementsByClass(cls):
                number = getattr(mark, "number", None)
                if number and float(number) > 0:
                    return float(number)
    except Exception:
        pass
    return float(default)
