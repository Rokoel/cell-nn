"""Recognize declared membrane stains; colors alone never determine roles."""
import re
from pathlib import Path


def membrane_stain_channels(path, channels):
    # Filenames in this collection identify stains by excitation/emission label.
    wavelengths=[int(v) for v in re.findall(r'(?:lam(?:in)?|lbr)[_-]?(405|488|514|543|546|561|633|635|640|644|647)',Path(path).stem,re.I)]
    if not wavelengths:return []
    matches=[]
    for i,c in enumerate(channels):
        excitation=c.excitation_nm
        if excitation is not None and any(abs(excitation-v)<=20 for v in wavelengths):matches.append(i)
    return matches
