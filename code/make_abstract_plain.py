"""LaTeX abstract -> paste-safe plain text (+ RTF) for the submission portal."""
import os, re, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN  = os.path.join(ROOT, "Batteries_MDPI", "manuscript")
SUB  = os.path.join(ROOT, "Batteries_MDPI", "SUBMISSION")

tex = open(os.path.join(MAN, "batteries.tex")).read()
nums = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(os.path.join(MAN, "numbers.tex")).read()))
a = tex.index(r"\abstract{") + len(r"\abstract{")
depth, i = 1, a
while depth:
    if tex[i] == "{": depth += 1
    elif tex[i] == "}": depth -= 1
    i += 1
s = tex[a:i-1]
for k, v in nums.items():                       # expand our own macros
    s = s.replace("\\" + k + "{}", v).replace("\\" + k, v)
s = re.sub(r"\\SI\{([^}]*)\}\{\\?(\w+)\}", lambda m: m.group(1) + {"percent": "%", "milli\\volt": " mV",
           "volt": " V", "second": " s", "ampere\\hour": " Ah"}.get(m.group(2), " " + m.group(2)), s)
s = s.replace(r"\SI{150}{\milli\volt}", "150 mV").replace(r"\SI{450}{\milli\volt}", "450 mV")
s = re.sub(r"\\SI\{([^}]*)\}\{[^}]*\}", r"\1", s)
s = s.replace("---", "—").replace("--", "–").replace(r"\%", "%").replace("$N$", "N")
s = re.sub(r"\$([^$]*)\$", r"\1", s)
s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
s = re.sub(r"\\[a-zA-Z]+", "", s)
s = re.sub(r"[{}]", "", s)
s = re.sub(r"\s+", " ", s).strip()
open(os.path.join(SUB, "abstract_plain.txt"), "w").write(s + "\n")
rtf = "{\\rtf1\\ansi\\ansicpg1252\\deff0{\\fonttbl{\\f0 Times New Roman;}}\\fs22 " + \
      s.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}") + "}"
open(os.path.join(SUB, "abstract_plain.rtf"), "w").write(rtf)
print(f"abstract: {len(s.split())} words -> SUBMISSION/abstract_plain.txt/.rtf")
