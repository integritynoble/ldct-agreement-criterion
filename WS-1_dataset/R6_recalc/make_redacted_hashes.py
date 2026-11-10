"""Write path-redacted copies of the R6 SHA-256 manifests.

The manifests record the integrity of the two data trees used by the paper, and of
the checkpoints the reported results were produced from. Their paths are absolute
locations on the machine that produced them, which names both a private working
directory and the layout of the restricted AAPM tree. This script replaces
everything before the tree root marker with `<LIDC_ROOT>/` or `<AAPM_ROOT>/`, and
reduces the checkpoint paths to their file name under `<CKPT_DIR>/`; hashes are
untouched, so every count the manuscript reports from a manifest is unchanged.

Run this before publishing the release clone:

    python3 WS-1_dataset/R6_recalc/make_redacted_hashes.py --check

`--check` only verifies that each redacted line parses back to the same hash and
the same tree-relative path as the committed manifest; without it the files are
written next to the originals as *_redacted.txt.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HASH_DIR = os.path.join(HERE, "hashes")

# (manifest, marker, placeholder, kind)
#   kind "tree": keep everything after `<marker>/`
#   kind "file": keep the file name only
TREES = [
    ("data_hashes.txt", "output_gpu", "<LIDC_ROOT>", "tree"),
    ("aapm_hashes.txt", "aapm_tree_v1", "<AAPM_ROOT>", "tree"),
    ("ckpt_hashes.txt", "checkpoints", "<CKPT_DIR>", "file"),
]


def redact_line(line: str, marker: str, placeholder: str, kind: str = "tree") -> str:
    parts = line.rstrip("\n").split(None, 1)
    if len(parts) != 2:
        raise SystemExit("unexpected manifest line: %r" % line)
    digest, path = parts
    path = path.strip().replace("\\", "/")
    if marker + "/" not in path:
        raise SystemExit("tree root %r not found in %r" % (marker, path))
    tail = path.split(marker + "/", 1)[1]
    if kind == "file":
        tail = tail.split("/")[-1]
        return "%s  %s/%s\n" % (digest, placeholder, tail)
    return "%s  %s/%s/%s\n" % (digest, placeholder, marker, tail)


def read(name: str):
    with open(os.path.join(HASH_DIR, name), encoding="utf-8") as f:
        return f.readlines()


def audit_redacted(path: str, marker: str, placeholder: str, kind: str = "tree") -> list:
    """Problems found in a redacted manifest, judged without the raw manifest.

    A release clone does not carry the raw manifests, so `--check` still has to be
    able to say something about the redacted copy it does carry: every line must
    keep the `digest  path` shape, name the tree root after the placeholder, and
    carry no absolute path, drive letter or home directory.
    """
    problems = []
    with open(path, encoding="utf-8-sig") as f:
        lines = f.readlines()
    prefix = placeholder + ("/" if kind == "file" else "/" + marker + "/")
    for i, line in enumerate(lines, 1):
        parts = line.rstrip("\n").split(None, 1)
        if len(parts) != 2:
            problems.append("line %d: not a 'digest  path' pair" % i)
            continue
        digest, p = parts
        if len(digest) != 64 or any(c not in "0123456789ABCDEFabcdef" for c in digest):
            problems.append("line %d: digest is not a SHA-256" % i)
        if not p.startswith(prefix):
            problems.append("line %d: path does not start with %s" % (i, prefix))
        if ":" in p or p.startswith("/") or p.startswith("<") and not p.startswith(placeholder):
            problems.append("line %d: path still looks absolute: %r" % (i, p))
        if "\\\\" in p:
            problems.append("line %d: path still uses Windows separators" % i)
        if kind == "file" and "/" in p[len(prefix):]:
            problems.append("line %d: checkpoint path is not a bare file name: %r" % (i, p))
    if not lines:
        problems.append("manifest is empty")
    return problems


def main() -> int:
    check = "--check" in sys.argv[1:]
    ok = True
    for name, marker, placeholder, kind in TREES:
        src = os.path.join(HASH_DIR, name)
        dst = os.path.join(HASH_DIR, name.replace(".txt", "_redacted.txt"))
        if check:
            if not os.path.isfile(dst):
                print("MISSING: %s; run make_redacted_hashes.py" % os.path.basename(dst),
                      file=sys.stderr)
                ok = False
                continue
            if not os.path.isfile(src):
                problems = audit_redacted(dst, marker, placeholder, kind)
                if problems:
                    print("BAD: %s" % os.path.basename(dst), file=sys.stderr)
                    for p in problems[:10]:
                        print("  " + p, file=sys.stderr)
                    ok = False
                else:
                    n = len(read(os.path.basename(dst)))
                    print("OK: %s (%d entries, no absolute path; the raw manifest is "
                          "not part of this tree, so only the redaction was checked)"
                          % (os.path.basename(dst), n))
                continue
            want = [redact_line(l, marker, placeholder, kind) for l in read(name)]
            got = read(os.path.basename(dst))
            if want != got:
                print("STALE: %s does not match the committed manifest"
                      % os.path.basename(dst), file=sys.stderr)
                ok = False
            else:
                print("OK: %s (%d entries, %s preserved)"
                      % (os.path.basename(dst), len(got),
                         "tree-relative paths" if kind == "tree" else "file names"))
            continue
        lines = [redact_line(l, marker, placeholder, kind) for l in read(name)]
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(lines)
        print("wrote %s (%d entries)" % (os.path.relpath(dst, HERE), len(lines)))
    if check:
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
