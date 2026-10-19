"""Create a dedicated cloud branch without changing the private local server config."""
import argparse
import subprocess
from _common import ROOT


def git(*args, input=None):
    # Binary pipes keep Windows from adding CR characters to Git tree filenames.
    payload = input.encode("utf-8") if input is not None else None
    result = subprocess.run(["git", *args], cwd=ROOT, input=payload, capture_output=True, check=True)
    return result.stdout.decode("utf-8").strip()


def tree_entries(ref):
    result = []
    for line in git("ls-tree", ref).splitlines():
        metadata, name = line.split("\t", 1)
        mode, kind, sha = metadata.split()
        result.append([mode, kind, sha, name])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="Refresh the derived branch from main, preserving its history")
    args = parser.parse_args()
    head = git("rev-parse", "main")
    previous = git("rev-parse", "streamlit-demo") if args.update else "0" * 40
    config = git("show", "main:.streamlit/config.toml")
    if 'address = "127.0.0.1"' not in config:
        raise ValueError("Expected the private local server binding before deriving cloud configuration.")
    cloud = config.replace('address = "127.0.0.1"', 'address = "0.0.0.0"') + "\n"
    blob = git("hash-object", "-w", "--stdin", input=cloud)
    entries = tree_entries("main:.streamlit")
    for entry in entries:
        if entry[3] == "config.toml":
            entry[2] = blob
    streamlit_tree = git("mktree", input="\n".join(f"{mode} {kind} {sha}\t{name}" for mode, kind, sha, name in entries) + "\n")
    root_entries = tree_entries("main")
    for entry in root_entries:
        if entry[3] == ".streamlit":
            entry[2] = streamlit_tree
    root_tree = git("mktree", input="\n".join(f"{mode} {kind} {sha}\t{name}" for mode, kind, sha, name in root_entries) + "\n")
    if git("diff-tree", "--no-commit-id", "--name-only", "-r", head, root_tree) != ".streamlit/config.toml":
        raise ValueError("Cloud tree must differ from main only in the Streamlit configuration.")
    parents = ["-p", head]
    if args.update and previous != head:
        parents.extend(["-p", previous])
    commit = git("commit-tree", root_tree, *parents, input="Configure the public-example-only Streamlit deployment branch\n")
    # Compare-and-swap prevents silently overwriting a concurrently changed branch.
    git("update-ref", "refs/heads/streamlit-demo", commit, previous)
    print(f"Prepared streamlit-demo at {commit}; working directory remains on main.")
