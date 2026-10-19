"""Create a dedicated cloud branch without changing the private local server config."""
import subprocess
from _common import ROOT


def git(*args, input=None):
    result = subprocess.run(["git", *args], cwd=ROOT, input=input, text=True, encoding="utf-8", capture_output=True, check=True)
    return result.stdout.strip()


def tree_entries(ref):
    result = []
    for line in git("ls-tree", ref).splitlines():
        metadata, name = line.split("\t", 1)
        mode, kind, sha = metadata.split()
        result.append([mode, kind, sha, name])
    return result


if __name__ == "__main__":
    head = git("rev-parse", "main")
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
    commit = git("commit-tree", root_tree, "-p", head, input="Configure the public-example-only Streamlit deployment branch\n")
    # Only a new branch is created; an existing branch is never silently moved.
    git("update-ref", "refs/heads/streamlit-demo", commit, "0" * 40)
    print(f"Created streamlit-demo at {commit}; working directory remains on main.")
