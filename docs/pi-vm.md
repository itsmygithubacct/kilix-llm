# Pi in a persistent QEMU VM

Reference for [kilix-llm](../README.md).

Run [Pi](https://pi.dev/) and its coding tools inside a Debian 13 VM while
Qwen3.5 4B inference uses the host GPU. The VM mounts a persistent host folder
for projects, Pi sessions and settings, and uv-managed Python. The guest has
network access; the model API uses an authenticated host loopback endpoint
reached through QEMU's `10.0.2.2` host address.

This option requires an x86-64 Linux host with read/write access to `/dev/kvm`.
On Debian/Ubuntu, its host packages are `qemu-system-x86`, `qemu-utils`,
`xorriso`, `openssh-client`, and `curl`. Run the launcher as your regular
user. After building the Vulkan runtime and downloading the 4B model above:

```sh
./kilix-llm vm setup                    # optional: provision the guest first
./kilix-llm vm start --device Vulkan1    # also provisions it when needed
./kilix-llm vm pi --new-tab             # open Pi in a new Kilix tab
./kilix-llm vm pi                       # or use this terminal
./kilix-llm vm pi -- --continue         # continue the last Pi session
./kilix-llm vm shell                    # guest login shell
./kilix-llm vm shell -- uv run --managed-python --python 3.12 python --version
./kilix-llm vm status
./kilix-llm vm stop                     # shut down; retain disk and shared files
```

First boot downloads the image and installs guest tools, which takes several
minutes. The guest uses four virtual CPUs, 4 GiB RAM, and a 24 GiB QCOW2 disk
that grows as used. The selected GPU is remembered for later starts. Exit any
standalone session using this model before starting its VM server: two copies
exceed a 6 GiB GPU's capacity.

VM state defaults to `~/.local/gpu_terminal/kilix-llm/vm/`, follows
`$GPU_TERMINAL_HOME/kilix-llm/vm/`, and can be overridden with
`KILIX_LLM_VM_DIR`. Its `shared/` directory is mounted at `/persist` using 9p:

| Host path relative to the VM directory | Guest path | Contents |
|---|---|---|
| `shared/workspace/` | `/persist/workspace/` | Projects; Pi starts here |
| `shared/pi/` | `/persist/pi/` | Pi sessions, settings, model configuration, API key |
| `shared/cache/uv/` | `/persist/cache/uv/` | uv download and package cache |
| `shared/python/` | `/persist/python/` | uv-managed Python installations |

Files changed under `/persist` are immediately visible in the host folder
and survive guest shutdown. The guest user's home, `/var/lib/kilix`, and
installed OS packages persist on `disk.qcow2`. Only `shared/` is exported;
the host's model weights, VM control files, and SSH private key remain outside
that mount. Pi sessions can be resumed with `vm pi -- -c` or browsed with
`vm pi -- --resume`.

Both the launcher and guest Python workflows use uv. Guest shell profiles
select managed Python and persistent caches; run Python with
`uv run --managed-python --python 3.12 python ...`. Pi itself uses Node and
is installed through npm with `--ignore-scripts`. Provisioning pins the
[Debian 13 image](https://cloud.debian.org/images/cloud/trixie/20260831-2587/),
[Node 22.23.2](https://nodejs.org/dist/v22.23.2/),
[uv 0.12.13](https://github.com/astral-sh/uv/releases/tag/0.12.13), and Pi 0.85.1;
image and binary archive downloads are checksum-verified.

The Pi profile uses 8,192 context tokens, a 2,048-token response limit,
thinking off, and compaction settings sized for that context.
`vm start --ctx N` can select a larger context after `vm stop`, subject to
available GPU memory. For startup details, inspect `qemu.log` and `server.log`
in the VM directory, or guest provisioning output with
`./kilix-llm vm shell -- sudo tail -80 /var/log/cloud-init-output.log`.

Validated on 2026-09-13: Pi read a shared file, ran uv-managed Python to write
the correct result, and read it back. Workspace files and the saved Pi session
remained byte-for-byte identical after VM shutdown and restart. This run
measured about 0.7 generated tokens/s in both Pi and the direct host API;
the cause of the slowdown from the earlier short benchmark is unresolved.
