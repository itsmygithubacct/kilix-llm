"""Cloud-init provisioning for the persistent QEMU Pi guest."""

from __future__ import annotations

import json


NODE_VERSION = "22.23.2"
NODE_SHA256 = "d60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307"
UV_VERSION = "0.12.13"
UV_SHA256 = "745765a3b6e360ad76743599ae5c42e9278c7edf8bbff9fc76d05bf2623a04dd"
PI_VERSION = "0.85.1"


ENVIRONMENT = """# Kilix's managed tools and host-mounted Pi state.
export PATH="/opt/node/bin:/opt/pi/bin:/opt/uv/bin:$PATH"
export PI_CODING_AGENT_DIR=/persist/pi
export PI_OFFLINE=1
export UV_CACHE_DIR=/persist/cache/uv
export UV_PYTHON_INSTALL_DIR=/persist/python
export UV_MANAGED_PYTHON=1
export UV_LINK_MODE=copy
if [ -r /persist/pi/api-key ]; then
    KILIX_LLM_API_KEY=$(cat /persist/pi/api-key)
    export KILIX_LLM_API_KEY
fi
"""


RUNNER = """#!/bin/bash
set -euo pipefail
if ! mountpoint -q /persist; then
    echo 'kilix-pi: the persistent host folder is not mounted at /persist' >&2
    exit 1
fi
if [ ! -f /var/lib/kilix-llm/ready ]; then
    echo 'kilix-pi: guest setup is incomplete; inspect /var/log/cloud-init-output.log' >&2
    exit 1
fi
source /etc/profile.d/kilix-llm.sh
cd /persist/workspace
exec /opt/pi/bin/pi "$@"
"""


def _bootstrap() -> str:
    pins = (
        f"NODE_VERSION={NODE_VERSION}\nNODE_SHA256={NODE_SHA256}\n"
        f"UV_VERSION={UV_VERSION}\nUV_SHA256={UV_SHA256}\n"
        f"PI_VERSION={PI_VERSION}\n"
    )
    return "#!/bin/bash\nset -euo pipefail\n" + pins + r"""
if ! mountpoint -q /persist; then
    echo 'Kilix setup stopped: /persist must be the mounted host folder.' >&2
    exit 1
fi
install -d -m 0755 /var/lib/kilix-llm /opt/node /opt/pi /opt/uv/bin
if ! command -v fd >/dev/null 2>&1; then
    ln -s /usr/bin/fdfind /usr/local/bin/fd
fi
rm -f /var/lib/kilix-llm/ready
install -d -o kilix -g kilix -m 0755 /persist/workspace
install -d -o kilix -g kilix -m 0700 /persist/pi /persist/cache /persist/cache/uv /persist/python
kilix_download_dir=$(mktemp -d /var/tmp/kilix-guest.XXXXXXXX)
trap 'rm -rf "$kilix_download_dir"' EXIT
cd "$kilix_download_dir"
curl --fail --location --retry 3 --output node.tar.xz \
    "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz"
printf '%s  node.tar.xz\n' "$NODE_SHA256" | sha256sum --check --status
tar --extract --xz --file node.tar.xz --directory /opt/node --strip-components=1
curl --fail --location --retry 3 --output uv.tar.gz \
    "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
printf '%s  uv.tar.gz\n' "$UV_SHA256" | sha256sum --check --status
tar --extract --gzip --file uv.tar.gz --directory /opt/uv/bin --strip-components=1
export PATH="/opt/node/bin:/opt/pi/bin:/opt/uv/bin:$PATH"
npm install --global --prefix /opt/pi --ignore-scripts --no-audit --no-fund \
    "@earendil-works/pi-coding-agent@${PI_VERSION}"
# uv reads configuration from its working directory; leave the root-only temp.
cd /persist/workspace
as_kilix=(runuser -u kilix -- env
    "PATH=$PATH"
    UV_CACHE_DIR=/persist/cache/uv
    UV_PYTHON_INSTALL_DIR=/persist/python
    UV_MANAGED_PYTHON=1
    UV_LINK_MODE=copy)
"${as_kilix[@]}" uv python install 3.12
"${as_kilix[@]}" uv run --managed-python --python 3.12 --no-project python -c \
    'import sys; print("Kilix managed Python:", sys.version.split()[0])'
/opt/node/bin/node --version
/opt/uv/bin/uv --version
PI_OFFLINE=1 /opt/pi/bin/pi --version
touch /var/lib/kilix-llm/ready
echo 'Kilix Pi guest is ready; persistent workspace: /persist/workspace'
"""


def cloud_config(public_key: str, uid: int, gid: int) -> str:
    """Render first-boot setup; the host supplies the writable 9p share."""
    public_key = public_key.strip()
    if not public_key or "\n" in public_key or "\r" in public_key:
        raise ValueError("a single SSH public key is required")
    if any(type(value) is not int or value <= 0 for value in (uid, gid)):
        raise ValueError("the guest requires positive host UID and GID values")
    config = {
        "hostname": "kilix-pi",
        "manage_etc_hosts": True,
        "ssh_pwauth": False,
        "disable_root": True,
        "bootcmd": [
            ["groupadd", "--force", "--non-unique", "--gid", str(gid), "kilix"],
            ["modprobe", "9pnet_virtio"],
            ["modprobe", "9p"],
        ],
        "users": [
            {
                "name": "kilix",
                "uid": uid,
                "primary_group": "kilix",
                "groups": ["sudo"],
                "homedir": "/var/lib/kilix",
                "shell": "/bin/bash",
                "lock_passwd": True,
                "sudo": ["ALL=(ALL) NOPASSWD:ALL"],
                "ssh_authorized_keys": [public_key],
            }
        ],
        "mounts": [
            [
                "kilix-persist",
                "/persist",
                "9p",
                "trans=virtio,version=9p2000.L,msize=262144,cache=none,_netdev",
                "0",
                "0",
            ]
        ],
        "package_update": True,
        "packages": [
            "curl", "ca-certificates", "git", "build-essential", "xz-utils",
            "sudo", "openssh-server", "ripgrep", "fd-find",
        ],
        "write_files": [
            {
                "path": "/etc/profile.d/kilix-llm.sh",
                "permissions": "0644",
                "content": ENVIRONMENT,
            },
            {
                "path": "/usr/local/bin/kilix-pi",
                "permissions": "0755",
                "content": RUNNER,
            },
            {
                "path": "/usr/local/sbin/kilix-guest-setup",
                "permissions": "0755",
                "content": _bootstrap(),
            },
            {
                "path": "/etc/modules-load.d/kilix-persist.conf",
                "permissions": "0644",
                "content": "9pnet_virtio\n9p\n",
            },
        ],
        "runcmd": [["/usr/local/sbin/kilix-guest-setup"]],
        "final_message": "Kilix guest cloud-init finished after $UPTIME seconds.",
    }
    return "#cloud-config\n" + json.dumps(config, indent=2) + "\n"
