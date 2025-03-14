#!/bin/bash
set -x
set -e

# Function to display usage instructions
usage() {
    echo "Usage: $0 -r <repository_url> [-b <branch_name>] [-d <clone_dir>]"
    exit 1
}

# Default values
default_branch_name="main"
default_clone_dir="eval-tool"

# Parse command line arguments
while getopts "r:b:d:" opt; do
    case "$opt" in
        r) repo_url=$OPTARG ;;
        b) branch_name=$OPTARG ;;
        d) clone_dir=$OPTARG ;;
        *) usage ;;
    esac
done

# Use default values if not provided
branch_name=${branch_name:-$default_branch_name}
clone_dir=${clone_dir:-$default_clone_dir}

# Clone the repository if it doesn't exist, otherwise ensure the desired branch is checked out
if [ ! -d "$clone_dir/.git" ]; then
    git clone --recurse-submodules --branch "$branch_name" "$repo_url" "$clone_dir"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to clone the repository."
        exit 1
    fi
    echo "Repository cloned successfully into $clone_dir."
else
    cd "$clone_dir" || exit
    git fetch --all
    git checkout "$branch_name"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to checkout branch $branch_name."
        exit 1
    fi
    git pull
    git submodule update --init --recursive
    cd ..
    echo "Repository already exists. Checked out and updated branch $branch_name."
fi

# Install dependencies
cd "$clone_dir" || exit
(sudo apt-get update || apt-get update) && (sudo apt install -y python3.12-venv lsof || apt install -y python3.12-venv lsof)
rm -rf .venv || true
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install sentencepiece
deactivate
echo "Dependencies installed."

# Create config files
mkdir -p ~/.nvevaltool
echo '{}' > ~/.nvevaltool/user_data.json
cat > ~/.nvevaltool/user_config.json <<EOL
{
    "resource": {
        "local": {
            "infra_type": "local",
            "infra_name": "local"
        }
    }
}
EOL

echo "Configuration file created at ~/.nvevaltool/user_config.json."
echo "evaltool setup complete."
exit 0
