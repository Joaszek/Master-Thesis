#!/bin/bash

################################################################################
# Prime Intellect - Automated Deployment Script for Elliptic2 GNN Training
################################################################################
#
# This script automates the complete setup process:
# 0. Clones project from GitHub
# 1. Creates folder structure
# 2. Installs kagglehub
# 3. Downloads Elliptic2 dataset
# 4. Moves data to correct location
# 5. Installs Python dependencies
# 6. Runs preprocessing
# 7. Starts training
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
#
################################################################################

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GITHUB_REPO="https://github.com/Joaszek/Master-Thesis.git"
PROJECT_ROOT="/ephemeral/elliptic2"
DATA_RAW="${PROJECT_ROOT}/data/raw"
DATA_PROCESSED="${PROJECT_ROOT}/data/processed"
CHECKPOINTS="${PROJECT_ROOT}/checkpoints"
RESULTS="${PROJECT_ROOT}/results"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

################################################################################
# Step 1: System Check
################################################################################

print_header "Step 1: System Check"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root"
   exit 1
fi

# Check CUDA
print_info "Checking CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. CUDA not available?"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
print_success "CUDA available"

# Check Python
print_info "Checking Python..."
python3 --version
print_success "Python available"

# Check Git
print_info "Checking Git..."
if ! command -v git &> /dev/null; then
    print_error "Git not found. Installing..."
    apt-get update && apt-get install -y git
fi
print_success "Git available"

# Check disk space
print_info "Checking disk space..."
df -h /ephemeral
AVAILABLE_GB=$(df -BG /ephemeral | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt 100 ]; then
    print_error "Less than 100GB available. Need more space!"
    exit 1
fi
print_success "Sufficient disk space: ${AVAILABLE_GB}GB available"

################################################################################
# Step 2: Clone Project from GitHub
################################################################################

print_header "Step 2: Clone Project from GitHub"

cd /ephemeral

# Check if project already exists
if [ -d "${PROJECT_ROOT}" ]; then
    print_info "Project directory already exists at ${PROJECT_ROOT}"
    read -p "Do you want to pull latest changes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "${PROJECT_ROOT}"
        print_info "Pulling latest changes..."
        git pull origin main || git pull origin master
        print_success "Repository updated"
    else
        print_info "Using existing code"
    fi
else
    print_info "Cloning repository from ${GITHUB_REPO}..."
    git clone "${GITHUB_REPO}" "${PROJECT_ROOT}"
    print_success "Repository cloned successfully"
fi

cd "${PROJECT_ROOT}"
print_info "Current branch: $(git branch --show-current)"
print_info "Latest commit: $(git log -1 --oneline)"

################################################################################
# Step 3: Create Additional Folder Structure
################################################################################

print_header "Step 3: Creating Additional Folder Structure"

print_header "Step 3: Creating Additional Folder Structure"

# Create data/runtime folders that aren't in git
mkdir -p "${DATA_RAW}"
mkdir -p "${DATA_PROCESSED}"
mkdir -p "${CHECKPOINTS}"
mkdir -p "${RESULTS}"

print_success "Additional folders created:"
tree -L 3 "${PROJECT_ROOT}" 2>/dev/null || find "${PROJECT_ROOT}" -maxdepth 3 -type d

################################################################################
# Step 4: Install kagglehub
################################################################################

print_header "Step 4: Installing kagglehub"

print_info "Installing kagglehub..."
pip install kagglehub --quiet
print_success "kagglehub installed"

################################################################################
# Step 5: Download Elliptic2 Dataset
################################################################################

print_header "Step 5: Downloading Elliptic2 Dataset"

print_info "This may take several minutes (dataset is ~77GB)..."

# Create Python script for downloading
cat > /tmp/download_elliptic2.py << 'DOWNLOAD_SCRIPT'
import kagglehub
import shutil
import os

print("Downloading Elliptic2 dataset from Kaggle...")
try:
    path = kagglehub.dataset_download("ellipticco/elliptic2-data-set")
    print(f"Dataset downloaded to: {path}")

    # Copy to target directory
    target = "/ephemeral/elliptic2/data/raw"
    os.makedirs(target, exist_ok=True)

    print(f"Copying files to {target}...")
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(target, item)

        if os.path.isfile(src):
            shutil.copy2(src, dst)
            size_gb = os.path.getsize(dst) / (1024**3)
            print(f"  Copied: {item} ({size_gb:.2f} GB)")
        elif os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  Copied directory: {item}")

    print("Dataset ready!")

except Exception as e:
    print(f"Error downloading dataset: {e}")
    exit(1)
DOWNLOAD_SCRIPT

python3 /tmp/download_elliptic2.py

if [ $? -eq 0 ]; then
    print_success "Dataset downloaded and moved to ${DATA_RAW}"
else
    print_error "Failed to download dataset"
    exit 1
fi

# Show what was downloaded
print_info "Files in ${DATA_RAW}:"
ls -lh "${DATA_RAW}"

################################################################################
# Step 6: Install Python Requirements
################################################################################

print_header "Step 6: Installing Python Requirements"

cd "${PROJECT_ROOT}"

if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found in ${PROJECT_ROOT}"
    print_info "Make sure your GitHub repo contains requirements.txt"
    exit 1
fi

print_info "Installing requirements..."
pip install -r requirements.txt

print_success "Python dependencies installed"

################################################################################
# Step 7: Verify Configuration
################################################################################

print_header "Step 7: Verifying Configuration"

if [ ! -f "config.yaml" ]; then
    print_error "config.yaml not found in ${PROJECT_ROOT}"
    exit 1
fi

print_info "Current config.yaml paths:"
grep -A 5 "paths:" config.yaml || echo "Could not parse config.yaml"

# Update config.yaml to use correct paths
print_info "Updating config.yaml paths..."

python3 << 'UPDATE_CONFIG'
import yaml

config_path = "/ephemeral/elliptic2/config.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Update paths
if 'paths' not in config:
    config['paths'] = {}

config['paths']['raw_data'] = "/ephemeral/elliptic2/data/raw"
config['paths']['processed_data'] = "/ephemeral/elliptic2/data/processed"
config['paths']['checkpoints'] = "/ephemeral/elliptic2/checkpoints"
config['paths']['results'] = "/ephemeral/elliptic2/results"

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("Config updated successfully")
UPDATE_CONFIG

print_success "Configuration verified and updated"

################################################################################
# Step 8: Run Preprocessing
################################################################################

print_header "Step 8: Running Preprocessing"

cd "${PROJECT_ROOT}"

# Check if preprocessing script exists
PREPROCESS_SCRIPT=""
if [ -f "src/preprocess/preprocess.py" ]; then
    PREPROCESS_SCRIPT="src/preprocess/preprocess.py"
elif [ -f "preprocess.py" ]; then
    PREPROCESS_SCRIPT="preprocess.py"
else
    print_error "Preprocessing script not found!"
    print_info "Looking for: src/preprocess/preprocess.py or preprocess.py"
    exit 1
fi

print_info "Running preprocessing: ${PREPROCESS_SCRIPT}"
print_info "This may take 30-60 minutes depending on data size..."

# Run preprocessing with output
python3 "${PREPROCESS_SCRIPT}" 2>&1 | tee "${RESULTS}/preprocessing.log"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_success "Preprocessing completed successfully"
    print_info "Preprocessing log saved to: ${RESULTS}/preprocessing.log"
else
    print_error "Preprocessing failed! Check log: ${RESULTS}/preprocessing.log"
    exit 1
fi

# Show what was created
print_info "Processed data:"
ls -lh "${DATA_PROCESSED}"

################################################################################
# Step 9: Start Training
################################################################################

print_header "Step 9: Starting Training"

cd "${PROJECT_ROOT}"

# Check if train script exists
TRAIN_SCRIPT=""
if [ -f "src/train/train.py" ]; then
    TRAIN_SCRIPT="src/train/train.py"
elif [ -f "train.py" ]; then
    TRAIN_SCRIPT="train.py"
else
    print_error "Training script not found!"
    print_info "Looking for: src/train/train.py or train.py"
    exit 1
fi

print_info "Starting training: ${TRAIN_SCRIPT}"
print_info "Training will run in background with output to: ${RESULTS}/training.log"

# Start training in background
nohup python3 "${TRAIN_SCRIPT}" > "${RESULTS}/training.log" 2>&1 &
TRAIN_PID=$!

echo $TRAIN_PID > "${PROJECT_ROOT}/train.pid"

print_success "Training started with PID: ${TRAIN_PID}"
print_info "PID saved to: ${PROJECT_ROOT}/train.pid"

################################################################################
# Final Instructions
################################################################################

print_header "Deployment Complete!"

echo ""
echo -e "${GREEN}All steps completed successfully!${NC}"
echo ""
echo "Training is now running in the background."
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo ""
echo "  Monitor GPU:"
echo "    watch -n 1 nvidia-smi"
echo ""
echo "  View training logs:"
echo "    tail -f ${RESULTS}/training.log"
echo ""
echo "  Check training process:"
echo "    ps aux | grep train.py"
echo ""
echo "  Kill training:"
echo "    kill \$(cat ${PROJECT_ROOT}/train.pid)"
echo ""
echo "  Backup checkpoints (from LOCAL computer):"
echo "    scp -P 1234 -r root@160.211.45.163:${CHECKPOINTS} ./backup/"
echo ""
echo -e "${YELLOW}Important:${NC}"
echo "  - Checkpoints are saved to: ${CHECKPOINTS}"
echo "  - Results/logs are in: ${RESULTS}"
echo "  - Remember to backup regularly!"
echo "  - /ephemeral is temporary - data lost on instance termination!"
echo ""
echo -e "${BLUE}Happy training! 🚀${NC}"
echo ""

# Show first few lines of training log
sleep 2
echo -e "${YELLOW}Initial training output:${NC}"
head -n 20 "${RESULTS}/training.log"
echo ""
echo "To continue monitoring: tail -f ${RESULTS}/training.log"