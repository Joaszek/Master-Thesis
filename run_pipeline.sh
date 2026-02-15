#!/bin/bash

################################################################################
# Fix Disk Space and Download Elliptic2 Dataset
################################################################################
# 
# This script fixes the "No space left on device" error by redirecting
# all cache and temporary files to /ephemeral instead of the small overlay disk.
#
# Usage:
#   chmod +x fix_disk_and_download.sh
#   ./fix_disk_and_download.sh
#
################################################################################

set -e  # Exit on any error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
# Step 1: Check Current Disk Usage
################################################################################

print_header "Step 1: Checking Disk Usage"

df -h
echo ""

OVERLAY_USAGE=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
EPHEMERAL_USAGE=$(df -h /ephemeral | tail -1 | awk '{print $5}' | sed 's/%//')

print_info "Overlay (/) usage: ${OVERLAY_USAGE}%"
print_info "Ephemeral usage: ${EPHEMERAL_USAGE}%"

################################################################################
# Step 2: Clean Up Overlay Disk
################################################################################

print_header "Step 2: Cleaning Up Overlay Disk"

print_info "Removing old kagglehub cache..."
rm -rf /root/.cache/kagglehub 2>/dev/null || true
print_success "Kagglehub cache removed"

print_info "Cleaning /tmp..."
rm -rf /tmp/* 2>/dev/null || true
print_success "/tmp cleaned"

print_info "Cleaning apt cache..."
apt-get clean 2>/dev/null || true
print_success "Apt cache cleaned"

print_info "Cleaning pip cache..."
pip cache purge 2>/dev/null || true
print_success "Pip cache cleaned"

echo ""
print_info "Disk usage after cleanup:"
df -h /
echo ""

################################################################################
# Step 3: Setup /ephemeral for Cache
################################################################################

print_header "Step 3: Setting Up /ephemeral for Cache"

print_info "Creating cache directories on /ephemeral..."
mkdir -p /ephemeral/kagglehub_cache
mkdir -p /ephemeral/.cache
mkdir -p /ephemeral/tmp
mkdir -p /root/.cache
print_success "Cache directories created"

print_info "Creating symbolic link..."
ln -sf /ephemeral/kagglehub_cache /root/.cache/kagglehub
print_success "Symbolic link created: /root/.cache/kagglehub -> /ephemeral/kagglehub_cache"

################################################################################
# Step 4: Set Environment Variables
################################################################################

print_header "Step 4: Setting Environment Variables"

export KAGGLE_CACHE_DIR=/ephemeral/kagglehub_cache
export XDG_CACHE_HOME=/ephemeral/.cache
export TMPDIR=/ephemeral/tmp

print_success "Environment variables set for current session"

# Add to .bashrc for persistence
if ! grep -q "KAGGLE_CACHE_DIR" /root/.bashrc; then
    print_info "Adding to /root/.bashrc for persistence..."
    cat >> /root/.bashrc << 'EOF'

# Cache directories on /ephemeral to avoid disk space issues
export KAGGLE_CACHE_DIR=/ephemeral/kagglehub_cache
export XDG_CACHE_HOME=/ephemeral/.cache
export TMPDIR=/ephemeral/tmp
EOF
    print_success "Added to .bashrc"
else
    print_info ".bashrc already configured"
fi

################################################################################
# Step 5: Install kagglehub
################################################################################

print_header "Step 5: Installing kagglehub"

if python3 -c "import kagglehub" 2>/dev/null; then
    print_info "kagglehub already installed"
else
    print_info "Installing kagglehub..."
    pip install kagglehub --quiet
    print_success "kagglehub installed"
fi

################################################################################
# Step 6: Download Elliptic2 Dataset
################################################################################

print_header "Step 6: Downloading Elliptic2 Dataset"

# Determine project root (current directory or /ephemeral/Master-Thesis)
if [ -f "config.yaml" ]; then
    PROJECT_ROOT=$(pwd)
elif [ -f "/ephemeral/Master-Thesis/config.yaml" ]; then
    PROJECT_ROOT="/ephemeral/Master-Thesis"
elif [ -f "/ephemeral/elliptic2/config.yaml" ]; then
    PROJECT_ROOT="/ephemeral/elliptic2"
else
    print_error "Cannot find project root with config.yaml"
    print_info "Please cd to your project directory and run this script again"
    exit 1
fi

print_info "Project root: ${PROJECT_ROOT}"

RAW_DATA="${PROJECT_ROOT}/data/raw"
mkdir -p "${RAW_DATA}"

print_info "Downloading dataset to /ephemeral (this will take several minutes)..."
print_info "Dataset size: ~77GB"
echo ""

# Create Python download script
cat > /tmp/download_elliptic2.py << 'DOWNLOAD_SCRIPT'
import kagglehub
import shutil
import os
import sys

# Force all cache to /ephemeral
os.environ['KAGGLE_CACHE_DIR'] = '/ephemeral/kagglehub_cache'
os.environ['XDG_CACHE_HOME'] = '/ephemeral/.cache'
os.environ['TMPDIR'] = '/ephemeral/tmp'

try:
    print("Downloading Elliptic2 dataset from Kaggle...")
    print("This may take 10-20 minutes depending on connection speed...")
    print("")
    
    path = kagglehub.dataset_download("ellipticco/elliptic2-data-set")
    print(f"\nDataset downloaded to: {path}")
    
    # Get target directory from command line arg
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "/ephemeral/Master-Thesis/data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    print(f"\nCopying files to {raw_dir}...")
    copied_files = []
    
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(raw_dir, item)
        
        if os.path.isfile(src):
            print(f"  Copying {item}...")
            shutil.copy2(src, dst)
            size_gb = os.path.getsize(dst) / (1024**3)
            copied_files.append((item, size_gb))
        elif os.path.isdir(src):
            print(f"  Copying directory {item}...")
            shutil.copytree(src, dst, dirs_exist_ok=True)
            copied_files.append((item, 0))
    
    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print("\nFiles in raw data folder:")
    for name, size in copied_files:
        if size > 0:
            print(f"  {name:40s} {size:6.2f} GB")
        else:
            print(f"  {name:40s} (directory)")
    
    print(f"\nTotal location: {raw_dir}")
    
except Exception as e:
    print(f"\nError downloading dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
DOWNLOAD_SCRIPT

# Run download script
python3 /tmp/download_elliptic2.py "${RAW_DATA}"

if [ $? -eq 0 ]; then
    print_success "Dataset downloaded successfully!"
else
    print_error "Failed to download dataset"
    exit 1
fi

# Clean up
rm -f /tmp/download_elliptic2.py

################################################################################
# Step 7: Verify Dataset
################################################################################

print_header "Step 7: Verifying Dataset"

print_info "Files in ${RAW_DATA}:"
ls -lh "${RAW_DATA}"

echo ""
print_info "Disk usage after download:"
df -h /ephemeral
echo ""

pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install -r requirements.txt
python3 src/preprocess/preprocess.py
python3 src/train/train.py
