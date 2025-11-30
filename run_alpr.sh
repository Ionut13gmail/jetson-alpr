#!/bin/bash
#
# ALPR Service Management Script for Jetson Nano
# Usage: ./run_alpr.sh [command]
#
# Commands:
#   start     - Start the ALPR service
#   stop      - Stop the ALPR service
#   restart   - Restart the ALPR service
#   update    - Pull latest code from GitHub and rebuild
#   build     - Build/rebuild the Docker image
#   logs      - Show service logs (follow mode)
#   status    - Show service status
#   test      - Run a test with sample image
#   setup     - Initial setup (create dirs, install vsftpd)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Defaults
CONTAINER_NAME="${CONTAINER_NAME:-alpr-service}"
IMAGE_NAME="${IMAGE_NAME:-alpr-service:latest}"
WATCH_DIR="${ALPR_WATCH_DIR:-/dev/shm/alpr_inbox}"
OUTPUT_DIR="${ALPR_OUTPUT_DIR:-/dev/shm/alpr_outbox}"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

create_dirs() {
    log_info "Creating watch directories in RAM..."
    sudo mkdir -p "$WATCH_DIR" "$OUTPUT_DIR"
    sudo chmod 777 "$WATCH_DIR" "$OUTPUT_DIR"
}

cmd_start() {
    log_info "Starting ALPR service..."

    # Create directories
    create_dirs

    # Check if container exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        # Container exists, start it
        docker start "$CONTAINER_NAME"
    else
        # Run new container
        docker run -d \
            --name "$CONTAINER_NAME" \
            --runtime nvidia \
            --restart unless-stopped \
            -v "$WATCH_DIR:/watch/inbox" \
            -v "$OUTPUT_DIR:/watch/outbox" \
            -v "$SCRIPT_DIR/models:/app/models:ro" \
            -e ALPR_FTP_HOST="${ALPR_FTP_HOST:-192.168.100.238}" \
            -e ALPR_FTP_USER="${ALPR_FTP_USER:-photo}" \
            -e ALPR_FTP_PASS="${ALPR_FTP_PASS:-photo}" \
            -e ALPR_FTP_DIR="${ALPR_FTP_DIR:-/}" \
            -e ALPR_CONF_THRESH="${ALPR_CONF_THRESH:-0.4}" \
            -e ALPR_DELETE_PROCESSED="${ALPR_DELETE_PROCESSED:-true}" \
            -e ALPR_LOG_LEVEL="${ALPR_LOG_LEVEL:-INFO}" \
            "$IMAGE_NAME"
    fi

    log_info "Service started. Use './run_alpr.sh logs' to view output."
}

cmd_stop() {
    log_info "Stopping ALPR service..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    log_info "Service stopped."
}

cmd_restart() {
    cmd_stop
    sleep 2

    # Remove old container to apply any env changes
    docker rm "$CONTAINER_NAME" 2>/dev/null || true

    cmd_start
}

cmd_build() {
    log_info "Building Docker image..."
    docker build -f Dockerfile.production -t "$IMAGE_NAME" .
    log_info "Build complete: $IMAGE_NAME"
}

cmd_update() {
    log_info "Updating from GitHub..."

    # Stash any local changes
    git stash 2>/dev/null || true

    # Pull latest
    git pull origin main

    # Rebuild
    cmd_build

    # Restart service
    cmd_restart

    log_info "Update complete!"
}

cmd_logs() {
    log_info "Showing logs (Ctrl+C to exit)..."
    docker logs -f "$CONTAINER_NAME"
}

cmd_status() {
    echo "=== ALPR Service Status ==="
    echo ""

    # Container status
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "Container: ${GREEN}RUNNING${NC}"
        docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Status}}\t{{.Ports}}"
    elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "Container: ${YELLOW}STOPPED${NC}"
    else
        echo -e "Container: ${RED}NOT FOUND${NC}"
    fi

    echo ""

    # Watch directory
    if [ -d "$WATCH_DIR" ]; then
        FILE_COUNT=$(ls -1 "$WATCH_DIR" 2>/dev/null | wc -l)
        echo "Watch dir: $WATCH_DIR ($FILE_COUNT files pending)"
    else
        echo -e "Watch dir: ${RED}NOT FOUND${NC}"
    fi

    echo ""

    # Configuration
    echo "=== Configuration ==="
    echo "FTP Host: ${ALPR_FTP_HOST:-192.168.100.238}"
    echo "FTP User: ${ALPR_FTP_USER:-photo}"
    echo "Confidence: ${ALPR_CONF_THRESH:-0.4}"
}

cmd_test() {
    log_info "Running test..."

    create_dirs

    if [ -f "assets/test_image.png" ]; then
        cp assets/test_image.png "$WATCH_DIR/"
        log_info "Copied test image to $WATCH_DIR"
        log_info "Waiting for processing..."
        sleep 5
        cmd_logs
    else
        log_error "Test image not found: assets/test_image.png"
        exit 1
    fi
}

cmd_setup() {
    log_info "=== Initial Setup ==="

    # Create directories
    create_dirs

    # Create .env if not exists
    if [ ! -f ".env" ]; then
        log_info "Creating .env file..."
        cp .env.example .env 2>/dev/null || cat > .env << 'EOF'
# ALPR Service Configuration
# Edit these values for your setup

# Destination FTP server (where annotated images are uploaded)
ALPR_FTP_HOST=192.168.100.238
ALPR_FTP_USER=photo
ALPR_FTP_PASS=photo
ALPR_FTP_DIR=/

# Detection settings
ALPR_CONF_THRESH=0.4

# Logging (DEBUG, INFO, WARNING, ERROR)
ALPR_LOG_LEVEL=INFO

# Watch directories (use /dev/shm for RAM-based storage)
ALPR_WATCH_DIR=/dev/shm/alpr_inbox
ALPR_OUTPUT_DIR=/dev/shm/alpr_outbox

# Container settings
CONTAINER_NAME=alpr-service
IMAGE_NAME=alpr-service:latest
EOF
        log_info "Created .env file. Please edit it with your settings."
    fi

    # Ask about vsftpd setup
    echo ""
    read -p "Setup vsftpd for camera uploads? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_vsftpd
    fi

    # Build image
    echo ""
    read -p "Build Docker image now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cmd_build
    fi

    log_info "Setup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Edit .env with your FTP settings"
    echo "  2. Run: ./run_alpr.sh start"
    echo "  3. Configure your camera to upload to this Jetson"
}

setup_vsftpd() {
    log_info "Setting up vsftpd..."

    # Install vsftpd
    sudo apt-get update
    sudo apt-get install -y vsftpd

    # Create FTP user with nologin shell (FTP-only access)
    sudo useradd -m -d /dev/shm/alpr_inbox -s /usr/sbin/nologin ftpcamera 2>/dev/null || true
    echo "ftpcamera:camera123" | sudo chpasswd

    # IMPORTANT: Add nologin to valid shells for PAM authentication
    # Without this, pam_shells.so rejects FTP login with "530 Login incorrect"
    if ! grep -q '/usr/sbin/nologin' /etc/shells 2>/dev/null; then
        log_info "Adding /usr/sbin/nologin to /etc/shells..."
        echo '/usr/sbin/nologin' | sudo tee -a /etc/shells > /dev/null
    fi

    # Configure vsftpd
    sudo tee /etc/vsftpd.conf > /dev/null << 'EOF'
listen=YES
listen_ipv6=NO
anonymous_enable=NO
local_enable=YES
write_enable=YES
local_umask=022
chroot_local_user=YES
allow_writeable_chroot=YES
pasv_enable=YES
pasv_min_port=21100
pasv_max_port=21110
userlist_enable=YES
userlist_file=/etc/vsftpd.userlist
userlist_deny=NO
EOF

    # Get local IP for passive mode
    LOCAL_IP=$(hostname -I | awk '{print $1}')
    echo "pasv_address=$LOCAL_IP" | sudo tee -a /etc/vsftpd.conf > /dev/null

    # Create user whitelist
    echo "ftpcamera" | sudo tee /etc/vsftpd.userlist > /dev/null

    # Create systemd service for directories
    sudo tee /etc/systemd/system/alpr-dirs.service > /dev/null << 'EOF'
[Unit]
Description=Create ALPR directories in RAM
Before=vsftpd.service docker.service

[Service]
Type=oneshot
ExecStart=/bin/mkdir -p /dev/shm/alpr_inbox /dev/shm/alpr_outbox
ExecStart=/bin/chmod 777 /dev/shm/alpr_inbox /dev/shm/alpr_outbox
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable alpr-dirs.service
    sudo systemctl start alpr-dirs.service

    # Restart vsftpd
    sudo systemctl restart vsftpd
    sudo systemctl enable vsftpd

    log_info "vsftpd configured!"
    echo ""
    echo "Camera FTP settings:"
    echo "  Host: $LOCAL_IP"
    echo "  Port: 21"
    echo "  User: ftpcamera"
    echo "  Pass: camera123"
}

cmd_deps() {
    log_info "Installing host dependencies for non-Docker use..."

    # System packages
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-opencv libopencv-dev

    # Python packages for Jetson
    pip3 install --upgrade pip
    pip3 install numpy pillow inotify

    # ONNX Runtime with CUDA/TensorRT support (Jetson-specific)
    # Check if already installed
    if python3 -c "import onnxruntime" 2>/dev/null; then
        log_info "ONNX Runtime already installed"
        python3 -c "import onnxruntime; print('Version:', onnxruntime.__version__)"
    else
        log_info "Installing ONNX Runtime for Jetson..."
        # For JetPack 4.6.x
        pip3 install onnxruntime-gpu==1.10.0 2>/dev/null || \
        pip3 install https://nvidia.box.com/shared/static/pmsqsiaw4pg9qrbeckcbymho6c01jj4z.whl 2>/dev/null || \
        log_warn "ONNX Runtime installation may require manual setup for your JetPack version"
    fi

    log_info "Dependencies installed."
    echo ""
    echo "Note: Docker deployment is recommended. Host dependencies are for testing only."
}

cmd_help() {
    echo "ALPR Service Management Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start     Start the ALPR service"
    echo "  stop      Stop the ALPR service"
    echo "  restart   Restart the ALPR service"
    echo "  update    Pull latest code and rebuild"
    echo "  build     Build/rebuild Docker image"
    echo "  logs      Show service logs"
    echo "  status    Show service status"
    echo "  test      Run test with sample image"
    echo "  setup     Initial setup wizard"
    echo "  deps      Install host dependencies (non-Docker)"
    echo "  help      Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 setup      # First time setup"
    echo "  $0 start      # Start service"
    echo "  $0 update     # Update from GitHub"
    echo "  $0 deps       # Install dependencies (non-Docker)"
}

# Main
case "${1:-help}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_restart ;;
    update)  cmd_update ;;
    build)   cmd_build ;;
    logs)    cmd_logs ;;
    status)  cmd_status ;;
    test)    cmd_test ;;
    setup)   cmd_setup ;;
    deps)    cmd_deps ;;
    help|*)  cmd_help ;;
esac
