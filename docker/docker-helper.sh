#!/bin/bash
# Docker helper scripts for LCA GNNs project
# Run this script from the project root directory

set -e

# Get the project root directory (parent of docker folder)
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
DOCKER_DIR="$(dirname "$(realpath "$0")")"

# Change to project root
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
}

# Build the Docker image
build() {
    print_status "Building LCA GNNs Docker image..."
    docker build -f docker/Dockerfile -t lca-gnn:latest .
    print_status "Build completed successfully!"
}

# Build development image
build_dev() {
    print_status "Building LCA GNNs development Docker image..."
    docker build -f docker/Dockerfile.dev -t lca-gnn:dev .
    print_status "Development build completed successfully!"
}

# Run inference on a single molecule
inference() {
    if [ $# -lt 2 ]; then
        print_error "Usage: $0 inference <model_path> <smiles> [additional_args...]"
        echo "Example: $0 inference trained_models/GNN_C_Gwi.pth 'CCO' --target_task Gwi --country_name Germany"
        exit 1
    fi
    
    local model_path=$1
    local smiles=$2
    shift 2
    
    print_status "Running inference for SMILES: $smiles"
    docker run --rm \
        -v "$(pwd)/trained_models:/app/trained_models:ro" \
        -v "$(pwd)/data:/app/data:ro" \
        lca-gnn:latest \
        python main.py --workflow inference \
        --model_path "$model_path" \
        --smiles "$smiles" \
        "$@"
}

# Run batch processing
batch() {
    if [ $# -lt 2 ]; then
        print_error "Usage: $0 batch <model_path> <data_path> [additional_args...]"
        echo "Example: $0 batch trained_models/GNN_C_multi_best.pth test_molecules.xlsx"
        exit 1
    fi
    
    local model_path=$1
    local data_path=$2
    shift 2
    
    print_status "Running batch processing for: $data_path"
    docker run --rm \
        -v "$(pwd)/trained_models:/app/trained_models:ro" \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/$data_path:/app/$data_path:ro" \
        lca-gnn:latest \
        python main.py --workflow batch \
        --model_path "$model_path" \
        --data_path "$data_path" \
        "$@"
}

# Start development environment
dev() {
    print_status "Starting development environment..."
    docker run -it --rm \
        -v "$(pwd):/app" \
        -p 8888:8888 \
        -p 8000:8000 \
        lca-gnn:dev
}

# Run tests in container
test() {
    print_status "Running tests in Docker container..."
    docker run --rm \
        -v "$(pwd):/app" \
        lca-gnn:dev \
        bash -c "cd /app && python -m pytest tests/ -v"
}

# Run quality checks
quality() {
    print_status "Running code quality checks in Docker container..."
    docker run --rm \
        -v "$(pwd):/app" \
        lca-gnn:dev \
        bash -c "cd /app && python -m ruff check src/ tests/ && python -m ruff format --check src/ tests/ && python -m pyright src/"
}

# Clean up Docker resources
clean() {
    print_status "Cleaning up Docker resources..."
    docker system prune -f
    print_status "Cleanup completed!"
}

# Show usage information
usage() {
    echo "LCA GNNs Docker Helper Script"
    echo ""
    echo "Usage: $0 <command> [arguments...]"
    echo ""
    echo "Commands:"
    echo "  build                    Build the production Docker image"
    echo "  build-dev                Build the development Docker image"
    echo "  inference <model> <smiles> [args...]  Run single molecule inference"
    echo "  batch <model> <data> [args...]        Run batch processing"
    echo "  dev                      Start development environment"
    echo "  test                     Run tests in container"
    echo "  quality                  Run code quality checks"
    echo "  clean                    Clean up Docker resources"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 inference trained_models/GNN_C_Gwi.pth 'CCO' --target_task Gwi"
    echo "  $0 batch trained_models/GNN_C_multi_best.pth test_data.xlsx"
    echo "  $0 dev"
}

# Main script logic
main() {
    check_docker
    
    case "${1:-}" in
        build)
            build
            ;;
        build-dev)
            build_dev
            ;;
        inference)
            shift
            inference "$@"
            ;;
        batch)
            shift
            batch "$@"
            ;;
        dev)
            dev
            ;;
        test)
            test
            ;;
        quality)
            quality
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            print_error "Unknown command: ${1:-}"
            echo ""
            usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
