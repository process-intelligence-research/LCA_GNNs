# Docker Deployment Guide for LCA GNNs

This guide provides comprehensive instructions for deploying and developing the LCA GNNs project using Docker.

**Note**: All Docker-related files are organized in this `docker/` folder for clean project structure. All Docker commands should reference this folder.

## Prerequisites

- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
- **Docker Compose** (included with Docker Desktop)
- At least 4GB RAM available for containers
- Pre-trained model files in `trained_models/` directory

## Quick Start

### 1. Build the Images

From the **project root directory**:

```bash
# Build production image
docker build -f docker/Dockerfile -t lca-gnn:latest .

# Build development image
docker build -f docker/Dockerfile.dev -t lca-gnn:dev .

# Or build with Docker Compose
docker-compose -f docker/docker-compose.yml build
```

### 2. Run Single Molecule Inference

From the **project root directory**:

```bash
# Using Docker Compose
docker-compose -f docker/docker-compose.yml run --rm lca-gnn \
    python main.py --workflow inference \
    --model_path trained_models/GNN_C_Gwi.pth \
    --smiles "CCO" \
    --target_task "Gwi" \
    --country_name "Germany"

# Or using Docker directly
docker run --rm \
    -v "$(pwd)/trained_models:/app/trained_models:ro" \
    -v "$(pwd)/data:/app/data:ro" \
    lca-gnn:latest \
    python main.py --workflow inference \
    --model_path trained_models/GNN_C_Gwi.pth \
    --smiles "CCO" \
    --target_task "Gwi" \
    --country_name "Germany"
```

### 3. Run Batch Processing

From the **project root directory**:

```bash
# Using Docker Compose
docker-compose -f docker/docker-compose.yml run --rm lca-gnn \
    python main.py --workflow batch \
    --model_path trained_models/GNN_C_multi_best.pth \
    --data_path test_molecules.xlsx

# Or using Docker directly
docker run --rm \
    -v "$(pwd)/trained_models:/app/trained_models:ro" \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/test_molecules.xlsx:/app/test_molecules.xlsx:ro" \
    lca-gnn:latest \
    python main.py --workflow batch \
    --model_path trained_models/GNN_C_multi_best.pth \
    --data_path test_molecules.xlsx
```

## Docker File Organization

This `docker/` folder contains all Docker-related configurations:

- **`Dockerfile`** - Production-optimized multi-stage build
- **`Dockerfile.dev`** - Development environment with tools (pytest, jupyter, etc.)
- **`docker-compose.yml`** - Complete orchestration with dev profiles and advanced settings
- **`docker-helper.sh/.bat`** - Comprehensive helper scripts for all platforms
- **`.env.example`** - Environment variable templates
- **`README.md`** - This comprehensive guide

All Docker commands should reference this folder (e.g., `docker-compose -f docker/docker-compose.yml` or `docker/docker-helper.sh`).

## Helper Scripts

### Linux/Mac

From the **project root directory**:

```bash
# Make executable
chmod +x docker/docker-helper.sh

# Build images
docker/docker-helper.sh build
docker/docker-helper.sh build-dev

# Run inference
docker/docker-helper.sh inference trained_models/GNN_C_Gwi.pth "CCO" --target_task Gwi --country_name Germany

# Run batch processing
docker/docker-helper.sh batch trained_models/GNN_C_multi_best.pth test_data.xlsx

# Start development environment
docker/docker-helper.sh dev

# Run tests
docker/docker-helper.sh test

# Run quality checks
docker/docker-helper.sh quality
```

### Windows

From the **project root directory**:

```cmd
REM Build images
docker\docker-helper.bat build
docker\docker-helper.bat build-dev

REM Run inference
docker\docker-helper.bat inference trained_models/GNN_C_Gwi.pth "CCO" --target_task Gwi --country_name Germany

REM Run batch processing
docker\docker-helper.bat batch trained_models/GNN_C_multi_best.pth test_data.xlsx

REM Start development environment
docker\docker-helper.bat dev
```

## Development Environment

### Start Development Container

From the **project root directory**:

```bash
# Using Docker Compose
docker-compose -f docker/docker-compose.yml --profile dev up lca-gnn-dev

# Or using helper script
docker/docker-helper.sh dev     # Linux/Mac
docker\docker-helper.bat dev    # Windows
```

This provides:
- **Live code mounting** - changes reflect immediately
- **Jupyter notebook** access on port 8888
- **Development tools** - pytest, ruff, pyright
- **Interactive shell** for debugging

### Development Workflow

```bash
# Inside the development container
cd /app

# Run tests
python -m pytest tests/ -v

# Run quality checks
python -m ruff check src/ tests/
python -m ruff format src/ tests/
python -m pyright src/

# Start Jupyter notebook (if needed)
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

## Production Deployment

### Using Docker Compose

Create a production docker-compose file:

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  lca-gnn:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: lca-gnn-prod
    volumes:
      - ../trained_models:/app/trained_models:ro
      - ../data:/app/data
      - ../reports:/app/reports
    environment:
      - PYTHONPATH=/app/src
      - LOG_LEVEL=INFO
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

```bash
# Deploy in production (from project root)
docker-compose -f docker/docker-compose.prod.yml up -d
```

### Scaling with Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack (from project root)
docker stack deploy -c docker/docker-compose.yml lca-gnn-stack

# Scale service
docker service scale lca-gnn-stack_lca-gnn=3
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
# From project root
cp docker/.env.example docker/.env
```

Key variables:
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `DEFAULT_MODEL_PATH`: Default model to use
- `DEFAULT_COUNTRY`: Default country for predictions
- `MEMORY_LIMIT`: Container memory limit
- `CPU_LIMIT`: Container CPU limit

### Volume Mounts

When running from project root, volume paths are:

| Host Path | Container Path | Purpose | Mode |
|-----------|----------------|---------|------|
| `./trained_models` | `/app/trained_models` | Model files | `ro` (read-only) |
| `./data` | `/app/data` | Input/output data | `rw` (read-write) |
| `./reports` | `/app/reports` | Generated reports | `rw` (read-write) |
| `./configs` | `/app/configs` | Configuration files | `ro` (read-only) |

## Troubleshooting

### Common Issues

**1. Docker daemon not running**
```bash
# Windows: Start Docker Desktop
# Linux: sudo systemctl start docker
# Mac: Start Docker Desktop application
```

**2. Permission errors (Linux)**
```bash
# Fix file permissions
sudo chown -R $USER:$USER ./data ./reports ./trained_models
```

**3. Out of memory errors**
```bash
# Increase Docker memory limits in Docker Desktop settings
# Or modify docker-compose.yml resource limits
```

**4. Model files not found**
```bash
# Ensure models exist locally
ls -la trained_models/
# Mount path should be absolute in docker run commands
```

### Performance Optimization

**1. Use Docker BuildKit**
```bash
export DOCKER_BUILDKIT=1
docker build -f docker/Dockerfile -t lca-gnn:latest .
```

**2. Multi-stage builds** (already implemented)
- Smaller production images
- Faster deployment

**3. Layer caching**
- Requirements installed in separate layer
- Code changes don't trigger full rebuild

### Monitoring

**1. Container health**
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

**2. Resource usage**
```bash
docker stats lca-gnn-app
```

**3. Logs**
```bash
docker logs lca-gnn-app --follow
```

## Security Considerations

1. **Non-root user** - Containers run as `appuser`
2. **Read-only mounts** - Model and config files mounted read-only
3. **No sensitive data** - Environment variables for secrets
4. **Network isolation** - Custom Docker networks
5. **Resource limits** - Prevent resource exhaustion

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Docker Build and Test

on: [push, pull_request]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: docker build -f docker/Dockerfile -t lca-gnn:test .
        
      - name: Run tests in container
        run: |
          docker run --rm \
            -v ${{ github.workspace }}:/app \
            lca-gnn:test \
            bash -c "cd /app && python -m pytest tests/ -v"
```

## Best Practices

1. **Use specific tags** - Avoid `latest` in production
2. **Multi-stage builds** - Optimize image size
3. **Health checks** - Monitor container health
4. **Volume mounts** - Persist important data
5. **Environment variables** - Configure without rebuilding
6. **Resource limits** - Prevent resource starvation
7. **Regular updates** - Keep base images updated

## Support

For Docker-related issues:
1. Check container logs: `docker logs <container_name>`
2. Verify Docker daemon: `docker info`
3. Test with minimal example
4. Review volume mount paths
5. Check resource availability

For application issues within containers, follow the standard troubleshooting guide in the main README.
