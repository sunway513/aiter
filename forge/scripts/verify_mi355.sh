#!/bin/bash
# Verify MI355X environment for AITER-Forge
# Run this ON the remote MI355X node (not locally).
# Usage: ssh mi355-gpu-9 'bash -s' < scripts/verify_mi355.sh
# Exit code: 0 = all checks passed, 1 = one or more required checks failed

FAILURES=0

fail() {
    echo "FAIL: $1"
    FAILURES=$((FAILURES + 1))
}

pass() {
    echo "PASS: $1"
}

echo "=== MI355X Environment Verification ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo ""

# --- Required checks (failure = exit 1) ---

echo "=== ROCm Version ==="
if cat /opt/rocm/.info/version 2>/dev/null; then
    pass "ROCm version file found"
else
    fail "ROCm version file not found at /opt/rocm/.info/version"
fi

echo ""
echo "=== GPU Hardware Verification ==="
# Use rocm-smi --showid Device Name (not rocminfo Marketing Name, which matches CPU first)
GPU_NAME=$(rocm-smi --showid 2>/dev/null | grep -i "Device Name" | head -1 || true)
if echo "$GPU_NAME" | grep -qi "MI355X\|MI355\|gfx950"; then
    pass "MI355X GPU detected:$GPU_NAME"
else
    if [ -n "$GPU_NAME" ]; then
        fail "Expected MI355X but found:$GPU_NAME"
    else
        fail "rocm-smi not available or no GPU detected"
    fi
fi

echo ""
echo "=== GPU Info ==="
rocm-smi --showid 2>/dev/null | head -20 || fail "rocm-smi not available"

echo ""
echo "=== Python ==="
if python3 --version 2>/dev/null; then
    pass "Python3 available"
else
    fail "Python3 not available"
fi

echo ""
echo "=== Required Python Packages ==="
if python3 -c "import triton; print(f'Triton version: {triton.__version__}')" 2>/dev/null; then
    pass "Triton importable"
else
    fail "Triton not importable"
fi

if python3 -c "import torch; print(f'PyTorch: {torch.__version__}, HIP available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')" 2>/dev/null; then
    pass "PyTorch importable with GPU support"
else
    fail "PyTorch not importable or no GPU support"
fi

echo ""
echo "=== AITER Check ==="
AITER_ROOT="${AITER_ROOT:-}"
if [ -z "$AITER_ROOT" ]; then
    for p in "$HOME/workspace/aiter" "$HOME/aiter" "/opt/aiter"; do
        if [ -d "$p" ]; then AITER_ROOT="$p"; break; fi
    done
fi
if [ -n "$AITER_ROOT" ] && [ -d "$AITER_ROOT" ]; then
    pass "AITER found at $AITER_ROOT"
else
    fail "AITER not found. Set AITER_ROOT env var."
fi

# --- Summary ---

echo ""
echo "=== Verification Summary ==="
if [ "$FAILURES" -eq 0 ]; then
    echo "ALL CHECKS PASSED"
    exit 0
else
    echo "$FAILURES CHECK(S) FAILED"
    exit 1
fi
