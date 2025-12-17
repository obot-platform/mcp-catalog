#!/bin/bash
# Compare MCP tool definitions between two package versions using mcptools
# Usage: ./compare-mcp-tools.sh <package_type> <package_name> <old_version> <new_version> <catalog_name>

set -e

PACKAGE_TYPE=$1  # node or python
PACKAGE_NAME=$2  # Full package name (e.g., @digitalocean/mcp)
OLD_VERSION=$3
NEW_VERSION=$4
CATALOG_NAME=$5  # Short name matching catalog YAML file (e.g., "digitalocean")

# Create temp directory for outputs
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

OLD_OUTPUT="$TEMP_DIR/old.json"
NEW_OUTPUT="$TEMP_DIR/new.json"
COMPARISON_OUTPUT="$TEMP_DIR/comparison.md"

echo "🔍 Comparing MCP tools: $PACKAGE_NAME ($OLD_VERSION → $NEW_VERSION)"
echo ""

# Check environment variable requirements from local catalog YAML
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run env check and capture exit code
python3 "$SCRIPT_DIR/get-env-requirements.py" "$CATALOG_NAME" > "$TEMP_DIR/env_check.json" 2>/dev/null
ENV_CHECK_EXIT=$?

if [ $ENV_CHECK_EXIT -eq 0 ] || [ $ENV_CHECK_EXIT -eq 1 ]; then
    # Exit 0 = can validate, Exit 1 = missing vars but found in catalog
    CAN_VALIDATE=$(jq -r '.can_validate' "$TEMP_DIR/env_check.json")
    MISSING_VARS=$(jq -r '.missing_env_vars[]?' "$TEMP_DIR/env_check.json" 2>/dev/null || echo "")
    
    if [ "$CAN_VALIDATE" = "false" ] && [ -n "$MISSING_VARS" ]; then
        echo "⚠️  Missing required environment variables:"
        echo "$MISSING_VARS" | while read -r var; do
            [ -n "$var" ] && echo "  - $var"
        done
        echo ""
        echo "Skipping validation - package requires credentials"
        echo ""
        
        # Create a helpful failure message
        FAILURE_MSG="## ⚠️ MCP Tools Validation\n\n"
        FAILURE_MSG+="Could not automatically validate the changes between versions.\n\n"
        FAILURE_MSG+="**Missing required environment variables:**\n"
        echo "$MISSING_VARS" | while read -r var; do
            if [ -n "$var" ]; then
                FAILURE_MSG+="- \`$var\`\n"
            fi
        done
        FAILURE_MSG+="\n**To enable validation:** Add these secrets to GitHub Actions.\n\n"
        FAILURE_MSG+="**Manual testing recommended** before merging.\n"
        
        echo -e "$FAILURE_MSG" > "$COMPARISON_OUTPUT"
        
        if [ -n "$GITHUB_OUTPUT" ]; then
            {
                echo "validation_result<<EOF"
                cat "$COMPARISON_OUTPUT"
                echo "EOF"
            } >> "$GITHUB_OUTPUT"
            echo "validation_success=false" >> "$GITHUB_OUTPUT"
        fi
        
        cat "$COMPARISON_OUTPUT"
        exit 0
    else
        echo "✓ All required environment variables are set"
        echo ""
    fi
elif [ $ENV_CHECK_EXIT -eq 2 ]; then
    echo "ℹ️  Package YAML not found in catalog - skipping env check"
    echo ""
else
    echo "ℹ️  Could not check environment requirements"
    echo ""
fi

# Determine command based on package type
if [ "$PACKAGE_TYPE" = "node" ]; then
    CMD_PREFIX="npx -y"
elif [ "$PACKAGE_TYPE" = "python" ]; then
    CMD_PREFIX="uvx"
else
    echo "❌ Unsupported package type: $PACKAGE_TYPE (must be 'node' or 'python')"
    exit 1
fi

# Function to run mcpt and capture output
run_mcpt() {
    local version=$1
    local output_file=$2
    
    echo "Testing version $version..."
    
    # Try to run mcpt tools with timeout
    if timeout 30s mcpt tools $CMD_PREFIX ${PACKAGE_NAME}@${version} --format json > "$output_file" 2>&1; then
        echo "✓ Version $version ran successfully"
        return 0
    else
        echo "⚠️  Version $version failed or timed out"
        return 1
    fi
}

# Run both versions
OLD_SUCCESS=false
NEW_SUCCESS=false

if run_mcpt "$OLD_VERSION" "$OLD_OUTPUT"; then
    OLD_SUCCESS=true
fi

if run_mcpt "$NEW_VERSION" "$NEW_OUTPUT"; then
    NEW_SUCCESS=true
fi

# If either version failed, create a failure note
if [ "$OLD_SUCCESS" = false ] || [ "$NEW_SUCCESS" = false ]; then
    echo ""
    echo "⚠️  Could not compare versions - one or both versions failed to run"
    
    FAILURE_MSG="## ⚠️ MCP Tools Validation\n\n"
    FAILURE_MSG+="Could not automatically validate the changes between versions.\n\n"
    FAILURE_MSG+="**Possible reasons:**\n"
    FAILURE_MSG+="- Package requires authentication credentials\n"
    FAILURE_MSG+="- Package initialization timed out\n"
    FAILURE_MSG+="- Package has errors in this version\n\n"
    FAILURE_MSG+="**Manual testing recommended** before merging.\n"
    
    echo -e "$FAILURE_MSG" > "$COMPARISON_OUTPUT"
    
    if [ -n "$GITHUB_OUTPUT" ]; then
        {
            echo "validation_result<<EOF"
            cat "$COMPARISON_OUTPUT"
            echo "EOF"
        } >> "$GITHUB_OUTPUT"
        echo "validation_success=false" >> "$GITHUB_OUTPUT"
    fi
    
    cat "$COMPARISON_OUTPUT"
    exit 0
fi

echo ""
echo "📊 Analyzing tool differences..."

# Try AI-powered analysis first
if python3 "$SCRIPT_DIR/ai-analyze-tools.py" "$OLD_OUTPUT" "$NEW_OUTPUT" "$PACKAGE_NAME" "$OLD_VERSION" "$NEW_VERSION" > "$COMPARISON_OUTPUT" 2>&1; then
    echo "✓ AI analysis completed successfully"
else
    echo "⚠️  AI analysis failed, falling back to basic comparison"
    
    # Fallback: Basic comparison
    OLD_TOOLS=$(jq -r '.tools[]?.name // empty' "$OLD_OUTPUT" 2>/dev/null | sort)
    NEW_TOOLS=$(jq -r '.tools[]?.name // empty' "$NEW_OUTPUT" 2>/dev/null | sort)
    
    OLD_COUNT=$(echo "$OLD_TOOLS" | grep -c . || echo "0")
    NEW_COUNT=$(echo "$NEW_TOOLS" | grep -c . || echo "0")
    
    ADDED=$(comm -13 <(echo "$OLD_TOOLS") <(echo "$NEW_TOOLS"))
    REMOVED=$(comm -23 <(echo "$OLD_TOOLS") <(echo "$NEW_TOOLS"))
    
    # Create basic summary
    SUMMARY="## 🔧 MCP Tools Comparison\n\n"
    SUMMARY+="### Summary\n"
    SUMMARY+="- Old version ($OLD_VERSION): **$OLD_COUNT tools**\n"
    SUMMARY+="- New version ($NEW_VERSION): **$NEW_COUNT tools**\n"
    SUMMARY+="- Net change: **$((NEW_COUNT - OLD_COUNT)) tools**\n\n"
    
    if [ -n "$ADDED" ] && [ "$(echo "$ADDED" | grep -c .)" -gt 0 ]; then
        ADDED_COUNT=$(echo "$ADDED" | grep -c .)
        SUMMARY+="### ✅ Added Tools ($ADDED_COUNT)\n"
        while IFS= read -r tool; do
            if [ -n "$tool" ]; then
                SUMMARY+="- \`$tool\`\n"
            fi
        done <<< "$ADDED"
        SUMMARY+="\n"
    fi
    
    if [ -n "$REMOVED" ] && [ "$(echo "$REMOVED" | grep -c .)" -gt 0 ]; then
        REMOVED_COUNT=$(echo "$REMOVED" | grep -c .)
        SUMMARY+="### ❌ Removed Tools ($REMOVED_COUNT)\n"
        while IFS= read -r tool; do
            if [ -n "$tool" ]; then
                SUMMARY+="- \`$tool\`\n"
            fi
        done <<< "$REMOVED"
        SUMMARY+="\n"
    fi
    
    if [ -z "$ADDED" ] && [ -z "$REMOVED" ]; then
        SUMMARY+="### No tool changes detected\n"
        SUMMARY+="All tools from the previous version are present in the new version.\n\n"
    fi
    
    SUMMARY+="---\n"
    SUMMARY+="*Note: AI analysis was not available. This is a basic comparison.*\n"
    
    echo -e "$SUMMARY" > "$COMPARISON_OUTPUT"
fi

# Output results
echo ""
echo "Results:"
cat "$COMPARISON_OUTPUT"

# Set output for GitHub Actions
if [ -n "$GITHUB_OUTPUT" ]; then
    {
        echo "validation_result<<EOF"
        cat "$COMPARISON_OUTPUT"
        echo "EOF"
    } >> "$GITHUB_OUTPUT"
    echo "validation_success=true" >> "$GITHUB_OUTPUT"
fi

exit 0

