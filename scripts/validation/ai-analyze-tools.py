#!/usr/bin/env python3
"""
AI-powered analysis of MCP tool differences using OpenAI.
Falls back to basic comparison if OpenAI is not available.
"""

import sys
import json
import os
from pathlib import Path

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def load_tools_json(filepath: str) -> dict:
    """Load and parse tools JSON from mcptools output."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}", file=sys.stderr)
        return {}


def extract_tools_info(tools_data: dict) -> list:
    """Extract relevant tool information for comparison."""
    tools = tools_data.get('tools', [])
    
    tool_info = []
    for tool in tools:
        info = {
            'name': tool.get('name', ''),
            'description': tool.get('description', ''),
            'input_schema': tool.get('inputSchema', {})
        }
        tool_info.append(info)
    
    return tool_info


def analyze_with_openai(old_tools: list, new_tools: list, package_name: str, 
                        old_version: str, new_version: str) -> str:
    """Use OpenAI to analyze the differences between tool versions."""
    
    if not HAS_OPENAI:
        raise Exception("OpenAI library not installed")
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise Exception("OPENAI_API_KEY not found in environment")
    
    client = OpenAI(api_key=api_key)
    
    # Prepare the comparison data
    comparison_data = {
        'package': package_name,
        'old_version': old_version,
        'new_version': new_version,
        'old_tools': old_tools,
        'new_tools': new_tools
    }
    
    prompt = f"""You are analyzing changes in an MCP (Model Context Protocol) server between two versions.

Package: {package_name}
Old Version: {old_version}
New Version: {new_version}

Old version tools ({len(old_tools)} total):
{json.dumps(old_tools, indent=2)}

New version tools ({len(new_tools)} total):
{json.dumps(new_tools, indent=2)}

Please provide a comprehensive analysis in Markdown format with the following sections:

## 🔧 MCP Tools Analysis: {old_version} → {new_version}

### 📊 Summary
Provide a brief overview of the changes (2-3 sentences).

### ✅ Added Tools
List any new tools with brief descriptions of what they do.

### ❌ Removed Tools
List any removed tools and note if this could be a breaking change.

### 🔄 Modified Tools
List tools that exist in both versions but have changed schemas or descriptions.
Highlight any breaking changes (required parameters added, parameters removed, type changes).

### 🎯 Impact Assessment
Provide a risk level (Low/Medium/High) and explain:
- Any breaking changes
- New capabilities
- Potential issues

### 💡 Recommendation
Should this update be merged? Any concerns or testing suggestions?

Keep the analysis concise but informative. Use emojis sparingly. Focus on practical implications for users."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cost-effective model
            messages=[
                {"role": "system", "content": "You are an expert at analyzing API and tool changes, identifying breaking changes, and assessing impact."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=2000
        )
        
        analysis = response.choices[0].message.content
        return analysis
        
    except Exception as e:
        raise Exception(f"OpenAI API call failed: {e}")


def main():
    if len(sys.argv) != 6:
        print("Usage: ai-analyze-tools.py <old_json> <new_json> <package_name> <old_version> <new_version>", 
              file=sys.stderr)
        sys.exit(1)
    
    old_json_path = sys.argv[1]
    new_json_path = sys.argv[2]
    package_name = sys.argv[3]
    old_version = sys.argv[4]
    new_version = sys.argv[5]
    
    # Load tool data
    old_data = load_tools_json(old_json_path)
    new_data = load_tools_json(new_json_path)
    
    old_tools = extract_tools_info(old_data)
    new_tools = extract_tools_info(new_data)
    
    # Try AI analysis
    try:
        analysis = analyze_with_openai(old_tools, new_tools, package_name, 
                                       old_version, new_version)
        print(analysis)
        sys.exit(0)
    except Exception as e:
        print(f"AI analysis failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

