#!/usr/bin/env python3
"""
Get environment variable requirements from local catalog YAML files.
This version is for use in the mcp-catalog repo where YAML files are local.
"""

import sys
import json
import yaml
from pathlib import Path


def get_env_requirements(package_name: str) -> dict:
    """
    Get environment variable requirements for a package from its local YAML file.
    
    Args:
        package_name: Name of the package (e.g., "digitalocean", "playwright")
        
    Returns:
        Dict with env requirements info
    """
    # Path to the catalog YAML file (in repo root)
    script_dir = Path(__file__).parent
    catalog_root = script_dir.parent.parent
    
    # Try multiple possible filenames
    possible_files = [
        catalog_root / f"{package_name}.yaml",
        catalog_root / f"{package_name}.yml",
        catalog_root / f"{package_name.replace('-', '_')}.yaml",
    ]
    
    yaml_path = None
    for path in possible_files:
        if path.exists():
            yaml_path = path
            break
    
    if not yaml_path:
        return {
            "package": package_name,
            "found": False,
            "env_vars": [],
            "required_env_vars": []
        }
    
    try:
        with open(yaml_path, 'r') as f:
            catalog_data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading {yaml_path}: {e}", file=sys.stderr)
        return {
            "package": package_name,
            "found": False,
            "env_vars": [],
            "required_env_vars": []
        }
    
    env_vars = catalog_data.get("env", [])
    required_vars = [
        {
            "key": var["key"],
            "name": var.get("name", var["key"]),
            "description": var.get("description", ""),
            "required": var.get("required", False),
            "sensitive": var.get("sensitive", False)
        }
        for var in env_vars
    ]
    
    return {
        "package": package_name,
        "found": True,
        "yaml_file": str(yaml_path.name),
        "env_vars": required_vars,
        "required_env_vars": [v["key"] for v in required_vars if v.get("required")]
    }


def check_env_availability(env_requirements: dict) -> dict:
    """
    Check which required env vars are actually set.
    
    Args:
        env_requirements: Output from get_env_requirements
        
    Returns:
        Dict with availability info
    """
    import os
    
    missing = []
    present = []
    
    for var_key in env_requirements.get("required_env_vars", []):
        if os.environ.get(var_key):
            present.append(var_key)
        else:
            missing.append(var_key)
    
    return {
        **env_requirements,
        "missing_env_vars": missing,
        "present_env_vars": present,
        "can_validate": len(missing) == 0 or len(env_requirements.get("required_env_vars", [])) == 0
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: get-env-requirements.py <package_name>", file=sys.stderr)
        print("\nExample: get-env-requirements.py digitalocean", file=sys.stderr)
        sys.exit(1)
    
    package_name = sys.argv[1]
    
    # Get requirements from local YAML file
    requirements = get_env_requirements(package_name)
    
    # Check availability
    result = check_env_availability(requirements)
    
    # Output as JSON
    print(json.dumps(result, indent=2))
    
    # Exit with appropriate code
    if not result["found"]:
        sys.exit(2)  # Package YAML not found
    elif not result["can_validate"]:
        sys.exit(1)  # Missing required env vars
    else:
        sys.exit(0)  # All good


if __name__ == "__main__":
    main()

