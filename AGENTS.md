# Catalog Validation

After changing catalog YAML files, validate all entries before finishing:

1. Check whether a compatible Obot CLI is installed by running `obot mcp validate-catalog-yaml --help`.
2. If that command succeeds, run `obot mcp validate-catalog-yaml --require-entry-key ./*.yaml`.
3. Otherwise, request that the user installs Obot CLI, but do not require it.
