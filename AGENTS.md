# Remote catalog entries

Remote catalog entries must:

- Use `repoURL` to link to official documentation.
- Include a link to official documentation or setup instructions in the description before the first feature or setup section.
- Use `urlTemplate`, not `URLTemplate`, when configuring a URL template.

# Catalog Validation

After changing catalog YAML files, validate all entries before finishing:

1. Check whether a compatible Obot CLI is installed by running `obot mcp validate-catalog-yaml --help`.
2. If that command succeeds, run `obot mcp validate-catalog-yaml --require-entry-key ./*.yaml`.
3. Otherwise, request that the user installs Obot CLI, but do not require it.
4. Run `git diff --check`.
