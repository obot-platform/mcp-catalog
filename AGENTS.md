# Catalog entries

Catalog entries must:

- Use `repoURL` to link to official documentation.
- Include a link to official documentation or setup instructions in the description before the first feature or setup section.
- Lead the description with concrete, product-specific functionality or outcomes.
- Avoid opening with protocol, hosting, or "official server" boilerplate.
- Use `MCP`, not the expanded term "Model Context Protocol".
- Mention hosting, authentication, or official status only when it adds useful context.
- Use `urlTemplate`, not `URLTemplate`, when configuring a URL template.
- Never interpolate credentials, API keys, tokens, passwords, or other sensitive values into `fixedURL` or `urlTemplate`, including query parameters. Use a sensitive `remoteConfig.headers` value or OAuth instead. Do not add an entry when its provider requires credentials in the URL.

# Catalog Validation

New catalog entries must use an `entryKey` prefixed with `obot-`.

After changing catalog YAML files, validate all entries before finishing:

1. Check whether a compatible Obot CLI is installed by running `obot mcp validate-catalog-yaml --help`.
2. If that command succeeds, run `obot mcp validate-catalog-yaml --require-entry-key .`.
3. Otherwise, request that the user installs Obot CLI, but do not require it.
4. Run `git diff --check`.
