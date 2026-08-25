---
name: claude-catalog-entry
description: Reviews remote Streamable HTTP connectors from the Claude directory and records verified Obot Catalog dispositions. Use when importing, matching, skipping, or continuing review of Claude directory MCP connectors in this repository.
---

# Claude Catalog Entry

Use the repository CLI for snapshots, selection, and every ledger mutation. Never edit `scripts/claude-directory/reviewed.yaml` directly.

## Start a review session

1. Refresh once at the beginning of the session:

   ```sh
   go -C scripts/claude-directory run . refresh
   ```

2. Select the highest-ranked unreviewed connector:

   ```sh
   go -C scripts/claude-directory run . select
   ```

3. Inspect `scripts/claude-directory/.state/current.json`. Use `show ID` to revisit another snapshot record. Do not refresh again while evaluating the batch.

## Evaluate the selection

1. Search catalog YAML recursively for product-name, hostname, and endpoint duplicates.
2. Start with authoritative provider documentation for the remote endpoint, setup, authentication method, scopes, and credential creation.
3. Confirm the endpoint is portable outside Claude and uses remote Streamable HTTP.
4. Verify authentication using the read-only public discovery flow below, then map it to supported Obot remote configuration.
5. If it is a duplicate, record `existing`. If it cannot be supported or ported, record `skipped`. Leave ambiguous cases unreviewed.
6. Otherwise create the smallest useful catalog entry under `remotes/`. Link the authoritative provider documentation in the entry's `description`, even when the same URL is used for `repoURL`. Do not add `toolPreview` during intake.
7. Before recording an `imported` disposition, verify the exact icon URL to be stored using the required image checks below. Icon verification is an import gate, not a follow-up task.
8. Validate catalog YAML before recording an imported disposition:

   ```sh
   obot mcp validate-catalog-yaml --require-entry-key .
   ```

## Confirm portability and documentation

A successful unauthenticated `initialize` response proves that an endpoint is reachable and speaks MCP; it does not by itself prove that the provider supports use outside Claude.

Before importing, require authoritative, endpoint-specific documentation that supports connecting from MCP-compatible clients generally or from the intended non-Claude client. A Claude directory listing, a Claude-only tutorial, or a Claude connect button is insufficient evidence of portability, even when Anthropic hosts the endpoint and it accepts standard MCP requests.

If the only available documentation is Claude-specific, do not import the connector. Record `skipped` when the connector is clearly limited to Claude; otherwise leave it unreviewed until portable support can be documented. Do not count it as verified based on live protocol behavior alone.

Do not catalog examples, demos, samples, or test deployments. A catalog entry must represent a provider-supported connector, not a reference implementation or showcase. Treat an `example`/`demo`/`sample` endpoint or documentation located under an examples directory as disqualifying evidence when it identifies the deployment as non-production; record `skipped` with that reason.

## Verify and map authentication

Treat provider documentation as the primary source, then use public OAuth endpoints to confirm the live behavior:

Before every request whose URL comes from a connector record or response, including documentation, MCP, metadata, issuer, and redirect URLs, enforce this public-destination policy:

- Require an absolute HTTPS URL without embedded credentials.
- Resolve the hostname immediately before the request. Reject the URL if any resolved address is loopback, private, link-local, multicast, unspecified, or otherwise non-public. Pin the request to a validated address while preserving the TLS hostname when the client supports it.
- Disable automatic redirects. Validate and resolve each redirect target with the same policy before following it.
- Leave the connector unreviewed if the client hides redirects, DNS resolution cannot be checked, or destination safety cannot otherwise be established.

1. Make an unauthenticated request to the exact MCP endpoint. Inspect the status and `WWW-Authenticate` header without sending credentials.
2. If the challenge contains `resource_metadata`, fetch that URL. Otherwise try the RFC 9728 endpoint-path URL (`https://HOST/.well-known/oauth-protected-resource/MCP_PATH`), then the root `https://HOST/.well-known/oauth-protected-resource` fallback.
3. Confirm protected-resource metadata identifies the MCP resource and lists `authorization_servers`.
4. For each advertised issuer, fetch its RFC 8414 authorization-server metadata and OIDC discovery fallback. Check the authorization and token endpoints, scopes, PKCE support, and `registration_endpoint`.
5. Do not POST to a registration endpoint during review. The public metadata is sufficient to confirm DCR support.

Choose the first verified Obot mapping:

- Prefer OAuth with Dynamic Client Registration when metadata advertises a `registration_endpoint`. Use `remoteConfig.fixedURL` or `urlTemplate`; do not add static headers or `staticOAuthRequired`.
- If OAuth requires a provider-created client ID/secret or allowlisted redirect URI, use static OAuth with `remoteConfig.staticOAuthRequired: true` and document the provider setup and required scopes.
- If provider docs specify a static token or API key, use `remoteConfig.headers`. Mark the value `required: true` and `sensitive: true`; use the documented header name and add a prefix such as `Bearer ` only when required.
- If documentation and live discovery conflict, or the auth scheme cannot be represented safely, leave the connector unreviewed. Never infer auth from a Claude-only connect button.

## Verify the icon before import

This check is required before recording an `imported` disposition. Prefer an official, stable image asset. Do not use a homepage URL or assume that `/favicon.ico` exists.

Apply the same public-destination and manual-redirect policy used for authentication requests, then make a GET request to the exact URL that will be stored in `icon`:

- Require a successful 2xx response without authentication.
- Require an `image/*` response content type and a non-empty body.
- If the candidate redirects, validate every redirect target and store the final direct image URL rather than the redirecting URL.
- Leave the connector unreviewed if no stable image URL can be verified.

## Record the verified disposition

Run exactly one applicable command after verification:

```sh
go -C scripts/claude-directory run . ledger add --id ID --status existing --catalog-entry remotes/FILE.yaml
go -C scripts/claude-directory run . ledger add --id ID --status imported --catalog-entry remotes/FILE.yaml
go -C scripts/claude-directory run . ledger add --id ID --status skipped --reason "Specific reason"
```

Use `ledger update` with the same flags to correct an existing record. Run `ledger check`, then run `select` again and repeat until the batch is done.
