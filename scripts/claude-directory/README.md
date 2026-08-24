# Claude directory intake

This utility creates a stable local snapshot of eligible Claude directory connectors and owns the tracked review ledger. Run it from the catalog root with `go -C scripts/claude-directory run . COMMAND`.

`refresh` is the only command that accesses the network. It writes `.state/directory.json`; `select` writes `.state/current.json`. Both files are generated and ignored by Git. The tracked `reviewed.yaml` file must only be changed through `ledger add` or `ledger update`.

The directory endpoint defaults to Anthropic's public Claude Directory feed. Override it with `--api-url` or `CLAUDE_DIRECTORY_API_URL` if Anthropic publishes the directory under a different origin.

Catalog references in `reviewed.yaml` are repository-relative paths. New remote imports belong under `remotes/`; existing matches may reference `remotes/`, `obot-remotes/`, or `obot-images/` according to their runtime source.

Run `go test ./...`, `go vet ./...`, and `go run . ledger check` from this directory before committing utility or ledger changes.
