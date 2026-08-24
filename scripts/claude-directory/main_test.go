package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/obot-platform/mcp-catalog/scripts/claude-directory/internal/intake"
)

func TestListAndSelectUseSnapshotWithoutNetwork(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	module := filepath.Join(root, "scripts", "claude-directory")
	state := filepath.Join(module, ".state")
	if err := os.MkdirAll(state, 0o755); err != nil {
		t.Fatal(err)
	}
	locations := paths{moduleDir: module, stateDir: state, snapshot: filepath.Join(state, "directory.json"), current: filepath.Join(state, "current.json"), ledger: filepath.Join(module, "reviewed.yaml"), catalogRoot: root}
	rawOne := json.RawMessage(`{"id":"one","name":"One","transport":"streamable_http","url":"https://one.test/mcp","detail":"complete"}`)
	rawTwo := json.RawMessage(`{"id":"two","name":"Two","transport":"streamable_http","url":"https://two.test/mcp"}`)
	snapshot := intake.Snapshot{Version: 1, Connectors: []intake.Connector{{Rank: 1, ID: "one", Name: "One", Record: rawOne}, {Rank: 2, ID: "two", Name: "Two", Record: rawTwo}}}
	if err := intake.WriteJSONAtomic(locations.snapshot, snapshot); err != nil {
		t.Fatal(err)
	}
	if err := intake.WriteLedgerAtomic(locations.ledger, intake.Ledger{Version: 1, Records: []intake.LedgerRecord{{ID: "one", Name: "One", Status: "skipped", Reason: "tested"}}}); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := list(locations, []string{"--json"}, &output, &output); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(output.String(), `"one"`) || !strings.Contains(output.String(), `"two"`) {
		t.Fatalf("default list output = %s", output.String())
	}
	output.Reset()
	if err := selectConnector(locations, nil, &output, &output); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(locations.current)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"id": "two"`) {
		t.Fatalf("current selection = %s", data)
	}
}

func TestSelectWritesCompleteRecordAndRejectsReviewedID(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	state := filepath.Join(root, ".state")
	locations := paths{stateDir: state, snapshot: filepath.Join(state, "directory.json"), current: filepath.Join(state, "current.json"), ledger: filepath.Join(root, "reviewed.yaml"), catalogRoot: root}
	raw := json.RawMessage(`{"id":"one","name":"One","nested":{"preserved":true}}`)
	if err := intake.WriteJSONAtomic(locations.snapshot, intake.Snapshot{Version: 1, Connectors: []intake.Connector{{Rank: 7, ID: "one", Name: "One", Record: raw}}}); err != nil {
		t.Fatal(err)
	}
	if err := intake.WriteLedgerAtomic(locations.ledger, intake.Ledger{Version: 1}); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := selectConnector(locations, []string{"--id", "one"}, &output, &output); err != nil {
		t.Fatal(err)
	}
	current, _ := os.ReadFile(locations.current)
	if !strings.Contains(string(current), `"rank": 7`) || !strings.Contains(string(current), `"preserved": true`) {
		t.Fatalf("current = %s", current)
	}
	if err := intake.WriteLedgerAtomic(locations.ledger, intake.Ledger{Version: 1, Records: []intake.LedgerRecord{{ID: "one", Name: "One", Status: "skipped", Reason: "done"}}}); err != nil {
		t.Fatal(err)
	}
	err := selectConnector(locations, []string{"--id", "one"}, &output, &output)
	if err == nil || !strings.Contains(err.Error(), "already been reviewed") {
		t.Fatalf("error = %v", err)
	}
}

func TestLedgerUpdateRepairsInvalidCatalogReference(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	state := filepath.Join(root, ".state")
	locations := paths{stateDir: state, snapshot: filepath.Join(state, "directory.json"), ledger: filepath.Join(root, "reviewed.yaml"), catalogRoot: root}
	snapshot := intake.Snapshot{Version: 1, Connectors: []intake.Connector{{ID: "one", Name: "One"}}}
	if err := intake.WriteJSONAtomic(locations.snapshot, snapshot); err != nil {
		t.Fatal(err)
	}
	invalid := intake.Ledger{Version: 1, Records: []intake.LedgerRecord{{ID: "one", Name: "One", Status: "imported", CatalogEntry: "missing.yaml"}}}
	if err := intake.WriteLedgerAtomic(locations.ledger, invalid); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "remotes"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "remotes", "replacement.yaml"), []byte("name: replacement\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var output bytes.Buffer
	args := []string{"update", "--id", "one", "--status", "imported", "--catalog-entry", "remotes/replacement.yaml"}
	if err := ledger(locations, args, &output, &output); err != nil {
		t.Fatal(err)
	}
	updated, err := intake.ReadLedger(locations.ledger)
	if err != nil {
		t.Fatal(err)
	}
	if got := updated.Records[0].CatalogEntry; got != "remotes/replacement.yaml" {
		t.Fatalf("catalog entry = %q", got)
	}
	if err := intake.ValidateLedger(updated, root); err != nil {
		t.Fatalf("updated ledger is invalid: %v", err)
	}
}
