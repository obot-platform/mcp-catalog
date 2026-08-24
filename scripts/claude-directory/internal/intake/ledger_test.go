package intake

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLedgerAddUpdateAndDeterministicSerialization(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	for _, name := range []string{"a.yaml", "b.yaml"} {
		if err := os.WriteFile(filepath.Join(directory, name), []byte("name: test\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	snapshot := Snapshot{Version: SnapshotVersion, Connectors: []Connector{{ID: "b", Name: "Bee"}, {ID: "a", Name: "Aye"}}}
	ledger := Ledger{Version: LedgerVersion}
	var err error
	ledger, err = MutateLedger(ledger, snapshot, directory, "add", LedgerRecord{ID: "b", Status: "skipped", Reason: "not portable"})
	if err != nil {
		t.Fatal(err)
	}
	ledger, err = MutateLedger(ledger, snapshot, directory, "add", LedgerRecord{ID: "a", Status: "existing", CatalogEntry: "a.yaml"})
	if err != nil {
		t.Fatal(err)
	}
	ledger, err = MutateLedger(ledger, snapshot, directory, "update", LedgerRecord{ID: "b", Status: "imported", CatalogEntry: "b.yaml"})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "reviewed.yaml")
	if err := WriteLedgerAtomic(path, ledger); err != nil {
		t.Fatal(err)
	}
	first, _ := os.ReadFile(path)
	if err := WriteLedgerAtomic(path, ledger); err != nil {
		t.Fatal(err)
	}
	second, _ := os.ReadFile(path)
	if string(first) != string(second) {
		t.Fatal("ledger serialization changed between writes")
	}
	if strings.Index(string(first), "id: a") > strings.Index(string(first), "id: b") {
		t.Fatalf("records are not sorted:\n%s", first)
	}
	loaded, err := ReadLedger(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Records[1].Name != "Bee" || loaded.Records[1].Status != "imported" {
		t.Fatalf("updated record = %+v", loaded.Records[1])
	}
}

func TestReadLedgerRejectsMultipleDocuments(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "reviewed.yaml")
	data := []byte("version: 1\nrecords: []\n---\nversion: 1\nrecords: []\n")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := ReadLedger(path)
	if err == nil || !strings.Contains(err.Error(), "multiple YAML documents") {
		t.Fatalf("error = %v", err)
	}
}

func TestLedgerMutationRejectsInvalidRecords(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	if err := os.WriteFile(filepath.Join(directory, "exists.yaml"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	snapshot := Snapshot{Version: SnapshotVersion, Connectors: []Connector{{ID: "known", Name: "Canonical"}}}
	tests := []struct {
		name      string
		ledger    Ledger
		operation string
		record    LedgerRecord
		contains  string
	}{
		{name: "unknown id", ledger: Ledger{Version: 1}, operation: "add", record: LedgerRecord{ID: "unknown", Status: "skipped", Reason: "x"}, contains: "unknown connector"},
		{name: "duplicate", ledger: Ledger{Version: 1, Records: []LedgerRecord{{ID: "known", Name: "Canonical", Status: "skipped", Reason: "x"}}}, operation: "add", record: LedgerRecord{ID: "known", Status: "skipped", Reason: "x"}, contains: "already reviewed"},
		{name: "update missing", ledger: Ledger{Version: 1}, operation: "update", record: LedgerRecord{ID: "known", Status: "skipped", Reason: "x"}, contains: "not in the ledger"},
		{name: "missing catalog", ledger: Ledger{Version: 1}, operation: "add", record: LedgerRecord{ID: "known", Status: "existing", CatalogEntry: "missing.yaml"}, contains: "does not exist"},
		{name: "catalog traversal", ledger: Ledger{Version: 1}, operation: "add", record: LedgerRecord{ID: "known", Status: "existing", CatalogEntry: "../outside.yaml"}, contains: "relative catalog YAML path"},
		{name: "existing needs catalog", ledger: Ledger{Version: 1}, operation: "add", record: LedgerRecord{ID: "known", Status: "existing"}, contains: "catalog_entry is required"},
		{name: "skipped needs reason", ledger: Ledger{Version: 1}, operation: "add", record: LedgerRecord{ID: "known", Status: "skipped"}, contains: "reason is required"},
		{name: "status field conflict", ledger: Ledger{Version: 1}, operation: "add", record: LedgerRecord{ID: "known", Status: "skipped", Reason: "x", CatalogEntry: "exists.yaml"}, contains: "not allowed"},
		{name: "invalid status", ledger: Ledger{Version: 1}, operation: "add", record: LedgerRecord{ID: "known", Status: "maybe"}, contains: "invalid status"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := MutateLedger(test.ledger, snapshot, directory, test.operation, test.record)
			if err == nil || !strings.Contains(err.Error(), test.contains) {
				t.Fatalf("error = %v, want containing %q", err, test.contains)
			}
		})
	}
}

func TestValidateLedgerFindsSchemaDuplicatesAndReferences(t *testing.T) {
	t.Parallel()
	ledger := Ledger{Version: 99, Records: []LedgerRecord{
		{ID: "same", Name: "One", Status: "existing", CatalogEntry: "missing.yaml"},
		{ID: "same", Name: "Two", Status: "skipped"},
	}}
	err := ValidateLedger(ledger, t.TempDir())
	for _, expected := range []string{"unsupported ledger version", "duplicate id", "does not exist", "reason is required"} {
		if err == nil || !strings.Contains(err.Error(), expected) {
			t.Fatalf("error %q does not contain %q", err, expected)
		}
	}
}
