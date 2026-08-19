package intake

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWriteJSONAtomicReplacesSnapshot(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), ".state", "directory.json")
	if err := WriteJSONAtomic(path, Snapshot{Version: SnapshotVersion, Eligible: 1}); err != nil {
		t.Fatal(err)
	}
	if err := WriteJSONAtomic(path, Snapshot{Version: SnapshotVersion, Eligible: 2}); err != nil {
		t.Fatal(err)
	}
	snapshot, err := ReadSnapshot(path)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Eligible != 2 {
		t.Fatalf("eligible = %d", snapshot.Eligible)
	}
	matches, err := filepath.Glob(filepath.Join(filepath.Dir(path), ".tmp-*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 0 {
		t.Fatalf("temporary files remain: %v", matches)
	}
	if info, err := os.Stat(path); err != nil || info.Mode().Perm() != 0o644 {
		t.Fatalf("mode = %v, err = %v", info.Mode().Perm(), err)
	}
}
