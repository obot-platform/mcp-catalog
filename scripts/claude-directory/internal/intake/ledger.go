package intake

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

const LedgerVersion = 1

type Ledger struct {
	Version int            `yaml:"version" json:"version"`
	Records []LedgerRecord `yaml:"records" json:"records"`
}

type LedgerRecord struct {
	ID           string `yaml:"id" json:"id"`
	Name         string `yaml:"name" json:"name"`
	Status       string `yaml:"status" json:"status"`
	CatalogEntry string `yaml:"catalog_entry,omitempty" json:"catalog_entry,omitempty"`
	Reason       string `yaml:"reason,omitempty" json:"reason,omitempty"`
}

func ReadLedger(path string) (Ledger, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Ledger{}, fmt.Errorf("read ledger %s: %w", path, err)
	}
	var ledger Ledger
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&ledger); err != nil {
		return Ledger{}, fmt.Errorf("decode ledger %s: %w", path, err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err == nil {
		return Ledger{}, fmt.Errorf("decode ledger %s: multiple YAML documents are not allowed", path)
	} else if !errors.Is(err, io.EOF) {
		return Ledger{}, fmt.Errorf("decode ledger %s trailing content: %w", path, err)
	}
	return ledger, nil
}

func WriteLedgerAtomic(path string, ledger Ledger) error {
	ledger.Version = LedgerVersion
	sort.Slice(ledger.Records, func(i, j int) bool {
		return ledger.Records[i].ID < ledger.Records[j].ID
	})
	var buffer bytes.Buffer
	encoder := yaml.NewEncoder(&buffer)
	encoder.SetIndent(2)
	if err := encoder.Encode(ledger); err != nil {
		return fmt.Errorf("encode ledger: %w", err)
	}
	if err := encoder.Close(); err != nil {
		return fmt.Errorf("finish ledger encoding: %w", err)
	}
	return writeAtomic(path, buffer.Bytes())
}

func MutateLedger(ledger Ledger, snapshot Snapshot, catalogDir, operation string, candidate LedgerRecord) (Ledger, error) {
	connector, found := FindConnector(snapshot, candidate.ID)
	if !found {
		return Ledger{}, fmt.Errorf("unknown connector id %q", candidate.ID)
	}
	candidate.Name = connector.Name
	index := -1
	for current := range ledger.Records {
		if ledger.Records[current].ID == candidate.ID {
			index = current
			break
		}
	}
	switch operation {
	case "add":
		if index >= 0 {
			return Ledger{}, fmt.Errorf("connector %q is already reviewed", candidate.ID)
		}
		ledger.Records = append(ledger.Records, candidate)
	case "update":
		if index < 0 {
			return Ledger{}, fmt.Errorf("connector %q is not in the ledger", candidate.ID)
		}
		ledger.Records[index] = candidate
	default:
		return Ledger{}, fmt.Errorf("unknown ledger operation %q", operation)
	}
	if err := ValidateLedger(ledger, catalogDir); err != nil {
		return Ledger{}, err
	}
	return ledger, nil
}

func ValidateLedger(ledger Ledger, catalogDir string) error {
	var problems []error
	if ledger.Version != LedgerVersion {
		problems = append(problems, fmt.Errorf("unsupported ledger version %d", ledger.Version))
	}
	seen := map[string]bool{}
	for index, record := range ledger.Records {
		label := fmt.Sprintf("record %d", index+1)
		if strings.TrimSpace(record.ID) == "" {
			problems = append(problems, fmt.Errorf("%s: id is required", label))
		} else if seen[record.ID] {
			problems = append(problems, fmt.Errorf("%s: duplicate id %q", label, record.ID))
		}
		seen[record.ID] = true
		if strings.TrimSpace(record.Name) == "" {
			problems = append(problems, fmt.Errorf("%s: name is required", label))
		}
		switch record.Status {
		case "existing", "imported":
			if record.CatalogEntry == "" {
				problems = append(problems, fmt.Errorf("%s: catalog_entry is required for %s", label, record.Status))
			} else if err := validateCatalogEntry(catalogDir, record.CatalogEntry); err != nil {
				problems = append(problems, fmt.Errorf("%s: %w", label, err))
			}
			if record.Reason != "" {
				problems = append(problems, fmt.Errorf("%s: reason is not allowed for %s", label, record.Status))
			}
		case "skipped":
			if strings.TrimSpace(record.Reason) == "" {
				problems = append(problems, fmt.Errorf("%s: reason is required for skipped", label))
			}
			if record.CatalogEntry != "" {
				problems = append(problems, fmt.Errorf("%s: catalog_entry is not allowed for skipped", label))
			}
		default:
			problems = append(problems, fmt.Errorf("%s: invalid status %q", label, record.Status))
		}
	}
	return errors.Join(problems...)
}

func validateCatalogEntry(catalogDir, entry string) error {
	if !filepath.IsLocal(entry) || filepath.Clean(entry) != entry || (filepath.Ext(entry) != ".yaml" && filepath.Ext(entry) != ".yml") {
		return fmt.Errorf("catalog_entry %q must be a relative catalog YAML path", entry)
	}
	info, err := os.Stat(filepath.Join(catalogDir, entry))
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("catalog entry %q does not exist", entry)
		}
		return fmt.Errorf("inspect catalog entry %q: %w", entry, err)
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("catalog entry %q is not a regular file", entry)
	}
	return nil
}

func ReviewedIDs(ledger Ledger) map[string]bool {
	reviewed := make(map[string]bool, len(ledger.Records))
	for _, record := range ledger.Records {
		reviewed[record.ID] = true
	}
	return reviewed
}
