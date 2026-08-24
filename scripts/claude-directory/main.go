package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/obot-platform/mcp-catalog/scripts/claude-directory/internal/intake"
)

type paths struct {
	moduleDir   string
	stateDir    string
	snapshot    string
	current     string
	ledger      string
	catalogRoot string
}

func main() {
	if err := run(context.Background(), os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		usage(stderr)
		return errors.New("a command is required")
	}
	locations, err := resolvePaths()
	if err != nil {
		return err
	}
	switch args[0] {
	case "refresh":
		return refresh(ctx, locations, args[1:], stdout, stderr)
	case "list":
		return list(locations, args[1:], stdout, stderr)
	case "select":
		return selectConnector(locations, args[1:], stdout, stderr)
	case "show":
		return show(locations, args[1:], stdout)
	case "ledger":
		return ledger(locations, args[1:], stdout, stderr)
	case "help", "-h", "--help":
		usage(stdout)
		return nil
	default:
		usage(stderr)
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func resolvePaths() (paths, error) {
	workingDirectory, err := os.Getwd()
	if err != nil {
		return paths{}, fmt.Errorf("get working directory: %w", err)
	}
	moduleDir := workingDirectory
	if filepath.Base(moduleDir) != "claude-directory" {
		candidate := filepath.Join(moduleDir, "scripts", "claude-directory")
		if _, err := os.Stat(filepath.Join(candidate, "go.mod")); err == nil {
			moduleDir = candidate
		}
	}
	if _, err := os.Stat(filepath.Join(moduleDir, "go.mod")); err != nil {
		return paths{}, errors.New("run from the catalog root or scripts/claude-directory")
	}
	catalogRoot := filepath.Clean(filepath.Join(moduleDir, "..", ".."))
	stateDir := filepath.Join(moduleDir, ".state")
	return paths{
		moduleDir: moduleDir, stateDir: stateDir,
		snapshot: filepath.Join(stateDir, "directory.json"),
		current:  filepath.Join(stateDir, "current.json"),
		ledger:   filepath.Join(moduleDir, "reviewed.yaml"), catalogRoot: catalogRoot,
	}, nil
}

func refresh(ctx context.Context, locations paths, args []string, stdout, stderr io.Writer) error {
	flags := flag.NewFlagSet("refresh", flag.ContinueOnError)
	flags.SetOutput(stderr)
	defaultURL := os.Getenv("CLAUDE_DIRECTORY_API_URL")
	if defaultURL == "" {
		defaultURL = intake.DefaultAPIURL
	}
	apiURL := flags.String("api-url", defaultURL, "Claude directory API URL")
	timeout := flags.Duration("timeout", 2*time.Minute, "overall fetch timeout")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return errors.New("refresh does not accept positional arguments")
	}
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	snapshot, err := (intake.Fetcher{}).Fetch(ctx, *apiURL)
	if err != nil {
		return err
	}
	if err := intake.WriteJSONAtomic(locations.snapshot, snapshot); err != nil {
		return err
	}
	fmt.Fprintf(stdout, "Refreshed %d eligible connectors from %d fetched records (directory total %d) across %d pages", snapshot.Eligible, snapshot.Fetched, snapshot.DirectoryTotal, snapshot.PageCount)
	if snapshot.DuplicateCount > 0 {
		fmt.Fprintf(stdout, "; ignored %d repeated IDs", snapshot.DuplicateCount)
	}
	fmt.Fprintf(stdout, "\nSnapshot: %s\n", locations.snapshot)
	return nil
}

func list(locations paths, args []string, stdout, stderr io.Writer) error {
	flags := flag.NewFlagSet("list", flag.ContinueOnError)
	flags.SetOutput(stderr)
	includeReviewed := flags.Bool("include-reviewed", false, "include connectors already in the ledger")
	limit := flags.Int("limit", 0, "maximum number of connectors (0 means all)")
	jsonOutput := flags.Bool("json", false, "write JSON instead of a table")
	output := flags.String("output", "table", "output format: table or json")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return errors.New("list does not accept positional arguments")
	}
	if *limit < 0 {
		return errors.New("limit cannot be negative")
	}
	if *jsonOutput {
		*output = "json"
	}
	if *output != "table" && *output != "json" {
		return fmt.Errorf("invalid output format %q", *output)
	}
	snapshot, ledgerValue, err := loadState(locations)
	if err != nil {
		return err
	}
	reviewed := intake.ReviewedIDs(ledgerValue)
	connectors := make([]intake.Connector, 0, len(snapshot.Connectors))
	for _, connector := range snapshot.Connectors {
		if !*includeReviewed && reviewed[connector.ID] {
			continue
		}
		connectors = append(connectors, connector)
		if *limit > 0 && len(connectors) == *limit {
			break
		}
	}
	if *output == "json" {
		encoder := json.NewEncoder(stdout)
		encoder.SetIndent("", "  ")
		return encoder.Encode(connectors)
	}
	w := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "RANK\tNAME\tID\tTIER\tPOPULARITY\tTRENDING\tREVIEWED")
	for _, connector := range connectors {
		fmt.Fprintf(w, "%d\t%s\t%s\t%s\t%s\t%s\t%t\n", connector.Rank, connector.Name, connector.ID, connector.Tier, score(connector.Popularity), score(connector.Trending), reviewed[connector.ID])
	}
	return w.Flush()
}

func selectConnector(locations paths, args []string, stdout, stderr io.Writer) error {
	flags := flag.NewFlagSet("select", flag.ContinueOnError)
	flags.SetOutput(stderr)
	id := flags.String("id", "", "specific unreviewed connector id")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return errors.New("select does not accept positional arguments")
	}
	snapshot, ledgerValue, err := loadState(locations)
	if err != nil {
		return err
	}
	reviewed := intake.ReviewedIDs(ledgerValue)
	var selected intake.Connector
	found := false
	for _, connector := range snapshot.Connectors {
		if (*id == "" || connector.ID == *id) && !reviewed[connector.ID] {
			selected, found = connector, true
			break
		}
	}
	if !found {
		if *id != "" {
			if _, exists := intake.FindConnector(snapshot, *id); !exists {
				return fmt.Errorf("unknown connector id %q", *id)
			}
			return fmt.Errorf("connector %q has already been reviewed", *id)
		}
		return errors.New("no unreviewed connectors remain")
	}
	var complete any
	if err := json.Unmarshal(selected.Record, &complete); err != nil {
		return fmt.Errorf("decode selected connector: %w", err)
	}
	workingFile := struct {
		Rank      int `json:"rank"`
		Connector any `json:"connector"`
	}{Rank: selected.Rank, Connector: complete}
	if err := intake.WriteJSONAtomic(locations.current, workingFile); err != nil {
		return err
	}
	fmt.Fprintf(stdout, "Selected %s (%s), rank %d\nWorking file: %s\n", selected.Name, selected.ID, selected.Rank, locations.current)
	return nil
}

func show(locations paths, args []string, stdout io.Writer) error {
	if len(args) != 1 {
		return errors.New("usage: show ID")
	}
	snapshot, err := intake.ReadSnapshot(locations.snapshot)
	if err != nil {
		return err
	}
	connector, found := intake.FindConnector(snapshot, args[0])
	if !found {
		return fmt.Errorf("unknown connector id %q", args[0])
	}
	var record any
	if err := json.Unmarshal(connector.Record, &record); err != nil {
		return err
	}
	encoder := json.NewEncoder(stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(record)
}

func ledger(locations paths, args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		return errors.New("usage: ledger add|update|check")
	}
	if args[0] == "check" {
		if len(args) != 1 {
			return errors.New("ledger check does not accept arguments")
		}
		ledgerValue, err := intake.ReadLedger(locations.ledger)
		if err != nil {
			return err
		}
		if err := intake.ValidateLedger(ledgerValue, locations.catalogRoot); err != nil {
			return err
		}
		fmt.Fprintf(stdout, "Ledger is valid (%d records)\n", len(ledgerValue.Records))
		return nil
	}
	if args[0] != "add" && args[0] != "update" {
		return fmt.Errorf("unknown ledger command %q", args[0])
	}
	flags := flag.NewFlagSet("ledger "+args[0], flag.ContinueOnError)
	flags.SetOutput(stderr)
	id := flags.String("id", "", "Claude directory connector id")
	status := flags.String("status", "", "existing, imported, or skipped")
	catalogEntry := flags.String("catalog-entry", "", "repository-relative catalog YAML path")
	reason := flags.String("reason", "", "reason the connector was skipped")
	if err := flags.Parse(args[1:]); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return errors.New("ledger mutations do not accept positional arguments")
	}
	if strings.TrimSpace(*id) == "" || strings.TrimSpace(*status) == "" {
		return errors.New("--id and --status are required")
	}
	snapshot, err := intake.ReadSnapshot(locations.snapshot)
	if err != nil {
		return err
	}
	ledgerValue, err := intake.ReadLedger(locations.ledger)
	if err != nil {
		return err
	}
	record := intake.LedgerRecord{ID: strings.TrimSpace(*id), Status: strings.TrimSpace(*status), CatalogEntry: strings.TrimSpace(*catalogEntry), Reason: strings.TrimSpace(*reason)}
	updated, err := intake.MutateLedger(ledgerValue, snapshot, locations.catalogRoot, args[0], record)
	if err != nil {
		return err
	}
	if err := intake.WriteLedgerAtomic(locations.ledger, updated); err != nil {
		return err
	}
	fmt.Fprintf(stdout, "Ledger %s: %s (%s) is %s\n", args[0], record.ID, canonicalName(snapshot, record.ID), record.Status)
	return nil
}

func loadState(locations paths) (intake.Snapshot, intake.Ledger, error) {
	snapshot, err := intake.ReadSnapshot(locations.snapshot)
	if err != nil {
		return intake.Snapshot{}, intake.Ledger{}, err
	}
	ledgerValue, err := intake.ReadLedger(locations.ledger)
	if err != nil {
		return intake.Snapshot{}, intake.Ledger{}, err
	}
	if err := intake.ValidateLedger(ledgerValue, locations.catalogRoot); err != nil {
		return intake.Snapshot{}, intake.Ledger{}, err
	}
	return snapshot, ledgerValue, nil
}

func canonicalName(snapshot intake.Snapshot, id string) string {
	connector, _ := intake.FindConnector(snapshot, id)
	return connector.Name
}

func score(value *float64) string {
	if value == nil {
		return "-"
	}
	return strconv.FormatFloat(*value, 'f', -1, 64)
}

func usage(output io.Writer) {
	fmt.Fprintln(output, `Usage: claude-directory COMMAND

Commands:
  refresh                 Fetch and atomically replace the directory snapshot
  list [flags]            List snapshot candidates without network access
  select [--id ID]        Write an unreviewed connector to .state/current.json
  show ID                 Print a connector's complete snapshot record
  ledger add [flags]      Record a new disposition
  ledger update [flags]   Correct an existing disposition
  ledger check            Validate the tracked ledger`)
}
