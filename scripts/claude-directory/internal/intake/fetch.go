package intake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

const (
	DefaultAPIURL = "https://api.anthropic.com/api/directory/servers"
	TierFilter    = "anthropic,partner,community"
)

type Fetcher struct {
	Client *http.Client
	Now    func() time.Time
}

func (f Fetcher) Fetch(ctx context.Context, apiURL string) (Snapshot, error) {
	if f.Client == nil {
		f.Client = http.DefaultClient
	}
	if f.Now == nil {
		f.Now = time.Now
	}
	base, err := url.Parse(apiURL)
	if err != nil {
		return Snapshot{}, fmt.Errorf("parse API URL: %w", err)
	}
	query := base.Query()
	query.Set("limit", "500")
	query.Set("verified_tier", TierFilter)
	base.RawQuery = ""

	var records []json.RawMessage
	seenCursors := map[string]bool{}
	cursor := ""
	pageCount := 0
	directoryTotal := 0
	for {
		requestURL := *base
		pageQuery := cloneValues(query)
		if cursor != "" {
			pageQuery.Set("cursor", cursor)
		}
		requestURL.RawQuery = pageQuery.Encode()

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL.String(), nil)
		if err != nil {
			return Snapshot{}, fmt.Errorf("create directory request: %w", err)
		}
		resp, err := f.Client.Do(req)
		if err != nil {
			return Snapshot{}, fmt.Errorf("fetch directory page %d: %w", pageCount+1, err)
		}
		body, readErr := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
		closeErr := resp.Body.Close()
		if readErr != nil {
			return Snapshot{}, fmt.Errorf("read directory page %d: %w", pageCount+1, readErr)
		}
		if closeErr != nil {
			return Snapshot{}, fmt.Errorf("close directory response: %w", closeErr)
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			return Snapshot{}, fmt.Errorf("directory page %d returned %s: %s", pageCount+1, resp.Status, strings.TrimSpace(string(body)))
		}

		pageRecords, next, total, err := decodePage(body)
		if err != nil {
			return Snapshot{}, fmt.Errorf("decode directory page %d: %w", pageCount+1, err)
		}
		records = append(records, pageRecords...)
		if directoryTotal == 0 && total > 0 {
			directoryTotal = total
		}
		pageCount++
		if next == "" {
			break
		}
		if seenCursors[next] || next == cursor {
			return Snapshot{}, fmt.Errorf("directory cursor loop detected at %q", next)
		}
		seenCursors[next] = true
		cursor = next
	}

	connectors := make([]Connector, 0, len(records))
	seenIDs := map[string]struct{}{}
	duplicateCount := 0
	for _, raw := range records {
		if !eligibleRemoteStreamableHTTP(raw) {
			continue
		}
		connector, err := connectorFromRaw(raw)
		if err != nil {
			return Snapshot{}, err
		}
		if _, seen := seenIDs[connector.ID]; seen {
			duplicateCount++
			continue
		}
		seenIDs[connector.ID] = struct{}{}
		connectors = append(connectors, connector)
	}
	rankConnectors(connectors)

	effectiveQuery := make(map[string]string, len(query))
	for key := range query {
		effectiveQuery[key] = query.Get(key)
	}
	return Snapshot{
		Version:        SnapshotVersion,
		FetchedAt:      f.Now().UTC(),
		SourceURL:      base.String(),
		Query:          effectiveQuery,
		PageCount:      pageCount,
		Fetched:        len(records),
		DirectoryTotal: directoryTotal,
		DuplicateCount: duplicateCount,
		Eligible:       len(connectors),
		Connectors:     connectors,
	}, nil
}

func cloneValues(source url.Values) url.Values {
	result := make(url.Values, len(source))
	for key, values := range source {
		result[key] = append([]string(nil), values...)
	}
	return result
}

// decodePage normalizes the response envelopes used by the Claude and Anthropic
// directory APIs. Connector records remain json.RawMessage so their complete
// source JSON is preserved without coercing numbers through float64.
func decodePage(body []byte) ([]json.RawMessage, string, int, error) {
	var root map[string]json.RawMessage
	if err := json.Unmarshal(body, &root); err != nil {
		return nil, "", 0, err
	}
	records, ok := rawArray(root, "data", "servers", "connectors", "items")
	if !ok {
		for _, key := range []string{"data", "result"} {
			var nested map[string]json.RawMessage
			if raw, exists := root[key]; exists && json.Unmarshal(raw, &nested) == nil {
				if found, exists := rawArray(nested, "servers", "connectors", "items", "data"); exists {
					records, ok = found, true
					break
				}
			}
		}
	}
	if !ok {
		return nil, "", 0, errors.New("response has no connector array")
	}
	next := rawString(root, "next_cursor", "nextCursor", "cursor")
	if next == "" {
		for _, key := range []string{"meta", "metadata", "pagination", "data"} {
			var nested map[string]json.RawMessage
			if raw, exists := root[key]; exists && json.Unmarshal(raw, &nested) == nil {
				if next = rawString(nested, "next_cursor", "nextCursor", "cursor"); next != "" {
					break
				}
			}
		}
	}
	return records, next, rawInt(root, "total"), nil
}

func rawInt(root map[string]json.RawMessage, key string) int {
	var value int
	if raw, ok := root[key]; ok {
		_ = json.Unmarshal(raw, &value)
	}
	return value
}

func rawArray(root map[string]json.RawMessage, keys ...string) ([]json.RawMessage, bool) {
	for _, key := range keys {
		var records []json.RawMessage
		if raw, ok := root[key]; ok && json.Unmarshal(raw, &records) == nil {
			return records, true
		}
	}
	return nil, false
}

func rawString(root map[string]json.RawMessage, keys ...string) string {
	for _, key := range keys {
		var value string
		if raw, ok := root[key]; ok && json.Unmarshal(raw, &value) == nil {
			return value
		}
	}
	return ""
}

func rankConnectors(connectors []Connector) {
	sort.Slice(connectors, func(i, j int) bool {
		left, right := connectors[i], connectors[j]
		leftRanked := left.Popularity != nil && left.Trending != nil
		rightRanked := right.Popularity != nil && right.Trending != nil
		if leftRanked != rightRanked {
			return leftRanked
		}
		if leftRanked {
			if compared := compareScores(left.Popularity, right.Popularity); compared != 0 {
				return compared < 0
			}
			if compared := compareScores(left.Trending, right.Trending); compared != 0 {
				return compared < 0
			}
		}
		if tierRank(left.Tier) != tierRank(right.Tier) {
			return tierRank(left.Tier) < tierRank(right.Tier)
		}
		if strings.ToLower(left.Name) != strings.ToLower(right.Name) {
			return strings.ToLower(left.Name) < strings.ToLower(right.Name)
		}
		return left.ID < right.ID
	})
	for index := range connectors {
		connectors[index].Rank = index + 1
	}
}

func compareScores(left, right *float64) int {
	if left == nil && right == nil {
		return 0
	}
	if left == nil {
		return 1
	}
	if right == nil {
		return -1
	}
	if *left > *right {
		return -1
	}
	if *left < *right {
		return 1
	}
	return 0
}

func tierRank(tier string) int {
	switch strings.ToLower(tier) {
	case "anthropic":
		return 0
	case "partner":
		return 1
	case "community":
		return 2
	default:
		return 3
	}
}
