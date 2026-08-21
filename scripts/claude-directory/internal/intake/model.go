package intake

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"
)

const SnapshotVersion = 1

type Snapshot struct {
	Version        int               `json:"version"`
	FetchedAt      time.Time         `json:"fetched_at"`
	SourceURL      string            `json:"source_url"`
	Query          map[string]string `json:"query"`
	PageCount      int               `json:"page_count"`
	Fetched        int               `json:"fetched_count"`
	DirectoryTotal int               `json:"directory_total"`
	DuplicateCount int               `json:"duplicate_count"`
	Eligible       int               `json:"eligible_count"`
	Connectors     []Connector       `json:"connectors"`
}

type Connector struct {
	Rank       int             `json:"rank"`
	ID         string          `json:"id"`
	Name       string          `json:"name"`
	Tier       string          `json:"verified_tier"`
	Popularity *float64        `json:"popularity_score"`
	Trending   *float64        `json:"trending_score"`
	Record     json.RawMessage `json:"record"`
}

func connectorFromRaw(raw json.RawMessage) (Connector, error) {
	var value map[string]any
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		return Connector{}, fmt.Errorf("decode connector: %w", err)
	}

	id := findString(value, "id", "uuid", "directory_id", "directory_uuid")
	name := findString(value, "display_name", "displayName", "title", "name")
	if id == "" || name == "" {
		return Connector{}, fmt.Errorf("connector is missing id or name")
	}

	return Connector{
		ID:         id,
		Name:       name,
		Tier:       strings.ToLower(findString(value, "verified_tier", "verifiedTier", "verification_tier")),
		Popularity: findNumber(value, "popularity_score", "popularityScore", "popularity"),
		Trending:   findNumber(value, "trending_score", "trendingScore", "trending"),
		Record:     append(json.RawMessage(nil), raw...),
	}, nil
}

func eligibleRemoteStreamableHTTP(raw json.RawMessage) bool {
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return false
	}
	return hasRemoteStreamableHTTP(value)
}

func hasRemoteStreamableHTTP(value any) bool {
	switch current := value.(type) {
	case map[string]any:
		streamable := false
		for key, child := range current {
			normalizedKey := normalize(key)
			if normalizedKey == "transport" || normalizedKey == "transporttype" || normalizedKey == "type" {
				if text, ok := child.(string); ok && normalize(text) == "streamablehttp" {
					streamable = true
				}
				if nested, ok := child.(map[string]any); ok && mapDeclaresStreamable(nested) {
					streamable = true
				}
			}
		}
		if streamable && mapHasEndpoint(current) {
			return true
		}
		for _, child := range current {
			if hasRemoteStreamableHTTP(child) {
				return true
			}
		}
	case []any:
		if slices.ContainsFunc(current, hasRemoteStreamableHTTP) {
			return true
		}
	}
	return false
}

func mapDeclaresStreamable(value map[string]any) bool {
	for key, child := range value {
		if normalizedKey := normalize(key); normalizedKey == "transport" || normalizedKey == "transporttype" || normalizedKey == "type" {
			if text, ok := child.(string); ok && normalize(text) == "streamablehttp" {
				return true
			}
		}
	}
	return false
}

func mapHasEndpoint(value map[string]any) bool {
	for key, child := range value {
		normalizedKey := normalize(key)
		if normalizedKey == "url" || normalizedKey == "serverurl" || normalizedKey == "endpointurl" {
			if text, ok := child.(string); ok && isPublicHTTPSEndpoint(text) {
				return true
			}
		}
	}
	return false
}

func isPublicHTTPSEndpoint(raw string) bool {
	endpoint, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || endpoint.Scheme != "https" || endpoint.Host == "" || endpoint.User != nil {
		return false
	}
	host := strings.TrimSuffix(strings.ToLower(endpoint.Hostname()), ".")
	if host == "" || host == "localhost" || strings.HasSuffix(host, ".localhost") || strings.Contains(host, "%") {
		return false
	}
	if ip := net.ParseIP(host); ip != nil {
		return ip.IsGlobalUnicast() && !ip.IsPrivate() && !ip.IsLoopback() && !ip.IsLinkLocalUnicast()
	}
	return true
}

func findString(value map[string]any, keys ...string) string {
	for _, key := range keys {
		if found := findStringKey(value, key); found != "" {
			return found
		}
	}
	return ""
}

func findStringKey(value map[string]any, key string) string {
	if text, ok := value[key].(string); ok && strings.TrimSpace(text) != "" {
		return strings.TrimSpace(text)
	}
	for _, child := range value {
		if nested, ok := child.(map[string]any); ok {
			if found := findStringKey(nested, key); found != "" {
				return found
			}
		}
	}
	return ""
}

func findNumber(value map[string]any, keys ...string) *float64 {
	for _, key := range keys {
		if found := findNumberKey(value, key); found != nil {
			return found
		}
	}
	return nil
}

func findNumberKey(value map[string]any, key string) *float64 {
	if number := numberValue(value[key]); number != nil {
		return number
	}
	for _, child := range value {
		if nested, ok := child.(map[string]any); ok {
			if found := findNumberKey(nested, key); found != nil {
				return found
			}
		}
	}
	return nil
}

func numberValue(value any) *float64 {
	var number float64
	switch typed := value.(type) {
	case json.Number:
		parsed, err := typed.Float64()
		if err != nil {
			return nil
		}
		number = parsed
	case float64:
		number = typed
	case string:
		parsed, err := strconv.ParseFloat(typed, 64)
		if err != nil {
			return nil
		}
		number = parsed
	default:
		return nil
	}
	return &number
}

func normalize(value string) string {
	return strings.Map(func(r rune) rune {
		if r >= 'A' && r <= 'Z' {
			return r + ('a' - 'A')
		}
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' {
			return r
		}
		return -1
	}, value)
}
