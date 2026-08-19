package intake

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestFetchPaginatesFiltersAndRanks(t *testing.T) {
	t.Parallel()
	fixedTime := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if got := r.URL.Query().Get("limit"); got != "500" {
			t.Errorf("limit = %q", got)
		}
		if got := r.URL.Query().Get("verified_tier"); got != TierFilter {
			t.Errorf("verified_tier = %q", got)
		}
		if got := r.URL.Query().Get("source"); got != "test" {
			t.Errorf("source = %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Query().Get("cursor") {
		case "":
			_, _ = w.Write([]byte(`{"data":[
                    {"id":"community","name":"Community","verified_tier":"community","popularity_score":10,"trending_score":null,"transport":"streamable_http","url":"https://example.com/mcp","extra":{"preserved":true}},
                    {"id":"sse","name":"SSE","verified_tier":"anthropic","popularity_score":99,"transport":"sse","url":"https://example.com/sse"},
                    {"id":"local","name":"Local","verified_tier":"anthropic","popularity_score":100,"transport":"streamable_http","command":"server"}
                ],"next_cursor":"page-2"}`))
		case "page-2":
			_, _ = w.Write([]byte(`{"data":[
                    {"uuid":"partner","display_name":"Partner","verification_tier":"partner","popularity":10,"trending":5,"connection":{"transport_type":"streamable-http","server_url":"https://partner.example/mcp"}},
                    {"id":"unknown-score","name":"Unknown Score","verified_tier":"anthropic","popularity_score":null,"trending_score":null,"transport":"streamable HTTP","endpoint_url":"https://unknown.example/mcp"}
                ]}`))
		default:
			t.Errorf("unexpected cursor %q", r.URL.Query().Get("cursor"))
		}
	}))
	defer server.Close()

	snapshot, err := (Fetcher{Client: server.Client(), Now: func() time.Time { return fixedTime }}).Fetch(context.Background(), server.URL+"?source=test")
	if err != nil {
		t.Fatal(err)
	}
	if requests != 2 || snapshot.PageCount != 2 || snapshot.Fetched != 5 || snapshot.Eligible != 3 {
		t.Fatalf("unexpected metadata: requests=%d snapshot=%+v", requests, snapshot)
	}
	if !snapshot.FetchedAt.Equal(fixedTime) {
		t.Fatalf("fetched_at = %v", snapshot.FetchedAt)
	}
	gotOrder := []string{snapshot.Connectors[0].ID, snapshot.Connectors[1].ID, snapshot.Connectors[2].ID}
	wantOrder := []string{"partner", "unknown-score", "community"}
	if strings.Join(gotOrder, ",") != strings.Join(wantOrder, ",") {
		t.Fatalf("order = %v, want %v", gotOrder, wantOrder)
	}
	for index, connector := range snapshot.Connectors {
		if connector.Rank != index+1 {
			t.Errorf("rank %d = %d", index, connector.Rank)
		}
	}
	var complete map[string]any
	community, found := FindConnector(snapshot, "community")
	if !found {
		t.Fatal("community connector not found")
	}
	if err := json.Unmarshal(community.Record, &complete); err != nil {
		t.Fatal(err)
	}
	if complete["extra"].(map[string]any)["preserved"] != true {
		t.Fatal("complete raw record was not preserved")
	}
}

func TestFetchUnderstandsAnthropicRegistryRecordShape(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"servers":[{"server":{"name":"app.linear/linear","title":"Linear","remotes":[{"type":"streamable-http","url":"https://mcp.linear.app/mcp"}]},"_meta":{"com.anthropic.api/mcp-registry":{"uuid":"directory-uuid","displayName":"Linear Directory Name","type":"remote"}}}],"metadata":{"count":1}}`))
	}))
	defer server.Close()
	snapshot, err := (Fetcher{Client: server.Client()}).Fetch(context.Background(), server.URL)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Eligible != 1 || snapshot.Connectors[0].ID != "directory-uuid" || snapshot.Connectors[0].Name != "Linear Directory Name" {
		t.Fatalf("connector = %+v", snapshot.Connectors)
	}
}

func TestFetchUnderstandsClaudeDirectoryRecordShape(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"servers":[
            {"id":"new-unscored","name":"New Unscored","verified_tier":"community","rank":1,"popularity_score":null,"trending_score":null,"remote":{"url":"https://new.example/mcp","transport":"streamable-http"}},
            {"id":"scored","name":"Scored","verified_tier":"community","rank":103,"popularity_score":6943,"trending_score":846987,"remote":{"url":"https://scored.example/mcp","transport":"streamable-http"}},
            {"id":"partial","name":"Partial","verified_tier":"anthropic","rank":2,"popularity_score":9000,"trending_score":null,"remote":{"url":"https://partial.example/mcp","transport":"streamable-http"}}
        ],"next_cursor":null,"total":2004}`))
	}))
	defer server.Close()

	snapshot, err := (Fetcher{Client: server.Client()}).Fetch(context.Background(), server.URL)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.DirectoryTotal != 2004 {
		t.Fatalf("directory total = %d", snapshot.DirectoryTotal)
	}
	got := []string{snapshot.Connectors[0].ID, snapshot.Connectors[1].ID, snapshot.Connectors[2].ID}
	want := []string{"scored", "partial", "new-unscored"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("order = %v, want %v", got, want)
	}
	if snapshot.Connectors[0].Tier != "community" || snapshot.Connectors[0].Popularity == nil || *snapshot.Connectors[0].Popularity != 6943 || snapshot.Connectors[0].Trending == nil || *snapshot.Connectors[0].Trending != 846987 {
		t.Fatalf("scored connector = %+v", snapshot.Connectors[0])
	}
}

func TestDefaultAPIUsesClaudeDirectoryFeed(t *testing.T) {
	t.Parallel()
	if DefaultAPIURL != "https://api.anthropic.com/api/directory/servers" {
		t.Fatalf("default API URL = %q", DefaultAPIURL)
	}
}

func TestFetchDetectsCursorLoop(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"data":[],"next_cursor":"same"}`))
	}))
	defer server.Close()
	_, err := (Fetcher{Client: server.Client()}).Fetch(context.Background(), server.URL)
	if err == nil || !strings.Contains(err.Error(), "cursor loop") {
		t.Fatalf("error = %v", err)
	}
}

func TestFetchDeduplicatesCursorOverlap(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("cursor") == "next" {
			_, _ = w.Write([]byte(`{"servers":[{"id":"same","name":"Same","popularity_score":1,"trending_score":1,"verified_tier":"community","remote":{"transport":"streamable-http","url":"https://same.example/mcp"}}],"total":1}`))
			return
		}
		_, _ = w.Write([]byte(`{"servers":[{"id":"same","name":"Same","popularity_score":2,"trending_score":2,"verified_tier":"community","remote":{"transport":"streamable-http","url":"https://same.example/mcp"}}],"next_cursor":"next","total":1}`))
	}))
	defer server.Close()

	snapshot, err := (Fetcher{Client: server.Client()}).Fetch(context.Background(), server.URL)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Fetched != 2 || snapshot.Eligible != 1 || snapshot.DuplicateCount != 1 {
		t.Fatalf("snapshot metadata = %+v", snapshot)
	}
	if got := *snapshot.Connectors[0].Popularity; got != 2 {
		t.Fatalf("first occurrence was not retained: popularity = %v", got)
	}
}

func TestFetchReportsAPIFailure(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
	}))
	defer server.Close()
	_, err := (Fetcher{Client: server.Client()}).Fetch(context.Background(), server.URL)
	if err == nil || !strings.Contains(err.Error(), "503") || !strings.Contains(err.Error(), "unavailable") {
		t.Fatalf("error = %v", err)
	}
}

func TestRankConnectorsDeterministicTiesAndNulls(t *testing.T) {
	t.Parallel()
	one := 1.0
	connectors := []Connector{
		{ID: "b", Name: "same", Tier: "unknown", Popularity: &one, Trending: &one},
		{ID: "z", Name: "Zulu", Tier: "community", Popularity: &one, Trending: &one},
		{ID: "a", Name: "same", Tier: "unknown", Popularity: &one, Trending: &one},
		{ID: "null", Name: "Null", Tier: "anthropic"},
		{ID: "partner", Name: "Partner", Tier: "partner", Popularity: &one, Trending: &one},
	}
	rankConnectors(connectors)
	got := make([]string, len(connectors))
	for index := range connectors {
		got[index] = connectors[index].ID
	}
	want := []string{"partner", "z", "a", "b", "null"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("order = %v, want %v", got, want)
	}
}

func TestFilteringRequiresTransportAndURLOnSameRemote(t *testing.T) {
	t.Parallel()
	raw := json.RawMessage(`{"id":"mixed","name":"Mixed","website":"https://example.com","package":{"type":"streamable-http"},"remotes":[{"type":"sse","url":"https://example.com/sse"}]}`)
	if eligibleRemoteStreamableHTTP(raw) {
		t.Fatal("unrelated website and transport fields must not make a connector eligible")
	}
}

func TestFilteringRequiresPublicHTTPSEndpoint(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		endpoint string
		eligible bool
	}{
		{name: "public hostname", endpoint: "https://mcp.example.com/mcp", eligible: true},
		{name: "public IPv4", endpoint: "https://8.8.8.8/mcp", eligible: true},
		{name: "relative", endpoint: "/mcp"},
		{name: "HTTP", endpoint: "http://mcp.example.com/mcp"},
		{name: "non-HTTP", endpoint: "file:///etc/passwd"},
		{name: "localhost", endpoint: "https://localhost/mcp"},
		{name: "localhost subdomain", endpoint: "https://mcp.localhost/mcp"},
		{name: "loopback IPv4", endpoint: "https://127.0.0.1/mcp"},
		{name: "private IPv4", endpoint: "https://10.0.0.1/mcp"},
		{name: "link-local IPv4", endpoint: "https://169.254.169.254/mcp"},
		{name: "loopback IPv6", endpoint: "https://[::1]/mcp"},
		{name: "private IPv6", endpoint: "https://[fd00::1]/mcp"},
		{name: "credentials", endpoint: "https://user@example.com/mcp"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			raw := json.RawMessage(fmt.Sprintf(`{"id":"test","name":"Test","remote":{"transport":"streamable-http","url":%q}}`, test.endpoint))
			if got := eligibleRemoteStreamableHTTP(raw); got != test.eligible {
				t.Fatalf("eligible = %v, want %v", got, test.eligible)
			}
		})
	}
}
