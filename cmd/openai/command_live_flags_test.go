package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func TestLiveRejectsUnsupportedInheritedFlags(t *testing.T) {
	var connections, requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		connections.Add(1)
		http.Error(w, "unexpected connection", http.StatusInternalServerError)
	}))
	defer server.Close()
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		requests.Add(1)
		return nil, errors.New("unexpected HTTP API request")
	})
	timeoutError := "--timeout applies to HTTP API requests; use --duration to limit the Live session"
	for _, tc := range []struct {
		name, flag string
		args       []string
	}{
		{"legacy", "legacy", []string{"--legacy"}},
		{"legacy true", "legacy", []string{"--legacy=true"}},
		{"legacy false", "legacy", []string{"--legacy=false"}},
		{"temporary", "temporary", []string{"--temporary"}},
		{"temporary true", "temporary", []string{"--temporary=true"}},
		{"temporary false", "temporary", []string{"--temporary=false"}},
		{"temporary shorthand", "temporary", []string{"-t"}},
		{"temporary shorthand false", "temporary", []string{"-t=false"}},
		{"history directory", "history-dir", []string{"--history-dir", "unused-history"}},
		{"history default", "history-dir", []string{"--history-dir", defaultHistoryDir()}},
		{"history empty", "history-dir", []string{"--history-dir="}},
		{"new session", "session", []string{"--session", "new"}},
		{"existing session", "session", []string{"--session", "unused-session"}},
		{"session empty", "session", []string{"--session="}},
		{"timeout", "timeout", []string{"--timeout", "30s"}},
		{"timeout default", "timeout", []string{"--timeout", "2m"}},
		{"timeout disabled", "timeout", []string{"--timeout", "0"}},
		{"timeout negative", "timeout", []string{"--timeout=-1s"}},
		{"legacy session conflict", "legacy", []string{"--legacy", "--session", "new"}},
		{"temporary session conflict", "temporary", []string{"-t", "--session", "new"}},
	} {
		for _, before := range []bool{false, true} {
			position := "after command"
			if before {
				position = "before command"
			}
			t.Run(tc.name+"/"+position, func(t *testing.T) {
				dir := t.TempDir()
				input := filepath.Join(dir, "input.pcm")
				if err := os.WriteFile(input, []byte{1, 2}, 0600); err != nil {
					t.Fatal(err)
				}
				args := []string{"live", "--input-pcm", input, "--output-pcm", filepath.Join(dir, "output.pcm"),
					"--event-log", filepath.Join(dir, "events.jsonl"), "--base-url", server.URL + "/v1/"}
				if before {
					args = append(append([]string{}, tc.args...), args...)
				} else {
					args = append(args, tc.args...)
				}
				out, _, err := executeTest(t, args, "", transport)
				want := "--" + tc.flag + " is not supported by live; omit this flag"
				if tc.flag == "timeout" {
					want = timeoutError
				}
				if err == nil || err.Error() != want || out != "" {
					t.Fatalf("stdout=%q err=%v; want %q", out, err, want)
				}
				entries, err := os.ReadDir(dir)
				if err != nil || len(entries) != 1 || entries[0].Name() != "input.pcm" {
					t.Fatalf("unexpected output or event files: %v, err=%v", entries, err)
				}
			})
		}
	}
	if connections.Load() != 0 || requests.Load() != 0 {
		t.Fatalf("made %d Live connections and %d HTTP API requests", connections.Load(), requests.Load())
	}
}

func TestLiveFlagValidationBeforeFileAccess(t *testing.T) {
	for _, tc := range []struct {
		args []string
		want string
	}{
		{[]string{"live", "--temporary=false"}, "--temporary is not supported by live"},
		{[]string{"live", "--timeout=0", "--input-pcm", "missing.pcm", "--output-pcm", "unused.pcm"}, "use --duration"},
	} {
		out, _, err := executeTest(t, tc.args, "", nil)
		if err == nil || !strings.Contains(err.Error(), tc.want) || out != "" {
			t.Fatalf("args=%v stdout=%q err=%v", tc.args, out, err)
		}
	}
}

func TestLiveUnsupportedFlagsExitCode(t *testing.T) {
	for _, flag := range []string{"--legacy=false", "-t=false", "--history-dir=", "--session=", "--timeout=0"} {
		var out, errOut bytes.Buffer
		code := run(t.Context(), []string{"live", flag}, strings.NewReader(""), &out, &errOut)
		if code != 1 || out.Len() != 0 || !strings.HasPrefix(errOut.String(), "openai: --") {
			t.Fatalf("flag=%s code=%d stdout=%q stderr=%q", flag, code, out.String(), errOut.String())
		}
		if !strings.Contains(errOut.String(), "is not supported by live") && !strings.Contains(errOut.String(), "use --duration") {
			t.Fatalf("flag=%s stderr=%q", flag, errOut.String())
		}
	}
}

func TestLiveSupportedFlagsAndUntouchedDefaults(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
	}{
		{"untouched defaults", nil},
		{"disabled response flags", []string{"--stream=false", "--web-search=false"}},
		{"Live options", []string{"--model", "synthetic-live", "--voice", "cedar", "--backend-model", "synthetic-backend",
			"--backend-max-output-tokens", "1024", "--sample-rate", "16000", "--duration", "1s",
			"--instructions", "Synthetic instructions", "--controls=false"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := liveFixtureServer(t)
			defer server.Close()
			dir := t.TempDir()
			input := filepath.Join(dir, "input.pcm")
			if err := os.WriteFile(input, []byte{1, 2}, 0600); err != nil {
				t.Fatal(err)
			}
			args := append([]string{"live", "--input-pcm", input, "--output-pcm", filepath.Join(dir, "output.pcm"),
				"--event-log", filepath.Join(dir, "events.jsonl"), "--base-url", server.URL + "/v1/",
				"--listen-after", "0", "-o", "json"}, tc.args...)
			out, _, err := executeTest(t, args, "", nil)
			if err != nil || !json.Valid([]byte(out)) || !strings.Contains(out, `"output_saved":true`) || !strings.Contains(out, `"event_log_saved":true`) {
				t.Fatalf("stdout=%q err=%v", out, err)
			}
		})
	}
}

func TestLiveResponseFlagsStillRejectEnabledValues(t *testing.T) {
	for _, tc := range []struct {
		flag, want string
	}{
		{"--stream=true", "Live already streams over its primary WebSocket; omit --stream"},
		{"--web-search=true", "web-search tools are not configured in this bounded Live client"},
	} {
		out, _, err := executeTest(t, []string{"live", "--input-pcm", "missing.pcm", "--output-pcm", "unused.pcm", tc.flag}, "", nil)
		if err == nil || err.Error() != tc.want || out != "" {
			t.Fatalf("flag=%s stdout=%q err=%v", tc.flag, out, err)
		}
	}
}

func TestLiveAndHTTPTimeoutHelp(t *testing.T) {
	for _, args := range [][]string{{"--help"}, {"chat", "--help"}, {"live", "--help"}, {"live", "--timeout=0", "--help"}} {
		out, _, err := executeTest(t, args, "", nil)
		if err != nil || !strings.Contains(out, "HTTP API request timeout (0 disables; Live uses --duration)") || !strings.Contains(out, "(default 2m0s)") {
			t.Fatalf("args=%v help=%q err=%v", args, out, err)
		}
		for _, name := range []string{"--legacy", "--temporary", "--history-dir", "--session"} {
			if !strings.Contains(out, name) {
				t.Errorf("args=%v help does not list %s", args, name)
			}
		}
		if args[0] == "live" {
			for _, want := range []string{"Use --duration to limit the Live session", "15-second bounds", "are not supported", "Live voice model", `(default "gpt-live-1")`} {
				if !strings.Contains(out, want) {
					t.Errorf("Live help does not contain %q: %s", want, out)
				}
			}
		}
	}
}
