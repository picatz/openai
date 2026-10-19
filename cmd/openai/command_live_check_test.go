package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/openai/openai-go/v3/option"
)

type unreadLiveControls struct{}

func (unreadLiveControls) Read([]byte) (int, error) { panic("preflight read stdin controls") }

func executeLiveCheck(t *testing.T, args []string, transport roundTripFunc) (string, error) {
	t.Helper()
	// Deliberately omit WithAPIKey: offline checks must work without credentials.
	cmd := newRootCommand(option.WithHTTPClient(&http.Client{Transport: transport}), option.WithMaxRetries(0))
	cmd.SetArgs(args)
	cmd.SetIn(unreadLiveControls{})
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(io.Discard)
	err := cmd.ExecuteContext(t.Context())
	return out.String(), err
}

func liveCheckFixture(t *testing.T) (string, []string) {
	t.Helper()
	dir := t.TempDir()
	input := filepath.Join(dir, "input.pcm")
	if err := os.WriteFile(input, []byte{1, 2, 3, 4}, 0600); err != nil {
		t.Fatal(err)
	}
	return dir, []string{"live", "--check", "--input-pcm", input,
		"--output-pcm", filepath.Join(dir, "output.pcm"), "--event-log", filepath.Join(dir, "events.jsonl")}
}

func assertLiveCheckUnchanged(t *testing.T, dir string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 1 || entries[0].Name() != "input.pcm" {
		t.Fatalf("unexpected output/log/temp files: %v, err=%v", entries, err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "input.pcm"))
	if err != nil || !bytes.Equal(data, []byte{1, 2, 3, 4}) {
		t.Fatalf("input changed: %v, err=%v", data, err)
	}
}

func TestLiveCheckOfflineReport(t *testing.T) {
	var dials, requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		dials.Add(1)
		http.Error(w, "unexpected connection", 500)
	}))
	defer server.Close()
	transport := roundTripFunc(func(*http.Request) (*http.Response, error) {
		requests.Add(1)
		return nil, errors.New("unexpected HTTP API request")
	})
	for _, key := range []string{"", "synthetic-secret-must-not-appear"} {
		for _, format := range []string{"text", "json"} {
			t.Run(format+"/credentials="+liveCheckCredentialState(key != ""), func(t *testing.T) {
				t.Setenv("OPENAI_API_KEY", key)
				dir, args := liveCheckFixture(t)
				args = append(args, "--base-url", server.URL+"/v1/", "--output", format, "--controls",
					"--sample-rate", "16000", "--model", "synthetic-live", "--backend-model", "synthetic-backend",
					"--voice", "synthetic-voice", "--backend-max-output-tokens", "16", "--duration", "10s", "--listen-after", "0")
				out, err := executeLiveCheck(t, args, transport)
				if err != nil {
					t.Fatal(err)
				}
				if strings.Contains(out, "synthetic-secret") {
					t.Fatal("report exposed a credential")
				}
				if format == "json" {
					var report livePreflightReport
					if err := json.Unmarshal([]byte(out), &report); err != nil {
						t.Fatal(err)
					}
					if report.Check != "offline" || report.LocalValidation != "passed" || report.InputBytes != 4 ||
						report.InputSeconds != 0.000125 || report.SampleRate != 16000 || report.Model != "synthetic-live" ||
						report.BackendModel != "synthetic-backend" || report.Voice != "synthetic-voice" ||
						report.BackendMaxOutputTokens != 16 || report.Duration != "10s" || report.ListenAfter != "0s" ||
						!report.Controls || len(report.Unverified) != 4 || !strings.Contains(report.FormatAssumption, "mono") {
						t.Fatalf("report=%+v", report)
					}
				} else {
					for _, want := range []string{"checks passed (offline)", "Format assumed:", "16000 Hz", "controls: true (not read)",
						"No network connection, output creation, or device access", "Unverified:", "write permissions", "credentials and service", "hardware"} {
						if !strings.Contains(out, want) {
							t.Errorf("report missing %q: %s", want, out)
						}
					}
				}
				assertLiveCheckUnchanged(t, dir)
			})
		}
	}
	if dials.Load() != 0 || requests.Load() != 0 {
		t.Fatalf("offline check made %d Live connections and %d HTTP requests", dials.Load(), requests.Load())
	}
}

func liveCheckCredentialState(value bool) string {
	if value {
		return "present"
	}
	return "absent"
}

func TestLiveCheckRejectsConfigurationBeforeFileAccess(t *testing.T) {
	for _, tc := range []struct {
		args []string
		want string
	}{
		{[]string{"extra"}, "unknown command"},
		{[]string{"--unknown-option"}, "unknown flag"},
		{[]string{"--legacy=false"}, "--legacy is not supported"},
		{[]string{"-t=false"}, "--temporary is not supported"},
		{[]string{"--history-dir="}, "--history-dir is not supported"},
		{[]string{"--session="}, "--session is not supported"},
		{[]string{"--timeout=0"}, "use --duration"},
		{[]string{"--stream"}, "Live already streams"},
		{[]string{"--web-search"}, "web-search tools"},
		{[]string{"--instructions", strings.Repeat("a", 16<<10+1)}, "instructions exceed"},
		{[]string{"--duration=0"}, "duration must be"},
		{[]string{"--duration=10m1s"}, "duration must be"},
		{[]string{"--listen-after=-1s"}, "duration must be"},
		{[]string{"--listen-after=61s"}, "duration must be"},
		{[]string{"--sample-rate=48000"}, "sample rate must be"},
		{[]string{"--model="}, "model, backend model, and voice are required"},
		{[]string{"--backend-model= "}, "model, backend model, and voice are required"},
		{[]string{"--voice="}, "model, backend model, and voice are required"},
		{[]string{"--backend-max-output-tokens=15"}, "between 16 and 16384"},
		{[]string{"--backend-max-output-tokens=16385"}, "between 16 and 16384"},
		{[]string{"--input-pcm=-"}, "explicit regular input/output"},
		{[]string{"--output-pcm="}, "explicit regular input/output"},
		{[]string{"--base-url=http://example.invalid"}, "only on loopback"},
		{[]string{"--base-url=https://user:pass@example.invalid"}, "without credentials"},
		{[]string{"--output=yaml"}, "output must be text or json"},
	} {
		t.Run(strings.Join(tc.args, " ")[:min(len(strings.Join(tc.args, " ")), 80)], func(t *testing.T) {
			args := append([]string{"live", "--check", "--input-pcm=missing.pcm", "--output-pcm=unused.pcm"}, tc.args...)
			out, err := executeLiveCheck(t, args, nil)
			if err == nil || !strings.Contains(err.Error(), tc.want) || out != "" {
				t.Fatalf("stdout=%q err=%v; want %q", out, err, tc.want)
			}
		})
	}
}

func TestLiveCheckRejectsInputWithoutArtifacts(t *testing.T) {
	for _, tc := range []struct {
		name string
		data []byte
		size int64
		want string
	}{
		{"empty", nil, 0, "complete PCM16 samples"},
		{"odd", []byte{1}, 0, "complete PCM16 samples"},
		{"WAV", []byte("RIFF0000WAVE"), 0, "input is WAV"},
		{"over duration", nil, 24000*2*60 + 2, "complete PCM16 samples"},
		{"large sparse", nil, 1 << 40, "complete PCM16 samples"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.size > 1<<30 && runtime.GOOS == "windows" {
				// Truncate does not mark Windows files sparse. Keep this fixture
				// from requiring a huge disk allocation on hosted Windows runners.
				t.Skip("large sparse fixture is Unix-only")
			}
			dir, args := liveCheckFixture(t)
			input := filepath.Join(dir, "input.pcm")
			if err := os.WriteFile(input, tc.data, 0600); err != nil {
				t.Fatal(err)
			}
			if tc.size > 0 {
				if err := os.Truncate(input, tc.size); err != nil {
					t.Fatal(err)
				}
			}
			before, _ := os.Stat(input)
			out, err := executeLiveCheck(t, args, nil)
			if err == nil || !strings.Contains(err.Error(), tc.want) || out != "" {
				t.Fatalf("stdout=%q err=%v", out, err)
			}
			after, _ := os.Stat(input)
			entries, err := os.ReadDir(dir)
			if err != nil || len(entries) != 1 || entries[0].Name() != "input.pcm" || before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) {
				t.Fatalf("preflight changed files: %v, err=%v", entries, err)
			}
		})
	}
}

func TestLiveCheckRejectsPathsWithoutArtifacts(t *testing.T) {
	for _, name := range []string{"missing input", "directory input", "output exists", "event log exists", "same destinations", "stdout event log", "missing output parent", "missing log parent", "file output parent"} {
		t.Run(name, func(t *testing.T) {
			dir, args := liveCheckFixture(t)
			input := filepath.Join(dir, "input.pcm")
			switch name {
			case "missing input":
				args = append(args, "--input-pcm", filepath.Join(dir, "missing.pcm"))
			case "directory input":
				args = append(args, "--input-pcm", dir)
			case "output exists":
				args = append(args, "--output-pcm", input)
			case "event log exists":
				args = append(args, "--event-log", input)
			case "same destinations":
				args = append(args, "--event-log", filepath.Join(dir, "output.pcm"))
			case "stdout event log":
				args = append(args, "--event-log=-")
			case "missing output parent":
				args = append(args, "--output-pcm", filepath.Join(dir, "missing", "output.pcm"))
			case "missing log parent":
				args = append(args, "--event-log", filepath.Join(dir, "missing", "events.jsonl"))
			case "file output parent":
				args = append(args, "--output-pcm", filepath.Join(input, "output.pcm"))
			}
			out, err := executeLiveCheck(t, args, nil)
			if err == nil || out != "" {
				t.Fatalf("stdout=%q err=%v", out, err)
			}
			assertLiveCheckUnchanged(t, dir)
		})
	}
}

func TestLiveCheckRequiredFlagsHelpAndExitCode(t *testing.T) {
	for _, args := range [][]string{{"live"}, {"live", "--check"}, {"live", "--check=false"}, {"live", "--check", "--input-pcm=missing.pcm"}} {
		var out, errOut bytes.Buffer
		if code := run(t.Context(), args, unreadLiveControls{}, &out, &errOut); code != 1 || out.Len() != 0 || !strings.Contains(errOut.String(), "required flag") {
			t.Fatalf("args=%v code=%d stdout=%q stderr=%q", args, code, out.String(), errOut.String())
		}
	}
	for _, args := range [][]string{{"live", "--help"}, {"live", "--check", "--help"}, {"live", "--check", "--temporary", "--help"}} {
		out, err := executeLiveCheck(t, args, nil)
		if err != nil {
			t.Fatal(err)
		}
		for _, want := range []string{"--check", "offline, without credentials", "output writability", "devices are not verified", "--input-pcm", "--output-pcm"} {
			if !strings.Contains(out, want) {
				t.Errorf("help missing %q: %s", want, out)
			}
		}
	}
}

func TestLiveCheckFalseStillRuns(t *testing.T) {
	server := liveFixtureServer(t)
	defer server.Close()
	dir, args := liveCheckFixture(t)
	args = append(args, "--check=false", "--base-url", server.URL+"/v1/", "--listen-after=0", "--output=json")
	out, _, err := executeTest(t, args, "", nil)
	if err != nil || !strings.Contains(out, `"output_saved":true`) || strings.Contains(out, "local_validation") {
		t.Fatalf("stdout=%q err=%v", out, err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "output.pcm"))
	if err != nil || !bytes.Equal(data, []byte{1, 2, 3, 4}) {
		t.Fatalf("normal run output=%v err=%v", data, err)
	}
}

func TestLiveCheckDefaultsAndRepeatedChecks(t *testing.T) {
	var dials atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		dials.Add(1)
		http.Error(w, "unexpected connection", 500)
	}))
	defer server.Close()
	dir, args := liveCheckFixture(t)
	args = append(args, "--event-log=", "--base-url", server.URL+"/v1/", "-ojson", "--stream=false", "--web-search=false")
	for range 2 {
		out, err := executeLiveCheck(t, args, nil)
		var report livePreflightReport
		if err != nil || json.Unmarshal([]byte(out), &report) != nil || report.SampleRate != 24000 ||
			report.Duration != "1m0s" || report.ListenAfter != "5s" || report.EventLogPath != "" || report.Controls {
			t.Fatalf("stdout=%q err=%v", out, err)
		}
		assertLiveCheckUnchanged(t, dir)
	}
	if dials.Load() != 0 {
		t.Fatalf("preflight made %d Live connections", dials.Load())
	}
}

func TestLiveRunPreservesPathErrorPrecedence(t *testing.T) {
	for _, check := range []bool{false, true} {
		t.Run(liveCheckCredentialState(check), func(t *testing.T) {
			dir, args := liveCheckFixture(t)
			// Historically, the normal run attempts the output temp file before
			// validating the event log. Preserve that error precedence and cleanup.
			args = append(args, "--check=false", "--event-log=-", "--output-pcm", filepath.Join(dir, "missing", "output.pcm"))
			want := "missing"
			if check {
				args = append(args, "--check")
				want = "event-log must be a new regular file path"
			}
			out, err := executeLiveCheck(t, args, nil)
			if err == nil || !strings.Contains(err.Error(), want) || out != "" {
				t.Fatalf("stdout=%q err=%v; want %q", out, err, want)
			}
			assertLiveCheckUnchanged(t, dir)
		})
	}
	dir, args := liveCheckFixture(t)
	args = append(args, "--check=false", "--event-log=-")
	out, err := executeLiveCheck(t, args, nil)
	if err == nil || !strings.Contains(err.Error(), "event-log must be a new regular file path") || out != "" {
		t.Fatalf("stdout=%q err=%v", out, err)
	}
	assertLiveCheckUnchanged(t, dir)
}

func TestLiveCheckPreservesExactHeaderValidation(t *testing.T) {
	// Less than a full RIFF/WAVE header remains eligible raw PCM, just as in a
	// normal run. It is an assumption, not proof of format or audio quality.
	dir, args := liveCheckFixture(t)
	if err := os.WriteFile(filepath.Join(dir, "input.pcm"), []byte("RIFF0000WA"), 0600); err != nil {
		t.Fatal(err)
	}
	args = append(args, "--output=json")
	out, err := executeLiveCheck(t, args, nil)
	if err != nil || !strings.Contains(out, `"local_validation":"passed"`) {
		t.Fatalf("stdout=%q err=%v", out, err)
	}
}
