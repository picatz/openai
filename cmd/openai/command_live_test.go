package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"github.com/coder/websocket"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func liveFixtureServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/live/sessions" || r.URL.RawQuery != "" {
			t.Error(r.URL)
		}
		socket, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Error(err)
			return
		}
		defer socket.CloseNow()
		send := func(s string) {
			if err := socket.Write(r.Context(), websocket.MessageText, []byte(s)); err != nil {
				t.Error(err)
			}
		}
		first := true
		sentAudio := false
		for {
			_, data, err := socket.Read(r.Context())
			if err != nil {
				return
			}
			var event map[string]json.RawMessage
			if err := json.Unmarshal(data, &event); err != nil {
				t.Error(err)
				return
			}
			var kind string
			json.Unmarshal(event["type"], &kind)
			if first && kind != "session.start" {
				t.Error("first message was not session.start")
			}
			first = false
			switch kind {
			case "session.start":
				if !strings.Contains(string(event["session"]), `"store":false`) || !strings.Contains(string(event["session"]), `"max_output_tokens":1024`) {
					t.Errorf("session=%s", event["session"])
				}
				send(`{"type":"session.started","session":{"id":"live_synthetic"}}`)
			case "session.input_audio.append":
				var audio string
				json.Unmarshal(event["audio"], &audio)
				decoded, err := base64.StdEncoding.DecodeString(audio)
				if err != nil || len(decoded)%2 != 0 {
					t.Error("invalid PCM")
				}
				if !sentAudio {
					send(`{"type":"session.output_audio.delta","delta":"AQIDBA=="}`)
					sentAudio = true
				}
			case "session.input_audio.mute", "session.input_audio.unmute":
				var id string
				json.Unmarshal(event["event_id"], &id)
				ack := "session.input_audio.muted"
				if kind == "session.input_audio.unmute" {
					ack = "session.input_audio.unmuted"
				}
				send(`{"type":"` + ack + `","client_event_id":"` + id + `"}`)
			case "session.close":
				send(`{"type":"session.closed","usage":{"seconds":0.1},"reason":"close_requested"}`)
				return
			}
		}
	}))
}
func TestLiveFileCommand(t *testing.T) {
	server := liveFixtureServer(t)
	defer server.Close()
	dir := t.TempDir()
	input := filepath.Join(dir, "input.pcm")
	output := filepath.Join(dir, "output.pcm")
	os.WriteFile(input, []byte{5, 6, 7, 8}, 0600)
	out, errOut, err := executeTest(t, []string{"live", "--input-pcm", input, "--output-pcm", output, "--base-url", server.URL + "/v1/", "--duration", "1s", "--listen-after", "20ms", "--output", "json"}, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !json.Valid([]byte(out)) || !strings.Contains(out, `"finalized":true`) || !strings.Contains(out, `"output_saved":true`) {
		t.Fatalf("out=%q", out)
	}
	if strings.Contains(errOut, "AQIDBA==") {
		t.Fatal("audio was copied into diagnostics")
	}
	data, err := os.ReadFile(output)
	if err != nil || !bytes.Equal(data, []byte{1, 2, 3, 4}) {
		t.Fatalf("audio=%v err=%v", data, err)
	}
}
func TestLiveRejectsInvalidBeforeConnection(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls.Add(1); http.Error(w, "unexpected", 500) }))
	defer server.Close()
	dir := t.TempDir()
	input := filepath.Join(dir, "input.pcm")
	output := filepath.Join(dir, "output.pcm")
	os.WriteFile(input, []byte{1, 2}, 0600)
	for _, extra := range [][]string{{"--duration", "0s"}, {"--duration", "11m"}, {"--sample-rate", "44100"}, {"--backend-max-output-tokens", "0"}, {"--output-pcm", "-"}, {"--stream"}} {
		args := append([]string{"live", "--input-pcm", input, "--output-pcm", output, "--base-url", server.URL + "/v1/"}, extra...)
		_, _, err := executeTest(t, args, "", nil)
		if err == nil {
			t.Fatalf("accepted %v", extra)
		}
	}
	if calls.Load() != 0 {
		t.Fatalf("made %d unexpected connections", calls.Load())
	}
}
func TestLiveControlsParse(t *testing.T) {
	var diagnostics bytes.Buffer
	controls, finish, err := liveControls(t.Context(), strings.NewReader("bad\nmute\nunmute\nclose\n"), &diagnostics)
	if err != nil {
		t.Fatal(err)
	}
	defer finish()
	var values []string
	for value := range controls {
		values = append(values, string(value))
	}
	if strings.Join(values, ",") != "invalid,mute,unmute,close" {
		t.Fatalf("values=%v logs=%q", values, diagnostics.String())
	}
}

func TestLiveEventLogDoesNotWriteToBlockedStderr(t *testing.T) {
	server := liveFixtureServer(t)
	defer server.Close()
	dir := t.TempDir()
	input := filepath.Join(dir, "input.pcm")
	output := filepath.Join(dir, "output.pcm")
	logPath := filepath.Join(dir, "events.jsonl")
	os.WriteFile(input, []byte{1, 2}, 0600)
	blockedReader, blockedWriter := io.Pipe()
	defer blockedReader.Close()
	defer blockedWriter.Close()
	cmd := newRootCommand()
	cmd.SetArgs([]string{"live", "--input-pcm", input, "--output-pcm", output, "--event-log", logPath, "--base-url", server.URL + "/v1/", "--duration", "1s", "--listen-after", "20ms", "--output", "json"})
	cmd.SetIn(strings.NewReader(""))
	cmd.SetErr(blockedWriter)
	var out bytes.Buffer
	cmd.SetOut(&out)
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- cmd.ExecuteContext(ctx) }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-ctx.Done():
		t.Fatal("blocked diagnostics held the Live command open")
	}
	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "session.started") || !strings.Contains(string(data), "session.closed") || strings.Contains(string(data), "AQIDBA==") {
		t.Fatalf("events=%s", data)
	}
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if !json.Valid([]byte(line)) {
			t.Fatalf("invalid event line %q", line)
		}
	}
}
