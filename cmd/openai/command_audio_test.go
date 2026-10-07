package main

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func syntheticWAV() []byte {
	data := make([]byte, 44+160)
	copy(data, "RIFF")
	binary.LittleEndian.PutUint32(data[4:], uint32(len(data)-8))
	copy(data[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(data[16:], 16)
	binary.LittleEndian.PutUint16(data[20:], 1)
	binary.LittleEndian.PutUint16(data[22:], 1)
	binary.LittleEndian.PutUint32(data[24:], 16000)
	binary.LittleEndian.PutUint32(data[28:], 32000)
	binary.LittleEndian.PutUint16(data[32:], 2)
	binary.LittleEndian.PutUint16(data[34:], 16)
	copy(data[36:], "data")
	binary.LittleEndian.PutUint32(data[40:], 160)
	return data
}
func TestFileTranscriptionMultipart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "synthetic.wav")
	data := syntheticWAV()
	os.WriteFile(path, data, 0600)
	for _, format := range []string{"text", "json"} {
		out, _, err := executeTest(t, []string{"audio", "transcribe", path, "--output", format}, "", func(r *http.Request) (*http.Response, error) {
			if r.Method != "POST" || r.URL.Path != "/v1/audio/transcriptions" {
				t.Error(r.Method, r.URL)
			}
			if err := r.ParseMultipartForm(1 << 20); err != nil {
				t.Fatal(err)
			}
			defer r.MultipartForm.RemoveAll()
			f, header, err := r.FormFile("file")
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			sent, _ := io.ReadAll(f)
			if header.Filename != "synthetic.wav" || header.Header.Get("Content-Type") != "audio/wav" || string(sent) != string(data) {
				t.Error("audio metadata/bytes changed")
			}
			if r.FormValue("model") != "gpt-transcribe" {
				t.Errorf("model=%q", r.FormValue("model"))
			}
			if _, ok := r.MultipartForm.Value["language"]; ok {
				t.Error("empty optional language should be omitted")
			}
			return fakeResponse(r, 200, "application/json", `{"text":"synthetic transcript","usage":{"input_tokens":5},"future":true}`), nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if format == "text" && out != "synthetic transcript\n" || format == "json" && (!json.Valid([]byte(out)) || !strings.Contains(out, `"future":true`)) {
			t.Fatalf("out=%q", out)
		}
	}
}
func TestSpeechWritesCompleteFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "speech.wav")
	data := syntheticWAV()
	out, _, err := executeTest(t, []string{"audio", "speech", "synthetic text", "--file", path, "--format", "wav"}, "", func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/v1/audio/speech" {
			t.Error(r.URL)
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["model"] != "gpt-4o-mini-tts" || body["voice"] != "marin" || body["input"] != "synthetic text" {
			t.Error(body)
		}
		return fakeResponse(r, 200, "audio/wav", string(data)), nil
	})
	if err != nil || out != path+"\n" {
		t.Fatalf("out=%q err=%v", out, err)
	}
	saved, err := os.ReadFile(path)
	if err != nil || string(saved) != string(data) {
		t.Fatalf("saved bytes differ: %v", err)
	}
}
func TestSpeechRejectsInvalidBeforeRequest(t *testing.T) {
	dir := t.TempDir()
	existing := filepath.Join(dir, "exists.mp3")
	os.WriteFile(existing, []byte("keep"), 0600)
	for _, args := range [][]string{
		{"audio", "speech", "text", "--file", existing},
		{"audio", "speech", "text", "--file", ""},
		{"audio", "speech", "text", "--file", filepath.Join(dir, "new"), "--format", "bad"},
		{"audio", "speech", "text", "--file", filepath.Join(dir, "new"), "--speed", "NaN"},
		{"audio", "speech", strings.Repeat("x", 4097), "--file", filepath.Join(dir, "new")},
	} {
		_, _, err := executeTest(t, args, "", func(r *http.Request) (*http.Response, error) { t.Fatal("unexpected request"); return nil, nil })
		if err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
	data, _ := os.ReadFile(existing)
	if string(data) != "keep" {
		t.Fatal("overwrote existing file")
	}
}
func TestSpeechConcurrentTargetPreserved(t *testing.T) {
	path := filepath.Join(t.TempDir(), "speech.wav")
	_, _, err := executeTest(t, []string{"audio", "speech", "text", "--file", path}, "", func(r *http.Request) (*http.Response, error) {
		os.WriteFile(path, []byte("another writer"), 0600)
		return fakeResponse(r, 200, "audio/mpeg", "synthetic audio"), nil
	})
	if err == nil {
		t.Fatal("overwrote concurrent target")
	}
	data, _ := os.ReadFile(path)
	if string(data) != "another writer" {
		t.Fatal("lost concurrent data")
	}
}

type failingAudioBody struct{}

func (failingAudioBody) Read(p []byte) (int, error) { return 0, errors.New("synthetic stream failure") }
func (failingAudioBody) Close() error               { return nil }
func TestFailedSpeechLeavesNoPartialFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "speech.wav")
	_, _, err := executeTest(t, []string{"audio", "speech", "text", "--file", path}, "", func(r *http.Request) (*http.Response, error) {
		resp := fakeResponse(r, 200, "audio/wav", "")
		resp.Body = failingAudioBody{}
		return resp, nil
	})
	if err == nil {
		t.Fatal("expected stream failure")
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 0 {
		t.Errorf("left partial files: %v", entries)
	}
}

func TestTranscriptionRequiresValidTextDocument(t *testing.T) {
	path := filepath.Join(t.TempDir(), "synthetic.wav")
	os.WriteFile(path, syntheticWAV(), 0600)
	for _, body := range []string{`null`, `{}`, `{"text":42}`, `{"text":null}`, `{"text":"ok"} trailing`} {
		_, _, err := executeTest(t, []string{"audio", "transcribe", path}, "", func(r *http.Request) (*http.Response, error) {
			return fakeResponse(r, 200, "application/json", body), nil
		})
		if err == nil {
			t.Errorf("accepted %s", body)
		}
	}
	out, _, err := executeTest(t, []string{"audio", "transcribe", path}, "", func(r *http.Request) (*http.Response, error) {
		return fakeResponse(r, 200, "application/json", `{"text":""}`), nil
	})
	if err != nil || out != "\n" {
		t.Fatalf("valid silence: out=%q err=%v", out, err)
	}
}
