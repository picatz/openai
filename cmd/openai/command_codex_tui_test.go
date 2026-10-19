package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestCodexTUIHelpAndPreflight(t *testing.T) {
	// Empty PATH is a tripwire: all cases must finish before executable launch.
	t.Setenv("PATH", t.TempDir())
	for _, tc := range []struct {
		args []string
		want string
		code int
	}{
		{[]string{"codex", "tui", "--help"}, "Unsupported server requests", 0},
		{[]string{"codex", "tui"}, "requires terminal input and output", 1},
		{[]string{"codex", "tui", "--output", "json"}, "--output is not supported", 1},
		{[]string{"codex", "tui", "--cd", " "}, "cd must not be empty", 1},
		{[]string{"codex", "tui", "--session", "latest"}, "--session is not supported", 1},
		{[]string{"codex", "tui", "--base-url", "https://invalid.test"}, "--base-url is not supported", 1},
		{[]string{"codex", "tui", "--timeout=-1s"}, "timeout must not be negative", 1},
		{[]string{"codex", "tui", "--model", " "}, "model must not be empty", 1},
		{[]string{"codex", "tui", "--resume", "anything"}, "unknown flag", 1},
		{[]string{"codex", "tui", "--dangerously-bypass-approvals-and-sandbox"}, "unknown flag", 1},
	} {
		t.Run(strings.Join(tc.args, " "), func(t *testing.T) {
			var out, errs bytes.Buffer
			code := run(context.Background(), tc.args, strings.NewReader(""), &out, &errs)
			if code != tc.code || !strings.Contains(out.String()+errs.String(), tc.want) {
				t.Fatalf("code=%d out=%q err=%q", code, out.String(), errs.String())
			}
		})
	}
}

// Reached only through an explicit synthetic subprocess environment. No native
// Codex executable, credentials, configuration, or service is used by this peer.
func runCodexTUITestHelper() {
	capture, err := os.Create(os.Getenv("OPENAI_CODEX_TUI_CAPTURE"))
	if err != nil {
		os.Exit(8)
	}
	defer capture.Close()
	scan := bufio.NewScanner(os.Stdin)
	for scan.Scan() {
		line := append([]byte(nil), scan.Bytes()...)
		_, _ = capture.Write(append(line, '\n'))
		var call struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
		}
		if json.Unmarshal(line, &call) != nil {
			os.Exit(9)
		}
		switch call.Method {
		case "initialize":
			fmt.Printf(`{"id":%s,"result":{"userAgent":"synthetic","codexHome":"/never/read","platformFamily":"unix","platformOs":"linux"}}`+"\n", call.ID)
		case "thread/start":
			fmt.Printf(`{"id":%s,"result":{"thread":{"id":"fake-thread"},"model":"fake-model","cwd":"/synthetic","approvalPolicy":"on-request","approvalsReviewer":"user","sandbox":{"type":"readOnly"}}}`+"\n", call.ID)
		case "turn/start":
			fmt.Printf(`{"id":%s,"result":{"turn":{"id":"fake-turn","status":"inProgress","items":[]}}}`+"\n", call.ID)
			fmt.Println(`{"method":"item/started","params":{"threadId":"fake-thread","turnId":"fake-turn","startedAtMs":0,"item":{"id":"fake-item","type":"agentMessage","text":""}}}`)
			fmt.Println(`{"method":"item/agentMessage/delta","params":{"threadId":"fake-thread","turnId":"fake-turn","itemId":"fake-item","delta":"synthetic answer"}}`)
			// Distinct uppercase bytes prevent the PTY renderer from reusing a
			// prefix of the lowercase streaming delta in its final-screen update.
			fmt.Println(`{"method":"item/completed","params":{"threadId":"fake-thread","turnId":"fake-turn","completedAtMs":1,"item":{"id":"fake-item","type":"agentMessage","text":"FINAL_RESULT_8427"}}}`)
			fmt.Println(`{"method":"turn/completed","params":{"threadId":"fake-thread","turn":{"id":"fake-turn","status":"completed","items":[]}}}`)
		}
	}
}
