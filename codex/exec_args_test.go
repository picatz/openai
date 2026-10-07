package codex

import (
	"reflect"
	"strings"
	"testing"
)

func TestCommandArgsCurrentCLI(t *testing.T) {
	disabled := false
	args := Args{
		Input: "must stay on stdin", BaseURL: `https://example.invalid/"quoted"`,
		ConfigFile: `legacy=true`, ConfigOverrides: []string{`model="old"`, `features.multi_agent=true`},
		Model: "test-model", ModelReasoningEffort: "high", ApprovalPolicy: ApprovalModeNever,
		WebSearchMode: WebSearchModeDisabled, NetworkAccessEnabled: &disabled,
		SandboxMode: SandboxModeReadOnly, WorkingDirectory: "work with spaces",
		AdditionalDirectories: []string{"extra one", "extra two"}, SkipGitRepoCheck: true,
		OutputSchemaFile: "schema.json", OutputLastMessage: "out.txt", Ephemeral: true,
		Enable: []string{"feature", ""}, ThreadID: "thread-123", Images: []string{"", "image one.png", "image two.png"},
	}
	got, err := args.commandArgs()
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"exec", "--json", "--config", "legacy=true", "--config", `model="old"`, "--config", "features.multi_agent=true",
		"--config", `openai_base_url="https://example.invalid/\"quoted\""`,
		"--config", `model_reasoning_effort="high"`, "--config", `approval_policy="never"`, "--config", `web_search="disabled"`,
		"--config", "sandbox_workspace_write.network_access=false",
		"--model", "test-model", "--sandbox", "read-only", "--cd", "work with spaces",
		"--add-dir", "extra one", "--add-dir", "extra two", "--skip-git-repo-check", "--ephemeral",
		"--output-schema", "schema.json", "--output-last-message", "out.txt", "--enable", "feature",
		"resume", "--image", "image one.png", "--image", "image two.png", "--", "thread-123", "-",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %#v\nwant %#v", got, want)
	}
}

func TestCommandArgsDefaultsAndInvalidLegacyOptions(t *testing.T) {
	got, err := (Args{}).commandArgs()
	if err != nil || !reflect.DeepEqual(got, []string{"exec", "--json", "--", "-"}) {
		t.Fatalf("default args = %v, %v", got, err)
	}
	for _, args := range []Args{{ConfigFile: "config.toml"}, {ConfigOverrides: []string{" =true"}}} {
		if _, err := args.commandArgs(); err == nil {
			t.Fatalf("expected invalid args: %+v", args)
		}
	}
	enabled := true
	for _, value := range []*bool{nil, &enabled} {
		got, err := (Args{NetworkAccessEnabled: value}).commandArgs()
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(strings.Join(got, " "), "network_access=true") != (value != nil) {
			t.Fatalf("args = %v", got)
		}
	}
}

func TestLegacyFlagsRemainPassthrough(t *testing.T) {
	args, err := (Args{FullAuto: true, IncludePlanTool: true}).commandArgs()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(args, []string{"exec", "--json", "--full-auto", "--include-plan-tool", "--", "-"}) {
		t.Fatalf("legacy args = %v", args)
	}
}

func TestResumeIDsCannotIntroduceCLIFlags(t *testing.T) {
	for _, id := range []string{"--last", "--dangerously-bypass-approvals-and-sandbox", "--config=approval_policy=never"} {
		t.Run(id, func(t *testing.T) {
			got, err := (Args{ThreadID: id, Images: []string{"image one.png"}, SandboxMode: SandboxModeReadOnly, ApprovalPolicy: ApprovalModeOnRequest}).commandArgs()
			if err != nil {
				t.Fatal(err)
			}
			want := []string{"exec", "--json", "--config", `approval_policy="on-request"`, "--sandbox", "read-only", "resume", "--image", "image one.png", "--", id, "-"}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("got %#v\nwant %#v", got, want)
			}
		})
	}
}
