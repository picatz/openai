package codex

import (
	"encoding/json"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing/object"
)

func TestBuildEnvironment(t *testing.T) {
	t.Setenv(internalOriginatorEnv, "")
	t.Setenv("OPENAI_BASE_URL", "existing")
	t.Setenv("CODEX_API_KEY", "existing")

	env := buildEnvironment("https://example.com", "test-key")

	assertEnvContains := func(key, expected string) {
		t.Helper()
		for _, entry := range env {
			if value, ok := strings.CutPrefix(entry, key+"="); ok {
				if value != expected {
					t.Fatalf("expected %s to be %q, got %q", key, expected, value)
				}
				return
			}
		}
		t.Fatalf("expected environment to contain %s", key)
	}

	assertEnvContains(internalOriginatorEnv, goSDKOriginator)
	assertEnvContains("OPENAI_BASE_URL", "https://example.com")
	assertEnvContains("CODEX_API_KEY", "test-key")
}

func TestExec_Run(t *testing.T) {
	dir := t.TempDir()

	repo, err := git.PlainCloneContext(t.Context(), dir, false, &git.CloneOptions{
		URL: "https://github.com/hashicorp/go-getter.git",
	})
	if err != nil {
		t.Fatalf("failed to clone repo: %v", err)
	}

	pathToCodex, err := findCodexPath()
	if err != nil {
		t.Fatalf("failed to find codex path: %v", err)
	}

	exec, err := NewExec(pathToCodex)
	if err != nil {
		t.Fatalf("failed to create exec: %v", err)
	}

	firstStream, err := exec.Run(t.Context(), Args{
		Input:       "What is the main purpose of this repository?",
		Model:       "gpt-5-codex",
		SandboxMode: SandboxModeWorkspaceWrite,
		WorkingDirectory: func() string {
			w, err := repo.Worktree()
			if err != nil {
				t.Fatalf("failed to get worktree: %v", err)
			}
			return w.Filesystem.Root()
		}(),
	})
	if err != nil {
		t.Fatalf("failed to run codex: %v", err)
	}
	defer firstStream.Close()

	decoder := json.NewDecoder(firstStream.Stdout())

	var threadId string
	for {
		var event ThreadEvent
		if err := decoder.Decode(&event); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("failed to decode event: %v", err)
		}
		if event.Type == EventTypeThreadStarted {
			threadId = event.ThreadID
		}

		t.Logf("Received event: %s: %s", threadId, event.String())
	}

	err = firstStream.Wait()
	if err != nil {
		t.Fatalf("codex execution failed: %v", err)
	}

	secondStream, err := exec.Run(t.Context(), Args{
		Input:    "What language is it written in?",
		ThreadID: threadId,
	})
	if err != nil {
		t.Fatalf("failed to run codex second time: %v", err)
	}
	defer secondStream.Close()

	decoder = json.NewDecoder(secondStream.Stdout())

	for {
		var event ThreadEvent
		if err := decoder.Decode(&event); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("failed to decode event: %v", err)
		}

		t.Logf("Received event: %s: %s", threadId, event.String())
	}

	err = secondStream.Wait()
	if err != nil {
		t.Fatalf("codex execution failed: %v", err)
	}

	thirdStream, err := exec.Run(t.Context(), Args{
		Input:    "Who is the main author of the project?",
		ThreadID: threadId,
	})
	if err != nil {
		t.Fatalf("failed to run codex third time: %v", err)
	}
	defer thirdStream.Close()

	for event, err := range EventStream(t.Context(), thirdStream) {
		if err != nil {
			t.Fatalf("failed to read event: %v", err)
		}
		if event == nil {
			break
		}

		t.Logf("Received event: %s: %s", threadId, event.String())
	}
}

func TestRun(t *testing.T) {
	for event, err := range Run(t.Context(), Args{
		Input:       "Hello!",
		Model:       "gpt-5-codex",
		SandboxMode: SandboxModeReadOnly,
	}) {
		if err != nil {
			t.Fatalf("failed to read event: %v", err)
		}
		if event == nil {
			break
		}

		t.Logf("Received event: %s", event.String())
	}
}

func TestRun_MakeMultipleChanges(t *testing.T) {
	dir := t.TempDir()

	repo, err := git.PlainCloneContext(t.Context(), dir, false, &git.CloneOptions{
		URL: "https://github.com/hashicorp/go-getter.git",
	})
	if err != nil {
		t.Fatalf("failed to clone repo: %v", err)
	}

	var threadID string

	for event, err := range Run(t.Context(), Args{
		Input:           "Hello!",
		Model:           "gpt-5-codex",
		SandboxMode:     SandboxModeReadOnly,
		IncludePlanTool: true,
		FullAuto:        true,
		ThreadID:        threadID,
		WorkingDirectory: func() string {
			w, err := repo.Worktree()
			if err != nil {
				t.Fatalf("failed to get worktree: %v", err)
			}
			return w.Filesystem.Root()
		}(),
	}) {
		if event.Type == EventTypeThreadStarted {
			threadID = event.ThreadID
		}
		if err != nil {
			t.Fatalf("failed to read event: %v", err)
		}
		t.Logf("Received event: %s", event.String())
	}

	for event, err := range Run(t.Context(), Args{
		Input: "Add an AGENTS.md file to this repository." +
			"https://agents.md/ describes the format. https://github.com/openai/agents.md is another useful resource." +
			" Make sure it is useful for an AI agents, but understandable for humans too, as it may be read by both.",
		Model:           "gpt-5-codex",
		SandboxMode:     SandboxModeWorkspaceWrite,
		IncludePlanTool: true,
		FullAuto:        true,
		ThreadID:        threadID,
		WorkingDirectory: func() string {
			w, err := repo.Worktree()
			if err != nil {
				t.Fatalf("failed to get worktree: %v", err)
			}
			return w.Filesystem.Root()
		}(),
	}) {
		if err != nil {
			t.Fatalf("failed to read event: %v", err)
		}
		t.Logf("Received event: %s", event.String())
	}

	// Read the file to verify it was created
	w, err := repo.Worktree()
	if err != nil {
		t.Fatalf("failed to get worktree: %v", err)
	}

	file, err := w.Filesystem.Open("AGENTS.md")
	if err != nil {
		t.Fatalf("failed to open AGENTS.md: %v", err)
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		t.Fatalf("failed to read AGENTS.md: %v", err)
	}

	t.Logf("AGENTS.md content:\n%s", string(content))

	// Now remove the file to demo cleanup
	for event, err := range Run(t.Context(), Args{
		Input:       "Remove the AGENTS.md file you just created.",
		Model:       "gpt-5-codex",
		SandboxMode: SandboxModeWorkspaceWrite,
		ThreadID:    threadID,
		WorkingDirectory: func() string {
			w, err := repo.Worktree()
			if err != nil {
				t.Fatalf("failed to get worktree: %v", err)
			}
			return w.Filesystem.Root()
		}(),
	}) {
		if err != nil {
			t.Fatalf("failed to read event: %v", err)
		}
		t.Logf("Received event: %s", event.String())
	}

	_, err = w.Filesystem.Stat("AGENTS.md")
	if err == nil {
		t.Fatalf("expected AGENTS.md to be removed, but it still exists")
	}

	// Make an output schema file to guide the changes
	schemaFilePath := w.Filesystem.Join(t.TempDir(), "output_schema.json")
	schemaFile, err := os.Create(schemaFilePath)
	if err != nil {
		t.Fatalf("failed to create output schema file: %v", err)
	}
	_, err = schemaFile.Write([]byte(`{
  "type": "object",
  "properties": {
	"workflows_modified": {	
	  "type": "array",
	  "items": {
		"type": "object",
		"properties": {
		  "path": { 
		  	"type": "string",
		  	"description": "The path to the modified workflow file."
		  },
		  "summary_of_changes": { 
		  	"type": "string",
		  	"description": "A summary of the changes made to the workflow file, with a focus on permissions, and the rationale behind them."
		  },
		  "permissions": {
		  	"type": "object",
			"properties": {
				"workflow": {
					"type": "array",
					"items": {
						"type": "string"
					}
				},
				"jobs": {
					"type": "array",
					"items": {
						"type": "object",
						"properties": {
							"id": {
								"type": "string"
							},
							"permissions": {
								"type": "array",
								"items": {
									"type": "string"
								}
							}
						},
						"required": ["id", "permissions"],
						"additionalProperties": false
					}
				}
			},
			"required": ["workflow", "jobs"],
			"additionalProperties": false
		  }
		},
		"required": ["path", "summary_of_changes", "permissions"],
		"additionalProperties": false
	  }
	}
  },
  "required": ["workflows_modified"],
  "additionalProperties": false
}`))
	if err != nil {
		t.Fatalf("failed to write output schema file: %v", err)
	}
	schemaFile.Close()

	// Now make sure all the GitHub Actions workflows have explicit permissions,
	// following GitHub best practices.
	for event, err := range Run(t.Context(), Args{
		Input: "Ensure all GitHub Actions workflows in the .github/workflows directory " +
			"have explicit permissions set as per GitHub best practices. " +
			"Make sure to only modify YAML files in the .github/workflows directory." +
			"You may need to create the permissions section if it does not exist, and investigate " +
			"the least privilege required for each workflow based on the actions it performs." +
			"Refer to https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#permissions " +
			"for more information on setting permissions." +
			"After making the changes, provide a brief summary of what was modified to make it easier to review." +
			"To understand the workflows, you may need to read multiple files and reason about them. " +
			"To assist you, here is a general guideline: " +
			"1. Identify the triggers for each workflow (e.g., push, pull request) and the events they respond to.\n" +
			"2. Determine the permissions required for the actions used in each workflow.\n" +
			"3. Create or update the permissions section in the workflow YAML file accordingly.\n" +
			"4. Test the workflows to ensure they function as expected with the new permissions.\n" +
			"5. Remember to follow the principle of least privilege when assigning permissions. " +
			"6. When in doubt, refer to the official GitHub documentation. " +
			"Take your time to ensure the permissions are set correctly without being overly permissive." +
			`https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#permissions
You can use permissions to modify the default permissions granted to the GITHUB_TOKEN, adding or removing access as required, so that you only allow the minimum required access. For more information, see Use GITHUB_TOKEN for authentication in workflows.

You can use permissions either as a top-level key, to apply to all jobs in the workflow, or within specific jobs. When you add the permissions key within a specific job, all actions and run commands within that job that use the GITHUB_TOKEN gain the access rights you specify. For more information, see jobs.<job_id>.permissions.

Owners of an organization can restrict write access for the GITHUB_TOKEN at the repository level. For more information, see Disabling or limiting GitHub Actions for your organization.

When a workflow is triggered by the pull_request_target event, the GITHUB_TOKEN is granted read/write repository permission, even when it is triggered from a public fork. For more information, see Events that trigger workflows.

For each of the available permissions, shown in the table below, you can assign one of the access levels: read (if applicable), write, or none. write includes read. If you specify the access for any of these permissions, all of those that are not specified are set to none.

Available permissions and details of what each allows an action to do:

Permission		Allows an action using GITHUB_TOKEN to:
---				---
actions			Work with GitHub Actions. For example, actions: write permits an action to cancel a workflow run. For more information, see Permissions required for GitHub Apps.
attestations	Work with artifact attestations. For example, attestations: write permits an action to generate an artifact attestation for a build. For more information, see Using artifact attestations to establish provenance for builds
checks			Work with check runs and check suites. For example, checks: write permits an action to create a check run. For more information, see Permissions required for GitHub Apps.
contents		Work with the contents of the repository. For example, contents: read permits an action to list the commits, and contents: write allows the action to create a release. For more information, see Permissions required for GitHub Apps.
deployments		Work with deployments. For example, deployments: write permits an action to create a new deployment. For more information, see Permissions required for GitHub Apps.
discussions		Work with GitHub Discussions. For example, discussions: write permits an action to close or delete a discussion. For more information, see Using the GraphQL API for Discussions.
id-token		Fetch an OpenID Connect (OIDC) token. This requires id-token: write. For more information, see OpenID Connect
issues			Work with issues. For example, issues: write permits an action to add a comment to an issue. For more information, see Permissions required for GitHub Apps.
models			Generate AI inference responses with GitHub Models. For example, models: read permits an action to use the GitHub Models inference API. See Prototyping with AI models.
packages		Work with GitHub Packages. For example, packages: write permits an action to upload and publish packages on GitHub Packages. For more information, see About permissions for GitHub Packages.
pages			Work with GitHub Pages. For example, pages: write permits an action to request a GitHub Pages build. For more information, see Permissions required for GitHub Apps.
pull-requests	Work with pull requests. For example, pull-requests: write permits an action to add a label to a pull request. For more information, see Permissions required for GitHub Apps.
security-events	Work with GitHub code scanning alerts. For example, security-events: read permits an action to list the code scanning alerts for the repository, and security-events: write allows an action to update the status of a code scanning alert. For more information, see Repository permissions for 'Code scanning alerts'.

## Defining access for the GITHUB_TOKEN scopes

You can define the access that the GITHUB_TOKEN will permit by specifying read, write, or none as the value of the available permissions within the permissions key.

permissions:
  actions: read|write|none
  attestations: read|write|none
  checks: read|write|none
  contents: read|write|none
  deployments: read|write|none
  id-token: write|none
  issues: read|write|none
  models: read|none
  discussions: read|write|none
  packages: read|write|none
  pages: read|write|none
  pull-requests: read|write|none
  security-events: read|write|none
  statuses: read|write|none

If you specify the access for any of these permissions, all of those that are not specified are set to none.

You can use the following syntax to define one of read-all or write-all access for all of the available permissions:

permissions: read-all

permissions: write-all

You can use the following syntax to disable permissions for all of the available permissions:

permissions: {}

## How permissions are calculated for a workflow job

The permissions for the GITHUB_TOKEN are initially set to the default setting for the enterprise, organization, or repository. 
If the default is set to the restricted permissions at any of these levels then this will apply to the relevant repositories. 
For example, if you choose the restricted default at the organization level then all repositories in that organization will use 
the restricted permissions as the default. The permissions are then adjusted based on any configuration within the workflow file, 
first at the workflow level and then at the job level. Finally, if the workflow was triggered by a pull request from a forked 
repository, and the Send write tokens to workflows from pull requests setting is not selected, the permissions are adjusted to 
change any write permissions to read only.

## Setting the GITHUB_TOKEN permissions for all jobs in a workflow

You can specify permissions at the top level of a workflow, so that the setting applies to all jobs in the workflow.

## Principle of least privilege
Any user with write access to your repository has read access to all secrets configured in your repository. 
Therefore, you should ensure that the credentials being used within workflows have the least privileges required.
Actions can use the GITHUB_TOKEN by accessing it from the github.token context. For more information, see Contexts 
reference. You should therefore make sure that the GITHUB_TOKEN is granted the minimum required permissions. 
It's good security practice to set the default permission for the GITHUB_TOKEN to read access only for repository contents. 
The permissions can then be increased, as required, for individual jobs within the workflow file. For more information, see 
Use GITHUB_TOKEN for authentication in workflows.

## Using the GITHUB_TOKEN in a workflow

You can use the GITHUB_TOKEN by using the standard syntax for referencing secrets: ${{ secrets.GITHUB_TOKEN }}. 
Examples of using the GITHUB_TOKEN include passing the token as an input to an action, or using it to make an authenticated 
GitHub API request.

### Important

An action can access the GITHUB_TOKEN through the github.token context even if the workflow does not explicitly pass the 
GITHUB_TOKEN to the action. As a good security practice, you should always make sure that actions only have the minimum access 
they require by limiting the permissions granted to the GITHUB_TOKEN. For more information, see Workflow syntax for GitHub Actions.

https://docs.github.com/en/actions/tutorials/authenticate-with-github_token#using-the-github_token-in-a-workflow

## Modifying the permissions for the GITHUB_TOKEN
Use the permissions key in your workflow file to modify permissions for the GITHUB_TOKEN for an entire workflow or for 
individual jobs. This allows you to configure the minimum required permissions for a workflow or job. As a good security practice, 
you should grant the GITHUB_TOKEN the least required access.

To see the list of permissions available for use and their parameterized names, see Managing your personal access tokens.
`,
		Model:             "gpt-5-codex",
		SandboxMode:       SandboxModeWorkspaceWrite,
		ThreadID:          threadID,
		OutputSchemaFile:  schemaFilePath,
		OutputLastMessage: "output_last_message.json",
		WorkingDirectory: func() string {
			w, err := repo.Worktree()
			if err != nil {
				t.Fatalf("failed to get worktree: %v", err)
			}
			return w.Filesystem.Root()
		}(),
	}) {
		if err != nil {
			t.Fatalf("failed to read event: %v", err)
		}
		t.Logf("Received event: %s", event.String())
	}

	// Read the last message to verify the output schema was followed
	lastMessageFile, err := os.Open("output_last_message.json")
	if err != nil {
		t.Fatalf("failed to open last message file: %v", err)
	}
	defer lastMessageFile.Close()

	content, err = io.ReadAll(lastMessageFile)
	if err != nil {
		t.Fatalf("failed to read last message file: %v", err)
	}

	t.Logf("Last message content:\n%s", string(content))

	var lastMessage struct {
		WorkflowsModified []struct {
			Path             string `json:"path"`
			SummaryOfChanges string `json:"summary_of_changes"`
			Permissions      struct {
				Workflow []string `json:"workflow"`
				Jobs     []struct {
					ID          string   `json:"id"`
					Permissions []string `json:"permissions"`
				}
			} `json:"permissions"`
		} `json:"workflows_modified"`
	}
	if err := json.Unmarshal(content, &lastMessage); err != nil {
		t.Fatalf("failed to unmarshal last message content: %v", err)
	}

	t.Logf("Workflows modified: %v", lastMessage.WorkflowsModified)

	for _, modified := range lastMessage.WorkflowsModified {
		t.Log(modified.SummaryOfChanges)
		t.Log(modified.Path)
		for _, perms := range modified.Permissions.Workflow {
			t.Logf("\tworkflow: %v", perms)
		}
		for _, job := range modified.Permissions.Jobs {
			t.Logf("\tjob %q: %q", job.ID, job.Permissions)
		}
		f, err := w.Filesystem.Open(modified.Path)
		if err != nil {
			t.Fatalf("failed to open modified workflow file: %v", err)
		}
		b, err := io.ReadAll(f)
		if err != nil {
			t.Fatalf("failed to reall all: %v", err)
		}
		t.Log("\n" + string(b) + "\n")
	}

	err = os.Remove("output_last_message.json")
	if err != nil {
		t.Fatal(err)
	}

	status, err := w.Status()
	if err != nil {
		t.Fatalf("failed to get worktree status: %v", err)
	}
	if status.IsClean() {
		t.Fatalf("expected workflow updates to produce changes")
	}

	for path, fileStatus := range status {
		if fileStatus.Worktree == git.Unmodified && fileStatus.Staging == git.Unmodified {
			continue
		}
		if fileStatus.Worktree == git.Deleted || fileStatus.Staging == git.Deleted {
			if _, err := w.Remove(path); err != nil {
				t.Fatalf("failed to remove %s: %v", path, err)
			}
			continue
		}
		if _, err := w.Add(path); err != nil {
			t.Fatalf("failed to stage %s: %v", path, err)
		}
	}

	signature := &object.Signature{
		Name:  "Codex Tests",
		Email: "codex-tests@example.com",
		When:  time.Now(),
	}
	commitHash, err := w.Commit("Use explicit workflow permissions", &git.CommitOptions{
		Author: signature,
	})
	if err != nil {
		t.Fatalf("failed to commit changes: %v", err)
	}
	t.Logf("Committed workflow updates: %s", commitHash.String())

	var summaryBuilder strings.Builder
	summaryBuilder.WriteString("## Summary\n\n")
	if len(lastMessage.WorkflowsModified) == 0 {
		summaryBuilder.WriteString("- No workflow permission updates detected\n")
	} else {
		for _, modified := range lastMessage.WorkflowsModified {
			summaryBuilder.WriteString("- `")
			summaryBuilder.WriteString(modified.Path)
			summaryBuilder.WriteString("`: ")
			summaryBuilder.WriteString(modified.SummaryOfChanges)
			summaryBuilder.WriteByte('\n')

			if len(modified.Permissions.Workflow) > 0 {
				summaryBuilder.WriteString("  - Workflow permissions: ")
				summaryBuilder.WriteString(strings.Join(modified.Permissions.Workflow, ", "))
				summaryBuilder.WriteByte('\n')
			}

			if len(modified.Permissions.Jobs) > 0 {
				summaryBuilder.WriteString("  - Job permissions:\n")
				for _, job := range modified.Permissions.Jobs {
					summaryBuilder.WriteString("    - `")
					summaryBuilder.WriteString(job.ID)
					summaryBuilder.WriteString("`: ")
					summaryBuilder.WriteString(strings.Join(job.Permissions, ", "))
					summaryBuilder.WriteByte('\n')
				}
			}
		}
	}

	prSummary := strings.TrimSuffix(summaryBuilder.String(), "\n")
	t.Logf("Pull request summary:\n%s", prSummary)
}
