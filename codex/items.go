package codex

import (
	"encoding/json"
	"fmt"
)

// ItemType identifies the kind of thread item.
type ItemType string

const (
	ItemTypeAgentMessage     ItemType = "agent_message"
	ItemTypeReasoning        ItemType = "reasoning"
	ItemTypeCommandExecution ItemType = "command_execution"
	ItemTypeFileChange       ItemType = "file_change"
	ItemTypeMcpToolCall      ItemType = "mcp_tool_call"
	ItemTypeWebSearch        ItemType = "web_search"
	ItemTypeTodoList         ItemType = "todo_list"
	ItemTypeError            ItemType = "error"
	ItemTypeUnknown          ItemType = "unknown"
)

// ThreadItem is implemented by all concrete thread item types.
type ThreadItem interface {
	ItemType() ItemType
}

// AgentMessageItem contains the assistant's text response.
type AgentMessageItem struct {
	ID   string   `json:"id"`
	Type ItemType `json:"type"`
	Text string   `json:"text"`
}

func (i *AgentMessageItem) ItemType() ItemType { return ItemTypeAgentMessage }

// ReasoningItem captures an agent's reasoning summary.
type ReasoningItem struct {
	ID   string   `json:"id"`
	Type ItemType `json:"type"`
	Text string   `json:"text"`
}

func (i *ReasoningItem) ItemType() ItemType { return ItemTypeReasoning }

// CommandExecutionStatus represents a command execution state.
type CommandExecutionStatus string

const (
	CommandExecutionStatusInProgress CommandExecutionStatus = "in_progress"
	CommandExecutionStatusCompleted  CommandExecutionStatus = "completed"
	CommandExecutionStatusFailed     CommandExecutionStatus = "failed"
)

// CommandExecutionItem records a shell command executed by the agent.
type CommandExecutionItem struct {
	ID               string                 `json:"id"`
	Type             ItemType               `json:"type"`
	Command          string                 `json:"command"`
	AggregatedOutput string                 `json:"aggregated_output"`
	ExitCode         *int                   `json:"exit_code,omitempty"`
	Status           CommandExecutionStatus `json:"status"`
}

func (i *CommandExecutionItem) ItemType() ItemType { return ItemTypeCommandExecution }

// PatchChangeKind indicates the type of file change.
type PatchChangeKind string

const (
	PatchChangeKindAdd    PatchChangeKind = "add"
	PatchChangeKindDelete PatchChangeKind = "delete"
	PatchChangeKindUpdate PatchChangeKind = "update"
)

// FileUpdateChange describes an individual file operation.
type FileUpdateChange struct {
	Path string          `json:"path"`
	Kind PatchChangeKind `json:"kind"`
}

// PatchApplyStatus indicates the result of applying a patch.
type PatchApplyStatus string

const (
	PatchApplyStatusCompleted PatchApplyStatus = "completed"
	PatchApplyStatusFailed    PatchApplyStatus = "failed"
)

// FileChangeItem aggregates a set of file modifications.
type FileChangeItem struct {
	ID      string             `json:"id"`
	Type    ItemType           `json:"type"`
	Changes []FileUpdateChange `json:"changes"`
	Status  PatchApplyStatus   `json:"status"`
}

func (i *FileChangeItem) ItemType() ItemType { return ItemTypeFileChange }

// McpToolCallStatus reflects the state of an MCP tool invocation.
type McpToolCallStatus string

const (
	McpToolCallStatusInProgress McpToolCallStatus = "in_progress"
	McpToolCallStatusCompleted  McpToolCallStatus = "completed"
	McpToolCallStatusFailed     McpToolCallStatus = "failed"
)

// McpToolCallItem represents an MCP tool call.
type McpToolCallItem struct {
	ID     string            `json:"id"`
	Type   ItemType          `json:"type"`
	Server string            `json:"server"`
	Tool   string            `json:"tool"`
	Status McpToolCallStatus `json:"status"`
}

func (i *McpToolCallItem) ItemType() ItemType { return ItemTypeMcpToolCall }

// WebSearchItem captures a web search initiated by the agent.
type WebSearchItem struct {
	ID    string   `json:"id"`
	Type  ItemType `json:"type"`
	Query string   `json:"query"`
}

func (i *WebSearchItem) ItemType() ItemType { return ItemTypeWebSearch }

// TodoItem describes a single checklist item.
type TodoItem struct {
	Text      string `json:"text"`
	Completed bool   `json:"completed"`
}

// TodoListItem models the agent's running plan.
type TodoListItem struct {
	ID    string     `json:"id"`
	Type  ItemType   `json:"type"`
	Items []TodoItem `json:"items"`
}

func (i *TodoListItem) ItemType() ItemType { return ItemTypeTodoList }

// ErrorItem reflects a non-fatal error surfaced to the user.
type ErrorItem struct {
	ID      string   `json:"id"`
	Type    ItemType `json:"type"`
	Message string   `json:"message"`
}

func (i *ErrorItem) ItemType() ItemType { return ItemTypeError }

// UnknownThreadItem preserves unrecognized item payloads.
type UnknownThreadItem struct {
	// Type holds the raw item type returned by the CLI.
	Type ItemType `json:"type"`
	// Raw retains the original JSON payload for callers that want to decode the item manually.
	Raw json.RawMessage `json:"-"`
}

func (i *UnknownThreadItem) ItemType() ItemType { return ItemTypeUnknown }

// UnmarshalThreadItem decodes a thread item into the corresponding Go type.
func UnmarshalThreadItem(data []byte) (ThreadItem, error) {
	var discriminator struct {
		Type ItemType `json:"type"`
	}
	if err := json.Unmarshal(data, &discriminator); err != nil {
		return nil, err
	}

	switch discriminator.Type {
	case ItemTypeAgentMessage:
		var item AgentMessageItem
		if err := json.Unmarshal(data, &item); err != nil {
			return nil, err
		}
		return &item, nil
	case ItemTypeReasoning:
		var item ReasoningItem
		if err := json.Unmarshal(data, &item); err != nil {
			return nil, err
		}
		return &item, nil
	case ItemTypeCommandExecution:
		var item CommandExecutionItem
		if err := json.Unmarshal(data, &item); err != nil {
			return nil, err
		}
		return &item, nil
	case ItemTypeFileChange:
		var item FileChangeItem
		if err := json.Unmarshal(data, &item); err != nil {
			return nil, err
		}
		return &item, nil
	case ItemTypeMcpToolCall:
		var item McpToolCallItem
		if err := json.Unmarshal(data, &item); err != nil {
			return nil, err
		}
		return &item, nil
	case ItemTypeWebSearch:
		var item WebSearchItem
		if err := json.Unmarshal(data, &item); err != nil {
			return nil, err
		}
		return &item, nil
	case ItemTypeTodoList:
		var item TodoListItem
		if err := json.Unmarshal(data, &item); err != nil {
			return nil, err
		}
		return &item, nil
	case ItemTypeError:
		var item ErrorItem
		if err := json.Unmarshal(data, &item); err != nil {
			return nil, err
		}
		return &item, nil
	case "":
		return nil, fmt.Errorf("thread item missing type discriminator")
	default:
		return &UnknownThreadItem{Type: discriminator.Type, Raw: json.RawMessage(data)}, nil
	}
}
