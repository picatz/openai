package openai

// ChatRole is a role that can be used in a chat session, either “system”, “user”, or “assistant”.
//
// https://platform.openai.com/docs/guides/chat/introduction
type ChatRole string

const (
	// ChatRoleUser is a user role.
	ChatRoleUser ChatRole = "user"

	// ChatRoleSystem is a system role.
	ChatRoleSystem ChatRole = "system"

	// ChatRoleAssistant is an assistant role.
	ChatRoleAssistant ChatRole = "assistant"
)
