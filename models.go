package openai

/*

$ op run -- sh -c 'curl -v https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"' | jq -r '.data[].id'

babbage
ada
text-davinci-002
davinci
text-embedding-ada-002
babbage-code-search-code
text-similarity-babbage-001
text-davinci-001
curie-instruct-beta
babbage-code-search-text
babbage-similarity
text-davinci-003
code-davinci-002
curie-search-query
code-search-babbage-text-001
code-cushman-001
code-search-babbage-code-001
text-ada-001
text-similarity-ada-001
text-davinci-insert-002
ada-code-search-code
ada-similarity
code-search-ada-text-001
text-search-ada-query-001
text-curie-001
text-davinci-edit-001
davinci-search-document
ada-code-search-text
text-search-ada-doc-001
code-davinci-edit-001
davinci-instruct-beta
text-babbage-001
text-similarity-curie-001
code-search-ada-code-001
ada-search-query
text-search-davinci-query-001
curie-similarity
davinci-search-query
text-davinci-insert-001
babbage-search-document
ada-search-document
curie
text-search-babbage-doc-001
text-search-curie-doc-001
text-search-curie-query-001
babbage-search-query
text-search-davinci-doc-001
text-search-babbage-query-001
curie-search-document
text-similarity-davinci-001
audio-transcribe-001
davinci-similarity
cushman:2020-05-03
ada:2020-05-03
babbage:2020-05-03
curie:2020-05-03
davinci:2020-05-03
if-davinci-v2
if-curie-v2
if-davinci:3.0.0
davinci-if:3.0.0
davinci-instruct-beta:2.0.0
text-ada:001
text-davinci:001
text-curie:001
text-babbage:001

*/

// Model is a known OpenAI model identifier.
type Model = string

// https://beta.openai.com/docs/models/finding-the-right-model
const (
	// ModelAda is the Ada model.
	//
	// Ada is usually the fastest model and can perform tasks like parsing text, address correction and certain kinds of classification
	// tasks that don’t require too much nuance. Ada’s performance can often be improved by providing more context.
	//
	// Good at: Parsing text, simple classification, address correction, keywords
	//
	// Note: Any task performed by a faster model like Ada can be performed by a more powerful model like Curie or Davinci.
	//
	// https://beta.openai.com/docs/models/ada
	ModelAda Model = "ada"

	// ModelBabbage is the Babbage model.
	//
	// Babbage can perform straightforward tasks like simple classification. It’s also quite capable when it comes to Semantic Search
	// ranking how well documents match up with search queries.
	//
	// Good at: Moderate classification, semantic search classification
	//
	// https://beta.openai.com/docs/models/babbage
	ModelBabbage Model = "babbage"

	// ModelCurie is the Curie model.
	//
	// Curie is extremely powerful, yet very fast. While Davinci is stronger when it comes to analyzing complicated text, Curie is q
	// uite capable for many nuanced tasks like sentiment classification and summarization. Curie is also quite good at answering
	// questions and performing Q&A and as a general service chatbot.
	//
	// Good at: Language translation, complex classification, text sentiment, summarization
	//
	// https://beta.openai.com/docs/models/curie
	ModelCurie Model = "curie"

	// ModelDavinci is the Davinci model.
	//
	// Davinci is the most capable model family and can perform any task the other models can perform and often with less instruction.
	// For applications requiring a lot of understanding of the content, like summarization for a specific audience and creative content
	// generation, Davinci is going to produce the best results. These increased capabilities require more compute resources, so Davinci
	// costs more per API call and is not as fast as the other models.
	//
	// Another area where Davinci shines is in understanding the intent of text. Davinci is quite good at solving many kinds of logic problems
	// and explaining the motives of characters. Davinci has been able to solve some of the most challenging AI problems involving cause and effect.
	//
	// Good at: Complex intent, cause and effect, summarization for audience
	//
	// https://beta.openai.com/docs/models/davinci
	ModelDavinci Model = "davinci"

	// https://beta.openai.com/docs/models/gpt-3

	// Most capable GPT-3 model. Can do any task the other models can do, often with higher quality, longer output and better instruction-following.
	// Also supports inserting completions within text.
	ModelTextDavinciEdit003 Model = "text-davinci-003"

	// Very capable, but faster and lower cost than Davinci.
	ModelTextCurie001 Model = "text-curie-001"

	// Capable of straightforward tasks, very fast, and lower cost.
	ModelBabbage001 Model = "text-babbage-001"

	// Capable of very simple tasks, usually the fastest model in the GPT-3 series, and lowest cost.
	ModelAda001 Model = "text-ada-001"

	// https://beta.openai.com/docs/models/codex

	// Most capable Codex model. Particularly good at translating natural language to code. In addition to completing code, also supports inserting completions within code.
	ModelCodeDavinci002 Model = "code-davinci-002"

	// Almost as capable as Davinci Codex, but slightly faster. This speed advantage may make it preferable for real-time applications.
	ModelCodeCushman001 Model = "code-cushman-001"

	// Used for the CreateEdit API endpoint.
	ModelTextDavinciEdit001 Model = "text-davinci-edit-001"
	ModelCodeDavinciEdit001 Model = "code-davinci-edit-001"

	// TODO: add more "known" models.
)
