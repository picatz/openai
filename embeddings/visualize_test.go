package embeddings

import (
	"context"
	"image/png"
	"os"
	"testing"
	"time"

	"github.com/picatz/openai"
)

func TestVisualizePNG(t *testing.T) {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	getEmbedding := func(t *testing.T, input string) []float64 {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		resp, err := client.CreateEmbedding(ctx, &openai.CreateEmbeddingRequest{
			Model: openai.ModelTextEmbeddingAda002,
			Input: input,
		})
		if err != nil {
			t.Fatalf("failed to create embedding: %v", err)
		}

		return resp.Data[0].Embedding
	}

	// a := getEmbedding(t, "Can you tell me about the history of the universe?")
	// b := getEmbedding(t, "How did the universe begin?")
	// c := getEmbedding(t, "I HATE KETCHUP NEVER GIVE ME KETCHUP WHY WOULD YOU DO THAT TO ME I HATE KETCHUP")
	//d := getEmbedding(t, "I love ketchup")
	e := getEmbedding(t, "This is the story all about how my life got flipped turned upside down, so I'd like to take a minute just sit right there, I'll tell you how I became the prince of a town called Bel-Air")
	f := getEmbedding(t, "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort.")
	g := getEmbedding(t, "Fly me to the moon, let me play among the stars, let me see what spring is like on Jupiter and Mars. In other words, hold my hand. In other words, baby, kiss me.")

	embeddings := [][]float64{
		// {},
		e,
		f,
		g,
		// a,
		// b,
		// c,
		// e,
	}

	img, err := Visualize(embeddings, 2, 256, 256)
	if err != nil {
		t.Fatalf("failed to visualize embeddings: %v", err)
	}

	// Delete the file if it already exists
	if _, err := os.Stat("embeddings.png"); err == nil {
		if err := os.Remove("embeddings.png"); err != nil {
			t.Fatalf("failed to remove file: %v", err)
		}
	}

	// Save the image to a file.
	fh, err := os.Create("embeddings.png")
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	defer fh.Close()

	// Encode the image as a PNG.
	if err := png.Encode(fh, img); err != nil {
		t.Fatalf("failed to encode image: %v", err)
	}
}
