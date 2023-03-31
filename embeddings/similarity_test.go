package embeddings

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/picatz/openai"
)

func TestCosignSimilariy(t *testing.T) {
	t.Run("return 1.0 for identical embeddings", func(t *testing.T) {
		sim, err := CosineSimilarity([]float64{1, 2, 3}, []float64{1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}

		if sim != 1.0 {
			t.Fatalf("expected similarity to be 1.0, got %f", sim)
		}
	})

	t.Run("return 0.0 for orthogonal embeddings", func(t *testing.T) {
		sim, err := CosineSimilarity([]float64{1, 0, 0}, []float64{0, 1, 0})
		if err != nil {
			t.Fatal(err)
		}

		if sim != 0.0 {
			t.Fatalf("expected similarity to be 0.0, got %f", sim)
		}
	})

	t.Run("return 0.0 for zero embeddings", func(t *testing.T) {
		sim, err := CosineSimilarity([]float64{0, 0, 0}, []float64{0, 0, 0})
		if err == nil {
			t.Fatal("expected error")
		}

		if sim != 0.0 {
			t.Fatalf("expected similarity to be 0.0, got %f", sim)
		}
	})

	ptrFloat64 := func(f float64) *float64 {
		return &f
	}

	t.Run("a,b", func(t *testing.T) {
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

		tests := []struct {
			name       string
			a          []float64
			b          []float64
			wantApprox *float64
			want       *float64
		}{
			{
				name: "return 1.0 for identical embeddings",
				a:    getEmbedding(t, "Hello world"),
				b:    getEmbedding(t, "Hello world"),
				want: ptrFloat64(1.0), // flakey? 0.9999999999999999
			},
			{
				name:       "hello world and moon",
				a:          getEmbedding(t, "Hello world"),
				b:          getEmbedding(t, "Hello moon"),
				wantApprox: ptrFloat64(0.8),
			},
			{
				name:       "hello world and hello",
				a:          getEmbedding(t, "Hello world"),
				b:          getEmbedding(t, "Hello"),
				wantApprox: ptrFloat64(0.6),
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				got, err := CosineSimilarity(tt.a, tt.b)
				if err != nil {
					t.Fatalf("error: %v", err)
				}

				if tt.want != nil && got != *tt.want {
					t.Fatalf("got: %v, want: %v", got, *tt.want)
				}

				if tt.wantApprox != nil && got < *tt.wantApprox {
					t.Fatalf("got: %v, want: %v", got, *tt.wantApprox)
				}
			})
		}
	})

	t.Run("which these is not like the other?", func(t *testing.T) {
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

		// Things that are similar, and things that are not
		tests := []struct {
			things   []string
			oddIndex int
		}{
			{
				things: []string{
					"I went to the store the other day, and a new brand of cereal caught my eye. Remembering the cereal I ate as a kid, I decided to buy a box.",
					"Chicago style pizza is delicious, but it is bascially a casserole.",
					"I like pancakes, but I prefer waffles. I like the crisp edges of waffles, with a lot of butter and syrup.",
					"Have you seen the price of eggs lately? They are expensive, but you can't make an omelette without them!",
					"The new cereal I bought was delicious. I ate it all in one sitting.",
				},
				oddIndex: 1,
			},
			{
				things: []string{
					"There's a snake in my boots!",
					"Howdy, partner!",
					"Can you help me find my hat?",
					"Sorry Dave, I'm afraid I can't do that.",
				},
				oddIndex: 3,
			},
			{
				things: []string{
					"In a hole in the ground there lived a hobbit.",
					"I'm going on an adventure!",
					"Fly, you fools!",
					"Keep it secret, keep it safe.",
					"Cybersecurity training is important in the modern workplace.",
				},
				oddIndex: 4,
			},
			{
				things: []string{
					"I want to bake a cake.",
					"I don't have any eggs.",
					"I do have some flour.",
					"I do have some sugar.",
					"I do have some butter.",
					"Where do I get a hammer in this game, under the mountain?",
				},
				oddIndex: 5,
			},
		}

		for _, tt := range tests {
			thingToEmbedding := map[string][]float64{}

			for _, thing := range tt.things {
				thingToEmbedding[thing] = getEmbedding(t, thing)
			}

			averages := map[string]float64{}

			for i, thing := range tt.things {
				var sum float64

				for j, otherThing := range tt.things {
					if i == j {
						continue
					}

					sim, err := CosineSimilarity(thingToEmbedding[thing], thingToEmbedding[otherThing])
					if err != nil {
						t.Fatalf("error: %v", err)
					}

					sum += sim

					// fmt.Printf("%s vs %s: %f\n", thing[:10], otherThing[:10], sim)
				}

				avg := sum / float64(len(tt.things)-1)

				averages[thing] = avg

				// fmt.Printf("%s avg: %f\n", thing[:10], avg)
			}

			var lowestAvg float64

			for thing, avg := range averages {
				t.Logf("%s avg: %f\n", thing[:10], avg)

				if lowestAvg == 0.0 || avg < lowestAvg {
					lowestAvg = avg
				}
			}

			var oddOneOut string

			for thing, avg := range averages {
				if avg == lowestAvg {
					oddOneOut = thing
					break
				}
			}

			t.Logf("odd one out: %s", oddOneOut)

			if tt.things[tt.oddIndex] != oddOneOut {
				t.Fatalf("got: %s, want: %s", oddOneOut, tt.things[tt.oddIndex])
			}
		}
	})
}
