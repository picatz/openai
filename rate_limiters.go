package openai

import (
	"time"

	"golang.org/x/time/rate"
)

// RateLimiters is a struct that holds all of the rate limiters for the OpenAI API
// that can be used by clients to rate limit their requests.
//
// These are not enforced by the client by default, but can be used to rate limit
// requests to the OpenAI API by calling the `Allow()` method on appropriate limiter
// before making a request.
//
// # Example
//
//	// If the rate limiter allows the request, make the request.
//	if openai.RateLimits.Chat.Requests.Allow() {
//	    resp, err := client.CreatChat(ctx, &openai.CreateChatRequest{
//	        ...
//	    })
//	    ...
//	}
//
//	 // Wait for the rate limiter to allow the request.
//	 for openai.RateLimits.Chat.Requests.Wait(ctx) {
//	     resp, err := client.CreatChat(ctx, &openai.CreateChatRequest{
//	         ...
//	     })
//	     ...
//	 }
type RateLimiters struct {
	Chat struct {
		Requests *rate.Limiter
		Tokens   *rate.Limiter
	}
	Text struct {
		Requests *rate.Limiter
		Tokens   *rate.Limiter
	}
	Embedding struct {
		Requests *rate.Limiter
		Tokens   *rate.Limiter
	}
	Images struct {
		Requests *rate.Limiter
	}
	Audio struct {
		Requests *rate.Limiter
	}
}

// RateLimits is the default rate limiters for the OpenAI API.
//
// # Multiple Organizations
//
// If using multiple organizations, users should create their
// own rate limiters using the `NewRateLimiters()` function.
var RateLimits = NewRateLimiters()

// NewRateLimiters returns a new set of rate limiters for the OpenAI API.
func NewRateLimiters() *RateLimiters {
	rl := &RateLimiters{}

	rl.Chat.Requests.Wait()

	rl.Chat.Requests = rate.NewLimiter(rate.Every(1*time.Minute), 3500)
	rl.Chat.Tokens = rate.NewLimiter(rate.Every(1*time.Minute), 90000)

	rl.Text.Requests = rate.NewLimiter(rate.Every(1*time.Minute), 3500)
	rl.Text.Tokens = rate.NewLimiter(rate.Every(1*time.Minute), 350000)

	rl.Embedding.Requests = rate.NewLimiter(rate.Every(1*time.Minute), 3500)
	rl.Embedding.Tokens = rate.NewLimiter(rate.Every(1*time.Minute), 350000)

	rl.Images.Requests = rate.NewLimiter(rate.Every(1*time.Minute), 50)

	rl.Audio.Requests = rate.NewLimiter(rate.Every(1*time.Minute), 50)

	return rl
}
