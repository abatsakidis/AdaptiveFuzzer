# AdaptiveFuzzer

AdaptiveFuzzer is a Python-based adaptive fuzzing tool that leverages deep learning (PyTorch) to intelligently generate and evolve fuzzing payloads for web application testing. It supports both GET and POST requests, uses a character-level RNN to predict the "interestingness" of payloads, and adapts its payload corpus based on feedback from the target application's responses.

---

## Features

- **Adaptive fuzzing** using a character-level RNN (GRU) implemented in PyTorch.
- Generates and evolves payloads based on their feedback score from HTTP responses.
- Supports both GET and POST methods, sending JSON payloads for POST.
- Parallel payload testing with configurable threading.
- Customizable HTTP headers, cookies, and proxy support.
- Detects error-related keywords and HTTP error codes in responses to score payloads.
- Simple mutation strategies: insert, delete, replace, swap characters in payloads.
- ML-guided payload selection for more effective fuzzing iterations.
- Debug mode to print detailed HTTP request/response info.
- Early stopping if no interesting payloads found in an iteration.

---

## Requirements

- Python 3.7+
- PyTorch
- requests

You can install dependencies via pip:

pip install torch requests

## Usage

python adaptive_fuzzer.py [OPTIONS] URL

# Positional arguments

    URL
    Target URL to fuzz. The tool will send payloads as GET query parameters or POST JSON bodies with the key "input".

# Optional arguments

*    --iterations INT
    Maximum number of fuzzing iterations (default: 20).

*    --delay FLOAT
    Delay in seconds between iterations (default: 1.0).

 *   --threads INT
    Number of parallel threads for sending requests (default: 5).

*    --debug-proxy
    Enable debug output for requests and responses.

 *   --proxy URL
    HTTP/HTTPS proxy URL (e.g. http://127.0.0.1:8080).

*    --cookie KEY=VALUE
    Cookies to include in requests. Can be specified multiple times.

*    --header "Key: Value"
    Additional HTTP headers to include. Can be specified multiple times.

## How It Works

###   Initialization:
    Starts with a predefined set of seed payloads known to trigger common vulnerabilities (SQL injection, XSS, path traversal, etc.).

###    Payload Testing:
    Each payload is sent as either a GET or POST request to the target URL. The response is scanned for error keywords and HTTP error status codes to assign a "score" indicating potential interest.

###    Model Training:
    The character-level RNN model is trained on payloads and their associated scores to learn which payloads are more likely to elicit interesting responses.

###   Payload Evolution:
    Using the model's predictions combined with historic feedback, the tool selects payloads to mutate via character insertion, deletion, replacement, or swapping, expanding the corpus.

###   Iteration:
    Steps 2–4 repeat for a configurable number of iterations or until no new interesting payloads are found.

## Example

python adaptive_fuzzer.py https://example.com/vulnerable-endpoint --iterations 10 --threads 10 --debug-proxy

This will run 10 iterations of adaptive fuzzing against the target URL with 10 concurrent threads and verbose debug output.

## Code Structure

*    CharRNN: PyTorch GRU-based character-level regression model predicting payload "interestingness" scores.

*   AdaptiveFuzzer: Main class that manages payload corpus, mutation, scoring, model training, and HTTP interactions.

*  Mutation methods: Insert, delete, replace, and swap characters in payload strings.

* Multi-threading: Uses ThreadPoolExecutor to send requests concurrently.

*  Scoring: Combines keyword presence and HTTP error codes in response to score payloads.

## Notes

    This tool is intended for authorized security testing and educational purposes only.
    Use responsibly and ensure you have permission to test the target web application.
    The RNN model is lightweight and trains incrementally during fuzzing iterations.


