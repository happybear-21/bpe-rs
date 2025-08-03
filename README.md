# Byte Pair Encoding (BPE) in Rust

This is a Rust implementation of **Byte Pair Encoding (BPE)**, an industry-standard tokenization algorithm used in NLP models like GPT and BERT.

The implementation in `main.rs` builds a vocabulary by iteratively merging the most frequent pair of tokens in a text corpus, enabling efficient text encoding and decoding.

---

## 🚀 Key Features

- ✅ Trains a BPE model with a specified vocabulary size  
- 🔒 Supports special tokens that are not merged  
- 🔁 Encodes text into token IDs and decodes back to text  
- 💾 Saves and loads models to/from files  
- 🧠 Handles unknown characters gracefully  

---

## 📦 Usage

### 1. Clone the Repository

```bash
git clone https://github.com/happybear-21/bpe-rs.git
```

### 2. Navigate to the Project

```bash
cd bpe-rs
```

### 3. Run the Program

```bash
cargo run
```

---

## 🧪 Example Code

```rust
use std::io;

fn main() -> io::Result<()> {
    let mut bpe = BPE::new();
    let training_text = "hello world hello there hello everyone";
    let vocab_size = 100;
    let special_tokens = vec!["<|endoftext|>".to_string()];
    
    bpe.train(training_text, vocab_size, special_tokens)?;
    let encoded = bpe.encode("hello world");
    let decoded = bpe.decode(&encoded);
    bpe.save("bpe_model.txt")?;
    
    Ok(())
}
```

---

## 🖨️ Sample Output

Running `cargo run` will produce output similar to:

```text
Training BPE model with text: Jack embraced beauty through art and life.
Text to encode: hello world
Encoded token IDs: [15, 121, 52, 52, 65, 22, 29, 65, 54, 52, 114]
Decoded text: hello world
Vocabulary size: 168
Learned merges: 38
Model saved to: vocab.json and bpe_merges.json
Model loaded from: vocab.json and bpe_merges.json
Re-encoded token IDs (after loading): [15, 121, 52, 52, 65, 22, 29, 65, 54, 52, 114]
Re-decoded text (after loading): hello world
```

---

## 📚 Notes
* Includes error handling for file operations and unknown characters.
* For production use, consider:
  * Adding more tests
  * Optimizing for large datasets